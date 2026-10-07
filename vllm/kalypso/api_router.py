import asyncio
import uuid

import pydantic
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.kalypso import interface, query, query_processor
from vllm.kalypso.pin_registry import PinnedRequestRegistry
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class UnpinPinnedRequestsRequest(pydantic.BaseModel):
    request_ids: list[str] | None = None


@router.post("/v1/semantic/dummy")
async def dummy(raw_request: Request):
    budget = await raw_request.app.state.engine_client.engine_core.call_utility_async(
        "get_kv_cache_budget"
    )

    return {"budget": budget}


@router.get("/v1/semantic/healthz")
async def semantic_healthz(raw_request: Request):
    return {
        "status": "ok",
        "component": "openai_api_server",
        "query_processor_initialized": hasattr(
            raw_request.app.state, "query_processor"
        ),
    }


@router.get("/v1/semantic/pinned")
async def get_pinned_requests(raw_request: Request):
    pinned_requests = PinnedRequestRegistry.instance().list()
    return {
        "count": len(pinned_requests),
        "pinned_requests": pinned_requests,
    }


@router.get("/v1/semantic/scheduler_state")
async def get_scheduler_state(raw_request: Request):
    try:
        scheduler_state = await asyncio.wait_for(
            raw_request.app.state.engine_client.engine_core.call_utility_async(
                "get_scheduler_state"
            ),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": "engine_unresponsive",
            },
        )

    return scheduler_state


@router.post("/v1/semantic/pinned/unpin")
async def unpin_pinned_requests(
    payload: UnpinPinnedRequestsRequest,
    raw_request: Request,
):
    current_pinned = PinnedRequestRegistry.instance().list()
    current_ids = [item["request_id"] for item in current_pinned]
    current_id_set = set(current_ids)

    request_ids = payload.request_ids if payload.request_ids is not None else current_ids
    request_ids = [
        request_id for request_id in request_ids if request_id in current_id_set
    ]

    if not request_ids:
        return {
            "unpinned_count": 0,
            "unpinned_request_ids": [],
            "remaining_pinned_requests": current_pinned,
        }

    try:
        await asyncio.wait_for(
            raw_request.app.state.engine_client.engine_core.call_utility_async(
                "unpin_requests",
                request_ids,
            ),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": "engine_unresponsive",
                "unpinned_count": 0,
                "requested_request_ids": request_ids,
                "remaining_pinned_requests": PinnedRequestRegistry.instance().list(),
            },
        )

    for request_id in request_ids:
        PinnedRequestRegistry.instance().remove(request_id)
    remaining_pinned = PinnedRequestRegistry.instance().list()

    return {
        "unpinned_count": len(request_ids),
        "unpinned_request_ids": request_ids,
        "remaining_pinned_requests": remaining_pinned,
    }


@router.post("/v1/semantic/query")
async def semantic_query(
    sem_request: interface.SemanticQueryRequest,
    raw_request: Request,
):
    started = asyncio.get_running_loop().time()
    print("=== Semantic Execute Request ===")
    print("Query:")
    print(sem_request.ops)
    print("data_path:")
    print(sem_request.data_path)
    print("================================")
    _query = query.Query(sem_request.ops, sem_request.data_path)
    print("query called")

    processor = raw_request.app.state.query_processor
    out_ctxs = await processor.execute(
        raw_request,
        _query,
        blocking=sem_request.blocking,
    )

    results = []
    for ctx in out_ctxs:
        outputs = []
        for item in ctx.output:
            if isinstance(item, dict) and len(item) == 1:
                op_name, value = next(iter(item.items()))
                outputs.append({"op": op_name, "value": value})
            else:
                outputs.append({"op": "unknown", "value": item})

        results.append(
            {
                "idx": getattr(ctx.state, "idx", None),
                "input_context": getattr(ctx.input, "data", None),
                "right_input_context": getattr(ctx.input, "right_data", None),
                "outputs": outputs,
            }
        )

    elapsed = asyncio.get_running_loop().time() - started
    return {
        "request_id": uuid.uuid4().hex,
        "model_name": processor.model_name,
        "data_path": sem_request.data_path,
        "ops": sem_request.ops,
        "blocking": sem_request.blocking,
        "num_output_rows": len(out_ctxs),
        "latency_sec": round(elapsed, 3),
        "results": results,
    }


@router.post("/v1/semantic/query_ref")
async def semantic_query_ref(
    request: interface.SemanticQueryRequest,
    raw_request: Request,
):
    print("=== Semantic Execute Request ===")
    print("Query:")
    print(request.query)
    print("data_path:")
    print(request.data_path)
    print("================================")
    processor = query_processor.QueryProcessor()
    _query = query.Query(request.query, request.data_path)
    print("query_ref called")
    await processor.execute_ref(raw_request, _query)
    return {
        "predicate_result": "yes",
        "request_id": "test",
    }


def attach_router(app: FastAPI):
    app.include_router(router)


async def init_kalypso_state(engine_client: EngineClient, state: State) -> None:
    logger.info("Starting Semantic Query processor")
    model_name = state.openai_serving_models.model_name()
    per_gpu_budget = await engine_client.engine_core.call_utility_async(
        "get_kv_cache_budget"
    )
    tp_size = engine_client.vllm_config.parallel_config.tensor_parallel_size
    budget = per_gpu_budget * tp_size
    gib = 1024**3
    logger.info(
        "Semantic Query processor KV budget: per_gpu=%.2f GiB tp_size=%s "
        "total=%.2f GiB",
        per_gpu_budget / gib,
        tp_size,
        budget / gib,
    )

    state.query_processor = query_processor.QueryProcessor(
        model_name=model_name, budget=budget
    )
    state.query_processor.start_stuck_monitor(engine_client)


def stop_kalypso_state(state: State) -> None:
    query_processor_ = getattr(state, "query_processor", None)
    if query_processor_ is not None:
        query_processor_.stop_stuck_monitor()
