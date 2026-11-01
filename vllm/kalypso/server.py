"""Standalone Kalypso coordinator.

Runs the semantic query processor in its own HTTP server and sends every LLM
request to one of N separate `vllm serve` processes.

    # workers (each its own process / GPU), started with the Kalypso vLLM fork:
    VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve <model> --port 8001 --enable-prefix-caching
    VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve <model> --port 8002 --enable-prefix-caching

    # VLLM_SERVER_DEV_MODE=1 exposes /server_info, which the coordinator reads once
    # per worker to record max_num_seqs and max_num_batched_tokens. It also enables
    # other dev endpoints, so use it only on trusted networks.

    # coordinator:
    python -m vllm.kalypso.server --port 8000 \
        --workers http://localhost:8001,http://localhost:8002 \
        --policy round_robin --record-path runs/requests.jsonl

Clients keep calling POST /v1/semantic/query on the coordinator exactly as
they called it on the single vLLM server.
"""

import argparse
import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request

from vllm.kalypso import interface
from vllm.kalypso import query as query_mod
from vllm.kalypso.instrumentation import (
    RequestRecorder,
    WorkerMetricsPoller,
    fetch_server_limits,
    per_worker_prefix_delta,
)
from vllm.kalypso.query_processor import QueryProcessor
from vllm.kalypso.results import format_results
from vllm.kalypso.workers.http_executor import HttpVLLMExecutor
from vllm.kalypso.workers.policy import make_policy
from vllm.kalypso.workers.pool import WorkerPool, load_config, parse_urls
from vllm.kalypso.workers.router import CallRouter


class CoordinatorQueryProcessor(QueryProcessor):
    """QueryProcessor with the in-process vLLM engine calls replaced by HTTP.

    KV snapshots sum fresh counters scraped from every worker's /metrics.
    """

    def __init__(self, *, poller: WorkerMetricsPoller, model_name, model_path=None, **kwargs):
        if not kwargs.get("virtual_pinning", True):
            raise ValueError(
                "Coordinator requires virtual_pinning=True; HTTP workers do not support real pinning"
            )
        # Served aliases need not be valid tokenizer/config paths.
        super().__init__(model_name=model_path or model_name, **kwargs)
        self.model_name = model_name
        self.poller = poller

    # -- metrics -------------------------------------------------------------
    def _kv_metric_snapshot(self) -> dict[str, float]:
        names = set(self._KV_METRICS.values()) | set(self._LLM_WORK_METRICS.values())
        totals = self.poller.totals(names)
        snap = {key: totals[name] for key, name in self._KV_METRICS.items()}
        snap["eviction_metric_available"] = float(
            self.poller.has(self._KV_METRICS["evicted_blocks"])
        )
        snap.update({key: totals[name] for key, name in self._LLM_WORK_METRICS.items()})
        snap["llm_work_metrics_available"] = float(
            all(self.poller.has(n) for n in self._LLM_WORK_METRICS.values())
        )
        # Per-worker prefix counters, so row affinity can be compared with the
        # per-call policies. The totals above stay summed across workers.
        snap["per_worker_prefix"] = self.poller.per_worker_prefix()
        return snap

    @staticmethod
    def _query_kv_delta(before, after):
        stats = QueryProcessor._query_kv_delta(before, after)
        stats["per_worker"] = per_worker_prefix_delta(
            before.get("per_worker_prefix", {}), after.get("per_worker_prefix", {})
        )
        return stats

    async def _query_kv_snapshot(self):
        await self.poller.poll_once()
        return self._kv_metric_snapshot()


def build_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import aiohttp

        recorder = RequestRecorder(args.record_path)
        from vllm.kalypso.budget import KVMemoryManager

        cfg = load_config(args.config) if args.config else None
        specs = cfg.workers if cfg else parse_urls(args.workers)
        pool = await WorkerPool.discover(
            specs,
            executor_factory=lambda url, model, name: HttpVLLMExecutor(url, model, name=name, recorder=recorder),
            manager_factory=lambda model, budget: KVMemoryManager(model, budget, torch.float16),
        )
        workers = pool.workers
        model_name = workers[0].model

        # Concurrency caps per worker, once at startup. Needs workers started
        # with VLLM_SERVER_DEV_MODE=1; if absent the values are null (not fatal).
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            limits = await asyncio.gather(*(fetch_server_limits(s, w.url) for w in workers))
        app.state.worker_limits = {}
        for w, lim in zip(workers, limits):
            info = {"worker": w.name, "url": w.url, "model": w.model, **lim}
            recorder.record_worker_info(**info)
            app.state.worker_limits[w.name] = lim
            print("[Coordinator] WORKER_LIMITS " + json.dumps(info, sort_keys=True))
            if lim["max_num_seqs"] is None:
                print(
                    f"[Coordinator] WARNING: no concurrency caps for {w.name} "
                    f"({lim['error']}); start workers with VLLM_SERVER_DEV_MODE=1"
                )

        poller = WorkerMetricsPoller(
            {w.name: w.url for w in workers},
            recorder,
            interval=args.poll_interval,
            extra_metrics=[
                *QueryProcessor._KV_METRICS.values(),
                *QueryProcessor._LLM_WORK_METRICS.values(),
            ],
        )

        policy = make_policy(args.policy or (cfg.policy if cfg else None) or "round_robin")
        total_budget = pool.total_capacity()
        if policy.sticky:
            # row_affinity: admit() places each row once; its later calls stay there.
            qp_pool, qp_policy, executor = pool, policy, workers[0].executor
            router = None
        else:
            # Per-call routing: tasks are admitted against the pool total,
            # and each LLM call picks its own worker.
            router = CallRouter(
                pool, policy,
                lambda prompt, max_tokens: (
                    KVMemoryManager.get_instance().token_length(prompt) + max_tokens
                ) * KVMemoryManager.get_instance().bytes_per_token,
            )
            qp_pool, qp_policy, executor = None, None, router

        qp = CoordinatorQueryProcessor(
            model_name=model_name,
            model_path=workers[0].model_path,
            budget=total_budget,
            pool=qp_pool,
            policy=qp_policy,
            poller=poller,
        )
        qp.executor = executor
        app.state.router = router
        app.state.query_processor = qp
        app.state.pool = pool
        app.state.recorder = recorder
        app.state.poller = poller
        await poller.poll_once()
        poller.start()
        print(
            f"[Coordinator] {len(workers)} workers, model={model_name}, "
            f"policy={policy.name}, records={args.record_path}"
        )
        try:
            yield
        finally:
            await poller.stop()
            for w in workers:
                await w.executor.close()
            recorder.close()

    app = FastAPI(lifespan=lifespan)
    app.state.query_lock = asyncio.Lock()

    @app.get("/v1/semantic/healthz")
    async def healthz(raw_request: Request):
        return {
            "status": "ok",
            "component": "kalypso_coordinator",
            "query_processor_initialized": hasattr(raw_request.app.state, "query_processor"),
        }

    @app.get("/v1/semantic/workers")
    async def workers_state(raw_request: Request):
        router = getattr(raw_request.app.state, "router", None)
        if router is not None:
            return router.stats()
        pool = raw_request.app.state.pool
        poller = raw_request.app.state.poller
        return {
            "workers": [
                {
                    "name": w.name,
                    "url": w.url,
                    "inflight": pool.manager(w).total_inflight(),
                    "capacity": pool.manager(w).capacity(),
                    "engine_waiting": poller.latest.get(w.name, {}).get(poller.WAITING),
                    "engine_running": poller.latest.get(w.name, {}).get(poller.RUNNING),
                    "scrape_failures": poller.failures.get(w.name, 0),
                }
                for w in pool.workers
            ]
        }

    @app.post("/v1/semantic/query")
    async def semantic_query(sem_request: interface.SemanticQueryRequest, raw_request: Request):
        started = asyncio.get_running_loop().time()
        _query = query_mod.Query(sem_request.ops, sem_request.data_path)
        processor = raw_request.app.state.query_processor
        async with raw_request.app.state.query_lock:
            out_ctxs = await processor.execute(
                raw_request, _query, blocking=sem_request.blocking
            )

        results = format_results(out_ctxs)

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

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser(description="Kalypso coordinator (multi-worker)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--workers", help="comma-separated worker base URLs")
    src.add_argument("--config", help="YAML/JSON worker pool config")
    p.add_argument("--policy", default=None)
    p.add_argument("--record-path", default=None, help="JSONL file for instrumentation rows")
    p.add_argument("--poll-interval", type=float, default=0.5)
    args = p.parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
