import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest

from vllm.kalypso import query_processor, server


def test_served_alias_uses_model_path_for_tokenizer_loading(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        query_processor.KVMemoryManager, "init", lambda model, budget: loaded.append(model)
    )
    monkeypatch.setattr(query_processor, "VLLMExecutor", lambda model: NS(model=model))
    processor = server.CoordinatorQueryProcessor(
        poller=object(), model_name="served-qwen", model_path="/cache/qwen", budget=1000
    )
    assert loaded == ["/cache/qwen"]
    assert processor.model_name == "served-qwen"


def test_coordinator_requires_virtual_pinning(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        query_processor.KVMemoryManager, "init", lambda model, budget: loaded.append(model)
    )
    monkeypatch.setattr(query_processor, "VLLMExecutor", lambda model: NS(model=model))
    with pytest.raises(ValueError, match="requires virtual_pinning=True"):
        server.CoordinatorQueryProcessor(
            poller=object(), model_name="served-qwen", model_path="/cache/qwen",
            budget=1000, virtual_pinning=False,
        )
    assert loaded == []


def test_both_query_routes_keep_the_same_result_shape():
    from vllm.entrypoints.openai import api_server

    app = server.build_app(NS())
    ctx = NS(state=NS(idx=0, worker=NS(name="w0")), input=NS(data="alpha"),
             output=[{"SemMap": "mapped"}])
    app.state.query_processor = NS(model_name="qwen", execute=AsyncMock(return_value=[ctx]))
    request = NS(app=app)
    payload = NS(ops=[], data_path="x.csv", blocking=False)
    results = []
    for routes in (app.routes, api_server.router.routes):
        route = next(r for r in routes if r.path == "/v1/semantic/query")
        results.append(asyncio.run(route.endpoint(payload, request))["results"])
    expected = [{"idx": 0, "worker": "w0", "input_context": "alpha",
                 "right_input_context": None, "outputs": [{"op": "SemMap", "value": "mapped"}]}]
    assert results == [expected, expected]
    assert json.dumps(results[0]) == json.dumps(results[1])
    paths = {r.path for r in api_server.router.routes}
    assert paths.isdisjoint({"/v1/semantic/kv/unpin", "/v1/semantic/kv/abort"})
    assert "/v1/semantic/pinned/unpin" in paths


def test_overlapping_queries_execute_serially():
    async def go():
        app = server.build_app(NS())
        entered, release, second_submitted = (asyncio.Event() for _ in range(3))
        calls = []

        async def execute(request, query, blocking=None):
            calls.append(query.data_path)
            if query.data_path == "first.csv":
                entered.set()
                await release.wait()
            return []

        app.state.query_processor = NS(model_name="qwen", execute=execute)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://coordinator"
        ) as client:
            first = asyncio.create_task(client.post(
                "/v1/semantic/query", json={"ops": [], "data_path": "first.csv"}
            ))
            await asyncio.wait_for(entered.wait(), 1)

            async def second_query():
                second_submitted.set()
                return await client.post(
                    "/v1/semantic/query", json={"ops": [], "data_path": "second.csv"}
                )

            second = asyncio.create_task(second_query())
            await second_submitted.wait()
            try:
                assert calls == ["first.csv"]
                assert not second.done()
            finally:
                release.set()
                responses = await asyncio.gather(first, second)
            assert calls == ["first.csv", "second.csv"]
            assert [r.status_code for r in responses] == [200, 200]
            assert [r.json()["results"] for r in responses] == [[], []]

    asyncio.run(go())


@pytest.mark.parametrize("failure", [RuntimeError("boom"), asyncio.CancelledError()])
def test_query_lock_released_after_failure(failure):
    async def go():
        app = server.build_app(NS())
        app.state.query_processor = NS(
            model_name="qwen", execute=AsyncMock(side_effect=[failure, []])
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://coordinator"
        ) as client:
            payload = {"ops": [], "data_path": "x.csv"}
            with pytest.raises(type(failure)):
                await client.post("/v1/semantic/query", json=payload)
            response = await asyncio.wait_for(
                client.post("/v1/semantic/query", json=payload), 1
            )
            assert response.status_code == 200

    asyncio.run(go())


@pytest.mark.parametrize("fail", [False, True])
def test_query_stats_refresh_after_execution(monkeypatch, capsys, fail):
    counters = {"engine": 10, "cached": 0}

    async def poll_once():
        counters["cached"] = counters["engine"]

    poller = NS(
        poll_once=AsyncMock(side_effect=poll_once),
        totals=lambda names: {name: counters["cached"] for name in names},
        has=lambda name: True,
        per_worker_prefix=lambda: {
            "w0": {"prefix_queries": counters["cached"], "prefix_hits": 0},
            "w1": {"prefix_queries": counters["cached"] * 2, "prefix_hits": counters["cached"]},
        },
    )
    processor = server.CoordinatorQueryProcessor.__new__(server.CoordinatorQueryProcessor)
    processor.__dict__.update(
        poller=poller, blocking=False, virtual_pinning=True,
        executor=object(), pool=None, policy=None,
    )

    async def execute(request, query):
        counters["engine"] = 17
        if fail:
            raise ValueError("bad query")
        return ["result"]

    monkeypatch.setattr(query_processor, "SemanticPlan", lambda *a, **kw: NS(execute=execute))
    monkeypatch.setattr(
        query_processor.KVMemoryManager, "get_instance",
        lambda: NS(reset_stages=lambda: None),
    )
    if fail:
        with pytest.raises(ValueError, match="bad query"):
            asyncio.run(processor.execute(object(), object()))
    else:
        assert asyncio.run(processor.execute(object(), object())) == ["result"]
    assert poller.poll_once.await_count == 2
    line = next(s for s in capsys.readouterr().out.splitlines() if "QUERY_KV_STATS " in s)
    stats = json.loads(line.split("QUERY_KV_STATS ", 1)[1])
    assert stats["prefix_tokens_queried"] == 7
    assert stats["prompt_tokens"] == 7
    # per-worker prefix-cache change over the query (counters went 10 -> 17)
    assert stats["per_worker"] == {
        "w0": {"prefix_tokens_queried": 7, "prefix_tokens_reused": 0, "prefix_hit_fraction": 0.0},
        "w1": {"prefix_tokens_queried": 14, "prefix_tokens_reused": 7, "prefix_hit_fraction": 0.5},
    }
