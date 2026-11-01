"""Tests for the instrumentation additions: per-worker KV usage and
prefix-cache counters on queue samples, worker limits from /server_info.

No GPU and no vllm package needed: instrumentation.py is loaded by path.
"""
import asyncio
import importlib.util
import pathlib

from aiohttp import web

_path = pathlib.Path(__file__).resolve().parents[1] / "vllm/kalypso/instrumentation.py"
_spec = importlib.util.spec_from_file_location("kalypso_instrumentation_under_test", _path)
instr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(instr)


def metrics_text(waiting, running, usage=None, queries=None, hits=None):
    lines = [
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {waiting}',
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{engine="0",model_name="m"}} {running}',
    ]
    if usage is not None:
        lines += [
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f'vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} {usage}',
        ]
    if queries is not None:
        lines += [
            "# TYPE vllm:prefix_cache_queries counter",
            f'vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} {queries}',
            f'vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} {hits}',
        ]
    return "\n".join(lines) + "\n"


class FakeWorker:
    """A worker that serves a fixed /metrics page and, optionally, /server_info."""

    def __init__(self, metrics, server_info=None):
        self.metrics = metrics
        self.server_info = server_info
        app = web.Application()
        app.router.add_get("/metrics", self._metrics)
        if server_info is not None:
            app.router.add_get("/server_info", self._info)
        self.app = app
        self.info_queries = []

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def stop(self):
        await self.runner.cleanup()

    async def _metrics(self, req):
        return web.Response(text=self.metrics)

    async def _info(self, req):
        self.info_queries.append(dict(req.query))
        return web.json_response(self.server_info)


def run(coro):
    return asyncio.run(coro)


def test_queue_samples_carry_per_worker_kv_usage_and_prefix_counters():
    async def go():
        a = FakeWorker(metrics_text(3, 2, usage=0.25, queries=1000, hits=400))
        b = FakeWorker(metrics_text(0, 5, usage=0.80, queries=50, hits=5))
        await a.start()
        await b.start()
        rec = instr.RequestRecorder()
        poller = instr.WorkerMetricsPoller({"w0": a.url, "w1": b.url}, rec)
        try:
            await poller.poll_once()
        finally:
            await poller.stop()
            await a.stop()
            await b.stop()
        by = {s["worker"]: s for s in rec.rows("queue_sample")}
        # per-worker values, never summed across workers
        assert (by["w0"]["kv_cache_usage"], by["w1"]["kv_cache_usage"]) == (0.25, 0.80)
        assert (by["w0"]["prefix_cache_queries"], by["w0"]["prefix_cache_hits"]) == (1000, 400)
        assert (by["w1"]["prefix_cache_queries"], by["w1"]["prefix_cache_hits"]) == (50, 5)
        # existing fields are unchanged
        assert (by["w0"]["waiting"], by["w0"]["running"]) == (3.0, 2.0)

    run(go())


def test_missing_metrics_are_null_not_zero():
    async def go():
        w = FakeWorker(metrics_text(1, 1))  # no kv usage, no prefix counters
        await w.start()
        rec = instr.RequestRecorder()
        poller = instr.WorkerMetricsPoller({"w0": w.url}, rec)
        try:
            await poller.poll_once()
        finally:
            await poller.stop()
            await w.stop()
        (s,) = rec.rows("queue_sample")
        assert s["kv_cache_usage"] is None
        assert s["prefix_cache_queries"] is None and s["prefix_cache_hits"] is None

    run(go())


def test_record_worker_info_row_type():
    rec = instr.RequestRecorder()
    rec.record_worker_info(worker="w0", max_num_seqs=256)
    (row,) = rec.rows("worker_info")
    assert row["worker"] == "w0" and row["max_num_seqs"] == 256


# ---------------------------------------------------------------------------
# Worker concurrency limits (/server_info)
# ---------------------------------------------------------------------------

SERVER_INFO = {
    "vllm_config": {
        "scheduler_config": {"max_num_seqs": 128, "max_num_batched_tokens": 8192},
        "cache_config": {"block_size": 16},
    }
}


def test_parse_server_info():
    assert instr.parse_server_info(SERVER_INFO) == {
        "max_num_seqs": 128,
        "max_num_batched_tokens": 8192,
    }
    nulls = {"max_num_seqs": None, "max_num_batched_tokens": None}
    assert instr.parse_server_info({"vllm_config": "text form, not json"}) == nulls
    assert instr.parse_server_info({"vllm_config": {}}) == nulls
    assert instr.parse_server_info(None) == nulls


def test_fetch_server_limits_asks_for_json_and_returns_caps():
    async def go():
        import aiohttp

        w = FakeWorker(metrics_text(0, 0), server_info=SERVER_INFO)
        await w.start()
        try:
            async with aiohttp.ClientSession() as s:
                lim = await instr.fetch_server_limits(s, w.url + "/")
        finally:
            await w.stop()
        assert lim == {"max_num_seqs": 128, "max_num_batched_tokens": 8192, "error": None}
        assert w.info_queries == [{"config_format": "json"}]

    run(go())


def test_fetch_server_limits_without_dev_mode_is_null_not_fatal():
    async def go():
        import aiohttp

        w = FakeWorker(metrics_text(0, 0))  # no /server_info route -> 404
        await w.start()
        try:
            async with aiohttp.ClientSession() as s:
                lim = await instr.fetch_server_limits(s, w.url)
        finally:
            await w.stop()
        assert lim["max_num_seqs"] is None and lim["max_num_batched_tokens"] is None
        assert "404" in lim["error"]

    run(go())


# ---------------------------------------------------------------------------
# Per-worker prefix-cache change over one query
# ---------------------------------------------------------------------------

def test_per_worker_prefix_delta_keeps_workers_separate():
    before = {
        "w0": {"prefix_queries": 100, "prefix_hits": 10},
        "w1": {"prefix_queries": 0, "prefix_hits": 0},
    }
    after = {
        "w0": {"prefix_queries": 300, "prefix_hits": 110},
        "w1": {"prefix_queries": 50, "prefix_hits": 0},
        "w2": {"prefix_queries": 8, "prefix_hits": 4},  # worker with no earlier snapshot
    }
    assert instr.per_worker_prefix_delta(before, after) == {
        "w0": {"prefix_tokens_queried": 200, "prefix_tokens_reused": 100, "prefix_hit_fraction": 0.5},
        "w1": {"prefix_tokens_queried": 50, "prefix_tokens_reused": 0, "prefix_hit_fraction": 0.0},
        "w2": {"prefix_tokens_queried": 8, "prefix_tokens_reused": 4, "prefix_hit_fraction": 0.5},
    }


def test_per_worker_prefix_delta_never_negative_and_no_divide_by_zero():
    before = {"w0": {"prefix_queries": 100, "prefix_hits": 50}}
    after = {"w0": {"prefix_queries": 100, "prefix_hits": 40}}  # counter reset / restart
    assert instr.per_worker_prefix_delta(before, after)["w0"] == {
        "prefix_tokens_queried": 0, "prefix_tokens_reused": 0, "prefix_hit_fraction": 0.0,
    }


def test_poller_per_worker_prefix_reads_latest_counters():
    async def go():
        a = FakeWorker(metrics_text(0, 0, usage=0.1, queries=10, hits=3))
        b = FakeWorker(metrics_text(0, 0, usage=0.2, queries=20, hits=9))
        await a.start()
        await b.start()
        poller = instr.WorkerMetricsPoller({"w0": a.url, "w1": b.url}, instr.RequestRecorder())
        try:
            await poller.poll_once()
        finally:
            await poller.stop()
            await a.stop()
            await b.stop()
        assert poller.per_worker_prefix() == {
            "w0": {"prefix_queries": 10.0, "prefix_hits": 3.0},
            "w1": {"prefix_queries": 20.0, "prefix_hits": 9.0},
        }

    run(go())
