"""Instrumentation v1 for multi-worker Kalypso.

Three kinds of JSONL rows are produced, sharing one clock (seconds since the
recorder was created, from time.monotonic):

  {"type": "request", ...}       one per LLM request (written by the executor)
  {"type": "queue_sample", ...}  one per worker per poll (written by the poller)
  {"type": "worker_info", ...}   one per worker at startup (written by the server)

Queue sample fields:
  t, worker, waiting, running,
  kv_cache_usage        vllm:kv_cache_usage_perc, a 0-1 fraction (null if the
                        worker does not export it)
  prefix_cache_queries  cumulative vllm:prefix_cache_queries_total (null if absent)
  prefix_cache_hits     cumulative vllm:prefix_cache_hits_total (null if absent)
  The prefix counters are cumulative: subtract the first sample of a run from
  the last to get that run's per-worker queries and hits.

Worker info fields:
  worker, url, model, max_num_seqs, max_num_batched_tokens (null if the worker
  was not started with VLLM_SERVER_DEV_MODE=1), error

Request row fields:
  request_id, worker, operator, stage, position, row_idx, kind ("completion"|"chat"),
  pin, priority, submit_t, start_t, finish_t, prompt_tokens,
  completion_tokens, finish_reason, status ("ok"|"error"|"cancelled"), error

`start_t` is always null in v1: a non-streaming vLLM response does not say when
the engine began running the request. Use the queue samples (waiting/running)
to reason about queueing, or add streaming later to get a first-token time.

No vllm imports here, so this file can be unit-tested without a GPU.
"""

import asyncio
import contextlib
import contextvars
import json
import threading
import time
from typing import Iterable

# ---------------------------------------------------------------------------
# Request metadata: which operator/stage/row is making the LLM call.
#
# Stage.run_task sets this around each operator call. asyncio tasks copy the
# context when created, so concurrent tasks never see each other's values, and
# the executor can read it without any operator signature changing.
# ---------------------------------------------------------------------------

_REQUEST_META: contextvars.ContextVar = contextvars.ContextVar(
    "kalypso_request_meta", default=None
)


@contextlib.contextmanager
def request_meta(**fields):
    token = _REQUEST_META.set(dict(fields))
    try:
        yield
    finally:
        _REQUEST_META.reset(token)


def current_request_meta() -> dict:
    return dict(_REQUEST_META.get() or {})


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class RequestRecorder:
    """Thread-safe, append-only record sink. Keeps rows in memory and, if a
    path is given, also appends them to a JSONL file as they arrive."""

    def __init__(self, path: str | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._t0_mono = time.monotonic()
        self._t0_wall = time.time()
        self._fh = open(path, "a", encoding="utf-8") if path else None
        self._write({"type": "header", "t0_wall": self._t0_wall})

    def now(self) -> float:
        """Seconds since this recorder was created."""
        return time.monotonic() - self._t0_mono

    def record_request(self, **fields) -> None:
        self._write({"type": "request", **fields})

    def record_queue_sample(self, **fields) -> None:
        self._write({"type": "queue_sample", **fields})

    def record_worker_info(self, **fields) -> None:
        self._write({"type": "worker_info", **fields})

    def _write(self, row: dict) -> None:
        with self._lock:
            self._rows.append(row)
            if self._fh is not None:
                self._fh.write(json.dumps(row, sort_keys=True) + "\n")
                self._fh.flush()

    def rows(self, type: str | None = None) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._rows if type is None or r["type"] == type]

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# ---------------------------------------------------------------------------
# Worker concurrency limits (from vLLM's /server_info dev endpoint)
# ---------------------------------------------------------------------------

def parse_server_info(payload) -> dict:
    """Pull the two concurrency caps out of a /server_info?config_format=json
    response. Missing or malformed input gives nulls rather than an error."""
    cfg = payload.get("vllm_config") if isinstance(payload, dict) else None
    sched = cfg.get("scheduler_config") if isinstance(cfg, dict) else None
    sched = sched if isinstance(sched, dict) else {}
    return {
        "max_num_seqs": sched.get("max_num_seqs"),
        "max_num_batched_tokens": sched.get("max_num_batched_tokens"),
    }


async def fetch_server_limits(session, base_url: str) -> dict:
    """GET {base_url}/server_info?config_format=json and return the caps.

    /server_info only exists when the worker was started with
    VLLM_SERVER_DEV_MODE=1. This never raises: on any failure the caps are null
    and `error` says why, so a missing endpoint cannot stop the coordinator.
    """
    try:
        async with session.get(
            f"{base_url.rstrip('/')}/server_info", params={"config_format": "json"}
        ) as resp:
            resp.raise_for_status()
            limits = parse_server_info(await resp.json())
        return {**limits, "error": None}
    except Exception as exc:
        return {"max_num_seqs": None, "max_num_batched_tokens": None, "error": repr(exc)}


# ---------------------------------------------------------------------------
# Per-worker prefix-cache stats for one query
# ---------------------------------------------------------------------------

def per_worker_prefix_delta(before: dict, after: dict) -> dict:
    """Per-worker prefix-cache change between two snapshots.

    Each snapshot is {worker: {"prefix_queries": float, "prefix_hits": float}}
    (cumulative vLLM counters). Key names match the query-level stats
    (prefix_tokens_queried, prefix_tokens_reused, prefix_hit_fraction) so a
    per-worker row reads the same as the whole-query row.
    """
    out = {}
    for worker, now in after.items():
        was = before.get(worker, {})
        queried = max(0, round(now.get("prefix_queries", 0.0) - was.get("prefix_queries", 0.0)))
        reused = max(0, round(now.get("prefix_hits", 0.0) - was.get("prefix_hits", 0.0)))
        out[worker] = {
            "prefix_tokens_queried": queried,
            "prefix_tokens_reused": reused,
            "prefix_hit_fraction": reused / queried if queried else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Per-worker /metrics poller
# ---------------------------------------------------------------------------

class WorkerMetricsPoller:
    """Scrapes each worker's Prometheus /metrics endpoint on a fixed interval.

    * Writes one queue_sample row per worker per poll with the engine's
      waiting and running request counts.
    * Keeps the latest value of any extra metric names in `extra_metrics`
      (summed over label sets) so other code can read cumulative counters
      without scraping again. See totals().
    """

    WAITING = "vllm:num_requests_waiting"
    RUNNING = "vllm:num_requests_running"
    KV_USAGE = "vllm:kv_cache_usage_perc"
    PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
    PREFIX_HITS = "vllm:prefix_cache_hits_total"

    def __init__(
        self,
        workers: dict[str, str],
        recorder: RequestRecorder,
        interval: float = 0.5,
        extra_metrics: Iterable[str] = (),
        scrape_timeout: float = 2.0,
    ):
        self.workers = dict(workers)  # name -> base url
        self.recorder = recorder
        self.interval = interval
        self.scrape_timeout = scrape_timeout
        self._wanted = {
            self.WAITING,
            self.RUNNING,
            self.KV_USAGE,
            self.PREFIX_QUERIES,
            self.PREFIX_HITS,
            *extra_metrics,
        }
        self.latest: dict[str, dict[str, float]] = {n: {} for n in self.workers}
        self.failures: dict[str, int] = {n: 0 for n in self.workers}
        self._session = None
        self._task: asyncio.Task | None = None

    # -- parsing (pure, testable) ------------------------------------------
    def parse(self, text: str) -> dict[str, float]:
        from prometheus_client.parser import text_string_to_metric_families

        sums: dict[str, float] = {}
        for family in text_string_to_metric_families(text):
            for s in family.samples:
                if s.name in self._wanted:
                    sums[s.name] = sums.get(s.name, 0.0) + float(s.value)
        return sums

    # -- scraping ----------------------------------------------------------
    async def _get_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.scrape_timeout)
            )
        return self._session

    async def _scrape(self, name: str, url: str) -> None:
        try:
            session = await self._get_session()
            async with session.get(f"{url}/metrics") as resp:
                resp.raise_for_status()
                text = await resp.text()
        except Exception:
            self.failures[name] += 1
            return
        sums = self.parse(text)
        self.latest[name] = sums
        self.recorder.record_queue_sample(
            t=self.recorder.now(),
            worker=name,
            waiting=sums.get(self.WAITING, 0.0),
            running=sums.get(self.RUNNING, 0.0),
            # None (not 0) when a metric is missing, so "unknown" is never
            # mistaken for "idle" or "no cache hits".
            kv_cache_usage=sums.get(self.KV_USAGE),
            prefix_cache_queries=sums.get(self.PREFIX_QUERIES),
            prefix_cache_hits=sums.get(self.PREFIX_HITS),
        )

    async def poll_once(self) -> None:
        await asyncio.gather(*(self._scrape(n, u) for n, u in self.workers.items()))

    def totals(self, names: Iterable[str]) -> dict[str, float]:
        """Latest value of each metric, summed across workers."""
        out = {n: 0.0 for n in names}
        for sums in self.latest.values():
            for n in out:
                out[n] += sums.get(n, 0.0)
        return out

    def per_worker_prefix(self) -> dict[str, dict[str, float]]:
        """Latest cumulative prefix-cache counters for each worker (not summed)."""
        return {
            name: {
                "prefix_queries": sums.get(self.PREFIX_QUERIES, 0.0),
                "prefix_hits": sums.get(self.PREFIX_HITS, 0.0),
            }
            for name, sums in self.latest.items()
        }

    def has(self, name: str) -> bool:
        return any(name in sums for sums in self.latest.values())

    # -- lifecycle ---------------------------------------------------------
    async def _run(self) -> None:
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._session is not None:
            await self._session.close()
            self._session = None
