import asyncio
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

from prometheus_client import REGISTRY

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.query import Query
from vllm.kalypso.controller import SemanticPlan
from vllm.kalypso.execution.vllm_executor import VLLMExecutor
from vllm.kalypso.pin_registry import PinnedRequestRegistry
from vllm.v1.engine.exceptions import EngineDeadError


class QueryProcessor:
    STUCK_CHECK_INTERVAL_SEC = 2.0
    STUCK_CONFIRMATION_COUNT = 2
    UNPIN_COOLDOWN_SEC = 10.0
    LOG_QUERY_KV_STATS = True

    _KV_METRICS = {
        "prefix_queries": "vllm:prefix_cache_queries_total",
        "prefix_hits": "vllm:prefix_cache_hits_total",
        # The histogram count is the number of observed block evictions.
        "evicted_blocks": "vllm:kv_block_lifetime_seconds_count",
    }
    _LLM_WORK_METRICS = {
        # Histogram sums aggregate the completed requests belonging to the
        # semantic query. Prompt tokens are the full logical prompt lengths;
        # prefill tokens computed exclude tokens served from the prefix cache.
        "prompt_tokens": "vllm:request_prompt_tokens_sum",
        "prefill_tokens_computed": (
            "vllm:request_prefill_kv_computed_tokens_sum"
        ),
        "decode_tokens": "vllm:generation_tokens_total",
        "prefill_time_seconds": "vllm:request_prefill_time_seconds_sum",
        "decode_time_seconds": "vllm:request_decode_time_seconds_sum",
        "inference_time_seconds": "vllm:request_inference_time_seconds_sum",
        "preemptions": "vllm:num_preemptions_total",
    }

    def __init__(
        self,
        model_name,
        budget,
        virtual_pinning: bool = True,
        blocking: bool = False,
    ):
        self.model_name = model_name
        self.virtual_pinning = virtual_pinning
        self.blocking = blocking
        KVMemoryManager.init(model_name, budget)
        self.executor = VLLMExecutor(model=model_name)
        self._stuck_monitor_task = None
        self._last_unpin_at = 0.0
        self._consecutive_stuck_checks = 0
        self._consecutive_timeout_checks = 0

    @classmethod
    def _kv_metric_snapshot(cls) -> dict[str, float]:
        """Read process-wide cumulative vLLM KV and execution metrics.

        Prefix query/hit counters are token counts. Evictions are exact when
        vLLM is started with ``--kv-cache-metrics-sample 1.0``; otherwise the
        histogram count is only the sampled number of evictions.
        """
        wanted = set(cls._KV_METRICS.values()) | set(cls._LLM_WORK_METRICS.values())
        totals = {name: 0.0 for name in wanted}
        found = set()
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name in wanted:
                    found.add(sample.name)
                    totals[sample.name] += float(sample.value)
        snapshot = {key: totals[name] for key, name in cls._KV_METRICS.items()}
        snapshot["eviction_metric_available"] = float(
            cls._KV_METRICS["evicted_blocks"] in found
        )
        snapshot.update(
            {key: totals[name] for key, name in cls._LLM_WORK_METRICS.items()}
        )
        snapshot["llm_work_metrics_available"] = float(
            all(name in found for name in cls._LLM_WORK_METRICS.values())
        )
        return snapshot

    @staticmethod
    def _query_kv_delta(
        before: dict[str, float], after: dict[str, float]
    ) -> dict[str, int | float]:
        queried = max(0, round(after["prefix_queries"] - before["prefix_queries"]))
        reused = max(0, round(after["prefix_hits"] - before["prefix_hits"]))
        evicted = max(0, round(after["evicted_blocks"] - before["evicted_blocks"]))
        prompt_tokens = max(0, round(after["prompt_tokens"] - before["prompt_tokens"]))
        prefill_tokens_computed = max(
            0,
            round(
                after["prefill_tokens_computed"]
                - before["prefill_tokens_computed"]
            ),
        )
        decode_tokens = max(
            0,
            round(after["decode_tokens"] - before["decode_tokens"]),
        )
        preemptions = max(
            0,
            round(after["preemptions"] - before["preemptions"]),
        )
        return {
            "prefix_tokens_queried": queried,
            "prefix_tokens_reused": reused,
            "prefix_hit_fraction": reused / queried if queried else 0.0,
            "kv_blocks_evicted": evicted,
            "kv_eviction_metric_available": bool(
                after.get("eviction_metric_available", 0.0)
            ),
            # vLLM exposes cache misses, but does not distinguish cold tokens
            # from tokens recomputed after an earlier eviction without keeping
            # extra history in the core.
            "prefix_tokens_computed_on_miss": max(0, queried - reused),
            "prompt_tokens": prompt_tokens,
            "prefill_tokens_computed": prefill_tokens_computed,
            "decode_tokens": decode_tokens,
            "prefill_time_seconds": max(
                0.0,
                after["prefill_time_seconds"] - before["prefill_time_seconds"],
            ),
            "decode_time_seconds": max(
                0.0,
                after["decode_time_seconds"] - before["decode_time_seconds"],
            ),
            "inference_time_seconds": max(
                0.0,
                after["inference_time_seconds"]
                - before["inference_time_seconds"],
            ),
            "preemptions": preemptions,
            "llm_work_metrics_available": bool(
                after.get("llm_work_metrics_available", 0.0)
            ),
        }


    def parse(self, query: Query):
        operations = [] # An operation consists of (data, operator) pairs 
        return operations


    # TODO: Organize operations into a plan
    def plan(self, query: Query):
        print(f"[QueryProcessor] Planning for query: {query.query}")
        return query


    def _data_source(self, raw_request, query: Query):
        path = Path(query.data_path)
 
        if path.suffix.lower() == ".csv":
            for ctx in self._csv_reader(raw_request, path):
                yield ctx

 
    async def execute(
        self,
        raw_request,
        query: Query,
        blocking: bool | None = None,
    ):
        effective_blocking = self.blocking if blocking is None else blocking
        plan = SemanticPlan(
            self.executor,
            virtual_pinning=self.virtual_pinning,
            blocking=effective_blocking,
        )
        owner_key = str(id(raw_request))
        query_id = f"query-{uuid4().hex}"
        started_at = monotonic()
        kv_before = self._kv_metric_snapshot()
        # Each query registers its own stages; drop those of earlier queries so
        # they neither hold capacity nor act as donors/receivers.
        KVMemoryManager.get_instance().reset_stages()
        try:
            return await plan.execute(raw_request, query)
        finally:
            await self._cleanup_query_pins(raw_request, owner_key)
            if self.LOG_QUERY_KV_STATS:
                stats = self._query_kv_delta(
                    kv_before,
                    self._kv_metric_snapshot(),
                )
                stats.update({
                    "query_id": query_id,
                    "elapsed_seconds": monotonic() - started_at,
                    "scope": "process_delta",
                })
                print(
                    "[QueryProcessor] QUERY_KV_STATS "
                    + json.dumps(stats, sort_keys=True)
                )

    def start_stuck_monitor(self, engine_client):
        if self._stuck_monitor_task is None or self._stuck_monitor_task.done():
            self._stuck_monitor_task = asyncio.create_task(
                self._monitor_stuck_scheduler(engine_client)
            )

    def stop_stuck_monitor(self):
        if self._stuck_monitor_task is not None and not self._stuck_monitor_task.done():
            self._stuck_monitor_task.cancel()

    async def _monitor_stuck_scheduler(self, engine_client):
        while True:
            await asyncio.sleep(self.STUCK_CHECK_INTERVAL_SEC)

            if engine_client.errored:
                print("[QueryProcessor] engine is dead; stopping stuck monitor")
                return

            try:
                scheduler_state = await asyncio.wait_for(
                    engine_client.engine_core.call_utility_async(
                        "get_scheduler_state"
                    ),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                self._consecutive_timeout_checks += 1
                if self._consecutive_timeout_checks < self.STUCK_CONFIRMATION_COUNT:
                    continue
                pinned_requests = PinnedRequestRegistry.instance().list()
                if not pinned_requests:
                    continue
                now = monotonic()
                if now - self._last_unpin_at < self.UNPIN_COOLDOWN_SEC:
                    continue
                request_ids = [item["request_id"] for item in pinned_requests]
                print(
                    "[QueryProcessor] scheduler-state RPC timed out repeatedly; "
                    f"unpinning {len(request_ids)} pinned requests"
                )
                try:
                    await asyncio.wait_for(
                        engine_client.engine_core.call_utility_async(
                            "unpin_requests",
                            request_ids,
                        ),
                        timeout=self.STUCK_CHECK_INTERVAL_SEC,
                    )
                except asyncio.TimeoutError:
                    print("[QueryProcessor] timed out while unpinning timeout-triggered pinned requests")
                    continue
                except Exception as exc:
                    print(f"[QueryProcessor] failed to unpin timeout-triggered pinned requests: {exc}")
                    continue

                registry = PinnedRequestRegistry.instance()
                for request_id in request_ids:
                    registry.remove(request_id)

                self._last_unpin_at = monotonic()
                self._consecutive_stuck_checks = 0
                self._consecutive_timeout_checks = 0
                continue
            except EngineDeadError:
                print("[QueryProcessor] engine is dead; stopping stuck monitor")
                return
            except Exception as exc:
                print(f"[QueryProcessor] stuck monitor failed to query scheduler state: {exc}")
                continue

            self._consecutive_timeout_checks = 0

            if scheduler_state.get("is_stuck"):
                self._consecutive_stuck_checks += 1
            else:
                self._consecutive_stuck_checks = 0
                continue

            if self._consecutive_stuck_checks < self.STUCK_CONFIRMATION_COUNT:
                continue

            pinned_requests = PinnedRequestRegistry.instance().list()
            if not pinned_requests:
                continue

            now = monotonic()
            if now - self._last_unpin_at < self.UNPIN_COOLDOWN_SEC:
                continue

            request_ids = [item["request_id"] for item in pinned_requests]
            print(
                "[QueryProcessor] detected stuck scheduler; "
                f"unpinning {len(request_ids)} pinned requests"
            )

            try:
                await asyncio.wait_for(
                    engine_client.engine_core.call_utility_async(
                        "unpin_requests",
                        request_ids,
                    ),
                    timeout=self.STUCK_CHECK_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                print("[QueryProcessor] timed out while unpinning stuck pinned requests")
                continue
            except Exception as exc:
                print(f"[QueryProcessor] failed to unpin stuck pinned requests: {exc}")
                continue

            registry = PinnedRequestRegistry.instance()
            for request_id in request_ids:
                registry.remove(request_id)

            self._last_unpin_at = monotonic()
            self._consecutive_stuck_checks = 0

    async def _cleanup_query_pins(self, raw_request, owner_key: str):
        pinned_requests = PinnedRequestRegistry.instance().list_by_owner(owner_key)
        if not pinned_requests:
            return

        request_ids = [item["request_id"] for item in pinned_requests]
        print(
            "[QueryProcessor] query finished; "
            f"unpinning {len(request_ids)} pinned requests"
        )
        try:
            await asyncio.wait_for(
                raw_request.app.state.engine_client.engine_core.call_utility_async(
                    "unpin_requests",
                    request_ids,
                ),
                timeout=self.STUCK_CHECK_INTERVAL_SEC,
            )
        except asyncio.TimeoutError:
            print("[QueryProcessor] timed out while cleaning up finished-query pinned requests")
            return
        except Exception as exc:
            print(f"[QueryProcessor] failed to clean up finished-query pinned requests: {exc}")
            return

        registry = PinnedRequestRegistry.instance()
        for request_id in request_ids:
            registry.remove(request_id)
