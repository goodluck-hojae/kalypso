from pathlib import Path
import asyncio
from time import monotonic

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.query import Query
from vllm.kalypso.controller import SemanticPlan
from vllm.kalypso.execution.vllm_executor import VLLMExecutor
from vllm.kalypso.pin_registry import PinnedRequestRegistry


class QueryProcessor:
    STUCK_CHECK_INTERVAL_SEC = 2.0
    STUCK_CONFIRMATION_COUNT = 2
    UNPIN_COOLDOWN_SEC = 10.0

    def __init__(self, model_name, budget, virtual_pinning: bool = True):
        self.model_name = model_name
        self.virtual_pinning = virtual_pinning
        KVMemoryManager.init(model_name, budget)
        self.executor = VLLMExecutor(model=model_name)
        self._stuck_monitor_task = None
        self._last_unpin_at = 0.0
        self._consecutive_stuck_checks = 0
        self._consecutive_timeout_checks = 0


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

 
    async def execute(self, raw_request, query: Query):
        plan = SemanticPlan(self.executor, virtual_pinning=self.virtual_pinning)
        owner_key = str(id(raw_request))
        try:
            return await plan.execute(raw_request, query)
        finally:
            await self._cleanup_query_pins(raw_request, owner_key)

    def start_stuck_monitor(self, engine_client):
        if self._stuck_monitor_task is None or self._stuck_monitor_task.done():
            self._stuck_monitor_task = asyncio.create_task(
                self._monitor_stuck_scheduler(engine_client)
            )

    async def _monitor_stuck_scheduler(self, engine_client):
        while True:
            await asyncio.sleep(self.STUCK_CHECK_INTERVAL_SEC)

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
