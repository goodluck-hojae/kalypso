import asyncio
# import math
import time

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso import op_usage
from vllm.kalypso.context import RetryTaskResult
from vllm.kalypso.controller.stage import Task
from vllm.kalypso.sem_ops import OpBehavior, ops


class PlanExecutor:

    def __init__(self):
        self.pipeline_executor = AsyncPipelineExecutor()
        self.blocking_executor = BlockingExecutor()

    async def execute(self, ctxs, plan):

        stage_stat_list = []
        for item in plan:

            # BLOCKING
            if isinstance(item, ops.BaseOp) and item.behavior == OpBehavior.BLOCKING:
                operator_name = item.__class__.__name__
                input_count = len(ctxs)
                started_at = time.perf_counter()
                print(
                    "[blocking-operator] start "
                    f"operator={operator_name} "
                    f"position={item.position} "
                    f"input_tuples={input_count}"
                )
                token = op_usage.current_op.set(f"b.{item.position}:{operator_name}")
                try:
                    ctxs = await item(ctxs)
                except Exception as exc:
                    elapsed = time.perf_counter() - started_at
                    print(
                        "[blocking-operator] failed "
                        f"operator={operator_name} "
                        f"position={item.position} "
                        f"input_tuples={input_count} "
                        f"elapsed_seconds={elapsed:.3f} "
                        f"error={type(exc).__name__}"
                    )
                    raise
                finally:
                    op_usage.current_op.reset(token)
                elapsed = time.perf_counter() - started_at
                print(
                    "[blocking-operator] done "
                    f"operator={operator_name} "
                    f"position={item.position} "
                    f"input_tuples={input_count} "
                    f"output_tuples={len(ctxs)} "
                    f"elapsed_seconds={elapsed:.3f}"
                )

            elif isinstance(item, ops.BaseOp) and item.behavior == OpBehavior.JOIN:
                next_ctxs = []
                for ctx in ctxs:
                    next_ctxs.extend(item(ctx) or [])
                ctxs = next_ctxs

            # STAGES
            else:
                ctxs, stage_stat = await self.pipeline_executor.execute_tasks(
                    ctxs,
                    item,
                )
                stage_stat_list.append(stage_stat)

        return ctxs, stage_stat_list


class JoinTracker:
    LOG_JOIN_TRACKER = False

    def __init__(
        self,
        parent_ctx,
        pending_children: int,
        manager: KVMemoryManager | None = None,
        stage_id: int | None = None,
        reserved_budget: int = 0,
        on_release=None,
    ):
        self.parent_ctx = parent_ctx
        self.pending_children = pending_children
        self.manager = manager
        self.stage_id = stage_id
        self.reserved_budget = reserved_budget
        self.on_release = on_release
        self.lock = asyncio.Lock()

    async def add_children(self, count: int):
        # A tracked descendant fanned out: it is replaced by `count` tasks that
        # each carry this tracker, so the parent stays pinned until all of them finish.
        async with self.lock:
            self.pending_children += count

    async def child_finished(self):
        async with self.lock:
            self.pending_children -= 1
            remaining = self.pending_children
            should_finalize = self.pending_children == 0

        if self.LOG_JOIN_TRACKER and self.stage_id is not None:
            print(
                "[join-tracker] "
                f"stage={self.stage_id} "
                f"parent_pin={self.parent_ctx.state.pin_req_id} "
                f"remaining_children={remaining} "
                f"reserved_budget={self.reserved_budget}"
            )

        if not should_finalize:
            return

        if self.parent_ctx.state.pin_req_id is not None:
            await self.parent_ctx.state.executor.unpin(
                self.parent_ctx.state.raw_request,
                self.parent_ctx.state.pin_req_id,
            )
            self.parent_ctx.state.pin_req_id = None

        if (
            self.manager is not None
            and self.stage_id is not None
            and self.reserved_budget > 0
        ):
            if self.LOG_JOIN_TRACKER:
                print(
                    "[join-tracker] "
                    f"release stage={self.stage_id} "
                    f"parent_pin={self.parent_ctx.state.pin_req_id} "
                    f"reserved_budget={self.reserved_budget}"
                )
            await self.manager.release_stage(self.stage_id, self.reserved_budget)
            if self.on_release is not None:
                self.on_release(self.stage_id, self.reserved_budget)


class AsyncPipelineExecutor:
    LOG_SCHEDULER = True
    LOG_SCHEDULER_INTERVAL_SEC = 1.0
    LOG_RETRY_TRACE = False
    LOG_DEFERRAL_TRACE = False
    # Pipeline-only ablation: submit every ready task directly to vLLM and let
    # vLLM perform its own queueing, preemption, and KV-cache management.
    MEMORY_ADMISSION_CONTROL = True

    def __init__(self):
        self.manager = KVMemoryManager.get_instance()

    async def execute_tasks(self, ctxs, stages):
        """
        ctxs: initial contexts
        stages: [stage1, stage2, ...]
        """
        stage_stat = {}
        out = []
        active = {}
        last_scheduler_log_at = 0.0
        deferred_parent_counts = {
            stage.stage_id: 0
            for stage in stages
        }
        deferred_parent_bytes = {
            stage.stage_id: 0
            for stage in stages
        }
        retry_counts = {
            stage.stage_id: 0
            for stage in stages
        }
        total_retries = 0

        def add_deferred_parent(stage_id: int, reserved_budget: int) -> None:
            deferred_parent_counts[stage_id] = (
                deferred_parent_counts.get(stage_id, 0) + 1
            )
            deferred_parent_bytes[stage_id] = (
                deferred_parent_bytes.get(stage_id, 0) + reserved_budget
            )

        def release_deferred_parent(stage_id: int, reserved_budget: int) -> None:
            deferred_parent_counts[stage_id] = max(
                0,
                deferred_parent_counts.get(stage_id, 0) - 1,
            )
            deferred_parent_bytes[stage_id] = max(
                0,
                deferred_parent_bytes.get(stage_id, 0) - reserved_budget,
            )

        def record_retry(stage_id: int) -> None:
            nonlocal total_retries
            retry_counts[stage_id] = retry_counts.get(stage_id, 0) + 1
            total_retries += 1

        def log_running_tasks(event: str):
            nonlocal last_scheduler_log_at
            if not self.LOG_SCHEDULER:
                return
            now = time.monotonic()
            if now - last_scheduler_log_at < self.LOG_SCHEDULER_INTERVAL_SEC:
                return
            last_scheduler_log_at = now
            # Small table per event, e.g.
            # [sched] 08:03:11 finish task=12 stage=2
            # stage   used/cap GB  wait   run
            #     1   25.7/25.8     325     3  FULL
            # defer GB = memory held by deferred tasks; FULL = has waiting tasks and memory >= 90% used.
            lines = [
                f"[sched] {time.strftime('%H:%M:%S')} {event}",
                f"{'stage':>5} {'used/cap GB':>13} {'wait':>5} {'run':>5} {'defer GB':>8}",
            ]
            for stage in stages:
                used, cap = self.manager.stage_usage(stage.stage_id)
                wait, run = len(stage.waiting_tasks), len(stage.running_tasks)
                full = "  FULL" if cap > 0 and used >= 0.9 * cap and wait > 0 else ""
                lines.append(
                    f"{stage.stage_id:>5}   {used / 1e9:5.1f}/{cap / 1e9:<5.1f} "
                    f"{wait:>5} {run:>5} "
                    f"{deferred_parent_bytes.get(stage.stage_id, 0) / 1e9:8.1f}{full}"
                )
            print("\n".join(lines))

        async def finalize_task(task: Task):
            for tracker in task.trackers:
                await tracker.child_finished()

        async def cancel_active_tasks():
            # A task failed (e.g. the client disconnected): stop the other
            # in-flight tasks so none outlive the query's pin cleanup.
            for future in active:
                future.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            for future, (stage, task) in list(active.items()):
                await stage.release_budget(stage.detach_budget(task), self.manager)
            active.clear()

        def record_input(stage_id: int):
            if stage_id in stage_stat:
                stage_stat[stage_id]["input"] += 1
            else:
                stage_stat[stage_id] = {"input": 1, "output": 0}

        def record_output(stage_id: int, count: int):
            if stage_id not in stage_stat:
                stage_stat[stage_id] = {"input": 0, "output": 0}
            stage_stat[stage_id]["output"] += count

        def has_pending_work():
            return any(stage.has_waiting_tasks() for stage in stages)

        def pick_donor(receiver):
            # 1st choice: starving stages, 2nd: busy stages (from the last stage);
            # saturated stages never donate. Only free capacity can move.
            for is_candidate in (lambda st: st.is_starving(), lambda st: st.is_busy()):
                for stage in reversed(stages):
                    if (
                        stage is not receiver
                        and is_candidate(stage)
                        and self.manager.stage_free(stage.stage_id) > 0
                    ):
                        return stage
            return None

        async def rebalance_stage_capacity():
            if not self.MEMORY_ADMISSION_CONTROL:
                return

            for stage in stages:
                stage.tune_thresholds()

            def no_free_mem(stage):
                head = stage.peek_task()
                if head is None:
                    return False
                used, cap = self.manager.stage_usage(stage.stage_id)
                return used + stage.estimate_budget(head) > cap

            asked = set()

            def memory_donor(receiver):
                for st in stages:
                    if (
                        st is not receiver
                        and st.stage_id not in asked
                        and self.manager.stage_free(st.stage_id) > 0
                    ):
                        return st
                candidates = [
                    st for st in reversed(stages)
                    if st is not receiver
                    and not no_free_mem(st)
                    and self.manager.stage_free(st.stage_id) > 0
                ]
                candidates.sort(key=lambda st: st.ready_count() >= st.low_threshold)
                return candidates[0] if candidates else None

            async def more_running(idx):
                if idx < 0:
                    return False
                stage = stages[idx]
                asked.add(stage.stage_id)
                moved = False
                if stage.ready_count() < stage.low_threshold:
                    moved = await more_running(idx - 1)
                if no_free_mem(stage):
                    donor = memory_donor(stage)
                    if donor is not None:
                        moved = await self.manager.transfer_capacity(
                            donor.stage_id,
                            stage.stage_id,
                            upstream=stages.index(donor) > idx,
                        ) or moved
                return moved

            changed = await more_running(len(stages) - 1)

            if changed:
                log_running_tasks("rebalance")

        def blocked_stage_details():
            details = {}
            for stage in stages:
                used, cap = self.manager.stage_usage(stage.stage_id)
                head_task = stage.peek_task()
                head_budget = None
                head_task_id = None
                admissible = None
                if head_task is not None:
                    head_task_id = head_task.task_id
                    head_budget = stage.estimate_budget(head_task)
                    admissible = used + head_budget <= cap
                details[stage.stage_id] = {
                    "used": used,
                    "cap": cap,
                    "waiting": len(stage.waiting_tasks),
                    "running": len(stage.running_tasks),
                    "head_task_id": head_task_id,
                    "head_budget": head_budget,
                    "head_admissible": admissible,
                }
            return details

        async def launch_ready_tasks():
            launched = False

            for stage in stages:
                while stage.has_waiting_tasks():
                    task = stage.peek_task()
                    if task is None:
                        break

                    # refresh_retry_max_tokens(stage, task)
                    if self.MEMORY_ADMISSION_CONTROL:
                        if await stage.accept(task, self.manager) is None:
                            break
                    else:
                        stage.accept_unconditionally(task)

                    record_input(stage.stage_id)
                    stage.pop_task()
                    log_running_tasks(
                        f"start task={task.task_id} stage={stage.stage_id}"
                    )
                    future = asyncio.create_task(stage.run_task(task))
                    active[future] = (stage, task)
                    launched = True

            return launched

        async def force_launch_blocked_tasks():
            launched = False

            for stage in stages:
                task = stage.peek_task()
                if task is None:
                    continue

                record_input(stage.stage_id)
                stage.pop_task()
                # refresh_retry_max_tokens(stage, task)
                await stage.force_accept(task, self.manager)
                log_running_tasks(
                    f"force-start task={task.task_id} stage={stage.stage_id}"
                )
                future = asyncio.create_task(stage.run_task(task))
                active[future] = (stage, task)
                launched = True

            return launched

        for stage in stages:
            stage.clear_tasks()

        if stages:
            for ctx in ctxs:
                stages[0].enqueue(Task(ctx=ctx, stage_index=0))

        while has_pending_work() or active:
            await rebalance_stage_capacity()
            await launch_ready_tasks()

            if not active:
                if has_pending_work():
                    forced = await force_launch_blocked_tasks()
                    if not forced:
                        raise RuntimeError(
                            "No runnable tasks remain, but stage queues are not empty. "
                            f"Blocked stage details: {blocked_stage_details()}"
                        )
                    continue
                break

            done, _ = await asyncio.wait(
                set(active.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for future in done:
                stage, task = active.pop(future)
                budget = stage.detach_budget(task)
                try:
                    result = future.result()
                except Exception:
                    await stage.release_budget(budget, self.manager)
                    log_running_tasks(
                        f"finish task={task.task_id} stage={stage.stage_id}"
                    )
                    await cancel_active_tasks()
                    raise

                if isinstance(result, RetryTaskResult):
                    if self.LOG_RETRY_TRACE:
                        print(
                            "[scheduler-retry] "
                            f"task={task.task_id} "
                            f"stage={stage.stage_id} "
                            f"op_index={result.op_index} "
                            f"budget={budget} "
                            f"retain_budget={result.retain_budget} "
                            f"retry_max_tokens={result.ctx.state.retry_max_tokens} "
                            f"pin_req_id={result.ctx.state.pin_req_id}"
                        )
                    retry_task = Task(
                        ctx=result.ctx,
                        stage_index=task.stage_index,
                        trackers=task.trackers,
                        op_index=result.op_index,
                        retry_priority=-(stage.priority_offset + result.op_index),
                        reserved_budget=budget if result.retain_budget else 0,
                    )
                    stages[task.stage_index].enqueue(retry_task)
                    record_retry(stage.stage_id)
                    if not result.retain_budget:
                        await stage.release_budget(budget, self.manager)
                    log_running_tasks(
                        f"retry task={task.task_id} stage={stage.stage_id} "
                        f"retain_budget={result.retain_budget}"
                    )
                    # await rebalance_stage_capacity()
                    continue

                if result is None:
                    await stage.release_budget(budget, self.manager)
                    log_running_tasks(
                        f"finish task={task.task_id} stage={stage.stage_id}"
                    )
                    # await rebalance_stage_capacity()
                    await finalize_task(task)
                    continue

                next_stage_index = task.stage_index + 1
                if stage.fanout_op is not None:
                    child_ctxs = stage.fanout_op(result) or []
                    record_output(stage.stage_id, len(child_ctxs))

                    if not child_ctxs:
                        if task.ctx.state.pin_req_id is not None:
                            await task.ctx.state.executor.unpin(
                                task.ctx.state.raw_request,
                                task.ctx.state.pin_req_id,
                            )
                            task.ctx.state.pin_req_id = None
                        await stage.release_budget(budget, self.manager)
                        log_running_tasks(
                            f"finish task={task.task_id} stage={stage.stage_id}"
                        )
                        # await rebalance_stage_capacity()
                        await finalize_task(task)
                        continue

                    # This task is replaced by its children, which inherit its
                    # trackers; count them before any child can finalize.
                    for tracker in task.trackers:
                        await tracker.add_children(len(child_ctxs) - 1)

                    deferred_release = False
                    trackers = task.trackers
                    # The parent's accumulated prompt is a prefix of every child's prompt.
                    shared_tokens = self.manager.token_length(result.input.data)
                    if task.ctx.state.pin_req_id is not None:
                        add_deferred_parent(stage.stage_id, budget)
                        if self.LOG_DEFERRAL_TRACE:
                            print(
                                "[scheduler] "
                                f"defer-release task={task.task_id} "
                                f"stage={stage.stage_id} "
                                f"children={len(child_ctxs)} "
                                f"budget={budget} "
                                f"pin_req_id={task.ctx.state.pin_req_id}"
                            )
                        trackers = trackers + (
                            JoinTracker(
                                task.ctx,
                                len(child_ctxs),
                                manager=self.manager,
                                stage_id=stage.stage_id,
                                reserved_budget=budget,
                                on_release=release_deferred_parent,
                            ),
                        )
                        deferred_release = True
                    elif budget > 0:
                        # Virtual pinning: children share the parent's prefix in the
                        # prefix cache; reserve it once until the last descendant ends.
                        shared_budget = min(budget, shared_tokens * self.manager.bytes_per_token)
                        await stage.release_budget(budget - shared_budget, self.manager)
                        add_deferred_parent(stage.stage_id, shared_budget)
                        trackers = trackers + (
                            JoinTracker(
                                task.ctx,
                                len(child_ctxs),
                                manager=self.manager,
                                stage_id=stage.stage_id,
                                reserved_budget=shared_budget,
                                on_release=release_deferred_parent,
                            ),
                        )
                        deferred_release = True

                    for child_ctx in child_ctxs:
                        child_ctx.state.shared_prefix_tokens = shared_tokens
                        child_task = Task(
                            ctx=child_ctx,
                            stage_index=next_stage_index,
                            trackers=trackers,
                        )
                        if next_stage_index >= len(stages):
                            out.append(child_ctx)
                            await finalize_task(child_task)
                        else:
                            stages[next_stage_index].enqueue(child_task)

                    if not deferred_release:
                        await stage.release_budget(budget, self.manager)

                    log_running_tasks(
                        f"finish task={task.task_id} stage={stage.stage_id} "
                        f"deferred_release={deferred_release}"
                    )
                    # await rebalance_stage_capacity()
                    continue

                record_output(stage.stage_id, 1)
                next_task = Task(
                    ctx=result,
                    stage_index=next_stage_index,
                    trackers=task.trackers,
                )
                if next_stage_index >= len(stages):
                    out.append(result)
                    await finalize_task(next_task)
                else:
                    stages[next_stage_index].enqueue(next_task)

                await stage.release_budget(budget, self.manager)
                log_running_tasks(
                    f"finish task={task.task_id} stage={stage.stage_id}"
                )
                # await rebalance_stage_capacity()

        return out, stage_stat


class BlockingExecutor:
    # Blocking/no-admission ablation: retain the fixed worker concurrency while
    # bypassing Kalypso's KV-memory budget checks.
    MEMORY_ADMISSION_CONTROL = True

    @staticmethod
    async def execute_tasks(
        seeds,
        task_builder,
        concurrency: int = 100,
    ):
        manager = KVMemoryManager.get_instance()

        queue = asyncio.Queue(maxsize=concurrency)
        capacity_cond = asyncio.Condition()
        results = []

        async def worker():
            while True:
                task = await queue.get()
                try:
                    out = await task()
                    results.append(out)
                finally:
                    if BlockingExecutor.MEMORY_ADMISSION_CONTROL:
                        async with capacity_cond:
                            await manager.release(task.budget)
                            capacity_cond.notify_all()
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(concurrency)
        ]

        for seed in seeds:
            task = task_builder(seed)

            if BlockingExecutor.MEMORY_ADMISSION_CONTROL:
                async with capacity_cond:
                    while not await manager.can_admit(task.budget):
                        await capacity_cond.wait()
                    await manager.allocate(task.budget)

            await queue.put(task)

        await queue.join()

        for w in workers:
            w.cancel()

        return results
