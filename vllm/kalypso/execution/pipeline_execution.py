import asyncio
# import math
import time

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.context import RetryTaskResult
# from vllm.kalypso.controller.map_estimator import MapRatioEstimator
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
                ctxs = await item(ctxs)

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
            stage_states = []
            for stage in stages:
                running_task_ids = sorted(stage.running_tasks.keys())
                waiting_task_ids = [task.task_id for task in stage.waiting_tasks]
                used, cap = self.manager.stage_usage(stage.stage_id)
                stage_states.append(
                    "stage="
                    f"{stage.stage_id} "
                    f"used={used:,} cap={cap:,} "
                    f"waiting={len(waiting_task_ids)} "
                    f"running={len(running_task_ids)} "
                    f"retries={retry_counts.get(stage.stage_id, 0)} "
                    f"deferred_parents={deferred_parent_counts.get(stage.stage_id, 0)} "
                    f"deferred_reserved={deferred_parent_bytes.get(stage.stage_id, 0):,}"
                )
            print(
                f"[scheduler] {event} total_retries={total_retries} | "
                + " | ".join(stage_states)
            )

        async def finalize_task(task: Task):
            for tracker in task.trackers:
                await tracker.child_finished()

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

        async def rebalance_stage_capacity():
            changed = False

            for stage in stages:
                stage.tune_thresholds()

            # Keep draining idle stages toward their floor so later stages
            # can absorb capacity when they are the bottleneck.
            for stage in stages:
                if stage.ready_count() == 0 and stage.running_count() == 0:
                    if await self.manager.return_stage_capacity(stage.stage_id, force_return=True):
                        changed = True

            # Grow any stage with a saturated queue. Prefer borrowing from the
            # immediate upstream stage first because that matches pipeline flow.
            for idx, stage in enumerate(stages[1:]):
                if not stage.is_saturated():
                    continue

                if await self.manager.rebalance_stage_capacity(
                    receiver_id=stages[-1].stage_id,
                    donor_hint=stages[idx].stage_id,
                ):
                    changed = True

            # If a later stage is empty while the previous stage still has work,
            # bias capacity upstream so the pipeline can produce tuples for it.
            for idx in range(1, len(stages)):
                stage = stages[idx]
                prev_stage = stages[idx - 1]
                # downstream_light = (
                #     stage.running_count() <= 1
                #     and stage.ready_count() <= stage.low_threshold
                # )
                # if not (stage.is_starving() and downstream_light):
                #     continue
    
                if not (stage.is_starving()):
                    continue
                head_task = prev_stage.peek_task()
                if head_task is None:
                    continue

                # head_budget = prev_stage.estimate_budget(head_task)
                # used, cap = self.manager.stage_usage(prev_stage.stage_id)
                # needed = used + head_budget - cap
                # if needed <= 0:
                #     continue

                if await self.manager.rebalance_stage_capacity(
                    receiver_id=prev_stage.stage_id,
                    donor_hint=stage.stage_id,
                ):
                    changed = True

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

        # def refresh_retry_max_tokens(stage, task):
        #     retry_position = task.ctx.state.retry_op_position
        #     if retry_position < 0 or task.op_index >= len(stage.operators):
        #         return
        #
        #     op = stage.operators[task.op_index]
        #     if not isinstance(op, ops.SemMap) or op.position != retry_position:
        #         return
        #
        #     ratio = MapRatioEstimator.instance().get_ratio(retry_position)
        #     if ratio is None:
        #         return
        #
        #     prompt = op._build_prompt(task.ctx)
        #     prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        #     prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        #     max_tokens = max(
        #         1,
        #         min(
        #             ops.SemMap.MAX_TOKEN_LIMIT,
        #             math.ceil(ratio * prompt_token_len),
        #         ),
        #     )
        #     task.ctx.state.retry_max_tokens = max_tokens

        async def launch_ready_tasks():
            launched = False

            for stage in stages:
                while stage.has_waiting_tasks():
                    task = stage.peek_task()
                    if task is None:
                        break

                    # refresh_retry_max_tokens(stage, task)
                    if await stage.accept(task, self.manager) is None:
                        break

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
                    await rebalance_stage_capacity()
                    continue

                if result is None:
                    await stage.release_budget(budget, self.manager)
                    log_running_tasks(
                        f"finish task={task.task_id} stage={stage.stage_id}"
                    )
                    await rebalance_stage_capacity()
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
                        await rebalance_stage_capacity()
                        await finalize_task(task)
                        continue

                    deferred_release = False
                    trackers = task.trackers
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

                    for child_ctx in child_ctxs:
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
                    await rebalance_stage_capacity()
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
                await rebalance_stage_capacity()

        return out, stage_stat


class BlockingExecutor:

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

            async with capacity_cond:
                while not await manager.can_admit(task.budget):
                    await capacity_cond.wait()
                await manager.allocate(task.budget)

            await queue.put(task)

        await queue.join()

        for w in workers:
            w.cancel()

        return results
