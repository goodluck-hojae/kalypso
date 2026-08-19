from dataclasses import dataclass, field
import math
from itertools import count
from typing import Any

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.context import RETRY_TASK, RetryTaskResult
from vllm.kalypso.sem_ops import OpBehavior, ops


_TASK_IDS = count()


@dataclass
class Task:
    ctx: Any
    stage_index: int
    trackers: tuple = ()
    op_index: int = 0
    retry_priority: int | None = None
    reserved_budget: int = 0
    task_id: int = field(default_factory=lambda: next(_TASK_IDS))


class Stage:
    def __init__(
        self,
        stage_id: int,
        operators,
        behavior: OpBehavior | None = None,
        fanout_op=None,
        priority_offset: int = 0,
    ):
        self.stage_id = stage_id
        self.operators = tuple(operators)
        self.behavior = behavior or self._infer_behavior()
        self.fanout_op = fanout_op
        self.priority_offset = priority_offset
        self.waiting_tasks = []
        self.running_tasks = {}
        self.bytes_per_token = KVMemoryManager.get_instance().bytes_per_token
        self.low_threshold = 1
        self.high_threshold = 5

    def _infer_behavior(self) -> OpBehavior:
        return OpBehavior.TUPLE_INDEPENDENT

    def task_priority(self, task: Task) -> int:
        if task.retry_priority is not None:
            return task.retry_priority
        return -(self.priority_offset + task.op_index)

    def enqueue(self, task: Task) -> None:
        task_priority = self.task_priority(task)
        insert_at = len(self.waiting_tasks)
        for idx, queued_task in enumerate(self.waiting_tasks):
            queued_priority = self.task_priority(queued_task)
            if (
                task_priority < queued_priority
                or (
                    task_priority == queued_priority
                    and task.task_id < queued_task.task_id
                )
            ):
                insert_at = idx
                break
        self.waiting_tasks.insert(insert_at, task)

    def clear_tasks(self) -> None:
        self.waiting_tasks.clear()
        self.running_tasks.clear()

    def has_waiting_tasks(self) -> bool:
        return bool(self.waiting_tasks)

    def ready_count(self) -> int:
        return len(self.waiting_tasks)

    def running_count(self) -> int:
        return len(self.running_tasks)

    def is_starving(self) -> bool:
        return self.ready_count() < self.low_threshold

    def is_saturated(self) -> bool:
        return self.ready_count() > self.high_threshold

    def tune_thresholds(self):
        manager = KVMemoryManager.get_instance()
        min_budget = int(manager._stage_min_capacity.get(self.stage_id, 0))
        if min_budget <= 0:
            min_budget = max(1, int(self.bytes_per_token))

        _, cur_cap = manager.stage_usage(self.stage_id)
        self.high_threshold = max(1, int(1 * cur_cap // min_budget))
        self.low_threshold = max(1, int(0.5 * cur_cap // min_budget))
        # print(f'{self.stage_id}-{self.low_threshold}, self.is_starving(): {self.is_starving()}, self.is_saturated(): {self.is_saturated()}')

    def peek_task(self) -> Task | None:
        if not self.waiting_tasks:
            return None
        return self.waiting_tasks[0]

    def pop_task(self) -> Task | None:
        if not self.waiting_tasks:
            return None
        return self.waiting_tasks.pop(0)

    def estimate_budget(self, task: Task) -> int:
        max_boundary = -1
        for op in self.operators[task.op_index:]:
            if isinstance(op, ops.CartesianProduct):
                break

            if not hasattr(op, "estimate_tokens"):
                raise AttributeError(f"{op} must define `estimate_tokens`")

            estimated_tokens = op.estimate_tokens(task.ctx)
            if estimated_tokens > max_boundary:
                max_boundary = estimated_tokens

        return max_boundary * self.bytes_per_token


    async def accept(self, task: Task, manager=None) -> int | None:
        manager = manager or KVMemoryManager.get_instance()
        if task.reserved_budget > 0:
            budget = task.reserved_budget
            task.reserved_budget = 0
            self.running_tasks[task.task_id] = budget
            return budget

        budget = self.estimate_budget(task)
        if not await manager.try_allocate_stage(self.stage_id, budget):
            return None
        self.running_tasks[task.task_id] = budget
        return budget


    async def force_accept(self, task: Task, manager=None) -> int:
        manager = manager or KVMemoryManager.get_instance()
        if task.reserved_budget > 0:
            budget = task.reserved_budget
            task.reserved_budget = 0
            self.running_tasks[task.task_id] = budget
            return budget

        budget = self.estimate_budget(task)
        await manager.force_allocate_stage(self.stage_id, budget)
        self.running_tasks[task.task_id] = budget
        return budget

    def detach_budget(self, task: Task) -> int:
        return self.running_tasks.pop(task.task_id, 0)

    async def release_budget(self, budget: int, manager=None) -> None:
        manager = manager or KVMemoryManager.get_instance()
        await manager.release_stage(self.stage_id, budget)

    # async def release(self, task: Task, manager=None) -> None:
    #     budget = self.detach_budget(task)
    #     await self.release_budget(budget, manager)

    async def run_task(self, task: Task):
        task.ctx.state.stage_id = self.stage_id

        keep_going = True
        for idx in range(task.op_index, len(self.operators)):
            op = self.operators[idx]
            if keep_going:
                priority = (
                    task.retry_priority
                    if idx == task.op_index and task.retry_priority is not None
                    else -(self.priority_offset + idx)
                )
                keep_going = await op(
                    task.ctx,
                    priority=priority,
                )
                if keep_going is RETRY_TASK:
                    return RetryTaskResult(
                        ctx=task.ctx,
                        op_index=idx,
                        retain_budget=False,
                    )
                if keep_going is False:
                    return None

        return task.ctx


def stage_builder(operators, stage_id, priority_offset: int = 0):
    return Stage(
        stage_id=stage_id,
        operators=operators,
        priority_offset=priority_offset,
    )
