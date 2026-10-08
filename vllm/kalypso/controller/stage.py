from vllm.kalypso import op_usage
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
    # Memory-based states with a middle band. demand = waiting tasks x task_size,
    # margin = STATE_MARGIN_RATIO x this stage's current cap:
    #   saturated: demand exceeds free memory by at least margin (asks for memory)
    #   starving:  free memory exceeds demand by at least margin (may give memory)
    #   balanced:  otherwise (neither)
    LOW_THRESHOLD_RATIO = 0.2
    STATE_MARGIN_RATIO = 0.1

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

    def task_size(self) -> int:
        head = self.peek_task()
        if head is not None:
            return max(1, self.estimate_budget(head))
        if self.running_tasks:
            return max(1, sum(self.running_tasks.values()) // len(self.running_tasks))
        manager = KVMemoryManager.get_instance()
        return max(1, int(manager._stage_min_capacity.get(self.stage_id, 0)), int(self.bytes_per_token))

    def free_memory(self) -> int:
        used, cap = KVMemoryManager.get_instance().stage_usage(self.stage_id)
        return max(0, int(cap - used))

    def _demand_free_margin(self):
        used, cap = KVMemoryManager.get_instance().stage_usage(self.stage_id)
        demand = self.ready_count() * self.task_size()
        return demand, max(0, int(cap - used)), self.STATE_MARGIN_RATIO * cap

    def is_saturated(self) -> bool:
        demand, free, margin = self._demand_free_margin()
        return demand - free >= margin

    def is_starving(self) -> bool:
        demand, free, margin = self._demand_free_margin()
        return free - demand >= margin

    def tune_thresholds(self):
        manager = KVMemoryManager.get_instance()
        min_budget = int(manager._stage_min_capacity.get(self.stage_id, 0))
        if min_budget <= 0:
            min_budget = max(1, int(self.bytes_per_token))

        _, cur_cap = manager.stage_usage(self.stage_id)
        self.low_threshold = max(1, int(self.LOW_THRESHOLD_RATIO * cur_cap // min_budget))
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

            # Exclude the prefix already reserved by the parent task.
            estimated_tokens = max(1, op.estimate_tokens(task.ctx) - getattr(task.ctx.state, "shared_prefix_tokens", 0))
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

    def accept_unconditionally(self, task: Task) -> int:
        # Accept a task without estimating or reserving KV-cache memory. This is for evaluation
        reserved_budget = 0
        task.reserved_budget = 0
        self.running_tasks[task.task_id] = reserved_budget
        return reserved_budget

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
                token = op_usage.current_op.set(op_usage.op_label(op, self.stage_id, idx))
                try:
                    keep_going = await op(
                        task.ctx,
                        priority=priority,
                    )
                finally:
                    op_usage.current_op.reset(token)
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
