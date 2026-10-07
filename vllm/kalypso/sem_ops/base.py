from enum import Enum, auto
from dataclasses import dataclass, field


class OpName:
    SEM_FILTER = "sem_filter"
    SEM_JOIN = "sem_join"
    SEM_CLASSIFY = "sem_classify"
    SEM_TOPK = "sem_topk"
    SEM_MAP = "sem_map"
    SEM_AGG = "sem_agg"
    JOIN = "join"
    CARTESIAN_PRODUCT = "cp"
    INDEXED_SEARCH = "indexed_search"


OPERATOR_LIST = [
    OpName.SEM_FILTER,
    OpName.SEM_JOIN,
    OpName.SEM_CLASSIFY,
    OpName.SEM_TOPK,
    OpName.SEM_MAP,
    OpName.SEM_AGG,
    OpName.JOIN,
    OpName.CARTESIAN_PRODUCT,
    OpName.INDEXED_SEARCH,
]


class OpBehavior(Enum):
    TUPLE_INDEPENDENT = auto()   # sem_filter, sem_map, sem_classify, sem_join
    BLOCKING = auto()     # sem_topk, sem_agg, 
    JOIN = auto()     # join, cartesian product


class BaseOp:
    def __init__(self, behavior, position, predicate=False):
        self.behavior = behavior
        self.position = position
        self.predicate = predicate

    async def __call__(self, ctx):
        raise NotImplementedError

    def predicate_passed(self, verdict):
        false_token = getattr(self, "FALSE", "false")
        passed = false_token not in str(verdict).strip().lower()
        if getattr(self, "negate", False):
            passed = not passed
        return passed

    async def handle_output(self, ctx, output):
        passed = output if isinstance(output, bool) else self.predicate_passed(output)

        if (getattr(self, "unpin", False) or not passed) and ctx.state.pin_req_id is not None:
            if getattr(self, "LOG", False):
                print(
                    "[sem-op] "
                    f"{self.__class__.__name__} unpin pin_req_id={ctx.state.pin_req_id} "
                    f"passed={passed} "
                    f"self_unpin={getattr(self, 'unpin', False)}"
                )
            await ctx.state.executor.unpin(ctx.state.raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return passed if self.predicate else ctx
     
