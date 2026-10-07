
from typing import Any, Dict, List
from dataclasses import dataclass, field, replace
from vllm.kalypso.execution.vllm_executor import LLMExecutor
from vllm.kalypso.sem_ops.prompt_utils import get_data_prompt

 
class SemanticInput:
    def __init__(self, data=None, token_len=-1, right_data=[]):
        if type(data) is str:
            data = get_data_prompt(data)
        self.data = data
        self.token_len = token_len
        self.right_data = []
        for i in range(len(right_data)):
            self.right_data += right_data[i]

    def add_right(self, value):
        self.right_data += value
        return self


class _RetrySignal:
    pass


RETRY_TASK = _RetrySignal()


@dataclass
class RetryTaskResult:
    ctx: Any
    op_index: int
    retain_budget: bool = False


@dataclass
class ExecutionState:
    raw_request: Any = None
    pin_req_id: Any = None
    executor: LLMExecutor = None
    predicate: bool = True
    helper_score: float | None = None
    idx: int = -1
    stage_id: int = -1
    retry_op_position: int = -1
    retry_max_tokens: int = 0


@dataclass
class SemContext:
    input: SemanticInput = None
    output: List[Dict[str, Any]] = field(default_factory=list)
    state: ExecutionState = None
