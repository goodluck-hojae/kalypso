import asyncio
import math
import os
from typing import Any, List

import requests

from .base import BaseOp, OpBehavior, OpName
from .prompt_utils import (
    add_assistant_prompt,
    get_data_prompt,
    get_prompt,
    get_system_prompt,
)
from vllm.kalypso.context import (
    RETRY_TASK,
    ExecutionState,
    SemContext,
    SemanticInput,
)
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.controller.map_estimator import MapRatioEstimator
from vllm.kalypso.execution.pipeline_execution import BlockingExecutor


class SemMap(BaseOp):
    MAX_TOKEN_LIMIT = 8192
    LOG = False
    LOG_RETRY_TRACE = True
    def __init__(
        self,
        instruction,
        max_tokens=MAX_TOKEN_LIMIT,
        expand=False,
        pin=False,
        unpin=False,
        position=-1,
        predicate=False,
    ):
        super().__init__(
            behavior=OpBehavior.TUPLE_INDEPENDENT,
            position=position,
            predicate=predicate,
        )
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.expand = expand
        self.pin = pin
        self.unpin = unpin
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + max_tokens

    def _planned_max_tokens(self, ctx, prompt_token_len=None) -> int:
        if (
            ctx.state.retry_op_position == self.position
            and ctx.state.retry_max_tokens > 0
        ):
            return ctx.state.retry_max_tokens

        if prompt_token_len is None:
            prompt = self._build_prompt(ctx)
            prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
            prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        ratio = MapRatioEstimator.instance().get_ratio(self.position)
        return int(ratio * prompt_token_len) if ratio else int(prompt_token_len)  #1

    def _build_prompt(self, ctx):
        return get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_MAP)
    

    def estimate_tokens(self, ctx):
        prompt = self._build_prompt(ctx)
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        self.max_tokens = self._planned_max_tokens(ctx, prompt_token_len)
        return prompt_token_len + self.max_tokens
    
    
    async def _run_single(self, ctx: SemContext, priority: int = 0):
        executor = ctx.state.executor
        raw_request = ctx.state.raw_request
        previous_pin_req_id = ctx.state.pin_req_id if self.pin else None

        prompt = self._build_prompt(ctx)
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        input_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        max_tokens = self._planned_max_tokens(ctx, input_token_len)

        output = await executor.execute(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=max_tokens,
            pin=self.pin,
            priority=priority,
        )
        
        if output.finish_reason == "length" and max_tokens < SemMap.MAX_TOKEN_LIMIT:
            # if self.pin:
            #     ctx.state.pin_req_id = output.request_id
            if self.pin:
                await executor.unpin(raw_request, output.request_id)
                if (
                    previous_pin_req_id is not None
                    and previous_pin_req_id != output.request_id
                ):
                    await executor.unpin(raw_request, previous_pin_req_id)
                ctx.state.pin_req_id = None
            ctx.state.retry_op_position = self.position
            ctx.state.retry_max_tokens = SemMap.MAX_TOKEN_LIMIT
            if self.LOG_RETRY_TRACE:
                print(
                    "[sem-map] "
                    f"retry-trigger stage={ctx.state.stage_id} "
                    f"position={self.position} "
                    f"request_id={output.request_id} "
                    f"pin_req_id={ctx.state.pin_req_id} "
                    f"from_max_tokens={max_tokens} "
                    f"to_max_tokens={ctx.state.retry_max_tokens} "
                    f"input_tokens={input_token_len}"
                )
            if self.LOG:
                print(
                    "[sem-map] "
                    f"retry-requeue stage={ctx.state.stage_id} "
                    f"position={self.position} "
                    f"request_id={output.request_id} "
                    f"from_max_tokens={max_tokens} "
                    f"to_max_tokens={SemMap.MAX_TOKEN_LIMIT}"
                )
            return RETRY_TASK

        if ctx.state.retry_op_position == self.position:
            ctx.state.retry_op_position = -1
            ctx.state.retry_max_tokens = 0

        appended_prompt, appended_prompt_str = add_assistant_prompt(prompt, output.text)
        ctx.input.data = appended_prompt
        # Update ratio
        MapRatioEstimator.instance().update(self.position, input_token_len, KVMemoryManager.get_instance().token_length(appended_prompt_str)-input_token_len)

        ctx.output.append({
            str(self.__class__): output.text
        })

        if self.pin:
            if (
                previous_pin_req_id is not None
                and previous_pin_req_id != output.request_id
            ):
                await executor.unpin(raw_request, previous_pin_req_id)
            ctx.state.pin_req_id = output.request_id
        elif self.unpin and ctx.state.pin_req_id:
            if self.LOG:
                print(
                    "[sem-op] "
                    f"SemMap unpin pin_req_id={ctx.state.pin_req_id}"
                )
            await executor.unpin(raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return ctx

    async def _run_blocking_single(self, ctx: SemContext) -> SemContext:
        while True:
            result = await self._run_single(ctx)
            if result is not RETRY_TASK:
                return result

    async def _run_blocking(self, ctxs: List[SemContext]) -> List[SemContext]:
        parent = self

        def build_task(ctx: SemContext):
            class MapTask:
                def __init__(self):
                    self.ctx = ctx
                    self.budget = (
                        parent.estimate_tokens(ctx)
                        * KVMemoryManager.get_instance().bytes_per_token
                    )

                async def __call__(self):
                    return await parent._run_blocking_single(self.ctx)

            return MapTask()

        return await BlockingExecutor.execute_tasks(
            seeds=ctxs,
            task_builder=build_task,
        )

    async def __call__(self, ctx: SemContext | List[SemContext], priority: int = 0):
        if isinstance(ctx, list):
            return await self._run_blocking(ctx)
        result = await self._run_single(ctx, priority=priority)
        if result is RETRY_TASK:
            return RETRY_TASK

        return await self.handle_output(
            result,
            result.output[-1].get(str(self.__class__), ""),
        )
