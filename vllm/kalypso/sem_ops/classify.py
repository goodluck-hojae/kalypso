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


class SemClassify(BaseOp):
    LOG = False

    def __init__(self, classes, pin=False, unpin=False, position=-1, predicate=False):
        super().__init__(
            behavior=OpBehavior.TUPLE_INDEPENDENT,
            position=position,
            predicate=predicate,
        )
        self.classes = list(classes)
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max(KVMemoryManager.get_instance().token_length(g) for g in self.classes) + 1

        self.instruction = "\n\n" \
                + "Choose exactly one class from the list below.\n" \
                + f"Class: {', '.join(self.classes)}\n" \
                + "Answer with the class name only:"
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + self.max_tokens + 1


    def _build_prompts(self, ctx):
        data_prompt = ctx.input.data if 'system' in ctx.input.data[0]['role'] else get_system_prompt() + ctx.input.data
        full_prompt = get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_FILTER)
        return data_prompt, full_prompt


    def estimate_tokens(self, ctx):
        _, prompt = self._build_prompts(ctx)
            
        prompt_str =KVMemoryManager.get_instance().apply_chat_template(prompt)
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def __call__(self, ctx: SemContext, priority: int = 0) -> SemContext: 
 
        data_part, prompt = self._build_prompts(ctx)

        data_result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=data_part,
                max_tokens=1,
                pin=self.pin,
                priority=priority,
        )

        output = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=False,
                priority=priority,
        )
        group_output = output.text.strip().lower()
        group = ""
        for g in self.classes:
            if g.lower() in group_output:
                group = g
                break 

        ctx.input.data = prompt[:-1]
        ctx.output.append({
            str(self.__class__): str(group)
        })

        if self.pin:
            ctx.state.pin_req_id = data_result.request_id

        elif self.unpin and ctx.state.pin_req_id is not None:
            if self.LOG:
                print(
                    "[sem-op] "
                    f"SemClassify unpin pin_req_id={ctx.state.pin_req_id}"
                )
            await ctx.state.executor.unpin(ctx.state.raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return await self.handle_output(ctx, group_output)
