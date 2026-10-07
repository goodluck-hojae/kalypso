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


class SemAgg(BaseOp):
    def __init__(self, instruction: str, max_tokens: int = 8192, concurrency: int = 8, position=-1):
        super().__init__(behavior=OpBehavior.BLOCKING, position=position)
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.concurrency = concurrency

    def _ctx_to_text(self, ctx: SemContext) -> str:
        parts = []
        for message in ctx.input.data:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    def _build_prompt(self, chunk: List[SemContext]):
        docs = []
        for i, ctx in enumerate(chunk, 1):
            docs.append(f"Document {i}:\n{self._ctx_to_text(ctx)}")
        data = [{
            "role": "user",
            "type": "text",
            "content": "\n\n".join(docs),
        }]
        return get_prompt(self.instruction, data, op=OpName.SEM_AGG)


    async def __call__(self, ctxs: List[SemContext]) -> List[SemContext]:
        working_set = list(ctxs)
        if not working_set:
            return []

        while len(working_set) > 1:
            chunks = self._chunk_by_tokens(working_set)

            reducible = [c for c in chunks if len(c) > 1]
            passthrough = [c[0] for c in chunks if len(c) == 1]

            if reducible:
                reduced = await BlockingExecutor.execute_tasks(
                    seeds=reducible,
                    task_builder=self._build_reducer,
                    concurrency=self.concurrency,
                )
                
                working_set = passthrough + reduced
            else:
                working_set = await BlockingExecutor.execute_tasks(
                    seeds=[working_set],
                    task_builder=self._build_reducer,
                    concurrency=1,
                )
                
        working_set[0].output.append({
            str(self.__class__): working_set[0].input.data
        })
        
        return working_set


    def _chunk_by_tokens(self, ctxs: List[SemContext]) -> List[List[SemContext]]:
        context_chunks = []
        current_chunk = []
        instruction_overhead = KVMemoryManager.get_instance().token_length(
            "\n\n" + self.instruction + "\n\n"
        )
        per_doc_overhead = KVMemoryManager.get_instance().token_length(
            "\n\nDocument 1:\n"
        )
        current_chunk_tokens = instruction_overhead

        for ctx in ctxs:
            ctx_tokens = KVMemoryManager.get_instance().token_length(ctx.input.data)
            chunk_item_tokens = ctx_tokens + per_doc_overhead
            # Allow an oversized single item as its own chunk instead of failing early.
            if current_chunk and current_chunk_tokens + chunk_item_tokens > self.max_tokens:
                context_chunks.append(current_chunk)
                current_chunk = []
                current_chunk_tokens = instruction_overhead

            current_chunk.append(ctx)
            current_chunk_tokens += chunk_item_tokens

        if current_chunk:
            context_chunks.append(current_chunk)

        return context_chunks


    def _build_reducer(self, chunk: List[SemContext]):
        parent = self
        
        class Reducer:
            def __init__(self, chunk: List[SemContext]):
                self.chunk = chunk

                prompt = parent._build_prompt(chunk)
                prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
                prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

                self.budget = (prompt_token_len + parent.max_tokens) * KVMemoryManager.get_instance().bytes_per_token 

            async def __call__(self) -> SemContext:
                return await parent._reduce_chunk(self.chunk)

        return Reducer(chunk)


    async def _reduce_chunk(self, chunk: List[SemContext]) -> SemContext:
        executor = chunk[0].state.executor
        raw_request = chunk[0].state.raw_request

        prompt = self._build_prompt(chunk)

        result = await executor.execute(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=False,
        )

        ctx = SemContext(
            input=SemanticInput(
                data=result.text,
                token_len=KVMemoryManager.get_instance().token_length(result.text)
            ),
            state=ExecutionState(
                raw_request=raw_request,
                pin_req_id=None,
                executor=executor
            ),
        )
        return ctx


class SemTopK(BaseOp):
    MERGE_RANK_THRESHOLD = 8

    def __init__(self, instruction: str, k: int= 10, concurrency: int = 20, position=-1):
        super().__init__(behavior=OpBehavior.BLOCKING, position=position)
        
        self.instruction = instruction
        self.k = k
        self.max_tokens = 5
        self.concurrency = concurrency

    def _ctx_to_text(self, ctx: SemContext) -> str:
        parts = []
        for message in ctx.input.data:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    def _build_compare_prompt(self, first: SemContext, second: SemContext):
        data = [{
            "role": "user",
            "type": "text",
            "content": (
                f"Document A:\n{self._ctx_to_text(first)}\n\n"
                f"Document B:\n{self._ctx_to_text(second)}"
            ),
        }]
        return get_prompt(self.instruction, data, op=OpName.SEM_TOPK)

    async def __call__(self, ctxs: List[SemContext]) -> List[SemContext]:
        if len(ctxs) <= self.k:
            return await self._rank(ctxs)

        topk = await self._quickselect(ctxs, self.k)
        return await self._rank(topk)


    async def _quickselect(self, ctxs: List[SemContext], k: int) -> List[SemContext]:
        if len(ctxs) <= k:
            return ctxs

        pivot = ctxs[0]
        others = ctxs[1:]

        better, worse = await self._partition(pivot, others)

        if len(better) >= k:
            return await self._quickselect(better, k)
        if len(better) + 1 == k:
            return better + [pivot]
        return better + [pivot] + await self._quickselect(worse, k - len(better) - 1)


    async def _partition(self, pivot: SemContext, others: List[SemContext]):
        executor = pivot.state.executor
        parent = self

        def build_task(other: SemContext):
            raw_request = pivot.state.raw_request
            max_tokens = self.max_tokens

            class CompareTask:
                def __init__(self):
                    self.prompt = parent._build_compare_prompt(pivot, other)
                    tokens = KVMemoryManager.get_instance().token_length(self.prompt)
                    self.budget = tokens * KVMemoryManager.get_instance().bytes_per_token


                async def __call__(self):
                    result = await executor.execute(
                        raw_request=raw_request,
                        prompt=self.prompt,
                        max_tokens=max_tokens,
                        pin=False,
                    )
                    return other, result.text.strip().upper()
                

            return CompareTask()

        results = await BlockingExecutor.execute_tasks(
            seeds=others,
            task_builder=build_task,
            concurrency=self.concurrency,
        )

        better, worse = [], []
        for other, verdict in results:
            if verdict.startswith("B"):
                better.append(other)
            else:
                worse.append(other)

        return better, worse


    async def _rank(self, ctxs: List[SemContext]) -> List[SemContext]:
        if len(ctxs) <= 1:
            return ctxs

        if len(ctxs) < self.MERGE_RANK_THRESHOLD:
            return await self._rank_insertion(ctxs)

        mid = len(ctxs) // 2
        left, right = await asyncio.gather(
            self._rank(ctxs[:mid]),
            self._rank(ctxs[mid:]),
        )
        return await self._merge_ranked(left, right)

    async def _rank_insertion(self, ctxs: List[SemContext]) -> List[SemContext]:
        ranked: List[SemContext] = []

        for ctx in ctxs:
            inserted = False
            for i, other in enumerate(ranked):
                better = await self._compare(ctx, other)
                if better:
                    ranked.insert(i, ctx)
                    inserted = True
                    break
            if not inserted:
                ranked.append(ctx)

        return ranked

    async def _merge_ranked(self, left: List[SemContext], right: List[SemContext]) -> List[SemContext]:
        merged: List[SemContext] = []
        i = 0
        j = 0

        while i < len(left) and j < len(right):
            better = await self._compare(left[i], right[j])
            if better:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        if i < len(left):
            merged.extend(left[i:])
        if j < len(right):
            merged.extend(right[j:])

        return merged

    async def _compare(self, a: SemContext, b: SemContext) -> bool:
        executor = a.state.executor 
        raw_request = a.state.raw_request
        prompt = self._build_compare_prompt(a, b)

        result = await executor.execute(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=False,
        )
        return result.text.strip().upper().startswith("A")
