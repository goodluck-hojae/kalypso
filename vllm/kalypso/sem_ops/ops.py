import asyncio
import math
import os
from .base import BaseOp, OpBehavior, OpName
from vllm.kalypso.context import (
    RETRY_TASK,
    ExecutionState,
    SemContext,
    SemanticInput,
)
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.execution.pipeline_execution import BlockingExecutor
from vllm.kalypso.controller.map_estimator import MapRatioEstimator
from .prompt_utils import get_prompt, get_system_prompt, add_assistant_prompt, get_data_prompt
import requests
from typing import Any, List


class SemFilter(BaseOp):

    TRUE = 'true'
    FALSE = 'false'
    LOG = False

    def __init__(
        self,
        instruction,
        negate=False,
        pin=False,
        unpin=False,
        max_tokens=8,
        position=-1,
        predicate=True,
    ):
        super().__init__(
            behavior=OpBehavior.TUPLE_INDEPENDENT,
            position=position,
            predicate=predicate,
        )
        self.instruction = instruction
        self.negate = negate
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max_tokens

    def _build_prompts(self, ctx):
        if ctx.input.right_data:
            joined_data = ctx.input.data + ctx.input.right_data
            data_prompt = joined_data if 'system' in joined_data[0]['role'] else get_system_prompt() + joined_data
            full_prompt = get_prompt(self.instruction, joined_data, op=OpName.SEM_JOIN)
        else:
            data_prompt = ctx.input.data if 'system' in ctx.input.data[0]['role'] else get_system_prompt() + ctx.input.data
            full_prompt = get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_FILTER)
        return data_prompt, full_prompt
    
    def estimate_tokens(self, ctx):
        _, prompt = self._build_prompts(ctx)
            
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def _run_single(self, ctx: SemContext, priority: int = 0):
        data_part, prompt = self._build_prompts(ctx)

        # Data part is only required to bin
        # data_result = await ctx.state.executor.execute(
        #         raw_request=ctx.state.raw_request,
        #         prompt=data_part,
        #         max_tokens=1,
        #         pin=self.pin,
        # )
        output = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=self.pin,
                priority=priority,
        )
        if self.pin:
            ctx.state.pin_req_id = output.request_id

        # appended_prompt, appended_prompt_str = add_assistant_prompt(prompt, output.text)
        verdict = output.text.strip().lower()
        
        ctx.input.data = prompt[:-1]
        
        if bool(ctx.input.right_data):
            ctx.input.right_data = []

        ctx.output.append({
            str(self.__class__): verdict
        })

        return await self.handle_output(ctx, verdict)

    async def _run_blocking(self, ctxs: List[SemContext]) -> List[SemContext]:
        parent = self

        def build_task(ctx: SemContext):
            class FilterTask:
                def __init__(self):
                    self.ctx = ctx
                    self.budget = (
                        parent.estimate_tokens(ctx)
                        * KVMemoryManager.get_instance().bytes_per_token
                    )

                async def __call__(self):
                    passed = await parent._run_single(self.ctx)
                    return self.ctx if passed else None

            return FilterTask()

        results = await BlockingExecutor.execute_tasks(
            seeds=ctxs,
            task_builder=build_task,
        )
        return [ctx for ctx in results if ctx is not None]

    async def __call__(self, ctx: SemContext | List[SemContext], priority: int = 0):
        if isinstance(ctx, list):
            return await self._run_blocking(ctx)
        return await self._run_single(ctx, priority=priority)


class ICPFilter(BaseOp):
    LOG = False
    STATS = {
        "called": 0,
        "fallback": 0,
        "accepted": 0,
        "rejected": 0,
        "helper_positive": 0,
        "helper_negative": 0,
        "fallback_accepted": 0,
        "fallback_rejected": 0,
    }

    @classmethod
    def reset_stats(cls):
        for key in cls.STATS:
            cls.STATS[key] = 0

    @classmethod
    def get_stats(cls):
        return dict(cls.STATS)

    @classmethod
    def _record(cls, resolved_by, passed):
        cls.STATS["called"] += 1
        cls.STATS["accepted" if passed else "rejected"] += 1

        if resolved_by == "icp_helper_positive":
            cls.STATS["helper_positive"] += 1
        elif resolved_by == "icp_helper_negative":
            cls.STATS["helper_negative"] += 1
        elif resolved_by == "icp_oracle_fallback":
            cls.STATS["fallback"] += 1
            cls.STATS["fallback_accepted" if passed else "fallback_rejected"] += 1

        if cls.LOG:
            print(f"[sem-op] ICPFilter stats={cls.STATS}")

    def __init__(
        self,
        instruction,
        low_threshold,
        high_threshold,
        max_tokens=8,
        position=-1,
        predicate=True,
    ):
        super().__init__(
            behavior=OpBehavior.TUPLE_INDEPENDENT,
            position=position,
            predicate=predicate,
        )
        if low_threshold is None or high_threshold is None:
            raise ValueError("ICPFilter requires both low_threshold and high_threshold.")
        if low_threshold > high_threshold:
            raise ValueError("ICPFilter requires low_threshold <= high_threshold.")
        self.instruction = instruction
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.max_tokens = max_tokens
        self.oracle_filter = SemFilter(
            instruction=instruction,
            max_tokens=max_tokens,
            position=position,
        )

    def estimate_tokens(self, ctx):
        score = ctx.state.helper_score
        if score is not None and (
            score >= self.high_threshold or score <= self.low_threshold
        ):
            return 1
        return self.oracle_filter.estimate_tokens(ctx)

    async def __call__(self, ctx: SemContext, priority: int = 0):
        score = ctx.state.helper_score
        if score is None:
            raise ValueError("ICPFilter expected helper_score on context state.")

        if score >= self.high_threshold:
            self._record("icp_helper_positive", True)
            ctx.output.append({
                str(self.__class__): {
                    "resolved_by": "icp_helper_positive",
                    "score": score,
                }
            })
            return await self.handle_output(ctx, True)

        if score <= self.low_threshold:
            self._record("icp_helper_negative", False)
            ctx.output.append({
                str(self.__class__): {
                    "resolved_by": "icp_helper_negative",
                    "score": score,
                }
            })
            return await self.handle_output(ctx, False)

        passed = await self.oracle_filter(ctx, priority=priority)
        self._record("icp_oracle_fallback", passed)
        ctx.output.append({
            str(self.__class__): {
                "resolved_by": "icp_oracle_fallback",
                "score": score,
            }
        })
        return await self.handle_output(ctx, passed)
    

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


class CartesianProduct(BaseOp):
    def __init__(self, right_table, position=-1):
        super().__init__(behavior=OpBehavior.JOIN, position=position)
        self.right_table = right_table

    def __call__(self, ctx):
        out = []

        for right in self.right_table:
            base_input = SemanticInput(
                data=ctx.input.data[:],
                right_data=[]
            )
            _input = base_input.add_right(right.input.data)
            new_ctx = SemContext(
                input=_input,
                state=ExecutionState(
                    raw_request=ctx.state.raw_request,
                    pin_req_id=None,
                    executor=ctx.state.executor
                ),
                )
            out.append(new_ctx)
        return out

 

class IndexedCartesianProduct(BaseOp):
    def __init__(
        self,
        right_table=None,
        service_address="127.0.0.1",
        service_port=8080,
        top_k=5,
        low_threshold=None,
        high_threshold=None,
        cp_id=None,
        position=-1,
    ):
        super().__init__(behavior=OpBehavior.JOIN, position=position)
        self.right_table = right_table or []
        self.service_address = service_address
        self.service_port = service_port
        self.top_k = top_k
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._indexed_rows = [
            (idx, right.input.data)
            for idx, right in enumerate(self.right_table)
        ]
        self.cp_id = cp_id or f"icp:{position}:{id(self)}"

    def _post_icp(self, path, payload):
        return requests.post(
            f"http://{self.service_address}:{self.service_port}/{path}",
            json=payload,
            timeout=60,
        )

    def _build_remote_index(self) -> None:
        if not self.right_table:
            return
        response = self._post_icp(
            "build_index",
            {
                "cp_id": self.cp_id,
                "right_table": [list(row) for row in self._indexed_rows],
            },
        )
        response.raise_for_status()

    def _query_retrieval(self, left_row: tuple[Any, ...]) -> list[dict[str, Any]]:
        payload = {
            "cp_id": self.cp_id,
            "left_tuple": list(left_row),
        }
        if self.low_threshold is not None or self.high_threshold is not None:
            payload["low_threshold"] = self.low_threshold
            payload["high_threshold"] = self.high_threshold
        else:
            payload["top_k"] = self.top_k

        response = self._post_icp("query", payload)
        if response.ok:
            return response.json()["results"]

        self._build_remote_index()

        retry_response = self._post_icp("query", payload)
        retry_response.raise_for_status()
        return retry_response.json()["results"]

    def _query_text_from_data(self, data):
        for message in reversed(data):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return str(data).strip()

    def __call__(self, ctx):
        query_data = self._query_text_from_data(ctx.input.data)

        left_row = (query_data,)
        try:
            ranked_rows = self._query_retrieval(left_row)
        except requests.RequestException:
            if not self.right_table:
                raise
            ranked_rows = [
                {"metadata": [idx, right.input.data], "score": 0.0}
                for idx, right in enumerate(self.right_table)
            ]

        out = []
        for row in ranked_rows:
            metadata = row.get("metadata", [])
            if len(metadata) < 2:
                continue
            _id, data = metadata[0], metadata[1]
            score = row.get("score")
            right_data = get_data_prompt(data)
            base_input = SemanticInput(
                data=ctx.input.data[:],
                right_data=[],
            )
            _input = base_input.add_right(right_data)
            new_ctx = SemContext(
                input=_input,
                state=ExecutionState(
                    raw_request=ctx.state.raw_request,
                    pin_req_id=None,
                    executor=ctx.state.executor,
                    helper_score=score,
                    idx=ctx.state.idx,
                ),
            )
            out.append(new_ctx)

        return out


class IndexedSearch(BaseOp):
    def __init__(
        self,
        right_table=None,
        service_address="127.0.0.1",
        service_port=8080,
        top_k=5,
        low_threshold=None,
        high_threshold=None,
        cp_id=None,
        position=-1,
    ):
        super().__init__(behavior=OpBehavior.TUPLE_INDEPENDENT, position=position)
        self.right_table = right_table or []
        self.service_address = service_address
        self.service_port = service_port
        self.top_k = top_k
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._indexed_rows = [
            (idx, right.input.data)
            for idx, right in enumerate(self.right_table)
        ]
        self.cp_id = cp_id or f"icp:{position}:{id(self)}"

    def _post_icp(self, path, payload):
        return requests.post(
            f"http://{self.service_address}:{self.service_port}/{path}",
            json=payload,
            timeout=60,
        )

    def _build_remote_index(self) -> None:
        if not self.right_table:
            return
        response = self._post_icp(
            "build_index",
            {
                "cp_id": self.cp_id,
                "right_table": [list(row) for row in self._indexed_rows],
            },
        )
        response.raise_for_status()

    def _query_retrieval(self, left_row: tuple[Any, ...]) -> list[dict[str, Any]]:
        payload = {
            "cp_id": self.cp_id,
            "left_tuple": list(left_row),
        }
        if self.low_threshold is not None or self.high_threshold is not None:
            payload["low_threshold"] = self.low_threshold
            payload["high_threshold"] = self.high_threshold
        else:
            payload["top_k"] = self.top_k

        response = self._post_icp("query", payload)
        if response.ok:
            return response.json()["results"]

        self._build_remote_index()

        retry_response = self._post_icp("query", payload)
        retry_response.raise_for_status()
        return retry_response.json()["results"]

    def _query_text_from_data(self, data):
        for message in reversed(data):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return str(data).strip()

    def estimate_tokens(self, ctx):
        return max(
            1,
            KVMemoryManager.get_instance().token_length(
                self._query_text_from_data(ctx.input.data)
            ),
        )

    async def __call__(self, ctx, priority: int = 0):
        query_data = self._query_text_from_data(ctx.input.data)
        ranked_rows = await asyncio.to_thread(
            self._query_retrieval,
            (query_data,),
        )

        retrieved_contexts = []
        for rank, row in enumerate(ranked_rows, start=1):
            metadata = row.get("metadata", [])
            if len(metadata) < 2:
                continue
            doc_id, text = metadata[0], metadata[1]
            retrieved_contexts.append(
                f"[{rank}] Document ID: {doc_id}\n{text}"
            )

        if not retrieved_contexts:
            return ctx

        retrieved_data = get_data_prompt("\n\n".join(retrieved_contexts))
        ctx.input.data += retrieved_data
        ctx.output.append({
            str(self.__class__): {
                "num_retrieved": len(retrieved_contexts),
                "cp_id": self.cp_id,
            }
        })

        return ctx




class CascadeOperator(BaseOp):
    TRUE = 'true'
    FALSE = 'false'
    LOG = True
    delegate_to_main_count = 0
    STATS = {
        "called": 0,
        "fallback": 0,
        "accepted": 0,
        "rejected": 0,
        "helper_accepted": 0,
        "helper_rejected": 0,
        "fallback_accepted": 0,
        "fallback_rejected": 0,
    }

    @classmethod
    def reset_stats(cls):
        cls.delegate_to_main_count = 0
        for key in cls.STATS:
            cls.STATS[key] = 0

    @classmethod
    def get_stats(cls):
        return dict(cls.STATS)

    @classmethod
    def _record(cls, resolved_by, passed):
        cls.STATS["called"] += 1
        cls.STATS["accepted" if passed else "rejected"] += 1

        if resolved_by == "main":
            cls.STATS["fallback"] += 1
            cls.STATS["fallback_accepted" if passed else "fallback_rejected"] += 1
        else:
            cls.STATS["helper_accepted" if passed else "helper_rejected"] += 1

    def __init__(
        self,
        instruction,
        negate=False,
        pin=False,
        unpin=False,
        model_name=None,
        api_base=None,
        api_port=None,
        max_tokens=8,
        temperature=0,
        seed=42,
        low_threshold=None,
        high_threshold=None,
        position=-1,
        predicate=True,
    ):
        super().__init__(
            behavior=OpBehavior.TUPLE_INDEPENDENT,
            position=position,
            predicate=predicate,
        )
        self.instruction = instruction
        self.negate = negate
        self.pin = pin
        self.unpin = unpin
        self.model_name = (
            model_name
            or os.environ.get("SEMOPS_CASCADE_MODEL")
            or os.environ.get("SEMOPS_ICP_VLLM_MODEL")
            or "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.api_base = (
            api_base
            or os.environ.get("SEMOPS_CASCADE_API_BASE")
            or os.environ.get("SEMOPS_ICP_VLLM_API_BASE")
            or os.environ.get("VLLM_8B_API_BASE")
            or (
                f"http://localhost:{api_port}/v1"
                if api_port is not None
                else "http://localhost:8004/v1"
            )
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        if (self.low_threshold is None) != (self.high_threshold is None):
            raise ValueError("low_threshold and high_threshold must be provided together.")
        if (
            self.low_threshold is not None
            and self.high_threshold is not None
            and self.low_threshold > self.high_threshold
        ):
            raise ValueError("low_threshold must be <= high_threshold.")

    def _build_prompts(self, ctx):
        if ctx.input.right_data:
            joined_data = ctx.input.data + ctx.input.right_data
            data_prompt = joined_data if 'system' in joined_data[0]['role'] else get_system_prompt() + joined_data
            full_prompt = get_prompt(self.instruction, joined_data, op=OpName.SEM_JOIN)
        else:
            data_prompt = ctx.input.data if 'system' in ctx.input.data[0]['role'] else get_system_prompt() + ctx.input.data
            full_prompt = get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_FILTER)
        return data_prompt, full_prompt

    def estimate_tokens(self, ctx):
        _, prompt = self._build_prompts(ctx)
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        return prompt_token_len + self.max_tokens

    @staticmethod
    def _get_attr_or_key(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _normalize_logprob_token(cls, token):
        if token is None:
            return ""
        return str(token).replace("Ġ", " ").replace("▁", " ").strip().lower()

    @classmethod
    def _extract_first_token_probs(cls, response):
        choices = cls._get_attr_or_key(response, "choices", [])
        if not choices:
            return None

        choice = choices[0]
        logprobs = cls._get_attr_or_key(choice, "logprobs")
        if logprobs is None:
            return None

        content = cls._get_attr_or_key(logprobs, "content")
        if not content:
            return None

        first = content[0]
        top_logprobs = cls._get_attr_or_key(first, "top_logprobs", []) or []
        token_logprobs = list(top_logprobs)
        if cls._get_attr_or_key(first, "token") is not None:
            token_logprobs.append(first)

        probs = {}
        top_token = cls._normalize_logprob_token(cls._get_attr_or_key(first, "token"))
        top_logprob = cls._get_attr_or_key(first, "logprob")
        for item in token_logprobs:
            token = cls._normalize_logprob_token(cls._get_attr_or_key(item, "token"))
            logprob = cls._get_attr_or_key(item, "logprob")
            if token in (cls.TRUE, cls.FALSE) and logprob is not None:
                probs[token] = max(probs.get(token, 0.0), math.exp(logprob))

        true_prob = probs.get(cls.TRUE, 0.0)
        false_prob = probs.get(cls.FALSE, 0.0)
        if true_prob > 0.0 and false_prob > 0.0:
            normalized_true_prob = true_prob / (true_prob + false_prob)
        else:
            normalized_true_prob = true_prob

        return {
            "top_token": top_token,
            "top_prob": math.exp(top_logprob) if top_logprob is not None else None,
            "true_prob": normalized_true_prob,
            "false_prob": false_prob,
        }

    def _call_vllm(self, messages):
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "CascadeOperator requires litellm in this environment."
            ) from exc

        use_thresholds = self.low_threshold is not None and self.high_threshold is not None
        completion_kwargs = {
            "model": f"hosted_vllm/{self.model_name}",
            "messages": messages,
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if use_thresholds:
            completion_kwargs.update({
                "logprobs": True,
                "top_logprobs": 20,
            })

        response = litellm.completion(**completion_kwargs)
        if not response.choices:
            return "", None

        choice = response.choices[0]
        message = self._get_attr_or_key(choice, "message", {})
        content = self._get_attr_or_key(message, "content", "") or ""
        return content, self._extract_first_token_probs(response) if use_thresholds else None

    async def _fallback_to_main_executor(self, ctx, prompt, priority):
        type(self).delegate_to_main_count += 1
        output = await ctx.state.executor.execute(
            raw_request=ctx.state.raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=self.pin,
            priority=priority,
        )
        if self.pin:
            ctx.state.pin_req_id = output.request_id
        return output.text.strip().lower(), "main"

    def _resolve_with_thresholds(self, helper_probs):
        if self.low_threshold is None or self.high_threshold is None:
            return None
        if helper_probs is None:
            return None

        top_token = helper_probs["top_token"]
        if top_token not in (self.TRUE, self.FALSE):
            return None

        true_prob = helper_probs["true_prob"]
        if true_prob >= self.high_threshold:
            return True, true_prob
        if true_prob <= self.low_threshold:
            return False, true_prob
        return None, true_prob

    async def __call__(self, ctx: SemContext, priority: int = 0):
        _, prompt = self._build_prompts(ctx)
        output_text, helper_probs = await asyncio.to_thread(self._call_vllm, prompt)
        verdict = output_text.strip().lower()

        if self.low_threshold is None and self.high_threshold is None:
            passed = self.FALSE not in verdict
            resolved_by = "helper"
            positive_prob = None
        else:
            resolved = self._resolve_with_thresholds(helper_probs)
            print(resolved)
            if resolved is None or resolved[0] is None:
                verdict, resolved_by = await self._fallback_to_main_executor(ctx, prompt, priority)
                passed = self.FALSE not in verdict
                positive_prob = resolved[1] if resolved is not None else None
            else:
                passed, positive_prob = resolved
                resolved_by = "helper"

            
            if self.LOG:
                print(
                    "[cascade-op] "
                    f"model={self.model_name}"
                    f"verdict={verdict!r} resolved_by={resolved_by} "
                    f"positive_prob={positive_prob}"
                )

        ctx.input.data = prompt[:-1]
        if bool(ctx.input.right_data):
            ctx.input.right_data = []

        if self.negate:
            passed = not passed
        type(self)._record(resolved_by, passed)

        ctx.output.append({
            str(self.__class__): verdict,
            "cascade_resolved_by": resolved_by,
            "cascade_positive_prob": positive_prob,
            "cascade_delegate_to_main_count": type(self).delegate_to_main_count,
        })

        return await self.handle_output(ctx, passed)


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
