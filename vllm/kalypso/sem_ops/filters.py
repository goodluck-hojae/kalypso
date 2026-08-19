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
