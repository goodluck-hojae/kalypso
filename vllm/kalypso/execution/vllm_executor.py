# execution/vllm_executor.py
import json
from fastapi import Request
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.entrypoints.openai.api_server import create_completion, create_chat_completion
from vllm.kalypso.pin_registry import PinnedRequestRegistry

from .executor import LLMExecutor, CompletionResult


class VLLMExecutor(LLMExecutor):

    UNPIN_FUNCTION = 'unpin_request'
    LOG = False

    @classmethod
    def _log(cls, message: str):
        if cls.LOG:
            print(message)

    @staticmethod
    def _owner_key(raw_request: Request) -> str:
        return str(id(raw_request))
    
    def __init__(self, model: str):
        self.model = model

    def _build_chat_request(self, message, max_tokens, pin, priority):
        return ChatCompletionRequest(
            model=self.model,
            messages=message,
            max_tokens=min(max_tokens, 16384),
            temperature=0.0,
            seed=42,
            top_p=1.0,
            frequency_penalty=0.5,
            repetition_penalty=1.3,
            priority=priority,
            vllm_xargs={"pinned": pin},
        )


    def _build_request(self, prompt, max_tokens, pin, priority):
        return CompletionRequest(
            model=self.model,
            prompt=prompt,
            max_tokens=min(max_tokens, 16384),
            temperature=0.0,
            seed=42,
            top_p=1.0,
            frequency_penalty=0.5,
            repetition_penalty=1.3,
            priority=priority,
            vllm_xargs={"pinned": pin},
        )

    async def execute(
        self,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        llm_func = self.complete if type(prompt) is str else self.chatcomplete
        return await llm_func(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=max_tokens,
            pin=pin,
            priority=priority,
        )


    async def chatcomplete(
        self,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        req = self._build_chat_request(prompt, max_tokens, pin, priority)
        prompt_items = len(prompt) if isinstance(prompt, list) else 1
        self._log(
            "[vllm-exec] "
            f"submit kind=chat "
            f"max_tokens={max_tokens} "
            f"pin={pin} "
            f"priority={priority} "
            f"prompt_items={prompt_items}"
        )

        gen = await create_chat_completion(
            request=req,
            raw_request=raw_request,
        )
        self._log(
            "[vllm-exec] "
            f"returned kind=chat "
            f"response_type={type(gen).__name__}"
        )

        raw = gen.body.decode("utf-8")
        data = json.loads(raw)
        request_id = data["id"]
        self._log(
            "[vllm-exec] "
            f"decoded kind=chat "
            f"request_id={request_id} "
            f"finish_reason={data['choices'][0].get('finish_reason', '')}"
        )
        if pin:
            PinnedRequestRegistry.instance().add(
                request_id,
                kind="chat",
                max_tokens=max_tokens,
                owner_key=self._owner_key(raw_request),
            )

        return CompletionResult(
            text=data['choices'][0]['message']['content'],
            request_id=request_id,
            finish_reason=data["choices"][0].get("finish_reason", "")
        )


    async def complete(
        self,
        *,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        req = self._build_request(prompt, max_tokens, pin, priority)
        prompt_len = len(prompt) if isinstance(prompt, str) else 1
        self._log(
            "[vllm-exec] "
            f"submit kind=completion "
            f"max_tokens={max_tokens} "
            f"pin={pin} "
            f"priority={priority} "
            f"prompt_len={prompt_len}"
        )

        gen = await create_completion(
            request=req,
            raw_request=raw_request,
        )
        self._log(
            "[vllm-exec] "
            f"returned kind=completion "
            f"response_type={type(gen).__name__}"
        )

        raw = gen.body.decode("utf-8")
        data = json.loads(raw)
        request_id = data["id"]
        self._log(
            "[vllm-exec] "
            f"decoded kind=completion "
            f"request_id={request_id} "
            f"finish_reason={data['choices'][0].get('finish_reason', '')}"
        )
        if pin:
            PinnedRequestRegistry.instance().add(
                request_id,
                kind="completion",
                max_tokens=max_tokens,
                owner_key=self._owner_key(raw_request),
            )

        return CompletionResult(
            text=data["choices"][0]["text"],
            request_id=request_id,
            finish_reason=data["choices"][0].get("finish_reason", ""),
        )

    async def unpin(self, raw_request: Request, request_id: str):
        engine = raw_request.app.state.engine_client
        self._log(f"[vllm-exec] unpin submit request_id={request_id}")
        await engine.engine_core.call_utility_async(
            VLLMExecutor.UNPIN_FUNCTION,
            request_id,
        )
        self._log(f"[vllm-exec] unpin done request_id={request_id}")
        PinnedRequestRegistry.instance().remove(request_id)

    async def abort(self, raw_request: Request, request_id: str):
        engine = raw_request.app.state.engine_client
        await engine.abort(request_id)
