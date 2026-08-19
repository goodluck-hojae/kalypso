import json

from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from fastapi import Request


def build_completion_request(prompt, max_tokens, pin=False):
    return CompletionRequest(
        # model="meta-llama/Llama-3.2-3B-Instruct",
        model="meta-llama/Llama-3.1-8B-Instruct",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
        vllm_xargs={
            "pin_kv": pin,
        }
    )

    
# def build_completion_request(messages, max_tokens, pin=False):
#     return ChatCompletionRequest(
#         # model="meta-llama/Llama-3.2-3B",
#         model="meta-llama/Llama-3.2-3B-Instruct",
#         messages=messages,
#         max_tokens=max_tokens,
#         temperature=0.0,
#         stream=False,
#         vllm_xargs={
#             "pin_kv": pin,
#         },
#     )

async def completion_call_internal(raw_request: Request, prompt, max_tokens, pin=False):
    # Import lazily to avoid a cycle while api_server imports Kalypso.
    from vllm.entrypoints.openai.api_server import create_completion

    req = build_completion_request(prompt, max_tokens, pin=pin)

    gen = await create_completion(
        request=req,
        raw_request=raw_request
    )
    
    raw = gen.body.decode("utf-8")
    data = json.loads(raw)
    return data, req
    
    
async def evict_request(raw_request, req_id: str):
    print('evicting request:', req_id)
    async_llm = raw_request.app.state.engine_client  # AsyncLLM
    await async_llm.abort(req_id)


async def unpin_request(engine, req_id: str):
    await engine.engine_core.call_utility_async(
        "unpin_request",
        req_id
    )
