from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec
from vllm.utils.hashing import sha256
from vllm.sampling_params import SamplingParams
# from vllm.vllm.v1.core.kv_trace import dump_trace

def make_request(request_id, token_ids, block_size):
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        eos_token_id=0,
        lora_request=None,
        cache_salt=None,
        block_hasher=lambda tokens, parent=None: sha256((parent, tuple(tokens))),
    )

def main():
    block_size = 8

    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=6,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(block_size, 1, 1, dtype=None),
                )
            ],
        ),
        max_model_len=128,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # Request 0: full miss
    req0 = make_request("req0", list(range(20)), block_size)
    computed, hit_tokens = manager.get_computed_blocks(req0)
    manager.allocate_slots(req0, num_new_tokens=20,
                           num_new_computed_tokens=hit_tokens,
                           new_computed_blocks=computed)

    # Request 1: prefix hit
    req1 = make_request("req1", list(range(16)), block_size)
    computed, hit_tokens = manager.get_computed_blocks(req1)
    manager.allocate_slots(req1, num_new_tokens=16 - hit_tokens,
                           num_new_computed_tokens=hit_tokens,
                           new_computed_blocks=computed)

    # Free req0
    manager.free(req0)

    # Trigger eviction pressure
    req2 = make_request("req2", list(range(32)), block_size)
    computed, hit_tokens = manager.get_computed_blocks(req2)
    manager.allocate_slots(req2, 32, hit_tokens, computed)

    manager.free(req1)
    manager.free(req2)

    dump_trace()

if __name__ == "__main__":
    main()
