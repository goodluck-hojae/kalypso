import time
from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.2-1B"

def make_join_prompts(prefix: str, suffixes: list[str]) -> list[str]:
    return [prefix + s for s in suffixes]


def run_join_benchmark(
    fanout: int,
    prefix_tokens: int = 512,
    suffix_tokens: int = 64,
    max_new_tokens: int = 32,
):
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.9,
    )

    prefix = "A " * prefix_tokens
    suffixes = [f"B{i} " * suffix_tokens for i in range(fanout)]
    prompts = make_join_prompts(prefix, suffixes)

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling)
    end = time.time()

    total_tokens = sum(
        len(o.outputs[0].token_ids) for o in outputs
    )

    print("=== JOIN BENCHMARK ===")
    print(f"Fanout: {fanout}")
    print(f"Prefix tokens: {prefix_tokens}")
    print(f"Suffix tokens: {suffix_tokens}")
    print(f"Generated tokens per request: {max_new_tokens}")
    print(f"Total requests: {len(prompts)}")
    print(f"Total generated tokens: {total_tokens}")
    print(f"Total time: {end - start:.2f}s")
    print(f"Throughput: {total_tokens / (end - start):.2f} tok/s")


if __name__ == "__main__":
    for fanout in [1, 4, 8, 16, 32]:
        run_join_benchmark(fanout)
        print()
