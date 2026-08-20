import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests

import scenarios
from cli_utils import parse_query_args


DEFAULT_ICP_ADDRESS = "127.0.0.1"
DEFAULT_ICP_PORT = 8080
DEFAULT_ICP_TOP_K = 17
# 6634
DEFAULT_ICP_HIGH_THRESHOLD = 0.95
DEFAULT_ICP_LOW_THRESHOLD = 0.865
DEFAULT_ICP_ORACLE_MAX_TOKENS = 8

DATA_ROOT = Path(__file__).resolve().parent / "sample_data"
DEFAULT_ARTICLE_DIR = DATA_ROOT / "articles_500"
DEFAULT_REACTION_DIR = DATA_ROOT / "reactions"
TIMER_INTERVAL_SECONDS = int(os.environ.get("QLLM_TIMER_INTERVAL_SECONDS", "60"))


def _response_summary(result: dict) -> dict:
    return {
        "request_id": result.get("request_id"),
        "predicate_result": result.get("predicate_result"),
        "num_output_rows": result.get("num_output_rows"),
        "latency_sec": result.get("latency_sec"),
        "response_keys": sorted(result.keys()),
    }


@contextmanager
def request_timer(label: str, interval_seconds: int = TIMER_INTERVAL_SECONDS):
    start = time.perf_counter()
    stop_event = threading.Event()

    def log_elapsed():
        while not stop_event.wait(interval_seconds):
            elapsed = time.perf_counter() - start
            print(f"  TIMER: {label} still running ({elapsed:.0f}s)", flush=True)

    timer_thread = threading.Thread(target=log_elapsed, daemon=True)
    timer_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        timer_thread.join(timeout=1)


class SemanticQueryBuilder:
    def __init__(self, data_path: str, model_name: str | None = None, blocking: bool = False):
        self.data_path = data_path
        self.model_name = model_name
        self.blocking = blocking
        self.plan = []

    def sem_map(self, prompt: str):
        self.plan.append(
            {
                "op": "sem_map",
                "args": {
                    "prompt": prompt,
                },
            }
        )
        return self

    def cartesian_product(
        self,
        right_table: str,
        *,
        icp: bool = False,
        icp_address: str = DEFAULT_ICP_ADDRESS,
        icp_port: int = DEFAULT_ICP_PORT,
        icp_top_k: int | None = DEFAULT_ICP_TOP_K,
        icp_low_threshold: float | None = None,
        icp_high_threshold: float | None = None,
        icp_oracle_fallback: bool = False,
        icp_oracle_instruction: str | None = None,
        icp_oracle_max_tokens: int = DEFAULT_ICP_ORACLE_MAX_TOKENS,
    ):
        args = {
            "right_table": right_table,
            "icp": icp,
            "icp_address": icp_address,
            "icp_port": icp_port,
        }
        if icp_low_threshold is not None or icp_high_threshold is not None:
            args["icp_low_threshold"] = icp_low_threshold
            args["icp_high_threshold"] = icp_high_threshold
        else:
            args["icp_top_k"] = icp_top_k

        if icp_oracle_fallback:
            args["icp_oracle_fallback"] = True
            args["icp_oracle_max_tokens"] = icp_oracle_max_tokens
            if icp_oracle_instruction is not None:
                args["icp_oracle_instruction"] = icp_oracle_instruction

        self.plan.append(
            {
                "op": "cp",
                "args": args,
            }
        )
        return self

    def build(self) -> dict:
        payload = {
            "data_path": self.data_path,
            "ops": self.plan,
            "blocking": self.blocking,
        }
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        return payload

    def execute(self, endpoint: str):
        payload = self.build()
        print(json.dumps(payload, indent=2))

        start = time.perf_counter()
        with request_timer("QLLM request"):
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        elapsed = time.perf_counter() - start

        try:
            response.raise_for_status()
        except requests.HTTPError:
            print("Server error response:")
            print(response.text)
            raise

        print(f"Request latency: {elapsed:.3f} seconds")
        return response.json(), elapsed


def main(*, blocking: bool = False):
    model_name, endpoint = parse_query_args()

    article_dir = os.environ.get("BIODEX_ARTICLE_DIR", str(DEFAULT_ARTICLE_DIR))
    reaction_dir = os.environ.get("BIODEX_REACTION_DIR", str(DEFAULT_REACTION_DIR))
    icp_address = os.environ.get("BIODEX_ICP_ADDRESS", DEFAULT_ICP_ADDRESS)
    icp_port = int(os.environ.get("BIODEX_ICP_PORT", str(DEFAULT_ICP_PORT)))
    icp_low_threshold = float(
        os.environ.get(
            "BIODEX_ICP_LOW_THRESHOLD",
            str(DEFAULT_ICP_LOW_THRESHOLD),
        )
    )
    icp_high_threshold = float(
        os.environ.get(
            "BIODEX_ICP_HIGH_THRESHOLD",
            str(DEFAULT_ICP_HIGH_THRESHOLD),
        )
    )
    icp_oracle_max_tokens = int(
        os.environ.get(
            "BIODEX_ICP_ORACLE_MAX_TOKENS",
            str(DEFAULT_ICP_ORACLE_MAX_TOKENS),
        )
    )

    query = (
        SemanticQueryBuilder(article_dir, model_name=model_name, blocking=blocking)
        .sem_map(scenarios.BIODEX_MAP_REACTIONS)
        .cartesian_product(
            reaction_dir,
            icp=True,
            icp_address=icp_address,
            icp_port=icp_port,
            icp_low_threshold=icp_low_threshold,
            icp_high_threshold=icp_high_threshold,
            icp_oracle_fallback=True,
            icp_oracle_instruction=scenarios.BIODEX_JOIN_REACTION,
            icp_oracle_max_tokens=icp_oracle_max_tokens,
        )
    )


    result, latency = query.execute(endpoint)

    print("\nResponse Summary:")
    print(json.dumps(_response_summary(result), indent=2))
    print(f"\nTotal request time: {latency:.3f} seconds")


if __name__ == "__main__":
    main()
