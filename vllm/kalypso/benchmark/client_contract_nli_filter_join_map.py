import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests

import scenarios
from cli_utils import parse_query_args


DATA_ROOT = Path(__file__).resolve().parent / "sample_data"
DEFAULT_CONTRACT_DIR = DATA_ROOT / "contract-nli" / "contracts"
DEFAULT_HYPOTHESIS_DIR = DATA_ROOT / "contract-nli" / "hypotheses"
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
    def __init__(self, data_path: str, model_name: str | None = None):
        self.data_path = data_path
        self.model_name = model_name
        self.plan = []

    def sem_filter(self, prompt: str):
        self.plan.append(
            {
                "op": "sem_filter",
                "args": {
                    "prompt": prompt,
                },
            }
        )
        return self

    def sem_join(self, instruction: str, right_table: str):
        self.plan.append(
            {
                "op": "join",
                "args": {
                    "instruction": instruction,
                    "right_table": right_table,
                },
            }
        )
        return self

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

    def sem_topk(self, instruction: str, k: int):
        self.plan.append(
            {
                "op": "sem_topk",
                "args": {
                    "instruction": instruction,
                    "k": k,
                },
            }
        )
        return self

    def build(self) -> dict:
        payload = {
            "data_path": self.data_path,
            "ops": self.plan,
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

        response.raise_for_status()
        print(f"Request latency: {elapsed:.3f} seconds")
        return response.json(), elapsed


if __name__ == "__main__":
    model_name, endpoint = parse_query_args()

    contract_dir = os.environ.get(
        "CONTRACT_NLI_CONTRACT_DIR",
        str(DEFAULT_CONTRACT_DIR),
    )
    hypothesis_dir = os.environ.get(
        "CONTRACT_NLI_HYPOTHESIS_DIR",
        str(DEFAULT_HYPOTHESIS_DIR),
    )

    query = (
        SemanticQueryBuilder(contract_dir, model_name=model_name)
        .sem_filter(scenarios.CONTRACT_NLI_VALID_CONTRACT)
        .sem_join(
            scenarios.CONTRACT_NLI_ENTAILMENT_JOIN,
            hypothesis_dir,
        )
        .sem_map(scenarios.CONTRACT_NLI_EXPLAIN_ENTAILMENT)
    )

    result, latency = query.execute(endpoint)

    print("\nResponse Summary:")
    print(json.dumps(_response_summary(result), indent=2))
    print(f"\nTotal request time: {latency:.3f} seconds")
