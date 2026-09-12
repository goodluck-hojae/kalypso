import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests

import scenarios
from cli_utils import parse_query_args


DEFAULT_MEDEC_CSV = (
    Path(__file__).resolve().parent
    / "sample_data"
    / "MEDEC-TrainingSet-1000.csv"
)
DEFAULT_CASCADE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_CASCADE_API_BASE = "http://localhost:8006/v1"
DEFAULT_CASCADE_LOW_THRESHOLD = 0.5
DEFAULT_CASCADE_HIGH_THRESHOLD = 1.0
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


def _results_head(result: dict, limit: int = 5):
    rows = result.get("results")
    if not isinstance(rows, list):
        return result
    return rows[:limit]


def _check_cascade_server(api_base: str):
    models_url = f"{api_base.rstrip('/')}/models"
    try:
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Cascade vLLM server is not reachable at {models_url}") from exc


class SemanticQueryBuilder:
    def __init__(self, data_path: str, model_name: str | None = None):
        self.data_path = data_path
        self.model_name = model_name
        self.plan = []

    def sem_filter(
        self,
        prompt: str,
        *,
        max_tokens: int = 8,
        cascade: bool = False,
        cascade_model: str | None = None,
        cascade_api_base: str | None = None,
        cascade_max_tokens: int = 8,
        cascade_low_threshold: float | None = None,
        cascade_high_threshold: float | None = None,
    ):
        args = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if cascade:
            args.update({
                "cascade": True,
                "cascade_model": cascade_model or DEFAULT_CASCADE_MODEL,
                "cascade_api_base": cascade_api_base or DEFAULT_CASCADE_API_BASE,
                "cascade_max_tokens": cascade_max_tokens,
                "cascade_low_threshold": (
                    cascade_low_threshold
                    if cascade_low_threshold is not None
                    else DEFAULT_CASCADE_LOW_THRESHOLD
                ),
                "cascade_high_threshold": (
                    cascade_high_threshold
                    if cascade_high_threshold is not None
                    else DEFAULT_CASCADE_HIGH_THRESHOLD
                ),
            })

        self.plan.append(
            {
                "op": "sem_filter",
                "args": args,
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

    medec_csv = os.environ.get("MEDEC_CSV", str(DEFAULT_MEDEC_CSV))
    filter_only = os.environ.get("MEDEC_FILTER_ONLY", "0") == "1"

    query = (
        SemanticQueryBuilder(medec_csv, model_name=model_name)
        .sem_filter(
            scenarios.MEDEC_ERROR_FILTER,
            max_tokens=128,
        )
        .sem_map(scenarios.MEDEC_ERROR_SENTENCE_ID_MAP)
        .sem_map(scenarios.MEDEC_CORRECTED_SENTENCE_MAP)
    )

    result, latency = query.execute(endpoint)

    print("\nResponse Summary:")
    print(json.dumps(_response_summary(result), indent=2))
    print(f"\nTotal request time: {latency:.3f} seconds")
