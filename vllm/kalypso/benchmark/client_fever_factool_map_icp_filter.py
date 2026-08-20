import json
import os
import time
from pathlib import Path

import requests

import scenarios
from cli_utils import parse_query_args


DEFAULT_ICP_ADDRESS = "127.0.0.1"
DEFAULT_ICP_PORT = 8080
DEFAULT_ICP_CP_ID = "wiki"
DEFAULT_ICP_TOP_K = 5
DEFAULT_CASCADE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_CASCADE_API_BASE = "http://localhost:8004/v1"

DATA_ROOT = Path(__file__).resolve().parent / "sample_data"
DEFAULT_FEVER_CLAIMS_CSV = DATA_ROOT / "fever_claims_sample_1000_data.csv"


def _response_summary(result: dict) -> dict:
    return {
        "request_id": result.get("request_id"),
        "predicate_result": result.get("predicate_result"),
        "num_output_rows": result.get("num_output_rows"),
        "latency_sec": result.get("latency_sec"),
        "response_keys": sorted(result.keys()),
    }


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
    ):
        icp_args = {
            "right_table": right_table,
            "icp": icp,
            "icp_address": icp_address,
            "icp_port": icp_port,
        }
        if icp_low_threshold is not None or icp_high_threshold is not None:
            icp_args["icp_low_threshold"] = icp_low_threshold
            icp_args["icp_high_threshold"] = icp_high_threshold
        else:
            icp_args["icp_top_k"] = icp_top_k

        self.plan.append(
            {
                "op": "cp",
                "args": icp_args,
            }
        )
        return self
    
    def sem_filter(
        self,
        prompt: str,
        *,
        cascade: bool = False,
        cascade_model: str | None = None,
        cascade_api_base: str | None = None,
        cascade_max_tokens: int = 8,
    ):
        args = {
            "prompt": prompt,
        }
        if cascade:
            args.update({
                "cascade": True,
                "cascade_model": cascade_model or DEFAULT_CASCADE_MODEL,
                "cascade_api_base": cascade_api_base or DEFAULT_CASCADE_API_BASE,
                "cascade_max_tokens": cascade_max_tokens,
            })

        self.plan.append(
            {
                "op": "sem_filter",
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

    fever_claims_csv = os.environ.get("FEVER_CLAIMS_CSV", str(DEFAULT_FEVER_CLAIMS_CSV))
    icp_address = os.environ.get("FEVER_ICP_ADDRESS", DEFAULT_ICP_ADDRESS)
    icp_port = int(os.environ.get("FEVER_ICP_PORT", str(DEFAULT_ICP_PORT)))
    icp_cp_id = os.environ.get("FEVER_ICP_CP_ID", DEFAULT_ICP_CP_ID)
    icp_top_k = int(os.environ.get("FEVER_ICP_TOP_K", str(DEFAULT_ICP_TOP_K)))
    cascade_model = os.environ.get("FEVER_CASCADE_MODEL", DEFAULT_CASCADE_MODEL)
    cascade_api_base = os.environ.get("FEVER_CASCADE_API_BASE", DEFAULT_CASCADE_API_BASE)
    cascade_max_tokens = int(os.environ.get("FEVER_CASCADE_MAX_TOKENS", "8"))
    _check_cascade_server(cascade_api_base)

    query = (
        SemanticQueryBuilder(fever_claims_csv, model_name=model_name, blocking=blocking)
        .sem_map(scenarios.FEVER_FACTOOL_QUERY_MAP)
        .cartesian_product(
            None,
            icp=True,
            icp_address=icp_address,
            icp_port=icp_port,
            icp_top_k=icp_top_k,
        )
        
        .sem_filter(scenarios.FEVER_FACTOOL_SUPPORT_FILTER)
    )

    result, latency = query.execute(endpoint)

    print("\nResponse Summary:")
    print(json.dumps(_response_summary(result), indent=2))
    print(f"\nTotal request time: {latency:.3f} seconds")


if __name__ == "__main__":
    main()
