import json
import os
from pathlib import Path

import scenarios
from cli_utils import parse_query_args
from client_contract_nli_filter_join_map import (
    DEFAULT_CONTRACT_DIR,
    DEFAULT_HYPOTHESIS_DIR,
    SemanticQueryBuilder,
    _response_summary,
)


DATA_ROOT = Path(__file__).resolve().parent / "sample_data"
DEFAULT_CATEGORY_DIR = DATA_ROOT / "contract-nli" / "obligation-categories"
DEFAULT_TOP_K = 5


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
    category_dir = os.environ.get(
        "CONTRACT_NLI_CATEGORY_DIR",
        str(DEFAULT_CATEGORY_DIR),
    )
    top_k = int(os.environ.get("CONTRACT_NLI_TOP_K", str(DEFAULT_TOP_K)))

    query = (
        SemanticQueryBuilder(contract_dir, model_name=model_name)
        .sem_filter(scenarios.CONTRACT_NLI_VALID_CONTRACT)
        .sem_join(
            scenarios.CONTRACT_NLI_ENTAILMENT_JOIN,
            hypothesis_dir,
        )
        .sem_map(scenarios.CONTRACT_NLI_EXPLAIN_ENTAILMENT)
        .sem_join(
            scenarios.CONTRACT_NLI_CATEGORY_JOIN,
            category_dir,
        )
        .sem_topk(
            scenarios.CONTRACT_NLI_TOPK_EVIDENCE,
            k=top_k,
        )
        .sem_map(scenarios.CONTRACT_NLI_SUMMARIZE_TOP_EVIDENCE)
    )

    result, latency = query.execute(endpoint)

    print("\nResponse Summary:")
    print(json.dumps(_response_summary(result), indent=2))
    print(f"\nTotal request time: {latency:.3f} seconds")
