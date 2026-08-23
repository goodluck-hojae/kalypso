import os

from client_contract_nli_multistage import main


if __name__ == "__main__":
    os.environ.setdefault("QLLM_OUTPUT_TAG", "blocking")
    main(blocking=True)
