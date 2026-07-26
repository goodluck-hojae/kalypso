# Kalypso: Relational LLM Serving

<p align="center">
  <img src="icon.png" alt="Kalypso logo" width="420">
</p>

This repository is built on top of vLLM `v0.13.0rc4` with an added semantic
query processor under `vllm/kalypso`.

Kalypso is a relational LLM serving system that executes semantic query plans
as memory-aware pipelines, reusing KV-cache state across operators to reduce
recomputation and improve query completion time.

## Install

Clone the repository and install it from source:

```bash
cd <repo-root>

pip install -U pip setuptools wheel ninja cmake packaging
pip install -r requirements/build.txt
pip install -r requirements/common.txt

pip install -e . --no-build-isolation
```

## Run vLLM

Start the vLLM OpenAI-compatible API server with Llama 3.3 70B:

```bash
cd <repo-root>

VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8003
```


## Run ICP and Cascade Services

Some benchmark pipelines use ICP/indexed retrieval. Start the ICP service before
running those clients.

### ICP Service

For BioDEX, use the default FAISS backend:

```bash
cd <repo-root>

python vllm/kalypso/icp/vector_service.py \
  --host 127.0.0.1 \
  --port 8080
```

For FEVER, use the ColBERT backend. Before starting the service, build a
ColBERT index over the Wikipedia data. Then start the ICP service with the
ColBERT backend:

```bash
cd <repo-root>

python vllm/kalypso/icp/vector_service.py \
  --host 127.0.0.1 \
  --port 8080 \
  --backend colbert
```

### Cascade Model

Cascade/proxy filtering should use a separate vLLM endpoint. For example, run a
Llama 8B server and configure benchmark clients with a separate
`cascade_api_base`. For the 70B setup above, pass `cascade_model` explicitly if
you use cascade operators.

```bash
cd <repo-root>

VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8004
```

## Benchmark

The example clients live in the benchmark directory:

```bash
cd <repo-root>/vllm/kalypso/benchmark
```

Each client sends a request to the semantic query endpoint. By default, the
clients use `meta-llama/Llama-3.3-70B-Instruct`.

Small benchmark datasets are bundled under:

```bash
vllm/kalypso/benchmark/sample_data
```

Included sample data contains 10 records per dataset:

- `fever_claims_sample_1000_data.csv`
- `MEDEC-TrainingSet-1000.csv`
- `articles_500/`
- `reactions/`
- `contract-nli/`

Full benchmark datasets are available as zip files on
[Google Drive](https://drive.google.com/drive/u/0/folders/1N2UvdBGyHPgq5FjdA_FDtCegItCdC8pd).

Run the FEVER Factool map, indexed search, and filter pipeline:

```bash
python client_fever_factool_map_search_filter_op.py
```

Run the FEVER Factool map, ICP, and filter pipeline:

```bash
python client_fever_factool_map_icp_filter.py
```

Run the MEDEC filter and two-map pipeline:

```bash
python client_medec_filter_map_map.py
```

Run the BioDEX map and ICP pipeline:

```bash
python client_biodex_map_icp.py
```

Run the Contract NLI filter, join, and map pipeline:

```bash
python client_contract_nli_filter_join_map.py
```

## Contact Us

- Hojae Son <hojaeson@umass.edu>
- Md Ashraful Islam <mdashrafulis@umass.edu>
- Huy Gia Cao <hcao@umass.edu>
- Hui Guan <huiguan@cs.umass.edu>
- Marco Serafini <marco@cs.umass.edu>
