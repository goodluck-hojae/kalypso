
<p align="center">
  <img src="icon.png" alt="Kalypso logo" width="300">
</p>
<h1>Kalypso: Relational LLM Serving</h1>



This repository is built on top of vLLM `v0.13.0rc4` with an added semantic
query processor under [`vllm/kalypso`](./vllm/kalypso).

Kalypso is a relational LLM serving system that executes semantic query plans
as memory-aware pipelines, reusing KV-cache state across operators to reduce
recomputation and improve query completion time.

## Installation

Clone the repository and install vLLM from source, Kalypso is built on top of vLLM as a Semantic Query Processing component:

```bash

pip install -U pip setuptools wheel ninja cmake packaging
pip install -r requirements/build.txt
pip install -r requirements/common.txt

pip install -e . --no-build-isolation
```

## Run vLLM

Start the vLLM OpenAI-compatible API server with Llama 3.3 70B:
```bash

VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8003
```


## Deploy ICP and Cascade Service

Some benchmark pipelines use ICP/indexed retrieval. Start the ICP service before
running those clients.

#### ICP Service

For BioDEX, use the default FAISS backend:

```bash

python vllm/kalypso/icp/vector_service.py
```

For FEVER, use the ColBERT backend. Before starting the service, build a
ColBERT index over the Wikipedia data. Then start the ICP service with the
ColBERT backend:

```bash

python vllm/kalypso/icp/vector_service.py --backend colbert
```

#### Cascade Model

Cascade/proxy filtering should use a separate vLLM proxy service. 
For example, run a Llama 8B server and configure benchmark clients with a separate `cascade_api_base` and `cascade_model` explicitly if you use cascade operators.

```bash

VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8004
```

## Benchmark

The example clients live in the benchmark directory[`vllm/kalypso/benchmark`](./vllm/kalypso/benchmark).

For detailed information on the experiment setup, please refer to the paper.

Full benchmark datasets are available as zip files on
[Google Drive](https://drive.google.com/drive/u/0/folders/1N2UvdBGyHPgq5FjdA_FDtCegItCdC8pd).


## Citation

If you use Kalypso, please cite our [paper](https://arxiv.org/abs/2607.23815):

```bibtex
@misc{son2026kalypsorelationalllmserving,
      title={Kalypso: Relational LLM Serving}, 
      author={Hojae Son and Md Ashraful Islam and Huy Gia Cao and Hui Guan and Marco Serafini},
      year={2026},
      eprint={2607.23815},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2607.23815}, 
}
```


## Contact Us

- Hojae Son <hojaeson@umass.edu>
- Md Ashraful Islam <mdashrafulis@umass.edu>
- Huy Gia Cao <hcao@umass.edu>
- Hui Guan <huiguan@cs.umass.edu>
- Marco Serafini <marco@cs.umass.edu>
