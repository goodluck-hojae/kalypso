# Kalypso evaluation handoff (2026-10-08)

Everything needed to continue the paper evaluation: scheduler setting, results so far, what is left, and exact steps.
Paper repo: `~/projects/pipelining-semantic-operator-tex` (`sections/5-evaluation.tex`). Code: this repo, branch `scheduler-fix-4`.

## 1. Scheduler setting to use ("fix-4", commit `0d5ae7766`)

This is the setting of the best dynamic 3S run (`fix4r2_3s_gate`, 2,123.9 s). Do not change it while running the paper evals.

Code: `vllm/kalypso/execution/pipeline_execution.py` (`rebalance_stage_capacity`), `vllm/kalypso/controller/stage.py`, `vllm/kalypso/controller/plan.py`.

| part | rule | where |
|---|---|---|
| start | every stage gets capacity / n; the last stage gets the remainder | `plan.py`, `equal_share` |
| when | rebalance at the top of the scheduler loop and after every finished task | `pipeline_execution.py`, `await rebalance_stage_capacity()` (6 calls) |
| walk | start at the last stage; go to the previous stage while the current one is starving (`waiting < low`, `low = max(1, 0.2 * cap // worst-case task budget)`) | `more_running`, `Stage.is_starving` |
| asks for memory | stages before the last: its next waiting task does not fit; last stage: its whole waiting queue (next task's budget x waiting) does not fit | `no_free_mem`, `backlog_needs_memory` |
| who gives (first match) | 1. a starving stage with free memory, searched from the last stage; 2. an earlier stage with free memory gives it to the later stage that asks; 3. a stage the walk did not reach, with free memory; 4. any stage whose next task fits and has free memory | `memory_donor` |
| how much (chunk) | 5% of capacity / (observed fan-out from this stage to the last stage) | `TRANSFER_QUANTUM = 0.05`, `transfer_quantum` |
| how often (timer) | after receiving, a stage waits 1 s x (same fan-out factor) before it can receive again | `RECEIVE_WAIT_S = 1.0`, `may_receive` |
| mode | chunk and timer both on, full fan-out scaling | `TRANSFER_MODE = "mix"`, `FANOUT_ALPHA = 1.0` |
| fixed | only free memory moves (deferred parents keep theirs until their last child finishes); no capacity cap | `stage_free` in `budget.py` |

Typical values on Llama-70B 3S: chunk about 4.1 GB / 1.5 GB / 0.1 GB and wait about 1 s / 2.7 s / 40 s for stages 3 / 2 / 1.
Logging: `REBALANCE_LOG = True` in `budget.py`; each `[sched]` table row shows used/cap, waiting, running, deferred GB and `SATURATED`/`STARVING`/`BLOCKED`.

Alternative that was tried (commit `a66336e6a`): memory-based states (starving = free memory fits a task and the queue cannot fill it). 2,298.8 s, but it ran while another user's CPU jobs were on the node (see 3), so it is not a fair comparison with 2,123.9 s. Not used for the paper runs.

## 2. Results so far (Llama-3.3-70B, TP 4, gpu-mem 0.6, KV-metrics server, node a0008)

All rows are in `eval_scripts/results_snapshot_20261008.tsv` (copy of `eval_runs/results.tsv`). Stats per run: `~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations/stats/<run>_{query_kv,op_tokens}.json`.

| run name | workload / config | time (s) | valid? |
|---|---|---|---|
| fix4r2_3s_gate | 3S, fix-4 dynamic | 2123.9 | yes (no other load on the node) |
| memstate3_3s | 3S, memory-based states | 2298.8 | under CPU load |
| static3_0.2_0.4_0.4_3s_a0008 | 3S, static 20/40/40 | 2487.7 | under CPU load (same split was ~2120 s without load) |
| ns_biodex_default | BioDEX default | 2065.2 | NO: vector service ran on CPU; rerun |
| ns_nli_default | NLI default | 1019.7 | under CPU load (old paper number 970.1) |
| ns_biodex_nopri | BioDEX, no priority | 589.6 | under CPU load (old 576.3) |
| qwen32b_static204040_3s | Qwen2.5-32B 3S, static 20/40/40 | 1098.2 | yes |
| qwen32b_dyn_3s | Qwen2.5-32B 3S, fix-4 dynamic | stopped at 10.5 min | partial |

Older reference numbers (paper, before this work): 3S 2043.4 (old adaptive, a0004), static 3S 20/40/40 2117.0 / 33/33/34 2185.7 / 10/30/60 2194.7 / 10/60/30 2197.1 (a0017), BioDEX 571.7, NLI 970.1.

## 3. Things that went wrong (avoid them)

1. **Other users on the node.** From 00:00 on 2026-10-08 another user's 3 CPU jobs (`train_sac.py`) ran on a0008 and slowed every run by 10-17%. Before each run check `ps -eo user,pcpu,args --sort=-pcpu | head`; if other users' jobs use CPU, note it in the run's config text or wait.
2. **BioDEX vector service on CPU** makes BioDEX ~3.6x slower (vLLM idles while the client waits). Always run it on GPU 0 (command in 5.4).
3. **Killing the server.** `kill` on the API server pid can leave `VLLM::EngineCore` / `VLLM::Worker_TP*` alive holding GPUs. Kill by process title: `ps -eo pid,args | awk '$2 ~ /^VLLM::/'` (keep the 8B helper's EngineCore if FEVER needs it). Never `pkill -f` a pattern that appears in your own command line.
4. **Switching branches** between `scheduler-fix-3` and `-4` deletes `vllm/vllm_flash_attn/__init__.py` and `flash_attn_interface.py`. Restore: `cp ~/kalypso/.deps/vllm-flash-attn-src/vllm_flash_attn/*.py ~/kalypso/vllm/vllm_flash_attn/`.
5. **Do not stop runs early.** Let every run finish; token-curve comparisons in the first 10 min were misleading several times.
6. **The job allocation ends.** The a0008 job ended at about 03:51 and killed the queue mid-run (3S no-priority at 12 min). Check the job time limit before starting a long queue.

## 4. What is left (in order)

`eval_scripts/paper_queue.sh` contains exactly these runs; each does server restart, warmup, run to completion, record:

| # | run name | workload | config |
|---|---|---|---|
| 1 | ns_3s_nopri | 3S | priority off |
| 2 | ns_nli_nopri | NLI | priority off |
| 3 | ns_biodex_pin | BioDEX | explicit pinning (virtual off) |
| 4 | ns_3s_pin | 3S | explicit pinning |
| 5 | ns_nli_pin | NLI | explicit pinning |
| 6-13 | ns_{biodex,nli}_mem{0.9,0.8,0.7,0.5} | BioDEX, NLI | gpu-memory-utilization 0.9 / 0.8 / 0.7 / 0.5 |
| 14 | ns_3s_blocking | 3S | blocking client |
| 15 | ns_biodex_default_gpuvec | BioDEX | default (rerun; vector service on GPU) |
| 16-19 | static3_{0.33_0.33_0.34, 0.1_0.3_0.6, 0.1_0.6_0.3, 0.4_0.4_0.2}_3s_a0008 | 3S | static splits |

Total about 10 h. Not in the queue: MEDEC (check whether rebalancing fires; old number 502.6), FEVER (needs the ColBERT index, see 6).
Paper updates after the runs: Section 4 algorithm text (section 1 above), Fig. 7, Tables 3/4, ablation ranges, sensitivity figure, abstract.

## 5. Exact steps on a new GPU node

### 5.1 Code and scripts
```bash
cd ~/kalypso && git fetch kalypso && git checkout scheduler-fix-4 && git log --oneline -1   # 0d5ae7766 or later
ls vllm/vllm_flash_attn/*.py || cp .deps/vllm-flash-attn-src/vllm_flash_attn/*.py vllm/vllm_flash_attn/
mkdir -p eval_runs && cp eval_scripts/* eval_runs/          # eval_runs/ is gitignored; scripts expect it
mkdir -p ~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations/{metrics,client,stats}
```
The server uses the existing build (vLLM 0.1.dev12361, `.so` files from 2026-08-19) and env `~/.conda/envs/py312`.

### 5.2 Data (symlinks; `semops-experiments/data/` is otherwise empty)
```bash
D=~/projects/semops-experiments/data; K=~/kalypso/vllm/kalypso/benchmark
mkdir -p $D/contract-nli
ln -sfn $K/data/contract-nli/contracts $D/contract-nli/contracts
ln -sfn $K/data/contract-nli/hypotheses $D/contract-nli/hypotheses
ln -sfn $K/sample_data/contract-nli/obligation-categories $D/contract-nli/obligation-categories
ln -sfn $K/data/biodex/articles_500 $D/articles_500
ln -sfn $K/data/biodex/reactions $D/reactions
ln -sfn $K/data/medec/MEDEC-TrainingSet-1000.csv $D/MEDEC-Full-TrainingSet-agreement-balanced-1000-with-ErrorType.csv
ln -sfn $K/data/fever/fever_claims_sample_1000_data.csv $D/fever_claims_sample_1000_data.csv
```
Check: a 3S run sends ~48,400 requests and ~124.8M prompt tokens (same as the reference runs).

### 5.3 tmux sessions
```bash
tmux new -s vllm   -d      # server
tmux new -s vllm2  -d      # client
tmux new -s lotus  -d      # warmup
tmux send-keys -t vllm2 'conda activate ~/.conda/envs/py312 && cd ~/projects/semops-experiments/pipelines/qllm' Enter
tmux send-keys -t lotus 'cd ~/projects/semops-experiments/pipelines/lotus' Enter
```
The base env has no pandas: warmup and clients must use `~/.conda/envs/py312` (the queue does this).

### 5.4 BioDEX vector service (port 8080, GPU 0)
```bash
(CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hojaeson_umass/datasets nohup ~/.conda/envs/py312/bin/python -u \
  ~/kalypso/vllm/kalypso/icp/vector_service.py --host 127.0.0.1 --port 8080 \
  > /scratch/hojaeson_umass/logs/vector_service_biodex_gpu.log 2>&1 &)
curl -fsS localhost:8080/health      # {"status":"ok","backend":"faiss"}
```
The 70B server uses GPUs 1-4 (`CUDA_VISIBLE_DEVICES=1,2,3,4` in `~/launch_vllm_py312_kvmetrics.sh`), so GPU 0 is free for this.

### 5.5 Run the queue
```bash
ps -eo user,pcpu,args --sort=-pcpu | head      # no other users' CPU-heavy jobs?
~/kalypso/eval_runs/paper_queue.sh > ~/kalypso/eval_runs/paper_queue.log 2>&1 &
tail -f ~/kalypso/eval_runs/paper_queue.log      # one "[name] DONE Total request time: ..." line per run
```
What one run does (`run` in `paper_queue.sh`): set `cfg.py priority/virtual`, kill the old server by process title, start `~/launch_vllm_py312_kvmetrics.sh` in tmux `vllm` with `VLLM_GPU_MEMORY_UTILIZATION`, wait for `/health`, restore the defaults on disk, 40 s Lotus MEDEC warmup in tmux `lotus`, wait for 0 running requests, start the client in tmux `vllm2` (`start_run.sh`), wait for it to exit, save metrics, `record_run.sh` (row in `eval_runs/results.tsv` + stats JSON).
Static splits (`static3` in the queue) insert a fixed-fraction block into `plan.py` before the server starts and restore `plan.py` (from `eval_runs/plan_current_backup.py`) right after it loads.

### 5.6 Watch a run
```bash
python3 ~/kalypso/eval_runs/sched_table.py <server_log> "<start YYYY-MM-DD HH:MM:SS>" 60   # per-minute stage caps/queues
~/kalypso/eval_runs/watch_vs_ref.sh <run> <client_pid> "<start>" ref_fix4r2_3s.json fix4 124.8 2123.9   # 3S tokens vs fix-4
```
Start time = mtime of `results/.../metrics/<run>_before.txt`.

## 6. Reference: data and vector databases

**Data location (checked 2026-10-06):** BioDEX / ContractNLI / MEDEC / FEVER-claims data are in `~/kalypso/vllm/kalypso/benchmark/data/` (`biodex/{articles_500 (500), reactions (11,271)}`, `contract-nli/`, `medec/`, `fever/fever_claims_sample_1000_data.csv`, plus `.zip` copies). `~/projects/semops-experiments/data/` and `pipelines/lotus/logs/` are empty (emptied 2026-10-02 18:47–18:50); `/scratch/hojaeson_umass/backup/semops-experiments/` has only the directory tree. Point clients at the kalypso copy, e.g. `BIODEX_ARTICLE_DIR=~/kalypso/vllm/kalypso/benchmark/data/biodex/articles_500 BIODEX_REACTION_DIR=~/kalypso/vllm/kalypso/benchmark/data/biodex/reactions`, or symlink it into `semops-experiments/data/`.

**FEVER corpus + ColBERT index: not found** in `~`, `/scratch/hojaeson_umass`, `/work`, or the HF cache (searched for `wikipedia.tsv`, `beir_fever_corpus_data.csv`, `ivf.pid.pt`, `doclens.0.json`, `0.codes.pt`, files > 1 GB). The HF cache does have the models (`colbert-ir/colbertv2.0`, `intfloat/e5-base-v2`). Possibly on an old node's local disk or deleted on 2026-10-02; otherwise rebuild (below).

Needed by the clients (`PROJECT_ROOT = ~/projects/semops-experiments`):

| Workload | Path | Notes |
|---|---|---|
| BioDEX | `data/articles_500/`, `data/reactions/` | override: `BIODEX_ARTICLE_DIR`, `BIODEX_REACTION_DIR` |
| NLI / 3S | `data/contract-nli/` | |
| MEDEC | MEDEC csv (sample: `benchmark/sample_data/MEDEC-TrainingSet-1000.csv`) | |
| FEVER claims | `data/fever_claims_sample_1000_data.csv` | sample copy in `benchmark/sample_data/` |
| FEVER corpus | `data/beir_fever_corpus_data.csv` | only to build the ColBERT index |
| FEVER index | `pipelines/lotus/logs/colbert_indexes/{collections/wikipedia.tsv, wikipedia/indexes/fever_factool_wikipedia_colbert/}` | read by `vector_service.py --backend colbert` |

#### BioDEX: nothing to prebuild
`vector_service.py` without `--backend` uses FAISS (`IndexFlatIP`, model `intfloat/e5-base-v2`). The BioDEX client calls `POST /build_index` with the reaction table at the start of each run, and the index is built in memory. You only need `data/reactions/`, plus the e5 model in `HF_HOME` (downloaded on first use).
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hojaeson_umass/datasets ~/.conda/envs/py312/bin/python -u ~/kalypso/vllm/kalypso/icp/vector_service.py --host 127.0.0.1 --port 8080   # GPU 0, NOT CPU
```

#### FEVER: build the ColBERT Wikipedia index once
Script: `~/projects/semops-experiments/pipelines/lotus/colbert_test.py`. It writes `collections/wikipedia.tsv` from the corpus CSV (text column `data`), then runs ColBERT `Indexer`. ColBERT source: `~/projects/semops-experiments/projects/ColBERT`.
The settings of the original build (from its `plan.json`): checkpoint `colbert-ir/colbertv2.0`, nbits 2, doc_maxlen 180, kmeans_niters 20, nranks 1, experiment `wikipedia`, index name `fever_factool_wikipedia_colbert`. Size: about 5.4M passages, 442M embeddings, 217 chunks, 262,144 partitions, so expect hours on one GPU.
```bash
cd ~/projects/semops-experiments/pipelines/lotus
# env: CUDA_HOME set (cuda/13.1.1 module) and nvcc on PATH, needed for ColBERT's torch extensions
COLBERT_INDEX_ROOT=$PWD/logs/colbert_indexes \
CUDA_VISIBLE_DEVICES=0 $PY -u colbert_test.py \
  --corpus-csv ../../data/beir_fever_corpus_data.csv --text-column data \
  --nbits 2 --nranks 1 --experiment wikipedia --index-name fever_factool_wikipedia_colbert
# result: logs/colbert_indexes/wikipedia/indexes/fever_factool_wikipedia_colbert/
```
`--overwrite reuse` (default) keeps an existing index; use `resume` to continue a partial build.
A stray partial build (only `plan.json`) is in `projects/ColBERT/experiments/wikipedia/`. It is not used.

**Unknown:** how `beir_fever_corpus_data.csv` was made. It is the BEIR FEVER corpus (5.42M docs, matching the 5.4M estimate) saved with a `data` text column, so probably `load_dataset("BeIR/fever", "corpus")` → CSV with `data = title + text`. Not verified; check the original file if you find it.

Then serve it:
```bash
CUDA_VISIBLE_DEVICES= $PY -u ~/kalypso/vllm/kalypso/icp/vector_service.py --host 127.0.0.1 --port 8080 --backend colbert
# other paths: --colbert-experiment-root, --colbert-collection, --colbert-root, --colbert-index-name
curl -fsS localhost:8080/health
```
FEVER also needs the 8B cascade helper on 8004 (`bash ~/launch_vllm_py312_8b.sh`).

## 7. Clients (tmux `vllm2`, dir `pipelines/qllm`)

| workload | client |
|---|---|
| NLI | `client_contract_nli_filter_join_map.py` |
| BioDEX | `client_biodex_map_icp.py` |
| 3S | `client_contract_nli_multistage.py` |
| 3S blocking | `client_contract_nli_multistage_blocking.py` |
| MEDEC | `client_medec_filter_map_map.py` |
| FEVER | `client_fever_factool_map_search_filter_cascade.py` (needs the 8B helper on 8004: `bash ~/launch_vllm_py312_8b.sh`) |

Other models: add `--model-name <hf id>` to the client and the warmup, and `VLLM_MODEL_NAME=<hf id>` to the server launch (see `eval_runs/run_qwen_3s.sh` for Qwen2.5-32B).
