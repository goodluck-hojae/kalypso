# Kalypso — working notes (2026-10-06)

Branch: `scheduler-fix-2`. Paper: `~/projects/pipelining-semantic-operator-tex`.
Local-only helpers (gitignored, copy them by hand if the new node has no shared home): `eval_runs/` (record_run.sh, start_run.sh, watch_*.sh, sched_view.py, sched_compare.py, cfg.py, ref_*.json, results.tsv, TODO.md).

## What was done

### Scheduler (this commit)
New rebalancing, replacing the old starving / saturated / idle-return rules and α:
- `low = max(1, β·cap/min_budget)`, β = 0.2 (`Stage.LOW_THRESHOLD_RATIO`).
- busy: `running ≥ low`; saturated: not busy and `waiting ≥ low`; starving: not busy and `waiting < low`.
- `MORE_RUNNING` walks from the last stage only: busy → stop; saturated → receives from `PICK_DONOR`, stop; starving → previous stage.
- `PICK_DONOR` (from the last stage): first starving with free > 0, then busy with free > 0. Saturated stages never donate.
- Transfer = min(5% capacity, donor free, receiver max); upstream transfers cap a non-last receiver at an equal share (`budget.transfer_capacity`).
- `free = cap − max(min_cap, used)`; used includes deferred (fan-out parent) memory, running excludes it.
- Rebalance now runs **once per scheduler wake-up** (top of the `while` loop). The five per-finished-task calls in `pipeline_execution.py` are commented out: they let several transfers happen before any admission (up to K×5% when K tasks finish together, and 2 decisions even for K = 1).
- Scheduler log table has a `defer GB` column.
- `eval_runs/` added to `.gitignore`.

None of this has been run yet.

### Paper
- Table 3: PFLOP Blocking / Pipeline / Reduction columns (FEVER 6.1%, MEDEC 40.4%, BioDEX 36.6%, NLI 44.3%, 3S TBD).
- Table 4: per-operator calls / tokens / KV GB / PFLOP.
- Static 3S: 20/40/40 will be replaced by 45/45/10 (needs run #16 below). Then the best-static margin at line 453 changes (33/33/34 = 2185.7 s is the best remaining unless 45/45/10 wins).

### Results so far (old default, commit b9cbed049)
3S 2043.4 s, BioDEX 571.7 s, NLI 970.1 s. Explicit pinning: NLI 1027.3 / BioDEX 590.1 / 3S 2400.2. No priority: NLI 969.4 / BioDEX 576.3 / 3S 2030.6. Sensitivity (BioDEX / NLI): 0.9 564.7/947.1, 0.8 564.9/945.6, 0.7 564.2/953.7, 0.5 592.6/1011.8. All will be rerun with the new scheduler.

## Open question to check in the gate log
A downstream stage waiting for children from upstream looks *starving*, so it can be picked as a donor just before its children arrive, then need the memory back. Unverified. Check the 3S gate log for: transfer `donor=<last> receiver=1` followed soon by the last stage SATURATED. Fix only if it shows up and the run is behind (idea: `inflow(s) = running(prev stage)`, starving iff `waiting + inflow < low`).

## Runs

| # | Phase | Run | Workload | Paper target | Status |
|---|---|---|---|---|---|
| 1 | 1 Gate | New default, 20 min vs `ref_newdefault_3s.json` | 3S | go / no-go | pending |
| 2 | 1 | New default, full | 3S | Fig. 7, Fig. 14 (rebalancing + defer GB), Table 3 pipeline, Table 4 3S rows | pending |
| 3 | 2 Defaults | New default | NLI | Fig. 7, speedups, abstract, Table 3/4 | pending |
| 4 | 2 | New default | BioDEX | Fig. 7, speedups, abstract, Table 3/4 | pending |
| 5 | 3 Ablations | No priority (`cfg.py priority off`) | NLI | ablation text/ranges | pending |
| 6 | 3 | No priority | BioDEX | ablation text/ranges | pending |
| 7 | 3 | No priority | 3S | ablation text/ranges | pending |
| 8 | 3 | Explicit pinning (`cfg.py virtual off`) | NLI | ablation text/ranges | pending |
| 9 | 3 | Explicit pinning | BioDEX | ablation text/ranges | pending |
| 10 | 3 | Explicit pinning | 3S | ablation text/ranges | pending |
| 11 | 3 | Sensitivity 0.9 | NLI, BioDEX | sensitivity fig/text | pending |
| 12 | 3 | Sensitivity 0.8 | NLI, BioDEX | sensitivity fig/text | pending |
| 13 | 3 | Sensitivity 0.7 | NLI, BioDEX | sensitivity fig/text | pending |
| 14 | 3 | Sensitivity 0.5 | NLI, BioDEX | sensitivity fig/text | pending |
| 15 | 3 | Blocking (`client_contract_nli_multistage_blocking.py`) | 3S | Table 3 blocking miss/hit/evict/PFLOP + Reduction | pending |
| 16 | 3 | Static 45/45/10 (`cfg.py plan3 0.45 0.45 0.1`), replaces 20/40/40 | 3S | Fig. 7 static panel, best-static margin (line 453) | pending |
| 17 | — | Does rebalancing fire? (grep `transfer` in sched log); rerun if yes | FEVER, MEDEC | Table 3, Fig. 7 | check |

Unchanged (no rerun): static 2-stage splits, other static 3S splits, blocking runs except 3S.
On hold: node check (a0004 vs a0017).

## Paper text

| # | Item | Status |
|---|---|---|
| T1 | Section 4: new algorithm (MORE_RUNNING / PICK_DONOR), β = 0.2, α removed | pending |
| T2 | Recompute derived numbers: abstract ×, ablation ranges, KV %, PFLOP reduction | pending (after 3–14) |
| T3 | Line 438 note + Fig. 7 3S labels for 45/45/10 | pending (after 16) |


## How to run

### Rules
- Order inside a phase: NLI → BioDEX → 3S (3S last). Never run the same dataset back to back on one server (NLI and 3S share the contract data).
- Every run on the KV-metrics server (`~/launch_vllm_py312_kvmetrics.sh`), including blocking runs.
- Config switches (`cfg.py`) are read at server start → restart after each change, reset to defaults afterwards (`cfg.py priority on`, `cfg.py virtual on`, `cfg.py plan reset`).
- One run per config.

### 1. Server (tmux `vllm`)
```bash
L=~/projects/semops-experiments/pipelines/qllm/logs; T=$(date +%Y%m%d-%H%M%S)
bash ~/launch_vllm_py312_kvmetrics.sh 2>&1 | tee $L/server_vllm_$(hostname -s)_<tag>_$T.log
# sensitivity: VLLM_GPU_MEMORY_UTILIZATION=0.8 bash ~/launch_vllm_py312_kvmetrics.sh ...
```
Uses GPUs 1–4 (`CUDA_VISIBLE_DEVICES=1,2,3,4`), port 8003. If Ctrl-C doesn't stop it: `pkill -f vllm.entrypoints` (then `kill -9`).

### 2. Warmup (tmux `lotus`, dir `pipelines/lotus`)
```bash
python3 medec_filter_map_map.py      # 40 s, then Ctrl-C until it exits
# if the next run is MEDEC, warm up with: python3 contract_nli_filter_join_map.py
```
Wait for 0 running requests: `curl -s localhost:8003/metrics | grep ^vllm:num_requests_running`.
Do not use Lotus `biodex.py` (too slow).

### 3. Vector service (port 8080) — needed for FEVER and BioDEX only
Script: `~/kalypso/vllm/kalypso/icp/vector_service.py`. Only one service can use port 8080, so restart it when switching between FEVER and BioDEX.
```bash
PY=/scratch/hojaeson_umass/miniforge3/envs/rtx/bin/python
pkill -f "vector_service.py.*--port 8080"
# BioDEX (faiss, default backend)
CUDA_VISIBLE_DEVICES= $PY -u ~/kalypso/vllm/kalypso/icp/vector_service.py --host 127.0.0.1 --port 8080
# FEVER (ColBERT Wikipedia index)
CUDA_VISIBLE_DEVICES= $PY -u ~/kalypso/vllm/kalypso/icp/vector_service.py --host 127.0.0.1 --port 8080 --backend colbert
curl -fsS localhost:8080/health   # ready check
```
ColBERT needs `CUDA_HOME` (cuda/13.1.1 module) for its extension build; see `~/exp_factool.sh` for the full env setup.

### 4. FEVER cascade helper (8B, port 8004, GPU 0)
```bash
bash ~/launch_vllm_py312_8b.sh     # VLLM_8B_CUDA_VISIBLE_DEVICES=0 by default
```
FEVER client: `client_fever_factool_map_search_filter_cascade.py` (thresholds 0.7/0.9).

### 5. Client (tmux `vllm2`, dir `pipelines/qllm`)
```bash
~/kalypso/eval_runs/start_run.sh <run_name> <client.py>    # prints start time + pid
pgrep -f '^python3 -u <client.py>'                           # confirm it is running
```
| Workload | Client |
|---|---|
| NLI | `client_contract_nli_filter_join_map.py` |
| BioDEX | `client_biodex_map_icp.py` |
| 3S | `client_contract_nli_multistage.py` |
| 3S blocking | `client_contract_nli_multistage_blocking.py` |
| FEVER | `client_fever_factool_map_search_filter_cascade.py` |
| MEDEC | `client_medec_filter_map_map.py` |

### 6. Monitor
```bash
~/kalypso/eval_runs/watch_vs_ref.sh <run_name> <pid> "<start>" ref_newdefault_3s.json "old default" <expected_tokens_M> 2043.4
python3 ~/kalypso/eval_runs/sched_view.py <server_log>
python3 ~/kalypso/eval_runs/sched_compare.py <server_log> "<start>" 40
```
Gate rule: if behind the reference at ~10 min, stop, analyze the sched log, fix, retest.

### 7. Record
```bash
A=~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations
curl -s localhost:8003/metrics | grep -E "^vllm:(request_success_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)" > $A/metrics/<run_name>_after.txt
~/kalypso/eval_runs/record_run.sh <run_name> "<config text>" <server_log>
```
Saves a `results.tsv` row, client log, `stats/<run>_query_kv.json` (miss/hit/decode/evictions) and `stats/<run>_op_tokens.json` (per-op). Check that the printed eviction metric says `on`.
