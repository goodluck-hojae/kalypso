# CPU testing environment and correctness harness

The launcher starts N CPU vLLM workers sequentially on BASE+1 through BASE+N, then the standalone coordinator on BASE (default 8100).
End-to-end runs need the rest of the team's pieces and a working CPU-compatible vLLM build; this harness is unit-tested with fakes.
Activate that environment first; the launcher prefers the checkout's `.venv/bin/python` when present.

```bash
KV_GIB=1 \
  vllm/kalypso/benchmark/launch_multi_worker.sh -n 2 -p round_robin --detach -o ./kalypso_run
python vllm/kalypso/benchmark/correctness_harness.py compare \
  --payload vllm/kalypso/benchmark/sample_data/filter_map_20.json \
  --baseline-endpoint http://localhost:8101/v1/semantic/query \
  --candidate-endpoint http://localhost:8100/v1/semantic/query \
  --out-dir kalypso_run --records kalypso_run/requests.jsonl
vllm/kalypso/benchmark/launch_multi_worker.sh --stop -o ./kalypso_run
```

Worker 1's in-process semantic endpoint is the single-worker baseline; the candidate is `python -m vllm.kalypso.server`.
Use `-n N`, `-m MODEL`, `-b BASE_PORT`, `-p POLICY`, and `-o OUT_DIR` to choose workers, model, coordinator port, policy, and output directory.
The policy defaults to `round_robin` and is passed through to the coordinator.
Worker readiness uses `WORKER_READY_TIMEOUT` (default 900 seconds); coordinator readiness uses `/v1/semantic/healthz` and `COORD_READY_TIMEOUT` (default 600 seconds).
Each worker uses `KV_GIB` GiB of KV cache (default 1), plus model weights and runtime memory.
The launcher sets `KALYPSO_COIN_FLIP_SEED` to 42 unless it is already set; every worker and the coordinator inherit it, and real inference still runs before seeded reply replacement.
Pass `--no-seed` to unset it so operators return real model output; use that mode to try the system, not for `compare`, since real replies can differ between endpoints.
Use the same payload, model, checkout, and seed for both endpoints.

The harness saves `baseline.json`, `candidate.json`, and their input CSVs in `--out-dir`.
Client and servers must share a filesystem because requests submit the CSV's absolute `data_path`.
It checks unique valid output indices, row mapping, and exact operator-value agreement; filtered-out rows are allowed when the baseline agrees.
It also checks non-pin calls exactly once with successful status and full input coverage and prints per-worker call counts.
The launcher clears `requests.jsonl` in the output directory at start; send exactly one candidate query per launch because the recorder appends.
Exit status is 0 on success and 1 on failure; `run` and `check` remain available separately.
Omit `--detach` to keep the launcher attached; Ctrl-C stops its child processes.

## Compare against the old checkout

Run `vllm/kalypso/benchmark/compare_old_new.sh -n 2 -p round_robin -o ./kalypso_old_vs_new` to compare old single-server Kalypso against N coordinator workers on this checkout, with the same payload, model, and seed (default 42).
The old side is `samuel/old-seeded-baseline` (team `main` plus seeded replies); run `git fetch` first, since the script also tries `REMOTE/samuel/old-seeded-baseline` for each remote.
The script checks that ref out as a detached worktree in `OUT_DIR/src_old` and copies this checkout's compiled `vllm/**/*.so` into it; pass `--old-ref REF` for another seeded commit, or `--old DIR` to use an existing checkout.
An existing `src_old` must be a clean worktree at the same commit; an explicitly empty `KALYPSO_COIN_FLIP_SEED` is rejected.
The script prints both commits, stops and waits for the old server before starting the new side, and cleans up servers on exit.
Results are saved as `old.json`, `new.json`, their input CSVs, `old_server.log`, and `new_run/` with request records and logs; the exit status comes from harness `check` using old as baseline.
Use `--help` for payload, model, port, and output options; relative paths resolve from this checkout's root.

## Instrumentation output

The coordinator writes one JSONL file (`--record-path`, `requests.jsonl` in a launch directory)
with three row types; the full field list is in the docstring of `vllm/kalypso/instrumentation.py`.

- `request`: one per LLM call (worker, operator, stage, row, submit/finish times, tokens, status).
- `queue_sample`: one per worker every 0.5 s: waiting, running, `kv_cache_usage` (0-1), and the
  cumulative `prefix_cache_queries` / `prefix_cache_hits`. For one run, subtract the first sample
  of a worker from its last to get that run's queries and hits.
- `worker_info`: one per worker at startup: `max_num_seqs` and `max_num_batched_tokens`.

`worker_info` needs the workers started with `VLLM_SERVER_DEV_MODE=1` (`launch_multi_worker.sh` does
this). Without it the two limits are null and the coordinator prints a warning; it still starts.
That flag also enables other dev endpoints, so use it only on a trusted network.

Each query's `QUERY_KV_STATS` log line also has a `per_worker` object with that worker's
`prefix_tokens_queried`, `prefix_tokens_reused` and `prefix_hit_fraction` for the query.
