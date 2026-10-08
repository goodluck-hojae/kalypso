#!/bin/bash
# Append one results.tsv row from the before/after metrics and the client log.
# Usage: record_run.sh <run_name> "<config text>" <server_log>
X=$1; CFG=$2; S=$3
A=~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations; L=~/projects/semops-experiments/pipelines/qllm/logs
d(){ b=$(awk "/^vllm:$1/{s+=\$2} END{printf \"%d\", s}" $A/metrics/${X}_before.txt); a=$(awk "/^vllm:$1/{s+=\$2} END{printf \"%d\", s}" $A/metrics/${X}_after.txt); echo $((a-b)); }
t=$(grep -oE "Total request time: [0-9.]+" $L/$X.log | tail -1 | grep -oE "[0-9.]+$")
rows=$(grep -oE '"num_output_rows": [0-9]+' $L/$X.log | grep -oE '[0-9]+$' | tail -1)
cp $L/$X.log $A/client/
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t%s\t%s\t%s\t%s\n" "$X" "$CFG" "$t" $(d request_success_total) $(d prompt_tokens_total) $(d generation_tokens_total) $(d num_preemptions_total) "$rows" "$A/client/$X.log" "$S" "$(date +'%m-%d %H:%M')" >> ~/kalypso/eval_runs/results.tsv
tail -1 ~/kalypso/eval_runs/results.tsv | cut -f1-9

# Also keep the run's full per-query stats (miss/hit/decode tokens, evictions,
# per-operator tokens) from the server log, for the KV and per-operator tables.
mkdir -p $A/stats
grep -o "QUERY_KV_STATS.*" "$S" | tail -1 | sed 's/^QUERY_KV_STATS //' > $A/stats/${X}_query_kv.json
grep -o "OP_TOKEN_STATS.*" "$S" | tail -1 | sed 's/^OP_TOKEN_STATS //' > $A/stats/${X}_op_tokens.json
python3 - "$A/stats/${X}_query_kv.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"  stats: miss {d['prefix_tokens_computed_on_miss']:,} hit {d['prefix_tokens_reused']:,} decode {d['decode_tokens']:,} evicted {d['kv_blocks_evicted']:,} (eviction metric {'on' if d['kv_eviction_metric_available'] else 'OFF'})")
except Exception as e:
    print("  stats: QUERY_KV_STATS not found:", e)
PY
