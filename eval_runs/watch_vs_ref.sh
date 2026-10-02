#!/bin/bash
# Per-minute tokens of a run vs a reference run's curve at the same minute.
# Usage: watch_vs_ref.sh <run_name> <pid> "<start>" <ref.json> "<ref label>" <expected_tokens_M> <ref_total_s>
X=$1; p=$2; START=$3; REF=$4; LAB=$5; EXP=$6; RT=$7
A=~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations; cd ~/kalypso/eval_runs
CUR=$(date -d "$START" +%s); P0=$(awk '/^vllm:prompt_tokens_total/{s+=$2} END {print s}' $A/metrics/${X}_before.txt)
R0=$(awk '/^vllm:request_success_total/{s+=$2} END {print s}' $A/metrics/${X}_before.txt)
i=0
while kill -0 $p 2>/dev/null; do
  sleep 20; i=$((i+1)); [ $((i % 3)) -eq 0 ] || continue
  now=$(date +%s); P=$(curl -s -m 10 localhost:8003/metrics | awk '/^vllm:prompt_tokens_total/{s+=$2} END {print s}')
  python3 -c "
import json,numpy as np
m=($now-$CUR)/60; c=($P-$P0)/1e6; a=np.array(json.load(open('$REF'))); r=float(np.interp(m,a[:,0],a[:,1]))
eta=${EXP}e6/(c*1e6/(m*60)) if c>0 else float('nan')
print(f'{m:.1f}\t{c:.1f}M\t$LAB {r:.1f}M\t{100*(c/r-1):+.0f}%\tETA ~{eta:.0f}s\t($LAB total ${RT}s)', flush=True)"
done
sleep 3; curl -s localhost:8003/metrics | grep -E "^vllm:(request_success_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)" > $A/metrics/${X}_after.txt
t=$(grep -oE "Total request time: [0-9.]+ seconds" ~/projects/semops-experiments/pipelines/qllm/logs/$X.log | tail -1)
r=$(awk -v r0=$R0 '/^vllm:request_success_total/{s+=$2} END {print s-r0}' $A/metrics/${X}_after.txt)
echo "CLIENT ENDED $X: ${t:-no total line} | requests: $r"
