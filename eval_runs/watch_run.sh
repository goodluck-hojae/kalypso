#!/bin/bash
# Per-minute progress of one client run, then the final time and request count.
# Usage: watch_run.sh <run_name> <client_pid> "<start YYYY-MM-DD HH:MM:SS>" <expected_prompt_tokens_M> "<reference text>"
X=$1; p=$2; START=$3; EXP=$4; REF=$5
A=~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations
CUR=$(date -d "$START" +%s)
P0=$(awk '/^vllm:prompt_tokens_total/{s+=$2} END {print s}' $A/metrics/${X}_before.txt)
R0=$(awk '/^vllm:request_success_total/{s+=$2} END {print s}' $A/metrics/${X}_before.txt)
i=0
while kill -0 $p 2>/dev/null; do
  sleep 20; i=$((i+1)); [ $((i % 3)) -eq 0 ] || continue
  now=$(date +%s); P=$(curl -s -m 10 localhost:8003/metrics | awk '/^vllm:prompt_tokens_total/{s+=$2} END {print s}')
  python3 -c "
el=$now-$CUR; c=$P-$P0
print(f'{el/60:.1f}\t{c/1e6:.1f}M of ~${EXP}M\tETA ~{${EXP}e6/(c/el) if c>0 else float(\"nan\"):.0f}s ($REF)', flush=True)"
done
sleep 3; curl -s localhost:8003/metrics | grep -E "^vllm:(request_success_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)" > $A/metrics/${X}_after.txt
t=$(grep -oE "Total request time: [0-9.]+ seconds" ~/projects/semops-experiments/pipelines/qllm/logs/$X.log | tail -1)
r=$(awk -v r0=$R0 '/^vllm:request_success_total/{s+=$2} END {print s-r0}' $A/metrics/${X}_after.txt)
echo "CLIENT ENDED $X: ${t:-no total line} ($REF) | requests: $r"
