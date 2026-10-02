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
