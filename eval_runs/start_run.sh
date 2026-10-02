#!/bin/bash
# Snapshot metrics, start a client in tmux vllm2, print "<start time> <pid>".
# Usage: start_run.sh <run_name> <client_script.py>
X=$1; C=$2; A=~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations
for i in $(seq 1 30); do r=$(curl -s localhost:8003/metrics | awk '/^vllm:num_requests_running/{print $2}'); [ "$r" = "0.0" ] && break; sleep 2; done
curl -s localhost:8003/metrics | grep -E "^vllm:(request_success_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)" > $A/metrics/${X}_before.txt
T=$(date +"%F %T"); tmux send-keys -t vllm2 "python3 -u $C 2>&1 | tee logs/$X.log" Enter; sleep 5
echo "$T $(pgrep -f "^python3 -u $C")"
