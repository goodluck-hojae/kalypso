#!/bin/bash
# Paper eval queue on a0008 with the frozen scheduler (commit 0d5ae7766, fix4r2 states).
# Each run: config switch, server restart,
# Lotus MEDEC warmup (40 s), client run to completion, record (results.tsv + stats).
E=~/kalypso/eval_runs
L=~/projects/semops-experiments/pipelines/qllm/logs
A=~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations
QLLM=~/projects/semops-experiments/pipelines/qllm

# Remaining paper runs (as of 2026-10-08 05:00). Run from a fresh shell on the GPU node:
#   ~/kalypso/eval_runs/paper_queue.sh > ~/kalypso/eval_runs/paper_queue.log 2>&1 &
# Needs: tmux sessions vllm / vllm2 / lotus (see note.md), BioDEX vector service on :8080 (GPU 0).

client_for() {
  case $1 in
    biodex) echo client_biodex_map_icp.py ;;
    nli) echo client_contract_nli_filter_join_map.py ;;
    3s) echo client_contract_nli_multistage.py ;;
    3s_blocking) echo client_contract_nli_multistage_blocking.py ;;
  esac
}

run() {  # run <name> <workload> <mem> <priority on|off> <virtual on|off>
  X=$1; W=$2; MEM=$3; PRI=$4; VIRT=$5
  python3 $E/cfg.py priority $PRI; python3 $E/cfg.py virtual $VIRT
  for p in $(ps -eo pid,args | awk '$2 ~ /^VLLM::/ {print $1}'); do [ "$p" = "917067" ] || kill $p 2>/dev/null; done; sleep 10
  for p in $(ps -eo pid,args | awk '$2 ~ /^VLLM::/ {print $1}'); do [ "$p" = "917067" ] || kill -9 $p 2>/dev/null; done; sleep 5
  tmux send-keys -t vllm:0 C-c; sleep 3
  T=$(date +%Y%m%d-%H%M%S); S_LOG=$L/server_vllm_$(hostname -s)_${X}_$T.log
  tmux send-keys -t vllm:0 -l "VLLM_GPU_MEMORY_UTILIZATION=$MEM bash ~/launch_vllm_py312_kvmetrics.sh 2>&1 | tee $S_LOG"
  tmux send-keys -t vllm:0 Enter
  echo "[$X] server $S_LOG ($(date +%T))"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8003/health)" = 200 ]; do sleep 5; done
  # switches are read at server start: restore the defaults on disk right away
  python3 $E/cfg.py priority on; python3 $E/cfg.py virtual on
  cp $E/plan_current_backup.py ~/kalypso/vllm/kalypso/controller/plan.py
  curl -s -m 3 localhost:8080/health >/dev/null || echo "[$X] WARNING vector service down"
  tmux send-keys -t lotus:0 -l '~/.conda/envs/py312/bin/python medec_filter_map_map.py'; tmux send-keys -t lotus:0 Enter
  sleep 40
  for i in 1 2 3 4 5 6; do ps -eo cmd | grep -q "[m]edec_filter_map_map.py" || break; tmux send-keys -t lotus:0 C-c; sleep 3; done
  until [ "$(curl -s localhost:8003/metrics | awk '/^vllm:num_requests_running/{print $2}')" = "0.0" ]; do sleep 2; done
  C=$(client_for $W)
  $E/start_run.sh $X $C
  P=$(pgrep -f "^python3 -u $C")
  echo "[$X] client pid $P ($(date +%T))"
  while kill -0 $P 2>/dev/null; do sleep 20; done
  sleep 3
  curl -s localhost:8003/metrics | grep -E "^vllm:(request_success_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)" > $A/metrics/${X}_after.txt
  $E/record_run.sh $X "new sched 0d5ae7766: $W mem=$MEM priority=$PRI virtual=$VIRT ($(hostname -s))" $S_LOG
  echo "[$X] DONE $(grep -oE 'Total request time: [0-9.]+' $L/$X.log | tail -1)"
}


static3() {  # static3 <f1> <f2> <f3>: Llama-70B 3S static split, full run
  K=~/kalypso/vllm/kalypso/controller/plan.py; B=$E/plan_current_backup.py
  cp $B $K
  python3 - "$1" "$2" "$3" <<'PY'
import sys
p="/home/hojaeson_umass/kalypso/vllm/kalypso/controller/plan.py"; s=open(p).read()
f=", ".join(sys.argv[1:4])
o="            elif num_stages > 1:\n                total_capacity = kv.capacity()"
blk=("            elif num_stages == 3:\n"
f"                # STATIC {f} (temporary)\n"
f"                for sid, fraction in zip(stage_ids, ({f})):\n"
"                    kv.register_stage(sid, fraction, min_fraction=fraction, max_fraction=fraction)\n")
assert s.count(o)==1; open(p,"w").write(s.replace(o, blk+o))
PY
  run static3_${1}_${2}_${3}_3s_$(hostname -s) 3s 0.6 on on
}

# Dataset order alternates (NLI and 3S share data; never back to back).
run ns_3s_nopri            3s          0.6 off on
run ns_nli_nopri           nli         0.6 off on
run ns_biodex_pin          biodex      0.6 on  off
run ns_3s_pin              3s          0.6 on  off
run ns_nli_pin             nli         0.6 on  off
run ns_biodex_mem0.9       biodex      0.9 on  on
run ns_nli_mem0.9          nli         0.9 on  on
run ns_biodex_mem0.8       biodex      0.8 on  on
run ns_nli_mem0.8          nli         0.8 on  on
run ns_biodex_mem0.7       biodex      0.7 on  on
run ns_nli_mem0.7          nli         0.7 on  on
run ns_biodex_mem0.5       biodex      0.5 on  on
run ns_3s_blocking         3s_blocking 0.6 on  on
run ns_nli_mem0.5          nli         0.5 on  on

# BioDEX default again: the first try (2,065 s) had the vector service on CPU
run ns_biodex_default_gpuvec biodex 0.6 on on
run ns_nli_default_a0007   nli    0.6 on  on
run ns_biodex_nopri_a0007  biodex 0.6 off on

# Remaining 3S static splits (moved to the end: long runs)
static3 0.33 0.33 0.34
static3 0.1 0.3 0.6
static3 0.1 0.6 0.3
static3 0.4 0.4 0.2
echo "PAPER QUEUE DONE $(date +%T)"
