#!/bin/bash
# One command on a fresh GPU node (inside the job allocation) to resume the fix-4 paper queue.
# Usage: ~/kalypso/eval_scripts/start_queue_on_node.sh
set -e
cd ~/kalypso
git checkout -q scheduler-fix-4
ls vllm/vllm_flash_attn/flash_attn_interface.py >/dev/null 2>&1 || cp .deps/vllm-flash-attn-src/vllm_flash_attn/*.py vllm/vllm_flash_attn/
mkdir -p eval_runs ~/projects/semops-experiments/results/2026-10-01_budgetfix_ablations/{metrics,client,stats} /scratch/hojaeson_umass/logs
cp -n eval_scripts/{start_run.sh,record_run.sh,watch_vs_ref.sh,cfg.py,sched_table.py,sched_compare.py,sched_view.py,ref_*.json,plan_current_backup.py} eval_runs/
cp eval_scripts/paper_queue.sh eval_runs/paper_queue.sh
for s in vllm vllm2 lotus; do tmux has-session -t $s 2>/dev/null || tmux new -s $s -d; done
tmux send-keys -t vllm2 'conda activate ~/.conda/envs/py312 && cd ~/projects/semops-experiments/pipelines/qllm' Enter
tmux send-keys -t lotus 'cd ~/projects/semops-experiments/pipelines/lotus' Enter
if ! curl -s -m 2 localhost:8080/health >/dev/null; then
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hojaeson_umass/datasets setsid nohup ~/.conda/envs/py312/bin/python -u \
    ~/kalypso/vllm/kalypso/icp/vector_service.py --host 127.0.0.1 --port 8080 \
    > /scratch/hojaeson_umass/logs/vector_service_biodex_$(hostname -s).log 2>&1 < /dev/null &
  for i in $(seq 1 40); do curl -s -m 2 localhost:8080/health && break; sleep 3; done; echo
fi
setsid nohup ~/kalypso/eval_runs/paper_queue.sh > ~/kalypso/eval_runs/paper_queue_$(hostname -s).log 2>&1 < /dev/null &
echo "queue started on $(hostname -s); log: ~/kalypso/eval_runs/paper_queue_$(hostname -s).log"
