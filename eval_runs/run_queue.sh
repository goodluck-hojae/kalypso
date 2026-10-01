#!/usr/bin/env bash
# Kalypso internal-knob evals with the budget fix (branch budget-fix).
# Each config change is committed before its runs; each finished run appends a row to
# eval_runs/results.tsv (committed) and saves client log, server-log segment and
# Prometheus snapshots under $A.
set -uo pipefail

K=/home/hojaeson_umass/kalypso
Q=/home/hojaeson_umass/projects/semops-experiments/pipelines/qllm
A=/home/hojaeson_umass/projects/semops-experiments/results/2026-10-01_budgetfix_ablations
S=$Q/logs/server_vllm_20261001-120129.log      # tmux pipe-pane log of the "vllm" pane
LAUNCH=/home/hojaeson_umass/launch_vllm_py312.sh
LAUNCH_KV=/home/hojaeson_umass/launch_vllm_py312_kvmetrics.sh
RES=$K/eval_runs/results.tsv
LOG=$K/eval_runs/runner.log
CFG="python3 $K/eval_runs/cfg.py"
mkdir -p "$A"/{client,server,metrics}
[ -f "$RES" ] || printf "job\tconfig\tlatency_s\tcalls\tprompt_tokens\tgen_tokens\tpreemptions\tretries\toutput_rows\tclient_log\tserver_log\tfinished\n" > "$RES"

log() { echo "$(date +%T) $*" | tee -a "$LOG"; }
snapshot() { curl -s -m 10 localhost:8003/metrics | grep -E "^vllm:(request_success_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total)" > "$1"; }
server_up() { curl -s -m 5 localhost:8003/v1/models > /dev/null; }
server_pid() { pgrep -f "vllm.entrypoints.openai.api_server.*Llama-3.3-70B" | head -1; }

commit() {
  git -C "$K" add eval_runs vllm/kalypso
  git -C "$K" commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit: $1"
}

restart_server() {  # $1 = env prefix (may be empty), $2 = launch script
  log "restart server: ${1:-defaults} $2"
  local pid; pid=$(server_pid)
  if [ -n "$pid" ]; then
    tmux send-keys -t vllm C-c
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    if kill -0 "$pid" 2>/dev/null; then tmux send-keys -t vllm C-c; sleep 15; fi
    if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; sleep 20; fi
    if kill -0 "$pid" 2>/dev/null; then log "ERROR: server $pid did not stop"; exit 1; fi
  fi
  sleep 5
  tmux send-keys -t vllm "cd ~ && $1 bash $2" Enter
  for _ in $(seq 1 120); do server_up && { log "server up (pid $(server_pid))"; return 0; }; sleep 10; done
  log "ERROR: server did not come up"; exit 1
}

record() {  # name config client_log server_log before after
  python3 - "$@" >> "$RES" <<'EOF'
import re, sys, time
name, config, clog, slog, before, after = sys.argv[1:7]
def load(p):
    d = {}
    for l in open(p):
        m = re.match(r'(vllm:\w+)\{.*\}\s+([\d.e+]+)', l)
        if m: d[m.group(1)] = d.get(m.group(1), 0) + float(m.group(2))
    return d
b, a = load(before), load(after)
dd = lambda k: int(a.get(k, 0) - b.get(k, 0))
txt = open(clog, errors="replace").read()
lat = re.findall(r"Total request time: ([\d.]+)", txt)
rows = re.findall(r'"num_output_rows": (\d+)', txt)
retries = open(slog, errors="replace").read().count("retry-trigger")
print("\t".join([name, config, lat[-1] if lat else "FAILED", str(dd("vllm:request_success_total")),
                 str(dd("vllm:prompt_tokens_total")), str(dd("vllm:generation_tokens_total")),
                 str(dd("vllm:num_preemptions_total")), str(retries), rows[-1] if rows else "-",
                 clog, slog, time.strftime("%m-%d %H:%M")]))
EOF
  log "result: $(tail -1 "$RES" | cut -f1-9)"
}

run_client() {  # name config client timeout_s
  local name=$1 config=$2 client=$3 to=$4
  local done="$A/client/.$name.done" clog="$A/client/$name.log" slog="$A/server/$name.log"
  local off0; off0=$(stat -c %s "$S")
  snapshot "$A/metrics/${name}_before.txt"
  rm -f "$done"
  log "start $name ($config): $client"
  tmux send-keys -t vllm2 "cd $Q && python3 -u $client 2>&1 | tee $clog; echo done > $done" Enter
  local t0; t0=$(date +%s)
  while [ ! -f "$done" ]; do
    sleep 20
    if (( $(date +%s) - t0 > to )); then log "TIMEOUT $name after ${to}s"; tmux send-keys -t vllm2 C-c; sleep 30; break; fi
  done
  snapshot "$A/metrics/${name}_after.txt"
  local off1; off1=$(stat -c %s "$S")
  tail -c +$((off0 + 1)) "$S" | head -c $((off1 - off0)) > "$slog"
  record "$name" "$config" "$clog" "$slog" "$A/metrics/${name}_before.txt" "$A/metrics/${name}_after.txt"
  commit "result: $name ($config)"
}

log "=== queue start ==="

# 0. 3S explicit pinning, started manually at 18:10:35 on the explicit-mode server
#    (commit ce9438fa6). Requests on the server before it: 35,836.
while pgrep -f "^python3 client_contract_nli_multistage.py" > /dev/null; do sleep 30; done
name=explicit_3s
tmux capture-pane -p -J -S - -t vllm2 > "$A/client/.pane.txt"
s=$(grep -n "python3 client_contract_nli_multistage.py" "$A/client/.pane.txt" | tail -1 | cut -d: -f1)
sed -n "${s},\$p" "$A/client/.pane.txt" | sed 's/[[:space:]]*$//' > "$A/client/$name.log"; rm -f "$A/client/.pane.txt"
n=$(sed 's/\x1b\[[0-9;]*m//g' "$S" | grep -n "INFO 10-01 18:1[0-9]" | head -1 | cut -d: -f1)
sed 's/\x1b\[[0-9;]*m//g' "$S" | tail -n +"$n" > "$A/server/$name.log"
printf 'vllm:request_success_total{x="y"} 35836\n' > "$A/metrics/${name}_before.txt"
snapshot "$A/metrics/${name}_after.txt"
record "$name" "explicit pinning, 3S" "$A/client/$name.log" "$A/server/$name.log" "$A/metrics/${name}_before.txt" "$A/metrics/${name}_after.txt"
commit "result: $name (explicit pinning, 3S)"

# 1. No priority (priority=-1), virtual pinning, adaptive budgets.
$CFG virtual on; $CFG priority off; $CFG plan reset
commit "config: no priority (priority=-1 in both request builders), virtual pinning"
restart_server "" "$LAUNCH"
run_client noprio_biodex "no priority" client_biodex_map_icp.py 2400
run_client noprio_nli2s "no priority" client_contract_nli_filter_join_map.py 3600
run_client noprio_3s "no priority" client_contract_nli_multistage.py 7200
$CFG priority on

# 2a. Static 2-stage splits (BioDEX, ContractNLI) at 0.6.
for f in "0.1 0.9" "0.3 0.7" "0.5 0.5" "0.7 0.3" "0.9 0.1"; do
  $CFG plan2 $f
  tag=$(echo "$f" | tr ' ' '_')
  commit "config: static 2-stage split ($f)"
  restart_server "" "$LAUNCH"
  run_client "static2_${tag}_biodex" "static 2-stage ($f)" client_biodex_map_icp.py 3600
  run_client "static2_${tag}_nli2s" "static 2-stage ($f)" client_contract_nli_filter_join_map.py 5400
done

# 2b. Static 3-stage splits (ContractNLI-3S) at 0.6.
for f in "0.1 0.3 0.6" "0.1 0.45 0.45" "0.1 0.6 0.3" "0.2 0.4 0.4" "0.33 0.33 0.34"; do
  $CFG plan3 $f
  tag=$(echo "$f" | tr ' ' '_')
  commit "config: static 3-stage split ($f)"
  restart_server "" "$LAUNCH"
  run_client "static3_${tag}_3s" "static 3-stage ($f)" client_contract_nli_multistage.py 9000
done
$CFG plan reset
commit "config: defaults (adaptive budgets, priority, virtual pinning)"

# 4. Resource sensitivity (BioDEX, ContractNLI) at the other GPU-memory settings.
for u in 0.9 0.8 0.7 0.5; do
  restart_server "VLLM_GPU_MEMORY_UTILIZATION=$u" "$LAUNCH"
  run_client "gpu${u}_biodex" "gpu util $u" client_biodex_map_icp.py 3600
  run_client "gpu${u}_nli2s" "gpu util $u" client_contract_nli_filter_join_map.py 5400
done

# 5. KV-block eviction counts (instrumented server; runtimes not used).
sed 's|^  --port "${VLLM_PORT:-8003}" \\$|  --port "${VLLM_PORT:-8003}" --kv-cache-metrics --kv-cache-metrics-sample 1.0 \\|' "$LAUNCH" > "$LAUNCH_KV"
grep -q -- "--kv-cache-metrics-sample 1.0" "$LAUNCH_KV" || { log "ERROR: could not build $LAUNCH_KV"; exit 1; }
restart_server "" "$LAUNCH_KV"
run_client kvmetrics_biodex "kv-cache-metrics" client_biodex_map_icp.py 3600
run_client kvmetrics_nli2s "kv-cache-metrics" client_contract_nli_filter_join_map.py 5400

restart_server "" "$LAUNCH"
log "=== queue done ==="
