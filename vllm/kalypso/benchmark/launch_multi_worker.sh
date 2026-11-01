#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
usage() {
    echo 'Usage: launch_multi_worker.sh -n N [-m MODEL] [-b BASE_PORT] [-p POLICY] [-o OUT_DIR] [--no-seed] [--detach]'
    echo '       launch_multi_worker.sh --stop [-o OUT_DIR]'
}
die() { echo "$*" >&2; exit 1; }
n=0 model=Qwen/Qwen2.5-0.5B-Instruct base=8100 policy=round_robin out=./kalypso_run detach=0 stop=0 seeded=1
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --detach) detach=1; shift ;;
        --stop) stop=1; shift ;;
        --no-seed) seeded=0; shift ;;
        -n|-m|-b|-p|-o)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                -n) n=$2 ;; -m) model=$2 ;;
                -b) base=$2 ;; -p) policy=$2 ;; -o) out=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
if ((stop)); then
    [[ -f "$out/pids" ]] || die "No PID file: $out/pids"
    while read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || die "Invalid PID: $pid"
        kill "$pid" 2>/dev/null || true
    done < "$out/pids"
    rm -f "$out/pids"
    exit 0
fi
[[ "$n" =~ ^[1-9][0-9]*$ && "$base" =~ ^[1-9][0-9]*$ ]] || die 'N and ports must be positive integers'
((n <= 65535 && base <= 65535 - n)) || die 'Ports must be at most 65535'
if ((seeded)); then
    export KALYPSO_COIN_FLIP_SEED="${KALYPSO_COIN_FLIP_SEED:-42}"
    echo "Seeded replies on (KALYPSO_COIN_FLIP_SEED=$KALYPSO_COIN_FLIP_SEED)"
else
    unset KALYPSO_COIN_FLIP_SEED
    echo 'Seeded replies off; operators return real model output'
fi
[[ ! -e "$out/pids" ]] || die "Run --stop first: $out/pids already exists"
mkdir -p "$out"
rm -f "$out/requests.jsonl"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python=python
[[ ! -x .venv/bin/python ]] || python="$PWD/.venv/bin/python"
command -v curl >/dev/null || die 'curl is required'
threads=$("$python" -c 'import os,sys; print(max(1, (os.cpu_count() or 1) // int(sys.argv[1])))' "$n")
pids=()
cleanup() {
    if ((${#pids[@]})); then kill "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready() {
    local url=$1 timeout=$2 log=$3 pid=$4 deadline=$((SECONDS + $2)) code
    while ((SECONDS < deadline)); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url") || code=000
        [[ "$code" != 200 ]] || return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    tail -n 30 "$log" >&2
    die "Readiness failed (timeout ${timeout}s or process exited): $url"
}
[[ "${WORKER_READY_TIMEOUT:-900}" =~ ^[1-9][0-9]*$ ]] || die 'WORKER_READY_TIMEOUT must be a positive integer'
[[ "${COORD_READY_TIMEOUT:-600}" =~ ^[1-9][0-9]*$ ]] || die 'COORD_READY_TIMEOUT must be a positive integer'
worker_urls=""
for ((i=1; i<=n; i++)); do
    started=$SECONDS
    VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=${KV_GIB:-1} OMP_NUM_THREADS=$threads nohup "$python" -m vllm.entrypoints.openai.api_server --model "$model" --enable-prefix-caching --max-model-len 4096 --port "$((base+i))" > "$out/worker_$i.log" 2>&1 < /dev/null &
    pids+=("$!")
    worker_urls+="${worker_urls:+,}http://localhost:$((base+i))"
    ready "http://localhost:$((base+i))/v1/models" "${WORKER_READY_TIMEOUT:-900}" "$out/worker_$i.log" "${pids[i-1]}"
    echo "Worker $i ready after $((SECONDS - started))s"
done
started=$SECONDS
PYTHONUNBUFFERED=1 nohup "$python" -m vllm.kalypso.server --port "$base" --workers "$worker_urls" --policy "$policy" --record-path "$out/requests.jsonl" > "$out/coordinator.log" 2>&1 < /dev/null &
coord_pid=$!
pids+=("$coord_pid")
ready "http://localhost:$base/v1/semantic/healthz" "${COORD_READY_TIMEOUT:-600}" "$out/coordinator.log" "$coord_pid"
echo "Coordinator ready after $((SECONDS - started))s"
echo "Coordinator: http://localhost:$base/v1/semantic/query"
echo "Single-worker baseline: http://localhost:$((base+1))/v1/semantic/query"
if ((detach)); then
    printf '%s\n' "${pids[@]}" > "$out/pids"
    trap - EXIT INT TERM
    exit 0
fi
wait
