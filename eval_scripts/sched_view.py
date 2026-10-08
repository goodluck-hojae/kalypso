"""Compact live view of Kalypso [scheduler] state from a server log.

Usage: python3 sched_view.py <server_log> [interval_s]
Every interval it prints the latest scheduler state as a small table, plus the
latest vLLM stats line. Ctrl-C to stop.
"""
import os
import re
import sys
import time

log = sys.argv[1]
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

STAGE = re.compile(
    r"stage=(\d+) used=([\d,]+) cap=([\d,]+) waiting=(\d+) running=(\d+) "
    r"retries=\d+ deferred_parents=(\d+) deferred_reserved=([\d,]+)"
)
VLLM = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py.*?Avg prompt throughput: ([\d.]+).*?"
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
)
sys.stdout.reconfigure(line_buffering=True)
ANSI = re.compile(r"\x1b\[[0-9;]*m|^\s*\([A-Za-z_]+[^)]*pid=\d+\)\s?")
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)/([\d.]+)\s+(\d+)\s+(\d+)(?:\s+[\d.]+)?((?:\s+(?:FULL|SATURATED|STARVING|BALANCED|BLOCKED))*)(?:\s+low=\d+)?\s*$")
COMPACT = re.compile(r"S(\d+): ([\d.]+)/([\d.]+) GB, wait (\d+), run (\d+)")
gb = lambda s: int(s.replace(",", "")) / 1e9

last_sched, last_vllm, block = None, None, None
with open(log, errors="replace") as f:
    # Read existing lines first so the latest state shows immediately.
    next_print = time.time()
    while True:
        line = f.readline()
        if line:
            row = ROW.match(ANSI.sub("", ANSI.sub("", line)))
            if row:
                sid, u, c, w, r, fl = row.groups()
                block = [x for x in (block or []) if x[0] != sid] + [(sid, float(u), float(c), w, r, fl.strip())]
                last_sched = sorted(block, key=lambda x: int(x[0]))
            elif "[sched]" in line and not COMPACT.search(line):
                block = []
            elif "[scheduler]" in line and "stage=" in line:
                last_sched = [(sid, gb(u), gb(c), w, r, None)
                              for sid, u, c, w, r, p, h in STAGE.findall(line)]
            elif "[sched]" in line:
                last_sched = [(sid, float(u), float(c), w, r, None)
                              for sid, u, c, w, r in COMPACT.findall(line)]
            else:
                m = VLLM.search(line)
                if m:
                    last_vllm = m.groups()
            continue
        if time.time() >= next_print and last_sched:
            next_print = time.time() + interval
            print("\n" + time.strftime("%H:%M:%S"))
            print(f"{'stage':>5} {'used/cap GB':>13} {'wait':>5} {'run':>5}")
            for sid, used, cap, wait, run, flags in last_sched:
                if flags is None:  # older logs carry no signals; derive FULL only
                    flags = "FULL" if cap > 0 and used >= 0.9 * cap and int(wait) > 0 else ""
                print(f"{sid:>5} {used:6.1f}/{cap:<6.1f} {wait:>5} {run:>5}  {flags}".rstrip())
            if last_vllm:
                ts, tput, run, wait, kv = last_vllm
                print(f"vLLM {ts}: run {run}, wait {wait}, KV {kv}%, prompt {float(tput)/1e3:.0f}k tok/s")
        time.sleep(0.2)
