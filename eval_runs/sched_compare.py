"""Per-minute scheduler state from a Kalypso server log.

Usage: python3 sched_compare.py <server_log> <run_start "YYYY-MM-DD HH:MM:SS"> [max_minutes]

Scheduler lines carry no timestamp, so each one is assigned the timestamp of the
latest preceding vLLM loggers.py stats line (emitted every ~10 s).
Prints, per minute since run start, the mean over scheduler lines of each stage's
cap / used (GB) / running / waiting tasks, plus vLLM running / waiting requests and
KV-cache usage from the stats lines.
"""
import re
import sys
from collections import defaultdict
from datetime import datetime

log, start = sys.argv[1], datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M:%S")
max_min = int(sys.argv[3]) if len(sys.argv) > 3 else 60
year = start.year

TS = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py")
VLLM = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
STAGE = re.compile(r"stage=(\d) used=([\d,]+) cap=([\d,]+) waiting=(\d+) running=(\d+)")
# Table format (newest servers): one row per stage after a "[sched]" header line
ANSI = re.compile(r"\x1b\[[0-9;]*m|^\s*\([A-Za-z_]+[^)]*pid=\d+\)\s?")
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)/([\d.]+)\s+(\d+)\s+(\d+)((?:\s+(?:FULL|SATURATED|STARVING))*)\s*$")
# Compact format (newer servers): S1 27.5/27.5G w539 r0 held27.5
COMPACT = re.compile(r"S(\d+): ([\d.]+)/([\d.]+) GB, wait (\d+), run (\d+)")

sched = defaultdict(lambda: defaultdict(list))  # minute -> key -> values
vllm = defaultdict(list)
now = None
for line in open(log, errors="replace"):
    m = TS.search(line)
    if m:
        now = datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S")
        v = VLLM.search(line)
        if v and now >= start:
            vllm[int((now - start).total_seconds() // 60)].append(
                (int(v.group(1)), int(v.group(2)), float(v.group(3))))
        continue
    row = ROW.match(ANSI.sub("", ANSI.sub("", line)))
    if row and now is not None and now >= start:
        sid, used, cap, waiting, running, _ = row.groups()
        d = sched[int((now - start).total_seconds() // 60)]
        d[f"cap{sid}"].append(float(cap)); d[f"used{sid}"].append(float(used))
        d[f"run{sid}"].append(int(running)); d[f"wait{sid}"].append(int(waiting))
        continue
    if not ("[scheduler]" in line or "[sched]" in line) or now is None or now < start:
        continue
    if "[sched]" in line and not COMPACT.search(line):
        continue
    minute = int((now - start).total_seconds() // 60)
    if "[sched]" in line:
        found = [(sid, float(u), float(c), w, r) for sid, u, c, w, r in COMPACT.findall(line)]
    else:
        found = [(sid, int(u.replace(",", "")) / 1e9, int(c.replace(",", "")) / 1e9, w, r)
                 for sid, u, c, w, r in STAGE.findall(line)]
    for sid, used, cap, waiting, running in found:
        d = sched[minute]
        d[f"cap{sid}"].append(cap)
        d[f"used{sid}"].append(used)
        d[f"run{sid}"].append(int(running))
        d[f"wait{sid}"].append(int(waiting))

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print("min | cap GB s1/s2/s3 | used GB s1/s2/s3 | running tasks s1/s2/s3 | waiting tasks s1/s2/s3 | vLLM run/wait/KV%")
for minute in range(max_min):
    d, v = sched.get(minute), vllm.get(minute)
    if not d and not v:
        continue
    f = lambda k, fmt: "/".join(fmt.format(mean(d[f"{k}{s}"])) for s in "123") if d else "-"
    vv = (f"{mean([x[0] for x in v]):.0f}/{mean([x[1] for x in v]):.0f}/{mean([x[2] for x in v]):.0f}"
          if v else "-")
    print(f"{minute:3d} | {f('cap', '{:.1f}')} | {f('used', '{:.1f}')} | "
          f"{f('run', '{:.0f}')} | {f('wait', '{:.0f}')} | {vv}")
