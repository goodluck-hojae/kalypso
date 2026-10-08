"""Scheduler state table in fixed time bins from a Kalypso server log.
Usage: python3 sched_table.py <server_log> "<start YYYY-MM-DD HH:MM:SS>" [bin_seconds] [ref.json]"""
import json, re, sys
from datetime import datetime
from collections import defaultdict
log, start = sys.argv[1], datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M:%S")
bin_s = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SCH = re.compile(r"\[sched\] (\d\d:\d\d:\d\d)")
ANSI = re.compile(r"\x1b\[[0-9;]*m|^\s*\([A-Za-z_]+[^)]*pid=\d+\)\s?")
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)/([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)")
b = defaultdict(lambda: defaultdict(list)); cur = None
for line in open(log, errors="replace"):
    l = ANSI.sub("", ANSI.sub("", line))
    m = SCH.search(l)
    if m:
        t = datetime.combine(start.date(), datetime.strptime(m.group(1), "%H:%M:%S").time())
        cur = int((t - start).total_seconds() // bin_s) if t >= start else None
        continue
    r = ROW.match(l)
    if r and cur is not None:
        s, d = r.group(1), b[cur]
        d["cap" + s].append(float(r.group(3))); d["w" + s].append(int(r.group(4)))
        d["r" + s].append(int(r.group(5))); d["d" + s].append(float(r.group(6)))
mean = lambda x: sum(x) / len(x) if x else float("nan")
print("| t (min) | cap s1 / s2 / s3 | waiting s1 / s2 / s3 | running s1 / s2 / s3 | deferred s1 / s2 |")
print("|---|---|---|---|---|")
for k in sorted(b):
    d = b[k]; f = lambda p, fmt: " / ".join(fmt.format(mean(d[p + s])) for s in "123")
    print(f"| {k * bin_s / 60:.1f} | {f('cap', '{:.1f}')} | {f('w', '{:.0f}')} | {f('r', '{:.0f}')} | {mean(d['d1']):.1f} / {mean(d['d2']):.1f} |")
