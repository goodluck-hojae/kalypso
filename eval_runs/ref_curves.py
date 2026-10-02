"""Cumulative prompt tokens (M) per minute since run start, rebuilt from vLLM
loggers.py throughput lines (every ~10 s). Usage: ref_curves.py <log> <start> -> JSON list."""
import json, re, sys
from datetime import datetime
log, start = sys.argv[1], datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M:%S")
pts, tot, prev = [(0.0, 0.0)], 0.0, start
for l in open(log, errors="replace"):
    m = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py.*Avg prompt throughput: ([\d.]+)", l)
    if not m:
        continue
    t = datetime.strptime(f"{start.year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S")
    if t <= start:
        continue
    tot += float(m.group(2)) * (t - prev).total_seconds(); prev = t
    pts.append(((t - start).total_seconds() / 60, tot / 1e6))
print(json.dumps(pts))
