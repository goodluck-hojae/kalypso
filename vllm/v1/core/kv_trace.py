import time
import json

_KV_TRACE = []

def kv_trace(event: str, **fields):
    _KV_TRACE.append({
        "ts": time.monotonic_ns(),
        "event": event,
        **fields,
    })

def dump_trace():
    for e in _KV_TRACE:
        print(json.dumps(e, indent=2))