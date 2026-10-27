"""Per-operator LLM token accounting.

Each LLM call is attributed to the operator that issued it (set via ``current_op``
around operator invocations). Per operator we sum the request ``usage`` returned by
vLLM: prompt tokens, prompt tokens served from the prefix cache (needs the server
flag ``--enable-prompt-tokens-details``), and generated tokens.

"Materialized" tokens = prompt - cached + generated: the KV that had to be
computed (and therefore written to the KV cache) for that operator.
"""
from collections import defaultdict
from contextvars import ContextVar

current_op: ContextVar[str] = ContextVar("kalypso_current_op", default="unknown")

_totals = defaultdict(lambda: defaultdict(int))


def op_label(op, stage_id=None, index=None) -> str:
    name = op.__class__.__name__
    if stage_id is None:
        return f"{index}:{name}" if index is not None else name
    return f"s{stage_id}.{index}:{name}"


def record(usage: dict | None) -> None:
    if not usage:
        return
    t = _totals[current_op.get()]
    prompt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    generated = int(usage.get("completion_tokens") or 0)
    t["calls"] += 1
    t["prompt_tokens"] += prompt
    t["cached_tokens"] += cached
    t["generated_tokens"] += generated
    t["materialized_tokens"] += prompt - cached + generated


def snapshot_and_reset() -> dict:
    out = {op: dict(v) for op, v in _totals.items()}
    _totals.clear()
    return out
