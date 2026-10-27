import csv
from pathlib import Path
import threading
import asyncio
import torch
from transformers import AutoConfig, AutoTokenizer


def compute_bytes_per_token(
    model_name: str,
    dtype: torch.dtype = torch.float16,
) -> int:
    cfg = AutoConfig.from_pretrained(model_name)

    num_layers = cfg.num_hidden_layers

    # GQA / MQA aware
    num_kv_heads = getattr(
        cfg,
        "num_key_value_heads",
        cfg.num_attention_heads,
    )

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    dtype_bytes = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
    }[dtype]

    return (
        2 *                 
        num_layers *
        num_kv_heads *
        head_dim *
        dtype_bytes
    )


import contextvars
import time
_QUERY_KV_MANAGER = contextvars.ContextVar("kalypso_query_kv_manager", default=None)

# Opt-in admission settings for multi-query tests (eval_runs/admit_cfg_<host>); all off by default.
_REAL_KV = {"t": 0.0, "v": 0.0}
_RECENT_ADMITS = []  # (time, bytes) admitted recently, not yet visible in vLLM's KV usage


def _real_kv_usage() -> float:
    """vLLM's actual KV-cache usage (0..1), read from this process's Prometheus registry; cached 0.25 s."""
    now = time.monotonic()
    if now - _REAL_KV["t"] > (0.02 if _UNSEEN or _ENGINE["client"] else 0.25):
        from prometheus_client import REGISTRY
        v = 0.0
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == "vllm:kv_cache_usage_perc":
                    v = max(v, float(sample.value))
        _REAL_KV.update(t=now, v=v)
    return _REAL_KV["v"]


# Live-memory admission (opt-in, "live=<line>"): one check shared by all queries against vLLM's actual
# KV use plus work that is admitted but not yet visible in it. asyncio runs the check and the recording
# without a switch in between, so two queries can never claim the same free memory.
_ENGINE = {"client": None}  # vLLM engine client, set at query start
_ADMIT_ENTRY = contextvars.ContextVar("kalypso_admit_entry", default=None)
_UNSEEN = {}  # entry -> [tokens, vLLM request id or None]: admitted, first LLM request not yet in vLLM
_next_entry = [0]


def _pending_tokens() -> int:
    """Tokens admitted but not yet in vLLM's KV usage: prompts vLLM has received but not computed,
    and admitted tasks whose first request vLLM has not received yet."""
    eng = _ENGINE["client"]
    states = getattr(getattr(eng, "output_processor", None), "request_states", None) or {}
    pending = sum((s.prompt_len or 0) for s in list(states.values()) if s.is_prefilling)
    for key, (tokens, rid) in list(_UNSEEN.items()):
        if rid is not None and rid in states:
            _UNSEEN.pop(key, None)  # now counted as a prefilling prompt, then by vLLM's usage
        else:
            pending += tokens
    return pending


def drop_admit_entry(entry):
    _UNSEEN.pop(entry, None)


class _QueryPool:
    """One KV pool shared by concurrent queries; each query keeps a guaranteed minimum."""

    def __init__(self, capacity, min_share):
        self.capacity = capacity
        self.min_bytes = capacity * min_share
        self.used = {}

    def fits(self, key, budget) -> bool:
        reserved_for_others = sum(
            max(0.0, self.min_bytes - u) for k, u in self.used.items() if k != key
        )
        return sum(self.used.values()) + budget <= self.capacity - reserved_for_others

    def add(self, key, budget):
        self.used[key] = max(0.0, self.used.get(key, 0.0) + budget)


class KVMemoryManager:
    ONE_TUPLE_TOKENS = 8192
    _instance = None
    _init_lock = threading.Lock()
    LOG = False
    REBALANCE_LOG = True
    # Opt-in (multi-query tests): admit beyond the budget while vLLM's real KV usage is below this line.
    real_admit = 0.0
    REAL_WINDOW = 2.0  # seconds an admission counts as not yet visible in the real usage
    _pool = None
    _total = None
    live_admit = 0.0  # opt-in: admit by live memory only (see _pending_tokens)
    _last_entry = None
    def __init__(self, model_name, kv_capacity, dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bytes_per_token = compute_bytes_per_token(model_name, dtype)

        self._capacity = kv_capacity 
        self._global_used = 0

        self._stage_capacity = {}
        self._stage_min_capacity = {}
        self._stage_max_capacity = {}
        self._stage_used = {}
        self._stage_inflight = {}

        self._cond = asyncio.Condition()

    # Stage API (Pipeline)
    @classmethod
    def fork_for_query(cls, share: float, pool_min: float = 0.0, real_admit: float = 0.0,
                       live_admit: float = 0.0):
        """Query-local manager with `share` of the global KV capacity and its own stages (multi-query mode).
        pool_min > 0: the query may use the whole capacity, from one pool shared by all queries, with
        `pool_min` of it guaranteed to each. real_admit > 0: see `real_admit`."""
        base = cls.get_instance()
        m = object.__new__(cls)
        m.tokenizer = base.tokenizer
        m.bytes_per_token = base.bytes_per_token
        m._total = base._capacity
        m.real_admit = real_admit
        m.live_admit = live_admit
        if pool_min > 0:
            share = 1.0
            if base.__dict__.get("_mq_pool") is None:
                base._mq_pool = _QueryPool(base._capacity, pool_min)
            m._pool = base._mq_pool
            m._pool.used[id(m)] = 0.0
        m._capacity = base._capacity * share
        m._global_used = 0
        m._stage_capacity, m._stage_min_capacity, m._stage_max_capacity = {}, {}, {}
        m._stage_used, m._stage_inflight = {}, {}
        m._cond = asyncio.Condition()
        return m

    def leave_pool(self):
        if self._pool is not None:
            self._pool.used.pop(id(self), None)

    def poll_interval(self):
        """How often the scheduler re-checks admission without a task finishing (None = only on finish)."""
        if self.live_admit > 0:
            return 0.05
        return 0.25 if (self.real_admit > 0 or self._pool is not None) else None

    def take_admit_entry(self):
        entry, self._last_entry = self._last_entry, None
        return entry

    def _admit_check(self, stage_id: int, budget: int) -> bool:
        if self.live_admit > 0:
            total = self._total or self._capacity
            used = _real_kv_usage() * total + _pending_tokens() * self.bytes_per_token
            if used + budget > self.live_admit * total:
                return False
            _next_entry[0] += 1
            self._last_entry = _next_entry[0]
            _UNSEEN[self._last_entry] = [budget / self.bytes_per_token, None]
            return True
        ok = self._stage_used[stage_id] + budget <= self._stage_capacity[stage_id]
        if ok and self._pool is not None:
            ok = self._pool.fits(id(self), budget)
        if self.real_admit > 0:
            now = time.monotonic()
            _RECENT_ADMITS[:] = [(t, b) for t, b in _RECENT_ADMITS if now - t < self.REAL_WINDOW]
            if not ok:
                room = (self.real_admit - _real_kv_usage()) * (self._total or self._capacity)
                ok = budget <= room - sum(b for _, b in _RECENT_ADMITS)
            if ok:
                _RECENT_ADMITS.append((now, budget))
        return ok

    def reset_stages(self):
        """Forget the stages of earlier queries. Called at the start of each query,
        so stale stages neither hold capacity nor act as donors or receivers."""
        for d in (
            self._stage_capacity, self._stage_min_capacity, self._stage_max_capacity,
            self._stage_used, self._stage_inflight,
        ):
            d.clear()

    def register_stage(
        self,
        stage_id: int,
        fraction: float,
        min_fraction: float | None = None,
        max_fraction: float | None = None,
        min_capacity_bytes: int | None = None,
    ):
        min_fraction = fraction if min_fraction is None else min_fraction
        max_fraction = fraction if max_fraction is None else max_fraction
        self._stage_capacity[stage_id] = self._capacity * fraction
        min_cap = self._capacity * min_fraction
        if min_capacity_bytes is not None:
            min_cap = max(min_cap, min_capacity_bytes)
        self._stage_min_capacity[stage_id] = min(min_cap, self._stage_capacity[stage_id])
        self._stage_max_capacity[stage_id] = self._capacity * max_fraction
        self._stage_used[stage_id] = 0
        self._stage_inflight[stage_id] = 0

    def _log_rebalance(self, event: str):
        if not self.REBALANCE_LOG:
            return
        stage_states = []
        for stage_id in sorted(self._stage_capacity):
            stage_states.append(
                f"S{stage_id}: "
                f"{self._stage_used.get(stage_id, 0) / 1e9:.1f}"
                f"/{self._stage_capacity.get(stage_id, 0) / 1e9:.1f} GB, "
                f"min {self._stage_min_capacity.get(stage_id, 0) / 1e9:.1f}, "
                f"max {self._stage_max_capacity.get(stage_id, 0) / 1e9:.1f}, "
                f"run {int(self._stage_inflight.get(stage_id, 0))}"
            )
        print(f"[rebalance] {event} | " + " | ".join(stage_states))

    def _last_stage_id(self) -> int | None:
        if not self._stage_capacity:
            return None
        return max(self._stage_capacity)

    async def rebalance_stage_capacity(
        self,
        receiver_id: int,
        donor_hint: int | None = None,
        quantum_fraction: float = 0.05,
        upstream: bool = False,
    ) -> bool:
        quantum = max(1, int(self._capacity * quantum_fraction))

        async with self._cond:
            receiver_cap = self._stage_capacity.get(receiver_id, 0)
            receiver_max = self._stage_max_capacity.get(receiver_id, receiver_cap)
            # Memory moved upstream (to an earlier stage) may not grow it beyond an
            # equal share of capacity: an early stage's memory ends up held by
            # fan-out parents until their descendants finish, so it cannot serve
            # the later stages that need it.
            if upstream and receiver_id != self._last_stage_id():
                receiver_max = min(receiver_max, self._capacity / len(self._stage_capacity))
            if receiver_cap >= receiver_max:
                return False

            donor_ids = []
            if donor_hint is not None and donor_hint != receiver_id:
                donor_ids.append(donor_hint)
            donor_ids.extend(
                stage_id
                for stage_id in self._stage_capacity
                if stage_id not in donor_ids and stage_id != receiver_id
            )

            for donor_id in donor_ids:
                donor_cap = self._stage_capacity.get(donor_id, 0)
                donor_min = self._stage_min_capacity.get(donor_id, donor_cap)
                donor_used = self._stage_used.get(donor_id, 0)
                donor_floor = max(donor_min, donor_used)
                borrowable = donor_cap - donor_floor
                if borrowable <= 0:
                    continue

                delta = min(
                    quantum,
                    borrowable,
                    receiver_max - receiver_cap,
                )
                if delta <= 0:
                    continue

                self._stage_capacity[donor_id] -= delta
                self._stage_capacity[receiver_id] += delta
                self._log_rebalance(
                    f"borrow receiver={receiver_id} donor={donor_id} moved {delta / 1e9:.1f} GB"
                )
                self._cond.notify_all()
                return True

            return False

    def stage_free(self, stage_id: int) -> int:
        """Capacity a stage can give away: its cap above max(min capacity, used).
        Used memory includes running and deferred tasks, so it never moves."""
        cap = self._stage_capacity.get(stage_id, 0)
        floor = max(self._stage_min_capacity.get(stage_id, cap), self._stage_used.get(stage_id, 0))
        return max(0, int(cap - floor))

    async def transfer_capacity(
        self,
        donor_id: int,
        receiver_id: int,
        quantum_fraction: float = 0.05,
        upstream: bool = False,
    ) -> bool:
        """Move up to one quantum of the donor's free capacity to the receiver.
        Memory moving upstream may not grow a non-last stage beyond an equal share."""
        quantum = max(1, int(self._capacity * quantum_fraction))
        async with self._cond:
            receiver_cap = self._stage_capacity.get(receiver_id, 0)
            receiver_max = self._stage_max_capacity.get(receiver_id, receiver_cap)
            delta = min(quantum, self.stage_free(donor_id), receiver_max - receiver_cap)
            if delta <= 0:
                return False
            self._stage_capacity[donor_id] -= delta
            self._stage_capacity[receiver_id] += delta
            self._log_rebalance(
                f"transfer donor={donor_id} receiver={receiver_id} moved {delta / 1e9:.1f} GB"
            )
            self._cond.notify_all()
            return True

    async def return_stage_capacity(
        self,
        stage_id: int,
        quantum_fraction: float = 0.05,
        force_return: bool = False,
    ) -> bool:
        quantum = max(1, int(self._capacity * quantum_fraction))

        async with self._cond:
            cur_cap = self._stage_capacity.get(stage_id, 0)
            min_cap = self._stage_min_capacity.get(stage_id, cur_cap)
            cur_used = self._stage_used.get(stage_id, 0)
            floor = cur_used if force_return else max(min_cap, cur_used)
            releasable = cur_cap - floor
            if releasable <= 0:
                return False

            receiver_id = self._last_stage_id()
            if receiver_id is None or receiver_id == stage_id:
                return False

            receiver_cap = self._stage_capacity.get(receiver_id, 0)
            receiver_max = self._stage_max_capacity.get(receiver_id, receiver_cap)
            growable = receiver_max - receiver_cap
            if growable <= 0:
                return False

            delta = min(quantum, releasable, growable)
            if delta <= 0:
                return False
            self._stage_capacity[stage_id] -= delta
            self._stage_capacity[receiver_id] += delta
            self._log_rebalance(
                f"return stage={stage_id} receiver={receiver_id} moved {delta / 1e9:.1f} GB"
            )
            self._cond.notify_all()
            return True

    async def can_admit_stage(self, stage_id: int, budget: int):
        async with self._cond:
            return (
                self._stage_used[stage_id] + budget
                <= self._stage_capacity[stage_id]
            )

    async def try_allocate_stage(self, stage_id: int, budget: int) -> bool:
        async with self._cond:
            if not self._admit_check(stage_id, budget):
                return False

            self._stage_used[stage_id] += budget
            self._stage_inflight[stage_id] += 1
            if self._pool is not None:
                self._pool.add(id(self), budget)
            return True

    async def allocate_stage(self, stage_id: int, budget: int):
        if KVMemoryManager.LOG:
            used, cap = self.stage_usage(stage_id)
            print("stage", stage_id, "inflight", self.stage_inflight(stage_id), "used", used, "cap", cap, "budget", budget)

        async with self._cond:
            while (
                self._stage_used[stage_id] + budget
                > self._stage_capacity[stage_id]
            ):
                await self._cond.wait()

            self._stage_used[stage_id] += budget
            self._stage_inflight[stage_id] += 1

    async def force_allocate_stage(self, stage_id: int, budget: int):
        if KVMemoryManager.LOG:
            used, cap = self.stage_usage(stage_id)
            print(
                "force stage",
                stage_id,
                "inflight",
                self.stage_inflight(stage_id),
                "used",
                used,
                "cap",
                cap,
                "budget",
                budget,
            )

        async with self._cond:
            self._stage_used[stage_id] += budget
            self._stage_inflight[stage_id] += 1
            if self._pool is not None:
                self._pool.add(id(self), budget)

    async def release_stage(self, stage_id: int, budget: int):
        if KVMemoryManager.LOG:
            used, cap = self.stage_usage(stage_id)
            print("stage", stage_id, "inflight", self.stage_inflight(stage_id), "used", used, "cap", cap, "budget", budget)

        async with self._cond: 
            if self._pool is not None:
                self._pool.add(id(self), -min(budget, self._stage_used[stage_id]))
            self._stage_used[stage_id] -= budget
            if self._stage_used[stage_id] < 0:
                self._stage_used[stage_id] = 0
            self._stage_inflight[stage_id] -= 1
            if self._stage_inflight[stage_id] < 0:
                self._stage_inflight[stage_id] = 0

            self._cond.notify_all()

    # async def wait_for_stage_capacity(self):
    #     async with self._cond:
    #         await self._cond.wait()
            
    def stage_inflight(self, stage_id: int) -> int:
        return self._stage_inflight.get(stage_id, 0)

    def stage_usage(self, stage_id: int) -> tuple[int, int]:
        return (
            int(self._stage_used.get(stage_id, 0)),
            int(self._stage_capacity.get(stage_id, 0)),
        )

    # Global API (Blocking)
    async def can_admit(self, budget: int):
        async with self._cond:
            return self._global_used + budget <= self._capacity

    async def allocate(self, budget: int):
        async with self._cond:
            while self._global_used + budget > self._capacity:
                await self._cond.wait()

            self._global_used += budget

    async def release(self, budget: int):
        async with self._cond:
            self._global_used -= budget
            if self._global_used < 0:
                self._global_used = 0

            self._cond.notify_all()


    def token_length(self, text) -> int:
        if type(text) is str:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            prompt = self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=False)
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    @classmethod
    def init(cls, model_name, kv_capacity, dtype=torch.float16):
        with cls._init_lock:
            if cls._instance is not None:
                raise RuntimeError("KVMemoryManager already initialized")
            cls._instance = cls(model_name, kv_capacity, dtype)

    @classmethod
    def get_instance(cls):
        # Multi-query mode (KALYPSO_QUERY_SHARE): a query-local manager owns a share of the capacity.
        local = _QUERY_KV_MANAGER.get()
        if local is not None:
            return local
        if cls._instance is None:
            raise RuntimeError("KVMemoryManager not initialized")
        return cls._instance
    
    def apply_chat_template(self, prompt):
        return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    
    def capacity(self):
        return self._capacity
