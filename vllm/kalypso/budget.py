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
    # Multimodal / nested configs (e.g. Qwen3.5) keep the language-model
    # fields under a sub-config instead of the top level.
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg

    num_layers = text_cfg.num_hidden_layers

    # Hybrid linear/full attention models (e.g. Qwen3.5, Qwen3-Next) only
    # keep a growing per-token KV cache for their full-attention layers;
    # linear-attention layers use a fixed-size recurrent state instead.
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        num_layers = sum(1 for lt in layer_types if lt == "full_attention")

    # GQA / MQA aware
    num_kv_heads = getattr(
        text_cfg,
        "num_key_value_heads",
        text_cfg.num_attention_heads,
    )

    head_dim = getattr(
        text_cfg,
        "head_dim",
        text_cfg.hidden_size // text_cfg.num_attention_heads,
    )

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


class KVMemoryManager:
    ONE_TUPLE_TOKENS = 8192
    _instance = None
    _init_lock = threading.Lock()
    LOG = False
    REBALANCE_LOG = False
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
                "stage="
                f"{stage_id} "
                f"used={int(self._stage_used.get(stage_id, 0)):,} "
                f"cap={int(self._stage_capacity.get(stage_id, 0)):,} "
                f"min={int(self._stage_min_capacity.get(stage_id, 0)):,} "
                f"max={int(self._stage_max_capacity.get(stage_id, 0)):,} "
                f"inflight={int(self._stage_inflight.get(stage_id, 0))}"
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
    ) -> bool:
        quantum = max(1, int(self._capacity * quantum_fraction))

        async with self._cond:
            receiver_cap = self._stage_capacity.get(receiver_id, 0)
            receiver_max = self._stage_max_capacity.get(receiver_id, receiver_cap)
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
                    f"borrow receiver={receiver_id} donor={donor_id} delta={int(delta)}"
                )
                self._cond.notify_all()
                return True

            return False

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
                f"return stage={stage_id} receiver={receiver_id} delta={int(delta)}"
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
            if (
                self._stage_used[stage_id] + budget
                > self._stage_capacity[stage_id]
            ):
                return False

            self._stage_used[stage_id] += budget
            self._stage_inflight[stage_id] += 1
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

    async def release_stage(self, stage_id: int, budget: int):
        if KVMemoryManager.LOG:
            used, cap = self.stage_usage(stage_id)
            print("stage", stage_id, "inflight", self.stage_inflight(stage_id), "used", used, "cap", cap, "budget", budget)

        async with self._cond: 
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
        if cls._instance is None:
            raise RuntimeError("KVMemoryManager not initialized")
        return cls._instance
    
    def apply_chat_template(self, prompt):
        return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    
    def capacity(self):
        return self._capacity
