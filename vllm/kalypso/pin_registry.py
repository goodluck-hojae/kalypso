import threading
from time import time
from typing import Any


class PinnedRequestRegistry:
    _instance = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}

    @classmethod
    def instance(cls) -> "PinnedRequestRegistry":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def add(
        self,
        request_id: str,
        *,
        kind: str,
        max_tokens: int,
        owner_key: str | None = None,
    ) -> None:
        with self._lock:
            self._entries[request_id] = {
                "request_id": request_id,
                "kind": kind,
                "max_tokens": max_tokens,
                "owner_key": owner_key,
                "pinned_at": time(),
            }

    def remove(self, request_id: str) -> bool:
        with self._lock:
            return self._entries.pop(request_id, None) is not None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(item)
                for item in sorted(
                    self._entries.values(),
                    key=lambda item: item["pinned_at"],
                )
            ]

    def list_by_owner(self, owner_key: str):
        with self._lock:
            return [
                dict(item)
                for item in sorted(
                    (
                        item
                        for item in self._entries.values()
                        if item.get("owner_key") == owner_key
                    ),
                    key=lambda item: item["pinned_at"],
                )
            ]
