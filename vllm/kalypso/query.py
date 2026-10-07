from typing import Any

class Query:
    def __init__(self, ops: list[dict[str, Any]], data_path: str):
        self.ops = ops
        self.data_path = data_path
