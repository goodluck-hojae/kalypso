from pydantic import BaseModel
from typing import Any


class SemanticQueryRequest(BaseModel):
    ops: list[dict[str, Any]]
    data_path: str
    model_name: str | None = None
    blocking: bool | None = None
