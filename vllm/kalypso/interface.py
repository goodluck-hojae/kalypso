from pydantic import BaseModel
from typing import Any


class SemanticQueryRequest(BaseModel):
    ops: list[dict[str, Any]]
    # Exactly one of data_path/records must be set. data_path names a file
    # already reachable on this server's filesystem; records lets a caller
    # with no shared filesystem (e.g. an external backend) submit rows
    # directly in the request body instead — either row objects (preserves
    # column names, e.g. for the Text/Sentences reader) or plain cell values.
    data_path: str | None = None
    records: list[dict[str, Any] | str | int | float] | None = None
    # Only meaningful with `records` of row objects: names which column
    # holds the text to run ops over, overriding _csv_reader's fixed
    # data/Text+Sentences/Text guessing order. Ignored for bare-cell records
    # (already unambiguous) and for data_path (the file's own columns rule).
    text_column: str | None = None
    model_name: str | None = None
