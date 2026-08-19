import asyncio
import math
import os
from typing import Any, List

import requests

from .base import BaseOp, OpBehavior, OpName
from .prompt_utils import (
    add_assistant_prompt,
    get_data_prompt,
    get_prompt,
    get_system_prompt,
)
from vllm.kalypso.context import (
    RETRY_TASK,
    ExecutionState,
    SemContext,
    SemanticInput,
)
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.controller.map_estimator import MapRatioEstimator
from vllm.kalypso.execution.pipeline_execution import BlockingExecutor


class CartesianProduct(BaseOp):
    def __init__(self, right_table, position=-1):
        super().__init__(behavior=OpBehavior.JOIN, position=position)
        self.right_table = right_table

    def __call__(self, ctx):
        out = []

        for right in self.right_table:
            base_input = SemanticInput(
                data=ctx.input.data[:],
                right_data=[]
            )
            _input = base_input.add_right(right.input.data)
            new_ctx = SemContext(
                input=_input,
                state=ExecutionState(
                    raw_request=ctx.state.raw_request,
                    pin_req_id=None,
                    executor=ctx.state.executor
                ),
                )
            out.append(new_ctx)
        return out


class IndexedCartesianProduct(BaseOp):
    def __init__(
        self,
        right_table=None,
        service_address="127.0.0.1",
        service_port=8080,
        top_k=5,
        low_threshold=None,
        high_threshold=None,
        cp_id=None,
        position=-1,
    ):
        super().__init__(behavior=OpBehavior.JOIN, position=position)
        self.right_table = right_table or []
        self.service_address = service_address
        self.service_port = service_port
        self.top_k = top_k
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._indexed_rows = [
            (idx, right.input.data)
            for idx, right in enumerate(self.right_table)
        ]
        self.cp_id = cp_id or f"icp:{position}:{id(self)}"

    def _post_icp(self, path, payload):
        return requests.post(
            f"http://{self.service_address}:{self.service_port}/{path}",
            json=payload,
            timeout=60,
        )

    def _build_remote_index(self) -> None:
        if not self.right_table:
            return
        response = self._post_icp(
            "build_index",
            {
                "cp_id": self.cp_id,
                "right_table": [list(row) for row in self._indexed_rows],
            },
        )
        response.raise_for_status()

    def _query_retrieval(self, left_row: tuple[Any, ...]) -> list[dict[str, Any]]:
        payload = {
            "cp_id": self.cp_id,
            "left_tuple": list(left_row),
        }
        if self.low_threshold is not None or self.high_threshold is not None:
            payload["low_threshold"] = self.low_threshold
            payload["high_threshold"] = self.high_threshold
        else:
            payload["top_k"] = self.top_k

        response = self._post_icp("query", payload)
        if response.ok:
            return response.json()["results"]

        self._build_remote_index()

        retry_response = self._post_icp("query", payload)
        retry_response.raise_for_status()
        return retry_response.json()["results"]

    def _query_text_from_data(self, data):
        for message in reversed(data):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return str(data).strip()

    def __call__(self, ctx):
        query_data = self._query_text_from_data(ctx.input.data)

        left_row = (query_data,)
        try:
            ranked_rows = self._query_retrieval(left_row)
        except requests.RequestException:
            if not self.right_table:
                raise
            ranked_rows = [
                {"metadata": [idx, right.input.data], "score": 0.0}
                for idx, right in enumerate(self.right_table)
            ]

        out = []
        for row in ranked_rows:
            metadata = row.get("metadata", [])
            if len(metadata) < 2:
                continue
            _id, data = metadata[0], metadata[1]
            score = row.get("score")
            right_data = get_data_prompt(data)
            base_input = SemanticInput(
                data=ctx.input.data[:],
                right_data=[],
            )
            _input = base_input.add_right(right_data)
            new_ctx = SemContext(
                input=_input,
                state=ExecutionState(
                    raw_request=ctx.state.raw_request,
                    pin_req_id=None,
                    executor=ctx.state.executor,
                    helper_score=score,
                    idx=ctx.state.idx,
                ),
            )
            out.append(new_ctx)

        return out


class IndexedSearch(BaseOp):
    def __init__(
        self,
        right_table=None,
        service_address="127.0.0.1",
        service_port=8080,
        top_k=5,
        low_threshold=None,
        high_threshold=None,
        cp_id=None,
        position=-1,
    ):
        super().__init__(behavior=OpBehavior.TUPLE_INDEPENDENT, position=position)
        self.right_table = right_table or []
        self.service_address = service_address
        self.service_port = service_port
        self.top_k = top_k
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._indexed_rows = [
            (idx, right.input.data)
            for idx, right in enumerate(self.right_table)
        ]
        self.cp_id = cp_id or f"icp:{position}:{id(self)}"

    def _post_icp(self, path, payload):
        return requests.post(
            f"http://{self.service_address}:{self.service_port}/{path}",
            json=payload,
            timeout=60,
        )

    def _build_remote_index(self) -> None:
        if not self.right_table:
            return
        response = self._post_icp(
            "build_index",
            {
                "cp_id": self.cp_id,
                "right_table": [list(row) for row in self._indexed_rows],
            },
        )
        response.raise_for_status()

    def _query_retrieval(self, left_row: tuple[Any, ...]) -> list[dict[str, Any]]:
        payload = {
            "cp_id": self.cp_id,
            "left_tuple": list(left_row),
        }
        if self.low_threshold is not None or self.high_threshold is not None:
            payload["low_threshold"] = self.low_threshold
            payload["high_threshold"] = self.high_threshold
        else:
            payload["top_k"] = self.top_k

        response = self._post_icp("query", payload)
        if response.ok:
            return response.json()["results"]

        self._build_remote_index()

        retry_response = self._post_icp("query", payload)
        retry_response.raise_for_status()
        return retry_response.json()["results"]

    def _query_text_from_data(self, data):
        for message in reversed(data):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return str(data).strip()

    def estimate_tokens(self, ctx):
        return max(
            1,
            KVMemoryManager.get_instance().token_length(
                self._query_text_from_data(ctx.input.data)
            ),
        )

    async def __call__(self, ctx, priority: int = 0):
        query_data = self._query_text_from_data(ctx.input.data)
        ranked_rows = await asyncio.to_thread(
            self._query_retrieval,
            (query_data,),
        )

        retrieved_contexts = []
        for rank, row in enumerate(ranked_rows, start=1):
            metadata = row.get("metadata", [])
            if len(metadata) < 2:
                continue
            doc_id, text = metadata[0], metadata[1]
            retrieved_contexts.append(
                f"[{rank}] Document ID: {doc_id}\n{text}"
            )

        if not retrieved_contexts:
            return ctx

        retrieved_data = get_data_prompt("\n\n".join(retrieved_contexts))
        ctx.input.data += retrieved_data
        ctx.output.append({
            str(self.__class__): {
                "num_retrieved": len(retrieved_contexts),
                "cp_id": self.cp_id,
            }
        })

        return ctx
