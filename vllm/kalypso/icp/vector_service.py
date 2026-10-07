import argparse
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vllm.kalypso.icp.colbert_wiki import ColbertWiki

DEFAULT_COLBERT_EXPERIMENT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "colbert_indexes"
)
DEFAULT_COLBERT_COLLECTION = os.path.join(
    DEFAULT_COLBERT_EXPERIMENT_ROOT, "collections", "wikipedia.tsv"
)
DEFAULT_COLBERT_ROOT = os.path.expanduser("~/ColBERT")


def _normalize_tuple(values: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(values)


import re
from typing import Any

_TEXT_FIELD_RE = re.compile(
    r'"text"\s*:\s*(.*?)(?:\n\s*}\s*$|$)',
    re.DOTALL,
)

def _extract_text_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""

        match = _TEXT_FIELD_RE.search(text)
        if match:
            text = match.group(1)

        return (
            text
            .strip()
            .rstrip(",")
            .strip()
            .strip('"')
            .strip()
        )

    if isinstance(value, dict):
        if "text" in value:
            return _extract_text_value(value["text"])
        if "content" in value:
            return _extract_text_value(value["content"])
        return " ".join(filter(None, (_extract_text_value(v) for v in value.values())))

    if isinstance(value, (list, tuple)):
        return " ".join(filter(None, (_extract_text_value(item) for item in value)))

    return str(value)

def _serialize_tuple(t: tuple[Any, ...]) -> str:
    if len(t) == 2 and isinstance(t[0], int):
        return _extract_text_value(t[1])
    return " ".join(filter(None, (_extract_text_value(value) for value in t)))


def _query_text(left_tuple: tuple[Any, ...]) -> str:
    if len(left_tuple) == 1:
        return _extract_text_value(left_tuple[0])
    if len(left_tuple) == 2:
        return _extract_text_value(left_tuple[1])
    return _serialize_tuple(left_tuple)


# -----------------------------
# Embedding Model
# -----------------------------
class SentenceTransformersRM:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


# -----------------------------
# FAISS Vector Store
# -----------------------------
class FaissVS:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[tuple[Any, ...]] = []

    def add(self, embeddings: np.ndarray, metadata: list[tuple[Any, ...]]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length.")
        if len(embeddings) == 0:
            return

        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata)

    def search_top_k(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []

        query = np.array([query_embedding]).astype("float32")
        D, I = self.index.search(query, k)

        results: list[dict[str, Any]] = []
        for idx in range(k):
            results.append({
                "metadata": list(self.metadata[I[0][idx]]),
                "score": float(D[0][idx]),
            })

        return results

    def search_threshold(
        self,
        query_embedding: np.ndarray,
        low_threshold: float,
        high_threshold: float,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be < high_threshold.")

        query = np.array([query_embedding]).astype("float32")
        lims, D, I = self.index.range_search(query, low_threshold)

        results: list[dict[str, Any]] = []
        start, end = lims[0], lims[1]
        for idx in range(start, end):
            score = float(D[idx])
            if score <= low_threshold or score >= high_threshold:
                continue
            results.append({
                "metadata": list(self.metadata[I[idx]]),
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# -----------------------------
# Vector DB Service
# -----------------------------
class VectorDBService:
    def __init__(self, rm: SentenceTransformersRM):
        self.rm = rm
        self.cache: dict[str, FaissVS] = {}
        self.table_cache: dict[str, list[tuple[Any, ...]]] = {}

    def build_index(
        self,
        cp_id: str,
        right_table: list[tuple[Any, ...]],
    ) -> FaissVS:
        cached_table = self.table_cache.get(cp_id)

        if cp_id in self.cache:
            if cached_table != right_table:
                raise ValueError(
                    f"Cached index for cp_id={cp_id!r} built with different table."
                )
            return self.cache[cp_id]

        if not right_table:
            raise ValueError("right_table must not be empty.")

        texts = [_serialize_tuple(row) for row in right_table]
        embeddings = self.rm.encode(texts)

        vs = FaissVS(embeddings.shape[1])
        vs.add(embeddings, right_table)

        self.cache[cp_id] = vs
        self.table_cache[cp_id] = list(right_table)
        return vs

    def query(
        self,
        cp_id: str,
        left_tuple: tuple[Any, ...],
        right_table: list[tuple[Any, ...]] | None,
        top_k: int | None = None,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if right_table is not None:
            vs = self.build_index(cp_id, right_table)
        else:
            vs = self.cache.get(cp_id)
            if vs is None:
                raise ValueError(f"No index for cp_id={cp_id!r}")

        query_text = _query_text(left_tuple)
        query_vec = self.rm.encode([query_text])[0]

        threshold_mode = (
            low_threshold is not None
            or high_threshold is not None
        )
        if threshold_mode:
            if low_threshold is None or high_threshold is None:
                raise ValueError(
                    "Both low_threshold and high_threshold are required "
                    "for threshold search."
                )
            return vs.search_threshold(query_vec, low_threshold, high_threshold)

        if top_k is None:
            raise ValueError(
                "top_k is required when thresholds are not specified."
            )
        return vs.search_top_k(query_vec, top_k)

    def clear_cp(self, cp_id: str) -> bool:
        removed = False
        if cp_id in self.cache:
            del self.cache[cp_id]
            removed = True
        if cp_id in self.table_cache:
            del self.table_cache[cp_id]
            removed = True
        return removed

    def cache_state(self) -> dict[str, int]:
        return {
            cp_id: vs.index.ntotal
            for cp_id, vs in self.cache.items()
        }


class ColBERTVectorDBService:
    def __init__(
        self,
        index_name: str,
        experiment_root: str,
        experiment: str,
        collection: str,
        colbert_root: str,
    ):
        self.index_name = index_name
        self.wiki = ColbertWiki(
            index_name=index_name,
            experiment_root=experiment_root,
            experiment=experiment,
            collection=collection,
            colbert_root=colbert_root,
        )

    def build_index(
        self,
        cp_id: str,
        right_table: list[tuple[Any, ...]],
    ) -> Any:
        return self

    def query(
        self,
        cp_id: str,
        left_tuple: tuple[Any, ...],
        right_table: list[tuple[Any, ...]] | None,
        top_k: int | None = None,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if low_threshold is not None or high_threshold is not None:
            raise ValueError("Threshold search is not supported by the ColBERT backend.")
        if top_k is None:
            raise ValueError("top_k is required for the ColBERT backend.")

        query_text = _query_text(left_tuple)
        return [
            {
                "metadata": [result["pid"], result["text"]],
                "score": result["score"],
            }
            for result in self.wiki.search(query_text, topk=top_k)
        ]

    def clear_cp(self, cp_id: str) -> bool:
        return False

    def cache_state(self) -> dict[str, int]:
        return {self.index_name: len(self.wiki.searcher.collection)}


# -----------------------------
# API Models
# -----------------------------
class BuildIndexRequest(BaseModel):
    cp_id: str
    right_table: list[list[Any]]


class QueryRequest(BaseModel):
    cp_id: str
    left_tuple: list[Any]
    right_table: list[list[Any]] | None = None
    top_k: int | None = Field(default=None, gt=0)
    low_threshold: float | None = None
    high_threshold: float | None = None


class ClearRequest(BaseModel):
    cp_id: str


# -----------------------------
# FastAPI App
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if app.state.backend == "colbert":
        app.state.vector_db = ColBERTVectorDBService(
            index_name=app.state.colbert_index_name,
            experiment_root=app.state.colbert_experiment_root,
            experiment=app.state.colbert_experiment,
            collection=app.state.colbert_collection,
            colbert_root=app.state.colbert_root,
        )
    else:
        app.state.vector_db = VectorDBService(
            SentenceTransformersRM(app.state.model_name)
        )
    yield


def create_app(
    model_name: str = "intfloat/e5-base-v2",
    backend: str = "faiss",
    colbert_index_name: str = "fever_factool_wikipedia_colbert",
    colbert_experiment_root: str = DEFAULT_COLBERT_EXPERIMENT_ROOT,
    colbert_experiment: str = "wikipedia",
    colbert_collection: str = DEFAULT_COLBERT_COLLECTION,
    colbert_root: str = DEFAULT_COLBERT_ROOT,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.model_name = model_name
    app.state.backend = backend
    app.state.colbert_index_name = colbert_index_name
    app.state.colbert_experiment_root = colbert_experiment_root
    app.state.colbert_experiment = colbert_experiment
    app.state.colbert_collection = colbert_collection
    app.state.colbert_root = colbert_root

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": app.state.backend}

    @app.get("/cache")
    async def cache():
        return {"cache": app.state.vector_db.cache_state()}

    @app.post("/build_index")
    async def build_index(request: BuildIndexRequest):
        try:
            right_table = [_normalize_tuple(r) for r in request.right_table]
            vs = app.state.vector_db.build_index(request.cp_id, right_table)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if app.state.backend == "colbert":
            return {
                "rows_indexed": len(app.state.vector_db.wiki.searcher.collection),
                "backend": "colbert",
                "note": "Using prebuilt ColBERT index; right_table ignored.",
            }
        return {"rows_indexed": vs.index.ntotal}

    @app.post("/query")
    async def query(request: QueryRequest):
        try:
            results = app.state.vector_db.query(
                request.cp_id,
                _normalize_tuple(request.left_tuple),
                None if request.right_table is None else [
                    _normalize_tuple(r) for r in request.right_table
                ],
                top_k=request.top_k,
                low_threshold=request.low_threshold,
                high_threshold=request.high_threshold,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        return {"results": results}

    @app.post("/clear")
    async def clear(request: ClearRequest):
        return {"cleared": app.state.vector_db.clear_cp(request.cp_id)}

    return app


# -----------------------------
# Run
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--backend", choices=("faiss", "colbert"), default="faiss")
    parser.add_argument("--colbert-index-name", default="fever_factool_wikipedia_colbert")
    parser.add_argument(
        "--colbert-experiment-root",
        default=DEFAULT_COLBERT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--colbert-experiment", default="wikipedia")
    parser.add_argument(
        "--colbert-collection",
        default=DEFAULT_COLBERT_COLLECTION,
    )
    parser.add_argument(
        "--colbert-root",
        default=DEFAULT_COLBERT_ROOT,
    )
    args = parser.parse_args()

    app = create_app(
        model_name=args.model_name,
        backend=args.backend,
        colbert_index_name=args.colbert_index_name,
        colbert_experiment_root=args.colbert_experiment_root,
        colbert_experiment=args.colbert_experiment,
        colbert_collection=args.colbert_collection,
        colbert_root=args.colbert_root,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
