"""Thin wrapper for searching a prebuilt ColBERT Wikipedia index."""

import os
import shutil
import sys
import time


def _add_colbert_repo_to_path(colbert_root: str) -> None:
    colbert_root = os.path.abspath(colbert_root)
    if not os.path.isdir(colbert_root):
        raise FileNotFoundError(
            f"ColBERT repository not found at {colbert_root}. "
            "Pass --colbert-root with the path to a ColBERT checkout."
        )
    if colbert_root not in sys.path:
        sys.path.insert(0, colbert_root)


def _ensure_torch_extension_build_tools_on_path() -> None:
    python_bin = os.path.dirname(sys.executable)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin and python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([python_bin, *path_entries])

    if shutil.which("ninja") is None:
        raise RuntimeError(
            "ColBERT requires ninja to compile PyTorch C++ extensions. "
            f"Install it with `{sys.executable} -m pip install ninja`."
        )


class ColbertWiki:
    """Load and query an existing ColBERT index and its text collection."""

    def __init__(
        self,
        index_name: str,
        experiment_root: str,
        experiment: str,
        collection: str,
        colbert_root: str,
    ):
        _add_colbert_repo_to_path(colbert_root)
        _ensure_torch_extension_build_tools_on_path()

        from colbert import Searcher
        from colbert.infra import Run, RunConfig

        self.index_name = index_name
        load_start = time.perf_counter()
        with Run().context(RunConfig(root=experiment_root, experiment=experiment)):
            self.searcher = Searcher(index=index_name, collection=collection)
        print(
            f"ColBERT: loaded index {index_name} from {experiment_root} "
            f"in {time.perf_counter() - load_start:.1f}s",
            flush=True,
        )

    def search(self, query: str, topk: int = 10) -> list[dict]:
        pids, ranks, scores = self.searcher.search(query, k=topk)
        return [
            {
                "pid": int(pid),
                "rank": int(rank),
                "score": float(score),
                "text": self.searcher.collection[int(pid)],
            }
            for pid, rank, score in zip(pids, ranks, scores)
        ]
