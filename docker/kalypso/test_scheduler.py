"""Exercise the patched upstream scheduler's finish path without CUDA imports.

Pass the prepared upstream source directory as the only argument.
"""
import ast
import sys
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace


source = Path(sys.argv.pop(1)) / "vllm/v1/core/sched/scheduler.py"
tree = ast.parse(source.read_text())
scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "Scheduler")
method = next(node for node in scheduler.body if isinstance(node, ast.FunctionDef)
              and node.name == "_free_request")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
    ast.alias(name="annotations")], level=0), method], type_ignores=[])


class Status(IntEnum):
    FINISHED_STOPPED = 1
    FINISHED_LENGTH_CAPPED = 2
    FINISHED_ABORTED = 3
    FINISHED_ERROR = 4


namespace = {"RequestStatus": Status}
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
finish = namespace["_free_request"]


class FinishTests(unittest.TestCase):
    def check_finish(self, key, status, expected_free):
        freed = []
        request = SimpleNamespace(request_id="random-internal-id", client_index=0,
                                  kalypso_pin_key=key, status=status,
                                  is_finished=lambda: True)
        scheduler = SimpleNamespace(
            _inflight_prefills=SimpleNamespace(discard=lambda _: None),
            _connector_finished=lambda _: (False, None), ec_connector=None,
            encoder_cache_manager=SimpleNamespace(free=lambda _: None),
            finished_req_ids=set(), finished_req_ids_dict=None,
            _free_blocks=lambda req: freed.append(req.request_id),
        )
        finish(scheduler, request)
        self.assertEqual(freed, [request.request_id] if expected_free else [])
        self.assertIn(request.request_id, scheduler.finished_req_ids)

    def test_finished_pinned_request_retains_blocks(self):
        for status in (Status.FINISHED_STOPPED, Status.FINISHED_LENGTH_CAPPED):
            with self.subTest(status=status):
                self.check_finish("owner", status, False)

    def test_failed_or_aborted_pinned_request_releases_blocks(self):
        for status in (Status.FINISHED_ABORTED, Status.FINISHED_ERROR):
            with self.subTest(status=status):
                self.check_finish("owner", status, True)

    def test_ordinary_completion_releases_blocks(self):
        self.check_finish(None, Status.FINISHED_STOPPED, True)


if __name__ == "__main__":
    unittest.main()
