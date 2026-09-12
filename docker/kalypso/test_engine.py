"""CPU-only ownership regressions for the vLLM 0.29 container port."""
import unittest
from types import SimpleNamespace

from engine import get_pinned_requests, unpin_requests


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.block = SimpleNamespace(ref_cnt=2, is_null=False)
        self.freed = []
        self.requests = {
            key: SimpleNamespace(
                request_id=key, kalypso_pin_key=owner,
                is_finished=lambda: True,
            )
            for key, owner in (("internal-a-random", "a"), ("internal-b-random", "b"))
        }

        def free(request):
            self.block.ref_cnt -= 1
            self.freed.append(request.request_id)
            del self.requests[request.request_id]

        self.scheduler = SimpleNamespace(
            requests=self.requests, _free_blocks=free,
            kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(
                get_blocks=lambda _: ([self.block],),
            )),
        )

    def test_unpin_resolves_owner_despite_randomized_internal_id(self):
        unpin_requests(self.scheduler, ["a"])
        self.assertEqual(self.freed, ["internal-a-random"])
        self.assertEqual(self.block.ref_cnt, 1)
        self.assertEqual(get_pinned_requests(self.scheduler), [
            {"request_id": "b", "pinned_blocks": 1, "total_blocks": 1},
        ])

    def test_duplicate_unpin_does_not_release_shared_owner(self):
        unpin_requests(self.scheduler, ["a", "a"])
        unpin_requests(self.scheduler, ["a", "unknown"])
        self.assertEqual(self.block.ref_cnt, 1)
        unpin_requests(self.scheduler, ["b"])
        self.assertEqual(self.block.ref_cnt, 0)
        self.assertFalse(self.requests)

    def test_unpin_active_request_leaves_blocks_until_scheduler_finishes(self):
        self.requests["internal-a-random"].is_finished = lambda: False
        unpin_requests(self.scheduler, ["a"])
        self.assertEqual(self.block.ref_cnt, 2)
        self.assertIsNone(self.requests["internal-a-random"].kalypso_pin_key)
        self.assertFalse(self.freed)


if __name__ == "__main__":
    unittest.main()
