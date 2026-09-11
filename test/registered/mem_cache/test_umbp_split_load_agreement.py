"""Page-set agreement for the UMBP direct linker's split load.

Split load has every TP rank read a disjoint window of the pages and all-gather
the rest, which is only sound while the ranks hold the same page set. The set is
settled after the tree insert has deduplicated whatever each rank already had,
so the linker intersects the sets before it cuts the windows. These are the
pure-logic tests for that intersection and for the plans it produces; the
collectives themselves are exercised by the multi-GPU linker tests.

    python -m pytest test/registered/mem_cache/test_umbp_split_load_agreement.py -v
"""

import multiprocessing
import os
import tempfile
import unittest

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    _POOL_IDS,
    UMBPDirectLinker,
    _split_windows,
)
try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # this tree predates the CI registry
    register_cpu_ci = None

if register_cpu_ci is not None:
    register_cpu_ci(est_time=300, suite="base-a-test-cpu")

KV = _POOL_IDS[PoolName.KV]
SWA = _POOL_IDS[PoolName.SWA]


class _FakeTransfer(msgspec.Struct):
    name: PoolName
    keys: list | None


class _StandIn:
    """The split methods plus only the linker state they read.

    ``UMBPDirectLinker`` cannot be constructed without a device pool group and a
    UMBP server, and these methods touch neither.
    """

    _parse_page_vector = staticmethod(UMBPDirectLinker._parse_page_vector)
    _intersect_page_vectors = staticmethod(UMBPDirectLinker._intersect_page_vectors)
    agree_common_pages = UMBPDirectLinker._agree_common_pages
    split_pool_plans = UMBPDirectLinker._split_pool_plans

    def __init__(self, tp_rank=0, split_world=1, sync_group=None):
        self._tp_rank = tp_rank
        self._split_world = split_world
        self._split_sync_group = sync_group


def _split_plans(tp_rank: int, split_world: int, **kwargs):
    """Run the plan cut with only the fields it touches, no linker construction."""
    return _StandIn(tp_rank=tp_rank, split_world=split_world).split_pool_plans(**kwargs)


def _vector(pools: dict) -> list:
    values = [len(pools)]
    for pool_id, pages in pools.items():
        values.extend([pool_id, len(pages)])
        values.extend(pages)
    return values


class TestSplitWindows(unittest.TestCase):
    def test_windows_tile_the_pages(self):
        for num_pages in range(0, 33):
            for world in (1, 2, 4, 8):
                windows = _split_windows(num_pages, world)
                self.assertEqual(len(windows), world)
                covered = []
                for start, end in windows:
                    self.assertLessEqual(start, end)
                    covered.extend(range(start, end))
                self.assertEqual(covered, list(range(num_pages)))

    def test_short_page_lists_leave_trailing_ranks_empty(self):
        self.assertEqual(_split_windows(2, 4), ((0, 1), (1, 2), (2, 2), (2, 2)))


class TestParsePageVector(unittest.TestCase):
    def test_round_trip(self):
        pools = {KV: [11, 22, 33], SWA: [44]}
        self.assertEqual(UMBPDirectLinker._parse_page_vector(_vector(pools)), pools)

    def test_padding_is_ignored(self):
        pools = {KV: [11, 22]}
        padded = _vector(pools) + [0] * 16
        self.assertEqual(UMBPDirectLinker._parse_page_vector(padded), pools)

    def test_empty_contribution(self):
        self.assertEqual(UMBPDirectLinker._parse_page_vector([0]), {})

    def test_malformed_vectors_are_rejected(self):
        for values in (
            [],  # nothing at all
            [-1],  # negative pool count
            [1, KV],  # truncated header
            [1, KV, 4, 1, 2],  # count runs past the end
            [1, KV, -1],  # negative page count
            [2, KV, 1, 7, KV, 1, 8],  # same pool twice
        ):
            with self.subTest(values=values):
                self.assertIsNone(UMBPDirectLinker._parse_page_vector(values))


class TestIntersectPageVectors(unittest.TestCase):
    def test_identical_ranks_keep_everything(self):
        pools = {KV: [1, 2, 3]}
        self.assertEqual(
            UMBPDirectLinker._intersect_page_vectors([pools, pools, pools]),
            {KV: [1, 2, 3]},
        )

    def test_divergent_ranks_keep_the_intersection_in_rank0_order(self):
        # Rank 1 deduplicated pages 2 and 3 away against nodes it already held.
        common = UMBPDirectLinker._intersect_page_vectors(
            [{KV: [1, 2, 3, 4]}, {KV: [4, 1]}, {KV: [1, 4, 9]}]
        )
        self.assertEqual(common, {KV: [1, 4]})

    def test_pool_missing_on_one_rank_is_dropped(self):
        common = UMBPDirectLinker._intersect_page_vectors(
            [{KV: [1, 2], SWA: [5]}, {KV: [1, 2]}]
        )
        self.assertEqual(common, {KV: [1, 2]})

    def test_disjoint_ranks_agree_on_nothing(self):
        self.assertEqual(
            UMBPDirectLinker._intersect_page_vectors([{KV: [1, 2]}, {KV: [3, 4]}]), {}
        )

    def test_empty_rank_empties_the_intersection(self):
        # A rank with nothing pending still joins the collective; its empty
        # contribution must turn the split off rather than strand anyone.
        self.assertEqual(
            UMBPDirectLinker._intersect_page_vectors([{KV: [1, 2]}, {}]), {}
        )

    def test_duplicate_pages_disable_the_pool(self):
        # A page named twice has no single row to scatter into.
        self.assertEqual(
            UMBPDirectLinker._intersect_page_vectors([{KV: [1, 1]}, {KV: [1, 1]}]), {}
        )

    def test_malformed_rank_disables_the_split(self):
        self.assertEqual(
            UMBPDirectLinker._intersect_page_vectors([{KV: [1, 2]}, None]), {}
        )


class TestPageHashesByPool(unittest.TestCase):
    def test_hashes_follow_the_plan_page_order(self):
        grouped = {
            PoolName.KV: [
                _FakeTransfer(PoolName.KV, ["aa" * 16, "bb" * 16]),
                _FakeTransfer(PoolName.KV, ["cc" * 16]),
            ]
        }
        hashes = UMBPDirectLinker._page_hashes_by_pool(grouped)
        self.assertEqual(len(hashes[PoolName.KV]), 3)
        self.assertEqual(len(set(hashes[PoolName.KV])), 3)

    def test_transfer_without_keys_contributes_nothing(self):
        grouped = {PoolName.KV: [_FakeTransfer(PoolName.KV, None)]}
        self.assertEqual(
            UMBPDirectLinker._page_hashes_by_pool(grouped), {PoolName.KV: []}
        )


class TestSplitPoolPlans(unittest.TestCase):
    """One pool, 6 pages, 2 objects per page, 3 ranks, 4 pages agreed."""

    page_hashes = [10, 11, 12, 13, 14, 15]
    locations = [100, 101, 102, 103, 104, 105]
    keys = [f"k{page}_{part}" for page in range(6) for part in ("a", "b")]
    agreed = [10, 12, 13, 15]

    def _plans(self, tp_rank: int):
        return _split_plans(
            tp_rank,
            3,
            name=PoolName.KV,
            keys=list(self.keys),
            locations=list(self.locations),
            page_hashes=list(self.page_hashes),
            entries_per_page=2,
            agreed=list(self.agreed),
        )

    def test_windows_tile_the_agreed_pages_once(self):
        seen = []
        for tp_rank in range(3):
            split = self._plans(tp_rank)[0]
            self.assertEqual(split.windows, ((0, 2), (2, 4), (4, 4)))
            self.assertTrue(split.split)
            seen.extend(split.locations)
        # Every agreed page is read exactly once across the group.
        self.assertEqual(sorted(seen), [100, 102, 103, 105])

    def test_all_locations_are_this_ranks_rows_in_agreed_order(self):
        for tp_rank in range(3):
            split = self._plans(tp_rank)[0]
            self.assertEqual(split.all_locations, [100, 102, 103, 105])

    def test_window_keys_match_its_pages(self):
        split = self._plans(1)[0]
        self.assertEqual(split.locations, [103, 105])
        self.assertEqual(split.keys, ["k3_a", "k3_b", "k5_a", "k5_b"])

    def test_unagreed_pages_are_loaded_whole(self):
        plans = self._plans(0)
        self.assertEqual(len(plans), 2)
        rest = plans[1]
        self.assertFalse(rest.split)
        self.assertEqual(rest.locations, [101, 104])
        self.assertEqual(rest.keys, ["k1_a", "k1_b", "k4_a", "k4_b"])

    def test_empty_trailing_window_still_carries_the_exchange(self):
        split = self._plans(2)[0]
        self.assertEqual(split.keys, [])
        self.assertEqual(split.locations, [])
        self.assertTrue(split.split)

    def test_all_pages_agreed_leaves_no_remainder(self):
        plans = _split_plans(
            0,
            3,
            name=PoolName.KV,
            keys=list(self.keys),
            locations=list(self.locations),
            page_hashes=list(self.page_hashes),
            entries_per_page=2,
            agreed=list(self.page_hashes),
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].all_locations, self.locations)

    def test_agreed_page_this_rank_lacks_is_a_bug_not_a_fallback(self):
        with self.assertRaises(AssertionError):
            _split_plans(
                0,
                3,
                name=PoolName.KV,
                keys=list(self.keys),
                locations=list(self.locations),
                page_hashes=list(self.page_hashes),
                entries_per_page=2,
                agreed=[10, 999],
            )


def _agree_worker(rank, world, rendezvous, page_sets, results):
    """One rank of the real collective, over gloo on CPU."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
    )
    try:
        stand_in = _StandIn(
            tp_rank=rank,
            split_world=world,
            sync_group=torch.distributed.group.WORLD,
        )
        agreed = stand_in.agree_common_pages(page_sets[rank])
        results.put((rank, {name.value: pages for name, pages in agreed.items()}))
    finally:
        torch.distributed.destroy_process_group()


class TestAgreeCommonPagesOverGloo(unittest.TestCase):
    """The collective itself: uneven page counts must gather and intersect."""

    def _agree(self, page_sets):
        world = len(page_sets)
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = os.path.join(directory, "rendezvous")
            workers = [
                context.Process(
                    target=_agree_worker,
                    args=(rank, world, rendezvous, page_sets, results),
                )
                for rank in range(world)
            ]
            for worker in workers:
                worker.start()
            collected = {}
            for _ in workers:
                rank, agreed = results.get(timeout=180)
                collected[rank] = agreed
            for worker in workers:
                worker.join(timeout=60)
                self.assertEqual(worker.exitcode, 0)
        return collected

    def test_identical_ranks_split_everything(self):
        pages = {PoolName.KV: [101, 102, 103]}
        agreed = self._agree([pages, pages, pages])
        self.assertEqual(agreed, {rank: {"kv": [101, 102, 103]} for rank in range(3)})

    def test_uneven_ranks_agree_on_the_intersection(self):
        # Padding to the widest vector must not leak into the answer, and every
        # rank must come back with the same list in the same order.
        agreed = self._agree(
            [
                {PoolName.KV: [101, 102, 103, 104]},
                {PoolName.KV: [104, 101]},
                {PoolName.KV: [101, 104, 109]},
            ]
        )
        self.assertEqual(agreed, {rank: {"kv": [101, 104]} for rank in range(3)})

    def test_rank_with_nothing_pending_turns_the_split_off(self):
        agreed = self._agree([{PoolName.KV: [101, 102]}, {}])
        self.assertEqual(agreed, {0: {}, 1: {}})


if __name__ == "__main__":
    unittest.main()
