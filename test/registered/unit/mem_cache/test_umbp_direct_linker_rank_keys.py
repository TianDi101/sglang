"""Unit tests for the UMBP direct linker's rank-replicated key handling."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import sglang.srt.mem_cache.storage.umbp.umbp_direct_linker as umbp_direct_linker
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    UMBPDirectLinker,
    _storage_suffix,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStorageSuffix(CustomTestCase):
    def test_replicated_group_drops_the_tp_term(self):
        # MLA/DSA pages are byte-identical on every attention TP rank, so every
        # rank has to name the same object or the store holds tp_size copies.
        suffixes = {
            _storage_suffix(
                rank_replicated=True,
                cp_replicated=False,
                tp_rank=rank,
                attn_cp_rank=0,
                pp_rank=0,
            )
            for rank in range(8)
        }
        self.assertEqual(suffixes, {"cp0_pp0"})

    def test_sharded_group_keeps_one_keyspace_per_rank(self):
        suffixes = [
            _storage_suffix(
                rank_replicated=False,
                cp_replicated=False,
                tp_rank=rank,
                attn_cp_rank=0,
                pp_rank=0,
            )
            for rank in range(8)
        ]
        self.assertEqual(len(set(suffixes)), 8)
        self.assertEqual(suffixes[3], "tp3_cp0_pp0")

    def test_cp_replicated_group_drops_the_cp_term(self):
        # DSv4 writes the whole sequence on every attention-CP rank.
        suffixes = {
            _storage_suffix(
                rank_replicated=True,
                cp_replicated=True,
                tp_rank=rank,
                attn_cp_rank=rank,
                pp_rank=0,
            )
            for rank in range(4)
        }
        self.assertEqual(suffixes, {"pp0"})

    def test_cp_and_pp_shard_even_when_replicated(self):
        # CP shards the sequence and PP shards layers, so neither may collapse.
        self.assertEqual(
            _storage_suffix(
                rank_replicated=True,
                cp_replicated=False,
                tp_rank=0,
                attn_cp_rank=1,
                pp_rank=2,
            ),
            "cp1_pp2",
        )
        self.assertNotEqual(
            _storage_suffix(
                rank_replicated=True,
                cp_replicated=False,
                tp_rank=0,
                attn_cp_rank=0,
                pp_rank=0,
            ),
            _storage_suffix(
                rank_replicated=True,
                cp_replicated=False,
                tp_rank=0,
                attn_cp_rank=1,
                pp_rank=0,
            ),
        )


class TestEveryRankWrites(CustomTestCase):
    """Collapsing the key must not also elide the write on non-zero ranks.

    A Local-mode UMBP tier is in-process and private to its rank, and a
    standalone server is per node, so a page written only by rank 0 is not
    reachable from the other ranks' stores. They then miss on a key nobody
    wrote for them, and the attention group's MIN over the restorable mask
    drives the hit to zero.
    """

    def test_offload_queues_the_write_on_every_rank(self):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker._offload_results = Queue()
        linker._offload_queue = Queue()
        linker._gc_frozen = True  # short-circuits _freeze_gc_once
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda _transfers, allow_partial: ["transfer"]
        )
        fake_device = SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda: None)
        )
        with mock.patch.object(umbp_direct_linker, "device_module", fake_device):
            self.assertTrue(linker.offload(["t"]))
        self.assertEqual(linker._offload_queue.qsize(), 1)
        # The result comes from the offload thread once the write lands, so
        # nothing may pre-post one here.
        self.assertEqual(linker._offload_results.qsize(), 0)

    def test_empty_transfer_set_produces_no_result(self):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker._offload_results = Queue()
        linker._offload_queue = Queue()
        linker._gc_frozen = True
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda _transfers, allow_partial: []
        )
        self.assertFalse(linker.offload([]))
        self.assertEqual(linker._offload_results.qsize(), 0)
        self.assertEqual(linker._offload_queue.qsize(), 0)


class TestPrefetch(CustomTestCase):
    def test_hints_only_whole_prefix_pools(self):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker._stats = {"prefetch": 0}
        linker._prefetch = mock.Mock()
        full = PoolTransfer(name=PoolName.KV, hit_policy=PoolHitPolicy.ALL_PAGES)
        tail = PoolTransfer(name=PoolName.SWA, hit_policy=PoolHitPolicy.TRAILING_PAGES)
        linker.pool_group = SimpleNamespace(resolve_transfers=lambda _: [full, tail])
        linker._object_keys_for_pages = lambda pages, t: (
            [f"{t.name.value}/{page}" for page in pages],
            1,
        )
        linker.prefetch("rid", [PoolTransfer(name=PoolName.KV, keys=["p0", "p1"])])
        linker._prefetch.submit.assert_called_once_with(
            linker._prefetch_keys,
            [f"{PoolName.KV.value}/p0", f"{PoolName.KV.value}/p1"],
        )


if __name__ == "__main__":
    unittest.main()
