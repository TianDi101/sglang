from __future__ import annotations

import contextlib
import copy
import ctypes
import logging
import math
import os
import threading
import time
import zlib
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable

import numpy as np
import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()

# Keep every control-plane RPC comfortably below gRPC's message-size limit.
# This is a logical-page count; existence queries carry keys only, no ranges.
CHUNK_PAGES = 64

# Log per-load-batch GPU stalls and group timings (measurement only).
LW_STATS = os.getenv("UMBP_LW_STATS", "0") == "1"

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _stage_copy_kernel(desc, src_rows, dst_rows, ELEM: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        # Program (block, page): desc row = src, dst addresses, size, stride, pages, first row.
        d = desc + tl.program_id(0) * 6
        k = tl.program_id(1)
        if k < tl.load(d + 4):
            size = tl.load(d + 2)
            i = tl.load(d + 5) + k
            src = (tl.load(d) + tl.load(src_rows + i) * size).to(tl.pointer_type(ELEM))
            dst = (tl.load(d + 1) + tl.load(dst_rows + i) * tl.load(d + 3)).to(tl.pointer_type(ELEM))
            n = (size // WIDTH).to(tl.int32)
            for start in range(0, n, BLOCK):
                o = start + tl.arange(0, BLOCK)
                m = o < n
                tl.store(dst + o, tl.load(src + o, mask=m), mask=m)

    _STAGE_ELEMS = {8: tl.int64, 4: tl.int32, 2: tl.int16, 1: tl.uint8}
except ImportError:
    triton = None

# Budget by ranges because pool layouts attach different counts per object;
# 8192 stays below gRPC's default message limit.
RANGES_PER_CALL = int(os.getenv("UMBP_RANGES_PER_CALL", "8192"))

# Stage-ahead: a load waits this long for its request's in-flight staging.
STAGE_WAIT_S = float(os.getenv("UMBP_STAGE_WAIT_S", "5"))
# A queued request is re-checked this often: its prefix keeps leaving HBM and landing on SSD.
STAGE_RECHECK_S = float(os.getenv("UMBP_STAGE_RECHECK_S", "2"))
# The queue head stages its HBM pages too: admitting a batch evicts the next requests' prefixes.
STAGE_HEAD = int(os.getenv("UMBP_STAGE_HEAD", "8"))
# Arena run size cap; a request stages as several runs.
STAGE_CHUNK = int(os.getenv("UMBP_STAGE_CHUNK_MB", "64")) << 20
STAGE_READS = int(os.getenv("UMBP_STAGE_READS", "4"))
# Direct second-stream reads in flight per rank; 0 reads one layer group at a time.
LOAD_READS = int(os.getenv("UMBP_LOAD_READS", "0"))
# Page split on one tiered server: its "dram:ssd" tier weights, so the second stream takes the SSD objects.
SPLIT_WEIGHTS = os.getenv("UMBP_SPLIT_WEIGHTS", "")
# The SSD member's min_object_size: smaller objects always place in DRAM.
SPLIT_SSD_MIN = int(os.getenv("UMBP_SPLIT_SSD_MIN_BYTES", "0"))
# Device scratch for staged host-to-device copies; a larger block gets its own.
STAGE_SCRATCH = int(os.getenv("UMBP_STAGE_SCRATCH_MB", "64")) << 20
_ARENA_ALIGN = 4096


def _fnv1a(keys: list[str]) -> np.ndarray:
    """UMBP's StablePlacementHash (64-bit FNV-1a) per key, a byte column at a time."""
    out = np.empty(len(keys), np.uint64)
    by_len: dict[int, list[int]] = defaultdict(list)
    raw = [key.encode() for key in keys]
    for i, b in enumerate(raw):
        by_len[len(b)].append(i)
    for length, idx in by_len.items():
        cols = np.frombuffer(b"".join(raw[i] for i in idx), np.uint8).reshape(len(idx), length)
        h = np.full(len(idx), 14695981039346656037, np.uint64)
        for col in cols.T:
            h ^= col
            h *= np.uint64(1099511628211)
        out[idx] = h
    return out


@dataclass(eq=False)
class _StageRun:
    """One request's staged second-server objects in one arena run.

    Every (pool, layer, component) is a page-major block, ordered group, pool, layer,
    so a load copies a pool's blocks for one layer group as one contiguous range.
    """

    rid: str
    offset: int
    size: int
    t0: float
    done: threading.Event
    pages: dict[PoolName, int]
    blocks: dict[tuple[PoolName, int, int], tuple[int, int]]
    keys: list[tuple[PoolName, str]]
    ready: bool = False
    busy: int = 0
    doomed: bool = False
    used: int = 0


def _install_stage_hook() -> None:
    """Hand queued requests to the linker at every prefill build, before admission."""
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_umbp_stage_hook", False):
        return
    build = Scheduler.get_new_batch_prefill

    def get_new_batch_prefill(self, *args, **kwargs):
        linker = getattr(getattr(self.tree_cache, "linker", None), "cache_linker", None)
        start = time.perf_counter()
        if getattr(linker, "_stage_arena", None) is not None and self.waiting_queue:
            try:
                linker.stage_ahead(self.waiting_queue, self.tree_cache)
            except Exception:
                logger.exception("UMBP stage-ahead failed")
        try:
            return build(self, *args, **kwargs)
        finally:
            if LW_STATS and linker is not None:
                linker._time("build", start)

    Scheduler.get_new_batch_prefill = get_new_batch_prefill
    Scheduler._umbp_stage_hook = True


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    # A rank-replicated group (MLA / DSA) holds byte-identical pages on every
    # attention TP rank, so a tp term stores tp_size copies of the same page.
    # Only the key collapses -- every rank still writes, because a Local-mode
    # tier is private to its rank and a standalone server is per node.
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


def _ordered_layers(entry) -> list[int]:
    component_lengths = {len(component) for component in entry.components}
    if len(component_lengths) != 1:
        raise ValueError(
            f"UMBP pool {entry.name} components have different layer counts."
        )
    pool_layer_count = component_lengths.pop()
    if pool_layer_count != len(entry.layer_mapping):
        raise ValueError(
            f"UMBP pool {entry.name} has {pool_layer_count} buffers per component "
            f"but {len(entry.layer_mapping)} mapped layers."
        )
    by_buffer = {
        buffer_index: logical_layer
        for logical_layer, buffer_index in entry.layer_mapping.items()
    }
    if sorted(by_buffer) != list(range(pool_layer_count)):
        raise ValueError(
            f"UMBP pool {entry.name} layer mapping is not a contiguous bijection."
        )
    return [by_buffer[index] for index in range(pool_layer_count)]


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._producer_index = -1
        self.consumer_index = -1
        self._futures: dict[int, list[Future]] = {}
        # LW_STATS: blocked layers per batch, the batch's load info, and
        # finished batches whose GPU events are not yet resolved.
        self._blocked: dict[int, list] = {}
        self.load_info: dict[int, dict] = {}
        self._reports: deque = deque()

    def update_producer(self) -> int:
        self._producer_index += 1
        self._futures[self._producer_index] = [Future() for _ in range(self.num_layers)]
        return self._producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self._futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self._futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self._futures.get(index)
        if futures is None:
            return
        try:
            if LW_STATS and not futures[threshold].done():
                self._timed_wait(index, threshold, futures[threshold])
            else:
                futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("UMBP layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self._futures.pop(index, None)
                if LW_STATS:
                    self._reports.append((index, self._blocked.pop(index, [])))
                    self._flush_reports()

    def _timed_wait(self, index: int, layer: int, future: Future) -> None:
        # Events bracket the wait on the forward stream: their gap is how long
        # the GPU sat idle for this layer, not how long the CPU waited.
        before = device_module.Event(enable_timing=True)
        before.record()
        start = time.perf_counter()
        future.result()
        cpu_ms = (time.perf_counter() - start) * 1e3
        after = device_module.Event(enable_timing=True)
        after.record()
        self._blocked.setdefault(index, []).append((layer, before, after, cpu_ms))

    def _flush_reports(self) -> None:
        while self._reports:
            index, blocked = self._reports[0]
            if blocked and not blocked[-1][2].query():
                return
            self._reports.popleft()
            info = self.load_info.pop(index, {})
            gpu = [b.elapsed_time(a) for _, b, a, _ in blocked]
            logger.info(
                "UMBPLW idx=%d stall_gpu_ms=%.1f stall_cpu_ms=%.1f blocked=%d "
                "first_blocked=%s worst=%s ready_ms=%.1f grp_ms=%s reqs=%s "
                "keys=%s bytes=%s grp2_ms=%s bytes2=%s",
                index,
                sum(gpu),
                sum(cpu for *_, cpu in blocked),
                len(blocked),
                blocked[0][0] if blocked else -1,
                f"{blocked[gpu.index(max(gpu))][0]}:{max(gpu):.1f}" if gpu else "-",
                info.get("ready_ms", -1.0),
                info.get("grp_ms", []),
                info.get("reqs", {}),
                info.get("keys", {}),
                info.get("bytes", {}),
                info.get("grp2_ms", []),
                info.get("bytes2", {}),
            )

    def reset(self) -> None:
        self._producer_index = -1
        self.consumer_index = -1
        self._futures.clear()
        self._blocked.clear()
        self.load_info.clear()
        self._reports.clear()


@dataclass
class _PoolRangePlan:
    """Object keys and locations for one pool load."""

    name: PoolName
    keys: list[str]
    locations: list[int]
    entries_per_page: int
    # Page split: per page, True when the page lives on the second server.
    routes: list[bool] | None = None


# One queued offload: the pools it resolved to, and the event guarding its KV.
_OffloadTask = tuple[list[PoolTransfer], object]


def _offload_task_pages(expanded: list[PoolTransfer]) -> int:
    """Pages this task puts into its widest pool, which is what sizes a plan."""
    return max((len(transfer.keys or ()) for transfer in expanded), default=0)


def _object_sizes_per_page(entry) -> list[int]:
    """Return per-page object sizes independently of emitted ranges.

    This keeps the tier's exact-tiling validation independent of range generation.
    """
    if entry.packed:
        return [
            sum(size for component in entry.buffer_meta for _, _, size in component)
        ]
    return [sum(size for _, _, size in component) for component in entry.buffer_meta]


def _config_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"UMBP linker config {key!r} must be boolean, got {value!r}.")


def _materialize_cpu_indices(indices: torch.Tensor) -> torch.Tensor:
    """Materialize CPU indices used to derive per-pool row locations."""
    return indices.detach().to(device="cpu", dtype=torch.int64).flatten()


def _parse_storage_extra_config(raw_config):
    # Keep the linker module importable in CPU-only unit tests. The hybrid
    # controller imports device-specific memory-pool modules transitively.
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )

    extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
        raw_config
    )
    return extra_config


class UMBPDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        _storage=None,
    ):
        self.page_size = params.page_size
        # Group layers to amortize per-object RPC overhead; 8 is the measured default.
        self.layer_group = max(1, int(os.getenv("UMBP_LAYER_GROUP", "8")))
        # Coalesce queued offload tasks up to this many pages; offload_nodes
        # queues one task per node, so they arrive a page or two at a time.
        self._offload_coalesce_pages = max(
            1, int(os.getenv("UMBP_OFFLOAD_COALESCE_PAGES", "1024"))
        )
        if _config_bool(os.getenv("UMBP_LOAD_SPLIT") or "0", "UMBP_LOAD_SPLIT"):
            raise ValueError(
                "UMBP_LOAD_SPLIT is not supported by the dedup-after-insert "
                "load flow: ranks may receive different page sets and deadlock."
            )

        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self._async_offload_index_snapshot = True
        self._offload_index_fallback_warned = False
        self._offload_index_stream = None
        self._offload_index_done = None
        self._offload_index_device = None
        self._offload_index_buffers: list[torch.Tensor | None] = []
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        tp_rank = 0
        if distributed:
            tp_rank = torch.distributed.get_rank(group=params.tp_cache_group)
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        rank_replicated = self.pool_group.rank_replicated
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        if self.num_layers <= 0:
            raise ValueError("UMBP requires at least one logical layer.")
        self.pool_layers = {
            name: _ordered_layers(entry) for name, entry in self.pools.items()
        }
        invalid_layers = {
            name: [layer for layer in layers if not 0 <= layer < self.num_layers]
            for name, layers in self.pool_layers.items()
        }
        invalid_layers = {
            name: layers for name, layers in invalid_layers.items() if layers
        }
        if invalid_layers:
            raise ValueError(
                f"UMBP pool mappings contain out-of-range logical layers: {invalid_layers}."
            )
        extra_config = _parse_storage_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        extra_config = dict(extra_config)
        standalone_requested = bool(
            extra_config.get("standalone_address")
            or os.getenv("UMBP_STANDALONE_ADDRESS")
        )
        if "ssd_enabled" in extra_config and _config_bool(
            extra_config["ssd_enabled"], "ssd_enabled"
        ):
            raise ValueError(
                "Direct UMBP requires ssd_enabled=false because its GPU path "
                "cannot use the corresponding host-memory fallback."
            )
        extra_config["ssd_enabled"] = False

        if "cache_remote_fetches" in extra_config and _config_bool(
            extra_config["cache_remote_fetches"], "cache_remote_fetches"
        ):
            raise ValueError(
                "Direct UMBP requires cache_remote_fetches=false because its GPU "
                "path cannot use the corresponding host-memory fallback."
            )
        if standalone_requested:
            extra_config.pop("cache_remote_fetches", None)
        else:
            extra_config["cache_remote_fetches"] = False

        min_object_size = min(
            min(
                pool.get_page_buffer_meta(
                    torch.arange(pool.page_size, dtype=torch.int64)
                )[1]
            )
            for pool in self.pools.values()
        )
        if standalone_requested:
            extra_config.pop("dram_page_size", None)
        else:
            dram_page_size = int(extra_config.get("dram_page_size", min_object_size))
            if not 0 < dram_page_size <= min_object_size:
                raise ValueError(
                    "Direct UMBP requires 0 < dram_page_size <= the smallest "
                    f"per-layer object ({min_object_size} bytes), got {dram_page_size}."
                )
            extra_config["dram_page_size"] = dram_page_size

        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=get_parallel().tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
        )

        if _storage is None:
            from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

            # per_rank_keyspace: every object key this class writes carries a
            # tp{rank} suffix (set just below), so the store must not put the
            # ranks into the shared-SSD leader/follower scheme meant for
            # deduplicating replicated MLA KV.
            self.storage = UMBPStore(
                storage_config, mem_pool_host=None, per_rank_keyspace=True
            )
        else:
            self.storage = _storage

        try:
            client = self.storage.client
            mode = client.get_deployment_mode()
            mode_type = type(mode)
            # What page-granular objects actually need is ranged multi-buffer
            # I/O. That used to be true only of StandaloneProcess, so the gate
            # was written as a mode test -- but mori now implements it in the
            # in-process client too ("Both media behind LocalStorageManager
            # implement ranged I/O now", standalone_client.h), and a mode test
            # would keep rejecting a client that can do the job.
            #
            # So ask the client what it supports instead of inferring it from
            # which mode it is. supports_ranged_io() is the capability this
            # code depends on, it is already consulted below, and a client that
            # answers truthfully cannot be wrongly admitted by it.
            supports_ranged = getattr(client, "supports_ranged_io", None)
            if not callable(supports_ranged) or not bool(supports_ranged()):
                raise ValueError(
                    f"Direct UMBP needs ranged multi-buffer I/O, which this "
                    f"{mode!r} client does not advertise. Upgrade mori; if the "
                    "server has a Distributed inner backend, set "
                    "UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES to a positive value "
                    "(both scratch arenas must be non-zero), and note that a "
                    "read-only SharedSSDFollower is whole-object by design."
                )
            get_backend_mode = getattr(client, "get_backend_mode", None)
            self.backend_mode = (
                get_backend_mode() if callable(get_backend_mode) else None
            )
            self.deployment_mode = mode
            self._standalone_process_mode = mode == mode_type.StandaloneProcess
            if getattr(self.storage, "_disable_zero_copy_register", False):
                raise ValueError(
                    "Direct UMBP cannot disable zero-copy memory registration."
                )

            self.storage.mem_pool_host = self.pool_group
            self.storage._kv_anchor_is_logical = True
            self.storage.registered_pools = self.pools
            rank_suffix = _storage_suffix(
                rank_replicated=rank_replicated,
                tp_rank=tp_rank,
                attn_cp_rank=params.attn_cp_rank,
                pp_rank=params.pp_rank,
            )
            self.storage.mla_suffix = rank_suffix
            self.storage.mha_suffix = rank_suffix
            self._register_buffers()
            # Report the mode rather than asserting it in the text: the line is
            # what every acceptance check greps to prove the linker attached at
            # all, and it used to say "standalone_process" unconditionally, so
            # it could not have shown an embedded run for what it was.
            logger.info(
                "UMBPDirectLinker topology=%s+%s ranged_io=yes "
                "rank_replicated=%s suffix=%s",
                mode.name,
                self.backend_mode.name if self.backend_mode is not None else None,
                rank_replicated,
                rank_suffix,
            )
        except BaseException:
            self.storage.close()
            raise

        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        # Tier-split prototype: a second (SSD-only) server holds each page's
        # back layers ("layer") or a hashed fraction of the pages ("page").
        self.split_mode = os.getenv("UMBP_SPLIT_MODE", "")
        self.storage2 = None
        self._stage_arena = None
        self._load_reads = (
            ThreadPoolExecutor(LOAD_READS, thread_name_prefix=f"umbp-loadrd-tp{tp_rank}") if LOAD_READS else None
        )
        if self.split_mode:
            try:
                self._init_split(storage_config, tp_rank)
                if int(os.getenv("UMBP_STAGE_ARENA_MB", "0")) > 0:
                    self._init_stage(tp_rank)
            except BaseException:
                self.storage.close()
                raise
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self._pending: dict[str, list[PoolTransfer]] = {}
        self._gc_frozen = False
        self._load_queue: Queue[
            tuple[int, list[str], list[_PoolRangePlan], object] | None
        ] = Queue()
        self._completed_loads: Queue[list[str]] = Queue()
        self._offload_queue: Queue[tuple[list[PoolTransfer], object] | None] = Queue()
        self._offload_results: Queue[bool] = Queue()
        self._stats = {
            "lookup": 0,
            "load": 0,
            "offload": 0,
            # offload / offload_batches is the coalescing actually achieved.
            "offload_batches": 0,
        }
        # LW_STATS: scheduler-thread ms in lookups and prefill builds.
        self._lk: dict[str, float] = defaultdict(float)
        self._lk_next_log = 0.0
        if LW_STATS:
            _install_stage_hook()
        self._load_thread = threading.Thread(
            target=self._load_thread_func,
            daemon=True,
            name=f"umbp-load-tp{tp_rank}",
        )
        self._offload_thread = threading.Thread(
            target=self._offload_thread_func,
            daemon=True,
            name=f"umbp-offload-tp{tp_rank}",
        )
        self._closed = False
        self._load_thread.start()
        self._offload_thread.start()

    def _init_split(self, storage_config, tp_rank: int) -> None:
        if self.split_mode not in ("layer", "page"):
            raise ValueError(f"UMBP_SPLIT_MODE must be layer or page, got {self.split_mode!r}.")
        from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

        address = os.environ["UMBP_SPLIT_ADDRESS"]
        fraction = float(os.getenv("UMBP_SPLIT_SSD_FRACTION", "0.667"))
        config = copy.copy(storage_config)
        config.extra_config = dict(storage_config.extra_config, standalone_address=address)
        self.storage2 = UMBPStore(config, mem_pool_host=None, per_rank_keyspace=True)
        for attr in ("mem_pool_host", "_kv_anchor_is_logical", "registered_pools",
                     "mla_suffix", "mha_suffix"):
            setattr(self.storage2, attr, getattr(self.storage, attr))
        for allocation in self._registered:
            if not self.storage2.client.register_memory(*allocation):
                raise RuntimeError(f"Failed to register a GPU KV buffer with {address}.")
        self._split_executor = ThreadPoolExecutor(1, thread_name_prefix=f"umbp-load2-tp{tp_rank}")
        self._exists_executor = ThreadPoolExecutor(2, thread_name_prefix=f"umbp-exists-tp{tp_rank}")
        self._page_threshold = int(fraction * 1000)
        self._placement = None
        if SPLIT_WEIGHTS:
            dram, ssd = (int(w) for w in SPLIT_WEIGHTS.split(":"))
            self._placement = (dram, dram + ssd)
            fraction = ssd / (dram + ssd)
        self.split_layer = None
        if self.split_mode == "layer":
            fraction = self._init_layer_split(fraction)
        logger.info(
            "UMBP tier split mode=%s second=%s ssd_fraction=%.3f split_layer=%s/%d",
            self.split_mode, address, fraction, self.split_layer, self.num_layers,
        )

    def _init_layer_split(self, fraction: float) -> float:
        """Pick the back layers that go to the second server; returns their byte share."""
        per_layer = [0] * self.num_layers
        for entry in self.pools.values():
            for layer, buffer_index in entry.layer_mapping.items():
                per_layer[layer] += sum(c[buffer_index][2] for c in entry.buffer_meta)
        # Largest back stack whose bytes stay within the SSD fraction.
        split, back = self.num_layers, 0
        while split > 1 and back + per_layer[split - 1] <= fraction * sum(per_layer):
            split -= 1
            back += per_layer[split]
        self.split_layer = split
        # Front and back objects repack their layers' ranges contiguously.
        self._part_offsets, self._part_sizes = {}, {}
        for name, entry in self.pools.items():
            for part, layers in (("front", range(split)), ("back", range(split, self.num_layers))):
                buffers = sorted(entry.layer_mapping[l] for l in layers if l in entry.layer_mapping)
                tables, sizes, offset = [], [], 0
                for component in entry.buffer_meta:
                    if not entry.packed:
                        offset = 0
                    table = {}
                    for b in buffers:
                        table[b] = offset
                        offset += component[b][2]
                    tables.append(table)
                    sizes.append(offset)
                self._part_offsets[(name, part)] = tables
                self._part_sizes[(name, part)] = [offset] if entry.packed else sizes
        return back / max(1, sum(per_layer))

    def _page_routes(self, page_keys: list[str], transfer: PoolTransfer) -> list[bool]:
        """Page split: which pages the second stream takes, the same on every rank."""
        if self._placement is None:
            return [zlib.crc32(key.encode()) % 1000 < self._page_threshold for key in page_keys]
        # The server places each object by its key; a page with any SSD object goes second.
        dram, total = self._placement
        keys, n = self._object_keys_for_pages(page_keys, transfer)
        ssd = (_fnv1a(keys) % np.uint64(total) >= dram).reshape(len(page_keys), n)
        ssd &= np.array(_object_sizes_per_page(self.pools[transfer.name])) >= SPLIT_SSD_MIN
        return ssd.any(axis=1).tolist()

    def _part_layers(self, name: PoolName, part: str) -> list[int]:
        layers = self.pool_layers[name]
        return [l for l in layers if (l < self.split_layer) == (part == "front")]

    # ---- stage-ahead: queued requests' second-server objects into a host arena ----

    def _init_stage(self, tp_rank: int) -> None:
        import mori.umbp as umbp_mod

        size = int(os.environ["UMBP_STAGE_ARENA_MB"]) << 20
        self._stage_alloc = umbp_mod.UMBPHostMemAllocator()
        handle = self._stage_alloc.alloc(
            size, umbp_mod.UMBPHostBufferBacking.AnonymousShm, 2 << 20, -1, True
        )
        if not handle:
            raise RuntimeError(f"UMBP stage arena alloc of {size} bytes failed.")
        self._stage_handle = handle
        ptr = int(handle.ptr)
        if not self.storage2.client.register_memory(ptr, size):
            raise RuntimeError("Failed to register the stage arena with the second server.")
        # Unpinned still works, only the per-group copies stop being async.
        self._stage_pinned = int(torch.cuda.cudart().cudaHostRegister(ptr, size, 0)) == 0
        arena = torch.frombuffer(
            (ctypes.c_byte * size).from_address(ptr), dtype=torch.uint8, count=size
        )
        self._init_stage_state(arena, tp_rank)
        _install_stage_hook()
        logger.info(
            "UMBP stage arena %d MiB pinned=%s depth=%d",
            size >> 20, self._stage_pinned, self._stage_depth,
        )

    def _init_stage_state(self, arena: torch.Tensor, tp_rank: int) -> None:
        self._stage_arena = arena
        self._stage_ptr = arena.data_ptr()
        self._stage_cv = threading.Condition()
        self._stage_free = [(0, arena.numel())]
        self._stage_runs: dict[str, list[_StageRun]] = {}
        # (pool, first object key of a page) -> (run, page index in the run)
        self._staged: dict[tuple[PoolName, str], tuple[_StageRun, int]] = {}
        self._stage_seen: dict[str, float] = {}
        # rid -> (page hashes, {pool: page routes}, {pool: [transfer, objects per page, keys, page known present]})
        self._stage_objs: dict[str, tuple] = {}
        self._stage_waiting: set[str] = set()
        self._stage_loading: set[str] = set()
        self._stage_gone: dict[str, float] = {}
        self._stage_depth = int(os.getenv("UMBP_STAGE_DEPTH", "64"))
        self._stage_lease = float(os.getenv("UMBP_STAGE_LEASE_S", "600"))
        self._stage_stats: dict[str, int] = defaultdict(int)
        self._stage_stream = None
        self._stage_next_expire = self._stage_next_log = time.monotonic()
        self._stage_exec = ThreadPoolExecutor(1, thread_name_prefix=f"umbp-stage-tp{tp_rank}")
        self._stage_reads = ThreadPoolExecutor(STAGE_READS, thread_name_prefix=f"umbp-stage-read-tp{tp_rank}")
        layer = self.split_mode == "layer"
        # Group-major like the second load stream, so a group's copy is contiguous.
        self._stage_groups = (
            self._layer_groups(self.split_layer, self.num_layers) if layer else self._layer_groups()
        )
        self._stage_part, self._stage_suffix = ("back", "|b") if layer else (None, "")
        self._stage_page_bytes = {
            name: sum(
                component[entry.layer_mapping[l]][2]
                for group in self._stage_groups
                for l in group
                if l in entry.layer_mapping
                for component in entry.buffer_meta
            )
            for name, entry in self.pools.items()
        }

    def stage_ahead(self, waiting_queue, cache) -> None:
        """Scheduler thread: hand queued requests to the stage thread, again every recheck."""
        now = time.monotonic()
        for pos, req in enumerate(waiting_queue[: self._stage_depth]):
            rid = req.rid
            if rid in self._stage_waiting or now - self._stage_seen.get(rid, -1e9) < STAGE_RECHECK_S:
                continue
            from sglang.srt.mem_cache.radix_cache import RadixKey

            self._stage_seen[rid] = now
            key = RadixKey(req.origin_input_ids, extra_key=req.extra_key, cache_salt=req.cache_salt)
            # Read-only and re-read every recheck: HBM pages are staged once evicted.
            device_len = 0 if pos < STAGE_HEAD else cache.tree_core.match_full_device_prefix(key)[0]
            key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
            with self._stage_cv:
                self._stage_waiting.add(rid)
            self._stage_exec.submit(self._stage_request, rid, key, device_len)
        if self._stage_runs or self._stage_waiting:
            queued = {req.rid for req in waiting_queue}
            with self._stage_cv:
                # Admitted without a load, or aborted: nothing will consume the runs.
                for rid in [*self._stage_runs, *self._stage_waiting]:
                    if rid not in queued and rid not in self._stage_loading:
                        self._stage_gone[rid] = now
                        self._stage_waiting.discard(rid)
                        self._stage_objs.pop(rid, None)
                        for run in list(self._stage_runs.get(rid, ())):
                            self._doom_locked(run)
        if now >= self._stage_next_expire:
            self._stage_next_expire = now + 5
            self._expire_stage(now)
        if now >= self._stage_next_log:
            self._stage_next_log = now + 60
            with self._stage_cv:
                used = self._stage_arena.numel() - sum(size for _, size in self._stage_free)
            logger.info("UMBPSTAGE used_mb=%d %s", used >> 20, dict(self._stage_stats))

    def _stage_request(self, rid: str, key, device_len: int) -> None:
        try:
            self._stage_one(rid, key, device_len)
        except Exception:
            self._stage_stats["errors"] += 1
            logger.exception("UMBP stage of rid=%s failed", rid)
        finally:
            with self._stage_cv:
                self._stage_waiting.discard(rid)

    def _stage_hashes(self, key) -> list[str]:
        from sglang.srt.mem_cache.utils import get_storage_hash_str

        n = len(key) // self.page_size * self.page_size
        return get_storage_hash_str(key[:n], None, page_size=self.page_size) if n else []

    def _stage_objects(self, key):
        hashes = self._stage_hashes(key)
        transfers = self.pool_group.resolve_transfers(
            [PoolTransfer(name=PoolName.KV, keys=hashes)], allow_partial=True
        )
        routes = (
            {t.name: self._page_routes(hashes, t) for t in transfers} if self.split_mode == "page" else None
        )
        pools = {}
        for transfer in transfers:
            keys, n = self._object_keys_for_pages(hashes, transfer)
            pools[transfer.name] = [transfer, n, keys, [False] * len(hashes)]
        return hashes, routes, pools

    def _stage_one(self, rid: str, key, device_len: int) -> None:
        if rid in self._stage_gone:
            self._stage_stats["late"] += 1
            return
        state = self._stage_objs.get(rid)
        if state is None:
            state = self._stage_objs[rid] = self._stage_objects(key)
        hashes, routes, pools = state
        # A load restores [HBM prefix, first page missing on either server).
        first, end = device_len // self.page_size, len(hashes)
        for transfer, _, _, have in pools.values():
            i = first
            while i < end:
                if have[i]:
                    i += 1
                    continue
                chunk = range(i, min(end, i + CHUNK_PAGES * 4))
                for j, present in zip(chunk, self._page_exists([hashes[j] for j in chunk], transfer, rid)):
                    have[j] = present
                    if not present:
                        end = j
                        break
                i = chunk[-1] + 1
        todo = {}
        for name, (_, n, keys, _) in pools.items():
            if self.split_mode == "layer" and not self._part_layers(name, "back"):
                continue
            with self._stage_cv:
                # A page another queued request staged is shared, not read twice.
                fresh = [
                    i for i in range(first, end)
                    if (routes is None or routes[name][i]) and (name, keys[i * n]) not in self._staged
                ]
            if fresh:
                todo[name] = (n, [keys[i * n + j] for i in fresh for j in range(n)])
        if not todo:
            self._stage_stats["nothing"] += 1
            return
        # Bounded chunks, prefix first: a full or fragmented arena still takes the head.
        per = max(1, STAGE_CHUNK // sum(self._stage_page_bytes[name] for name in todo))
        most = max(len(keys) // n for n, keys in todo.values())
        reads = []
        try:
            for start in range(0, most, per):
                chunk = {
                    name: (n, keys[start * n : (start + per) * n])
                    for name, (n, keys) in todo.items()
                    if len(keys) > start * n
                }
                run = self._stage_begin(rid, chunk)
                if run is None:
                    break
                # Chunks read in parallel: a short queue leaves little time to stage.
                reads.append(self._stage_reads.submit(self._stage_run, run, chunk))
        finally:
            for read in reads:
                read.result()

    def _stage_begin(self, rid: str, todo: dict[PoolName, tuple[int, list[str]]]):
        """Reserve a run and claim its pages; None when the arena has no room."""
        counts = {name: len(keys) // n for name, (n, keys) in todo.items()}
        blocks, size = self._stage_layout(counts)
        reserved = self._stage_reserve(rid, size)
        if reserved is None:
            return None
        run = _StageRun(rid, *reserved, time.monotonic(), threading.Event(), counts, blocks, [])
        with self._stage_cv:
            for name, (n, keys) in todo.items():
                for i in range(counts[name]):
                    page_key = (name, keys[i * n])
                    if page_key not in self._staged:
                        self._staged[page_key] = (run, i)
                        run.keys.append(page_key)
            self._stage_runs.setdefault(rid, []).append(run)
        return run

    def _stage_run(self, run: _StageRun, todo: dict[PoolName, tuple[int, list[str]]]) -> None:
        size, t0 = run.size, run.t0
        failed = []
        try:
            for name, (n, keys) in todo.items():
                ok = self._stage_get(name, n, keys, run)
                failed.extend((name, keys[i * n]) for i, good in enumerate(ok) if not good)
        except BaseException:
            failed = list(run.keys)
            raise
        finally:
            with self._stage_cv:
                for page_key in failed:
                    if self._staged.get(page_key, (None,))[0] is run:
                        del self._staged[page_key]
                run.ready = True
                run.done.set()
                self._stage_stats["runs"] += 1
                self._stage_stats["bytes_mb"] += -(-size >> 20)
                self._stage_stats["failed_pages"] += len(failed)
                self._stage_stats["stage_ms"] += int((time.monotonic() - t0) * 1e3)
                if run.doomed and not run.busy:
                    self._free_run_locked(run)

    def _stage_layout(self, counts: dict[PoolName, int]):
        blocks, pos = {}, 0
        for group in self._stage_groups:
            for name, pages in counts.items():
                entry = self.pools[name]
                for layer in group:
                    b = entry.layer_mapping.get(layer)
                    if b is None:
                        continue
                    for c, component in enumerate(entry.buffer_meta):
                        blocks[(name, layer, c)] = (pos, component[b][2])
                        pos += pages * component[b][2]
        return blocks, pos

    def _stage_get(self, name: PoolName, n: int, keys: list[str], run: _StageRun) -> list[bool]:
        """Read one pool's staged objects into the run's blocks; per-page success."""
        entry = self.pools[name]
        part = self._stage_part
        tables = self._part_offsets[(name, part)] if part else entry._component_offsets
        base = self._stage_ptr + run.offset
        items = []
        for group in self._stage_groups:
            for layer in group:
                b = entry.layer_mapping.get(layer)
                if b is not None:
                    items.append([
                        (base + run.blocks[(name, layer, c)][0], component[b][2], tables[c][b])
                        for c, component in enumerate(entry.buffer_meta)
                    ])
        pages = len(keys) // n
        idx = np.arange(pages, dtype=np.int64)
        if entry.packed:
            flat = [item for row in items for item in row]
            ptr0 = np.array([item[0] for item in flat], np.int64)
            size = np.array([item[1] for item in flat], np.int64)
            ptrs = (ptr0[None, :] + idx[:, None] * size[None, :]).tolist()
            sizes = [[item[1] for item in flat]] * pages
            offsets = [[item[2] for item in flat]] * pages
        else:
            ptr0 = np.array([[row[c][0] for row in items] for c in range(n)], np.int64)
            size = np.array([[row[c][1] for row in items] for c in range(n)], np.int64)
            ptrs = (ptr0[None] + idx[:, None, None] * size[None]).reshape(pages * n, -1).tolist()
            sizes = [[row[c][1] for row in items] for c in range(n)] * pages
            offsets = [[row[c][2] for row in items] for c in range(n)] * pages
        suffix = self._stage_suffix
        keys = [key + suffix for key in keys] if suffix else keys
        step = self._entries_per_call(sizes)
        ok = []
        for start in range(0, len(keys), step):
            end = start + step
            ok.extend(
                bool(value)
                for value in self.storage2.client.batch_get_ranges_into_ptr(
                    keys[start:end], ptrs[start:end], sizes[start:end], offsets[start:end]
                )
            )
        return [all(ok[i * n : (i + 1) * n]) for i in range(pages)]

    def _stage_reserve(self, rid: str, size: int) -> tuple[int, int] | None:
        """First-fit arena run; when full, the next recheck retries."""
        size = -(-size // _ARENA_ALIGN) * _ARENA_ALIGN
        with self._stage_cv:
            if rid in self._stage_gone:
                self._stage_stats["late"] += 1
                return None
            for i, (offset, free) in enumerate(self._stage_free):
                if free >= size:
                    if free == size:
                        del self._stage_free[i]
                    else:
                        self._stage_free[i] = (offset + size, free - size)
                    return offset, size
            self._stage_stats["no_space"] += 1
            return None

    def _doom_locked(self, run: _StageRun) -> None:
        run.doomed = True
        if run.ready and not run.busy:
            self._free_run_locked(run)

    def _free_run_locked(self, run: _StageRun) -> None:
        if run.offset < 0:
            return
        self._stage_stats["pages_unused"] += max(0, len(run.keys) - run.used)
        for page_key in run.keys:
            if self._staged.get(page_key, (None,))[0] is run:
                del self._staged[page_key]
        runs = self._stage_runs.get(run.rid, [])
        if run in runs:
            runs.remove(run)
            if not runs:
                del self._stage_runs[run.rid]
        merged = []
        for offset, size in sorted([*self._stage_free, (run.offset, run.size)]):
            if merged and sum(merged[-1]) == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        self._stage_free = merged
        run.offset = -1
        self._stage_cv.notify_all()

    def _expire_stage(self, now: float) -> None:
        with self._stage_cv:
            for run in [run for runs in self._stage_runs.values() for run in runs]:
                if run.ready and now - run.t0 > self._stage_lease:
                    self._stage_stats["expired"] += 1
                    self._doom_locked(run)
            for marks in (self._stage_seen, self._stage_gone):
                for rid in [rid for rid, t in marks.items() if now - t > 2 * self._stage_lease]:
                    del marks[rid]
            for rid in [rid for rid in self._stage_objs if rid not in self._stage_seen]:
                del self._stage_objs[rid]

    def _take_staged(self, stream, rids: list[str]):
        """Split the second stream's pages into staged and direct; pins the runs used."""
        storage, plans, groups, part, suffix = stream
        with self._stage_cv:
            inflight = [run for rid in rids for run in self._stage_runs.get(rid, ()) if not run.ready]
        t0 = time.monotonic()
        for run in inflight:
            run.done.wait(max(0.0, t0 + STAGE_WAIT_S - time.monotonic()))
        direct, staged = [], {}
        with self._stage_cv:
            self._stage_stats["wait_ms"] += int((time.monotonic() - t0) * 1e3)
            for plan in plans:
                n = plan.entries_per_page
                keep, by_run = [], {}
                for i, row in enumerate(plan.locations):
                    hit = self._staged.get((plan.name, plan.keys[i * n]))
                    if hit is None or not hit[0].ready:
                        keep.append(i)
                        continue
                    _, src, dst = by_run.setdefault(id(hit[0]), (hit[0], [], []))
                    hit[0].used += 1
                    src.append(hit[1])
                    dst.append(row)
                for run, src, dst in by_run.values():
                    staged.setdefault(id(run), (run, []))[1].append((plan.name, src, dst))
                if keep:
                    direct.append(_PoolRangePlan(
                        plan.name,
                        [plan.keys[i * n + j] for i in keep for j in range(n)],
                        [plan.locations[i] for i in keep],
                        n,
                    ))
                self._stage_stats["pages_staged"] += len(plan.locations) - len(keep)
                self._stage_stats["pages_direct"] += len(keep)
            for run, _ in staged.values():
                run.busy += 1
        return (storage, direct, groups, part, suffix), list(staged.values())

    def _release_stage(self, staged, rids: list[str]) -> None:
        now = time.monotonic()
        with self._stage_cv:
            runs = [run for run, _ in staged]
            for run in runs:
                run.busy -= 1
            for rid in rids:
                self._stage_loading.discard(rid)
                self._stage_gone[rid] = now
                self._stage_objs.pop(rid, None)
                for run in self._stage_runs.get(rid, ()):
                    run.doomed = True
                    runs.append(run)
            for run in runs:
                if run.doomed and run.ready and not run.busy:
                    self._free_run_locked(run)

    def _copy_staged_group(self, staged, group: list[int], scratch, rows) -> int:
        """Enqueue one group's staged pages: a copy per run, then a scatter per scratch fill."""
        nbytes, used, desc = 0, 0, []
        for run, spans in staged:
            blocks = []
            for name, first, n in spans:
                entry = self.pools[name]
                for layer in group:
                    b = entry.layer_mapping.get(layer)
                    if b is None:
                        continue
                    for c, meta in enumerate(entry.buffer_meta):
                        pos, size = run.blocks[(name, layer, c)]
                        blocks.append((pos, run.pages[name] * size, size, *meta[b][:2], n, first))
            if not blocks:
                continue
            # A run's group is contiguous (group-major layout).
            lo = min(pos for pos, *_ in blocks)
            hi = max(pos + extent for pos, extent, *_ in blocks)
            if used + hi - lo > scratch.numel():
                self._stage_scatter(scratch, desc, rows)
                used, desc = 0, []
            scratch[used : used + hi - lo].copy_(
                self._stage_arena[run.offset + lo : run.offset + hi], non_blocking=True
            )
            base = scratch.data_ptr() + used - lo
            desc += [
                (base + pos, ptr, size, stride, n, first)
                for pos, _, size, ptr, stride, n, first in blocks
            ]
            used += hi - lo
            nbytes += hi - lo
        if desc:
            self._stage_scatter(scratch, desc, rows)
        return nbytes

    def _stage_scatter(self, scratch, desc, rows) -> None:
        """Each (block, page): size bytes from the scratch to row * stride of its buffer (addresses)."""
        if scratch.device.type == "cpu":
            # Tests: the kernel's copies, by memmove.
            src_rows, dst_rows = (r.tolist() for r in rows)
            for src, dst, size, stride, n, first in desc:
                for k in range(first, first + n):
                    ctypes.memmove(dst + dst_rows[k] * stride, src + src_rows[k] * size, size)
            return
        table = torch.tensor(desc, dtype=torch.int64).pin_memory()
        table = table.to(scratch.device, non_blocking=True)
        grid = (len(desc), max(n for *_, n, _ in desc))
        # The widest element every address, size and stride is a multiple of.
        align = math.gcd(8, *(v for row in desc for v in row[:4]))
        _stage_copy_kernel[grid](table, *rows, ELEM=_STAGE_ELEMS[align], WIDTH=align, BLOCK=1024)

    def _register_buffers(self) -> None:
        seen = set()
        self._registered: list[tuple[int, int]] = []
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                if not self.storage.client.register_memory(*allocation):
                    raise RuntimeError(
                        "Failed to register a GPU KV buffer with UMBP: "
                        f"ptr=0x{allocation[0]:x}, size={allocation[1]}."
                    )
                self._registered.append(allocation)

    def _object_keys_for_pages(
        self, page_keys: list[str], transfer: PoolTransfer
    ) -> tuple[list[str], int]:
        component_keys, multiplier = self.storage._get_hybrid_page_component_keys(
            page_keys, transfer
        )
        entry = self.pools[transfer.name]
        # One key names one stored object, and a packed entry stores a whole
        # page as one object regardless of how many components it has.
        entries_per_page = 1 if entry.packed else len(entry.components)
        if multiplier != entries_per_page:
            raise ValueError(
                f"UMBP pool {transfer.name} produced {multiplier} keys per page "
                f"but its layout yields {entries_per_page} objects per page "
                f"(packed={entry.packed}, components={len(entry.components)})."
            )
        # No layer suffix: one object per page (or per page component) holds
        # every layer, and a layer is read back as a byte range inside it.
        return component_keys, multiplier

    def _page_exists(
        self, page_keys: list[str], transfer: PoolTransfer, rid: str | None = None, acc: list | None = None
    ) -> list[bool]:
        if self.split_mode == "layer":
            # A page is a hit only when every part it was written as survives.
            parts = [
                (storage, suffix)
                for storage, suffix, part in ((self.storage, "", "front"), (self.storage2, "|b", "back"))
                if self._part_layers(transfer.name, part)
            ]
            # Both servers at once: lookups block the lockstep DP ranks.
            results = [
                self._exists_executor.submit(self._exists_unstaged, s, page_keys, transfer, suffix, rid, acc)
                if s is self.storage2
                else self._exists_executor.submit(self._exists_on, s, page_keys, transfer, suffix, acc)
                for s, suffix in parts
            ]
            return [all(values) for values in zip(*(r.result() for r in results))]
        if self.split_mode == "page":
            routes = self._page_routes(page_keys, transfer)
            asks = []
            for storage, routed in ((self.storage, False), (self.storage2, True)):
                keys = [k for k, r in zip(page_keys, routes) if r == routed]
                ask = (
                    self._exists_executor.submit(self._exists_unstaged, storage, keys, transfer, "", rid, acc)
                    if routed
                    else self._exists_executor.submit(self._exists_on, storage, keys, transfer, "", acc)
                )
                asks.append((keys, ask))
            found = {k: v for keys, ask in asks for k, v in zip(keys, ask.result())}
            return [found[k] for k in page_keys]
        return self._exists_on(self.storage, page_keys, transfer, "", acc)

    def _exists_unstaged(
        self, storage, page_keys: list[str], transfer: PoolTransfer, suffix: str, rid: str | None,
        acc: list | None = None,
    ) -> list[bool]:
        """Second server: pages this request already staged are present; ask only for the rest."""
        if self._stage_arena is None or rid is None or not page_keys:
            return self._exists_on(storage, page_keys, transfer, suffix, acc)
        keys, n = self._object_keys_for_pages(page_keys, transfer)
        start = time.perf_counter()
        with self._stage_cv:
            if acc is not None:
                acc.append(("lock", time.perf_counter() - start))
            # Own runs only: they live until this request's load, another's may be freed first.
            staged = [
                (hit := self._staged.get((transfer.name, keys[i * n]))) is not None
                and hit[0].ready and hit[0].rid == rid
                for i in range(len(page_keys))
            ]
        ask = [key for key, done in zip(page_keys, staged) if not done]
        found = iter(self._exists_on(storage, ask, transfer, suffix, acc) if ask else ())
        return [done or next(found) for done in staged]

    def _exists_on(
        self, storage, page_keys: list[str], transfer: PoolTransfer, suffix: str = "",
        acc: list | None = None,
    ) -> list[bool]:
        entry = self.pools[transfer.name]
        objects_per_page = 1 if entry.packed else len(entry.components)
        max_objects = CHUNK_PAGES * self.num_layers
        pages_per_call = max(1, max_objects // objects_per_page)

        page_exists = []
        for start in range(0, len(page_keys), pages_per_call):
            chunk_pages = page_keys[start : start + pages_per_call]
            object_keys, _ = self._object_keys_for_pages(chunk_pages, transfer)
            object_keys = [key + suffix for key in object_keys]
            start = time.perf_counter()
            exists = list(storage.client.batch_exists(object_keys))
            if acc is not None:
                acc.append(("rpc", time.perf_counter() - start))
            if len(exists) != len(object_keys):
                raise RuntimeError(
                    f"UMBP exists result-size mismatch for pool {transfer.name}: "
                    f"expected={len(object_keys)} actual={len(exists)}."
                )
            page_exists.extend(
                all(exists[index : index + objects_per_page])
                for index in range(0, len(exists), objects_per_page)
            )
        return page_exists

    @staticmethod
    def _apply_hit_policy(
        valid_pages: list[int], page_exists: list[bool], transfer: PoolTransfer
    ) -> list[int]:
        present_prefix = [0]
        for present in page_exists:
            present_prefix.append(present_prefix[-1] + int(present))

        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            return [end for end in valid_pages if present_prefix[end] == end]
        if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys or ()))
            return [
                end
                for end in valid_pages
                if present_prefix[end] - present_prefix[max(0, end - trailing)]
                == end - max(0, end - trailing)
            ]
        raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        start = time.perf_counter()
        try:
            return self._lookup(rid, transfers)
        finally:
            if LW_STATS:
                self._time("lookup", start)

    def _time(self, name: str, start: float) -> None:
        now = time.perf_counter()
        ms = (now - start) * 1e3
        self._lk[f"{name}_ms"] += ms
        self._lk[f"{name}s"] += 1
        self._lk[f"{name}_max_ms"] = max(self._lk[f"{name}_max_ms"], ms)
        if now >= self._lk_next_log:
            self._lk_next_log = now + 60
            logger.info("UMBPLK %s", {k: round(v, 1) for k, v in self._lk.items()})

    def _lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys or [])
        if not page_keys:
            return []

        valid_pages = list(range(1, len(page_keys) + 1))
        # Lookup split for LW_STATS: scheduler time in _page_exists vs. RPC and stage-lock time under it.
        acc = [] if LW_STATS else None
        for transfer in expanded:
            # Probe only as far as the surviving boundary: no hit policy reads
            # past its own end offset, so pages beyond the longest candidate
            # cannot change the answer. This runs synchronously inside the
            # scheduler's prefill batch build, and DP-attention ranks are
            # lockstep, so every extra key is stall charged to all of them.
            start = time.perf_counter()
            page_exists = self._page_exists(page_keys[: valid_pages[-1]], transfer, rid, acc)
            if acc is not None:
                acc.append(("exists", time.perf_counter() - start))
            valid_pages = self._apply_hit_policy(valid_pages, page_exists, transfer)
            if not valid_pages:
                break

        self._stats["lookup"] += 1
        for name, seconds in acc or ():
            self._lk[f"{name}_ms"] += seconds * 1e3
        if valid_pages:
            logger.debug(
                "UMBP direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                valid_pages[-1],
                len(valid_pages),
            )
        return valid_pages

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Lookup establishes a restorable boundary before insert de-duplicates
        # resident pages. The remaining transfer can therefore contain only a
        # side pool such as SWA, with no KV transfer at all.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self._pending:
            raise RuntimeError(f"UMBP load for rid={rid} is already queued.")
        self._pending[rid] = expanded
        if self._stage_arena is not None:
            with self._stage_cv:
                self._stage_loading.add(rid)
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # The tree node is already visible in L1. Dropping its transfer would
        # leave a device hit pointing at slots that were never populated.
        return False

    def num_completed_loads(self) -> int:
        return self._completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self._completed_loads.get_nowait()

    def start_layer_wise_loading(self) -> int:
        if not self._pending:
            return -1
        self._freeze_gc_once()
        pending = self._pending
        rids = list(pending)
        plans = self._build_load_plans(list(pending.values()))
        ready_event = device_module.Event()
        ready_event.record()
        counter_index = self.layer_done_counter.update_producer()
        if LW_STATS:
            # A request whose transfers are all SWA pools only restores side
            # state for a prefix that already hit in HBM.
            kv = sum(
                any("SWA" not in t.name.name and t.keys for t in transfers)
                for transfers in pending.values()
            )
            self.layer_done_counter.load_info[counter_index] = {
                "reqs": {"kv": kv, "side_only": len(pending) - kv},
                "keys": {plan.name.name: len(plan.keys) for plan in plans},
            }
        self._load_queue.put((counter_index, rids, plans, ready_event))
        self._pending = {}
        self._stats["load"] += len(pending)
        return counter_index

    def _build_load_plans(
        self,
        request_transfers: list[list[PoolTransfer]],
        *,
        materialize_indices: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> list[_PoolRangePlan]:
        """Build a batch plan shared by load and offload."""
        grouped: dict[PoolName, list[PoolTransfer]] = {}
        for transfers in request_transfers:
            for transfer in transfers:
                grouped.setdefault(transfer.name, []).append(transfer)

        plans = []
        # One logical source can fan out to several physical pools (for
        # example GLM's KV and INDEXER pools). Snapshot it once, then let each
        # pool independently validate and derive its own row geometry.
        cpu_indices: dict[int, torch.Tensor] = {}
        for name, transfers in grouped.items():
            entry = self.pools[name]
            entries_per_page = 1 if entry.packed else len(entry.components)
            keys: list[str] = []
            locations: list[int] = []
            routes: list[bool] = []
            for transfer in transfers:
                page_keys = list(transfer.keys or [])
                if self.split_mode == "page":
                    routes.extend(self._page_routes(page_keys, transfer))
                transfer_keys, multiplier = self._object_keys_for_pages(
                    page_keys, transfer
                )
                if multiplier != entries_per_page:
                    raise ValueError(
                        f"UMBP pool {name} emits {multiplier} keys per page but "
                        f"its layout yields {entries_per_page} objects per page "
                        f"(packed={entry.packed})."
                    )
                if len(transfer_keys) != len(page_keys) * entries_per_page:
                    raise ValueError(
                        f"UMBP pool {name} key count mismatch: "
                        f"keys={len(transfer_keys)} pages={len(page_keys)}."
                    )
                keys.extend(transfer_keys)
                indices = transfer.host_indices
                if indices is None:
                    raise ValueError(f"UMBP pool {name} transfer has no indices.")
                source_id = id(indices)
                prepared_indices = cpu_indices.get(source_id)
                if prepared_indices is None:
                    prepared_indices = (
                        materialize_indices(indices)
                        if materialize_indices is not None
                        else _materialize_cpu_indices(indices)
                    )
                    cpu_indices[source_id] = prepared_indices
                locations.extend(entry.prepare_locations(prepared_indices))
            if len(keys) != len(locations) * entries_per_page:
                raise ValueError(
                    f"UMBP pool {name} plan mismatch: keys={len(keys)} "
                    f"rows={len(locations)} per_page={entries_per_page}."
                )
            plans.append(
                _PoolRangePlan(name, keys, locations, entries_per_page, routes or None)
            )

        if not plans or not plans[0].keys:
            raise ValueError("Layer-wise UMBP load has no object keys.")
        return plans

    def _materialize_offload_indices(
        self, indices: torch.Tensor, slot: int
    ) -> torch.Tensor:
        if not self._async_offload_index_snapshot or indices.device.type == "cpu":
            return _materialize_cpu_indices(indices)
        if (
            indices.dtype != torch.int64
            or indices.ndim != 1
            or not indices.is_contiguous()
        ):
            return self._fallback_offload_indices(indices, "tensor shape or dtype")
        if (
            self._offload_index_device is not None
            and indices.device != self._offload_index_device
        ):
            return self._fallback_offload_indices(indices, "device mismatch")

        try:
            with device_module.device(indices.device):
                if self._offload_index_stream is None:
                    self._offload_index_device = indices.device
                    self._offload_index_stream = device_module.Stream(
                        device=indices.device
                    )
                    self._offload_index_done = device_module.Event()
                while len(self._offload_index_buffers) <= slot:
                    self._offload_index_buffers.append(None)
                count = indices.numel()
                buffer = self._offload_index_buffers[slot]
                if buffer is None or buffer.numel() < count:
                    buffer = self._allocate_offload_index_buffer(count)
                    self._offload_index_buffers[slot] = buffer
                source = indices.detach()
                with device_module.stream(self._offload_index_stream):
                    # Register before enqueue so a failed copy cannot leave an
                    # untracked side-stream read of the source allocation.
                    source.record_stream(self._offload_index_stream)
                    buffer[:count].copy_(source, non_blocking=True)
                    self._offload_index_done.record(self._offload_index_stream)
                self._offload_index_done.synchronize()
                return buffer[:count]
        except RuntimeError:
            self._async_offload_index_snapshot = False
            logger.exception(
                "UMBP async index snapshot failed; falling back to synchronous D2H"
            )
            return _materialize_cpu_indices(indices)

    @staticmethod
    def _allocate_offload_index_buffer(count: int) -> torch.Tensor:
        return torch.empty(count, dtype=torch.int64, device="cpu", pin_memory=True)

    def _fallback_offload_indices(
        self, indices: torch.Tensor, reason: str
    ) -> torch.Tensor:
        if not self._offload_index_fallback_warned:
            self._offload_index_fallback_warned = True
            logger.warning(
                "UMBP offload index snapshot is using synchronous fallback: "
                "%s (device=%s dtype=%s shape=%s contiguous=%s)",
                reason,
                indices.device,
                indices.dtype,
                tuple(indices.shape),
                indices.is_contiguous(),
            )
        return _materialize_cpu_indices(indices)

    def _load_thread_func(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, rids, plans, ready_event = task
                try:
                    self._run_layer_wise_batch(counter_index, plans, ready_event, rids)
                finally:
                    self._completed_loads.put(rids)
            finally:
                self._load_queue.task_done()

    def _all_layer_ranges(self, plan: _PoolRangePlan):
        """Every layer's ranges, accumulated per object.

        Offload requires one call to carry ranges that tile the object exactly,
        so an object's ranges must never be split across calls.

        The group is the pool's own layer stack, so it always covers the pool
        and the None return is unreachable; rejecting it keeps a future caller
        that passes a foreign plan from failing on a tuple unpack instead.
        """
        meta = self._layer_group_ranges(plan, self.pool_layers[plan.name])
        if meta is None:
            raise ValueError(
                f"UMBP pool {plan.name} covers none of its own layers "
                f"({self.pool_layers[plan.name]})."
            )
        return meta

    @staticmethod
    def _plans_covering(
        by_layer: dict[int, list[_PoolRangePlan]], group: list[int]
    ) -> list[_PoolRangePlan]:
        """Plans touching any layer of the group, each listed once, in order."""
        seen: set[int] = set()
        plans = []
        for logical_layer in group:
            for plan in by_layer.get(logical_layer, ()):
                if id(plan) in seen:
                    continue
                seen.add(id(plan))
                plans.append(plan)
        return plans

    def _range_items(self, plan: _PoolRangePlan, layers: list[int], part: str | None = None):
        """(base_ptr, row_stride, size, offset) per emitted range, in wire order.

        Grouped by layer, and within a layer by component. Every object emits
        this same tuple sequence; the only thing that varies across objects is
        the pointer, by ``row * row_stride``. That invariant is what the
        vectorized builder rests on.
        """
        entry = self.pools[plan.name]
        tables = self._part_offsets[(plan.name, part)] if part else entry._component_offsets
        items: list[list[tuple[int, int, int, int]]] = []
        for logical_layer in layers:
            buffer_index = entry.layer_mapping.get(logical_layer)
            if buffer_index is None:
                continue
            items.append(
                [
                    (*component[buffer_index], offsets[buffer_index])
                    for component, offsets in zip(entry.buffer_meta, tables)
                ]
            )
        return items

    def _layer_group_ranges(
        self, plan: _PoolRangePlan, layers: list[int], part: str | None = None
    ):
        """One group of layers' ranges, one nested list per object.

        Built column-wise rather than object by object. A 256K restore emits
        ~63k ranges across its pools, and assembling those lists one page at a
        time cost ~43 ms per load -- serialized ahead of every group's transfer,
        on the same thread, so it landed directly on TTFT. Two redundancies pay
        for that: ``sizes`` and ``offsets`` do not depend on the row yet were
        rebuilt for every page, and ``ptrs`` is affine in the row so the whole
        column can be computed at once.

        Moving the work to a helper thread was tried and does not pay: the load
        thread has to re-acquire the GIL between blocking transfers and gives the
        saving straight back. It has to go away rather than move.

        Returns None when this pool covers none of the group, so the caller can
        skip the call.
        """
        items = self._range_items(plan, layers, part)
        if not items:
            return None
        rows = np.asarray(plan.locations, dtype=np.int64)
        if not rows.size:
            return [], [], []
        expected = len(rows) * plan.entries_per_page
        if expected != len(plan.keys):
            # One object per key, so a mismatch means pointers would be paired
            # with the wrong objects rather than anything failing loudly.
            raise ValueError(
                f"UMBP pool {plan.name} has {len(plan.keys)} keys for "
                f"{len(rows)} rows at {plan.entries_per_page} per page."
            )

        if self.pools[plan.name].packed:
            # One object per page, its ranges running (layer, component).
            flat = [item for layer_items in items for item in layer_items]
            base = np.fromiter((item[0] for item in flat), np.int64, len(flat))
            stride = np.fromiter((item[1] for item in flat), np.int64, len(flat))
            ptrs = (rows[:, None] * stride[None, :] + base[None, :]).tolist()
            # One list shared by every object instead of a copy each: the client
            # only reads these.
            sizes = [item[2] for item in flat]
            offsets = [item[3] for item in flat]
            return ptrs, [sizes] * len(rows), [offsets] * len(rows)

        # One object per (page, component), its ranges running over the layers.
        components = len(items[0])
        base = np.array([[i[0] for i in layer] for layer in items], np.int64).T
        stride = np.array([[i[1] for i in layer] for layer in items], np.int64).T
        ptrs = (
            (rows[:, None, None] * stride[None, :, :] + base[None, :, :])
            .reshape(len(rows) * components, -1)
            .tolist()
        )
        sizes = [[layer[index][2] for layer in items] for index in range(components)]
        offsets = [[layer[index][3] for layer in items] for index in range(components)]
        return ptrs, sizes * len(rows), offsets * len(rows)

    @staticmethod
    def _entries_per_call(sizes: list[list[int]]) -> int:
        """Objects per RPC, budgeted by the ranges they actually carry.

        Counted from the ranges that were built, not from the layer count. A
        packed pool puts one range per component per layer on an object, so a
        packed K/V pool carries twice what the layer count suggests and the
        budget would be overshot by that factor.
        """
        ranges_per_object = max((len(entry) for entry in sizes), default=1)
        return max(1, RANGES_PER_CALL // max(1, ranges_per_object))

    def _run_layer_wise_batch(
        self,
        counter_index: int,
        plans: list[_PoolRangePlan],
        ready_event: object,
        rids: list[str] = (),
    ) -> None:
        staged = []
        try:
            t0 = time.perf_counter()
            ready_event.synchronize()
            ready_ms = (time.perf_counter() - t0) * 1e3
            streams = self._load_streams(plans)
            stats = [([], defaultdict(int)) for _ in streams]
            # A group is released once every stream that covers it is done.
            pending: dict[int, int] = defaultdict(int)
            for stream in streams:
                for group in stream[2]:
                    pending[group[0]] += 1
            lock = threading.Lock()

            def finish(group: list[int]) -> None:
                with lock:
                    pending[group[0]] -= 1
                    if pending[group[0]]:
                        return
                if LW_STATS and group[-1] == self.num_layers - 1:
                    # Before the last completion, so the report can see it.
                    info = dict(ready_ms=ready_ms, grp_ms=stats[0][0], bytes=dict(stats[0][1]))
                    if len(stats) > 1:
                        info.update(grp2_ms=stats[1][0], bytes2=dict(stats[1][1]))
                    self.layer_done_counter.load_info.setdefault(counter_index, {}).update(info)
                # Only now is every layer in the group readable, so they are
                # released together. A group wider than 1 trades overlap
                # granularity for fewer times each object is named on the wire.
                for logical_layer in group:
                    self.layer_done_counter.complete(counter_index, logical_layer)

            def run_second() -> None:
                stream = streams[1]
                if self._stage_arena is not None:
                    # Here, not before the first stream: waiting on staging must not hold back layer 0.
                    stream, taken = self._take_staged(stream, rids)
                    staged.extend(taken)
                self._run_stream(stream, stats[1], finish, staged)

            second = self._split_executor.submit(run_second) if len(streams) > 1 else None
            try:
                self._run_stream(streams[0], stats[0], finish)
            finally:
                if second is not None:
                    second.result()
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("UMBP layer-wise load batch failed")
        finally:
            if self._stage_arena is not None:
                self._release_stage(staged, rids)

    def _load_streams(self, plans: list[_PoolRangePlan]):
        """(storage, plans, layer groups, part, key suffix) per parallel stream."""
        if self.split_mode == "layer":
            return [
                (self.storage, plans, self._layer_groups(0, self.split_layer), "front", ""),
                (self.storage2, plans, self._layer_groups(self.split_layer, self.num_layers), "back", "|b"),
            ]
        if self.split_mode == "page":
            first, second = self._route_plans(plans)
            return [
                (self.storage, first, self._layer_groups(), None, ""),
                (self.storage2, second, self._layer_groups(), None, ""),
            ]
        return [(self.storage, plans, self._layer_groups(), None, "")]

    def _route_plans(self, plans: list[_PoolRangePlan]):
        """Page split: the plans' pages on the first server, then on the second."""
        first, second = [], []
        for plan in plans:
            n = plan.entries_per_page
            routes = plan.routes or [False] * len(plan.locations)
            for target, routed in ((first, False), (second, True)):
                pages = [i for i, r in enumerate(routes) if r == routed]
                if pages:
                    target.append(
                        _PoolRangePlan(
                            plan.name,
                            [plan.keys[i * n + j] for i in pages for j in range(n)],
                            [plan.locations[i] for i in pages],
                            n,
                        )
                    )
        return first, second

    def _run_stream(self, stream, stat, finish: Callable[[list[int]], None], staged=()) -> None:
        storage, plans, groups, part, suffix = stream
        grp_ms, nbytes = stat
        by_layer: dict[int, list[_PoolRangePlan]] = defaultdict(list)
        for plan in plans:
            for logical_layer in self.pool_layers[plan.name]:
                by_layer[logical_layer].append(plan)

        copy_ctx, scratch = contextlib.nullcontext, None
        if staged:
            device = next(iter(self.pools.values())).components[0][0].device
            if device.type != "cpu":
                if self._stage_stream is None:
                    self._stage_stream = device_module.Stream(device=device)

                @contextlib.contextmanager
                def copy_ctx():
                    with device_module.device(device), device_module.stream(self._stage_stream):
                        yield

            with copy_ctx():
                src_rows, dst_rows, spans = [], [], []
                for run, pools in staged:
                    spans.append((run, [(name, len(src_rows), len(src)) for name, src, _ in pools]))
                    for _, src, dst in pools:
                        src_rows += src
                        dst_rows += dst
                rows = tuple(
                    torch.tensor(r, dtype=torch.int64).pin_memory().to(device, non_blocking=True)
                    if device.type != "cpu" else torch.tensor(r, dtype=torch.int64)
                    for r in (src_rows, dst_rows)
                )
                staged = spans
                size = max(STAGE_SCRATCH, max(run.size for run, _ in staged))
                scratch = torch.empty(size, dtype=torch.uint8, device=device)

        calls = [self._group_calls(by_layer, group, part, suffix) for group in groups]
        if LW_STATS:
            for call in (c for group_calls in calls for c in group_calls):
                nbytes[call[0].name] += sum(sum(s) for s in call[3])
        pending = None
        # The stream that reads SSD: the second in a split, else the only one.
        reads = self._load_reads if self.storage2 is None or storage is self.storage2 else None
        if reads is not None and any(calls):
            # Queue every group's reads now, group 0 first, so the drives never idle between groups.
            pending = [
                [reads.submit(self._get_ranges, storage, group, call) for call in group_calls]
                for group, group_calls in zip(groups, calls)
            ]
        try:
            for i, group in enumerate(groups):
                t_group = time.perf_counter()
                if staged:
                    with copy_ctx():
                        copied = self._copy_staged_group(staged, group, scratch, rows)
                    if LW_STATS:
                        nbytes["STAGED"] += copied
                if pending is None:
                    for call in calls[i]:
                        self._get_ranges(storage, group, call)
                else:
                    for future in pending[i]:
                        future.result()
                if staged and self._stage_stream is not None:
                    self._stage_stream.synchronize()
                grp_ms.append(round((time.perf_counter() - t_group) * 1e3, 1))
                finish(group)
        finally:
            if pending is not None:
                # A failed batch must not leave reads landing in KV pages it no longer owns.
                wait([future for group_futures in pending for future in group_futures])

    def _group_calls(self, by_layer, group: list[int], part, suffix: str) -> list[tuple]:
        """One layer group's direct reads: (pool, keys, ptrs, sizes, offsets) per call."""
        calls = []
        for plan in self._plans_covering(by_layer, group):
            meta = self._layer_group_ranges(plan, group, part)
            if meta is None:
                continue
            ptrs, sizes, offsets = meta
            step = self._entries_per_call(sizes)
            keys = [key + suffix for key in plan.keys] if suffix else plan.keys
            for start in range(0, len(keys), step):
                end = start + step
                calls.append((plan.name, keys[start:end], ptrs[start:end], sizes[start:end], offsets[start:end]))
        return calls

    @staticmethod
    def _get_ranges(storage, group: list[int], call: tuple) -> None:
        name, keys, ptrs, sizes, offsets = call
        results = list(storage.client.batch_get_ranges_into_ptr(keys, ptrs, sizes, offsets))
        if len(results) != len(keys) or not all(results):
            where = f"layer={group[0]}" if len(group) == 1 else f"layers={group[0]}..{group[-1]}"
            raise RuntimeError(
                f"UMBP get failed for pool={name}, {where}: "
                f"success={sum(bool(value) for value in results)}/{len(keys)}."
            )

    def _layer_groups(self, start: int = 0, stop: int | None = None) -> list[list[int]]:
        stop = self.num_layers if stop is None else stop
        return [
            list(range(first, min(first + self.layer_group, stop)))
            for first in range(start, stop, self.layer_group)
        ]

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self._freeze_gc_once()
        ready_event = device_module.Event()
        ready_event.record()
        self._offload_queue.put((expanded, ready_event))
        return True

    def _take_offload_batch(self) -> tuple[list[_OffloadTask], bool]:
        """Block for one task, then take whatever else is already queued.

        Returns the tasks and whether the stop sentinel came with them; each
        item taken here needs one ``task_done()`` from the caller. Taking only
        what is already queued keeps a task from waiting on an unsubmitted peer.
        """
        first = self._offload_queue.get()
        if first is None:
            return [], True
        tasks = [first]
        pages = _offload_task_pages(first[0])
        while pages < self._offload_coalesce_pages:
            try:
                task = self._offload_queue.get_nowait()
            except Empty:
                break
            if task is None:
                return tasks, True
            tasks.append(task)
            pages += _offload_task_pages(task[0])
        return tasks, False

    def _offload_thread_func(self) -> None:
        while True:
            tasks, stopping = self._take_offload_batch()
            taken = len(tasks) + int(stopping)
            try:
                if tasks:
                    self._offload_batch(tasks)
            finally:
                for _ in range(taken):
                    self._offload_queue.task_done()
            if stopping:
                return

    def _offload_batch(self, tasks: list[_OffloadTask]) -> None:
        success = False
        try:
            success = self._run_offload(tasks)
        except BaseException:
            logger.exception("UMBP offload failed")
            success = False
        finally:
            # One result per task in submission order: the tree pairs them
            # positionally. A batch resolves as a unit because the first failed
            # pool stops the rest, leaving every task in it incomplete.
            for _ in tasks:
                self._offload_results.put(success)

    def _run_offload(self, tasks: list[_OffloadTask]) -> bool:
        for _, ready_event in tasks:
            ready_event.synchronize()
        next_slot = 0

        def materialize_indices(indices: torch.Tensor) -> torch.Tensor:
            nonlocal next_slot
            slot = next_slot
            next_slot += 1
            return self._materialize_offload_indices(indices, slot)

        plans = self._build_load_plans(
            [expanded for expanded, _ in tasks],
            materialize_indices=materialize_indices,
        )
        # Over plans, not transfers: a plan already carries every task's keys
        # for its pool, so walking transfers would put that pool once per task.
        for storage, plan, layers, part, suffix in self._offload_puts(plans):
            if not self._put_plan(storage, plan, layers, part, suffix):
                return False

        self._stats["offload"] += len(tasks)
        self._stats["offload_batches"] += 1
        return True

    def _offload_puts(self, plans: list[_PoolRangePlan]):
        """(storage, plan, layers, part, key suffix) per put."""
        if self.split_mode == "layer":
            return [
                (storage, plan, layers, part, suffix)
                for plan in plans
                for storage, part, suffix in ((self.storage, "front", ""), (self.storage2, "back", "|b"))
                for layers in [self._part_layers(plan.name, part)]
                if layers
            ]
        if self.split_mode == "page":
            first, second = self._route_plans(plans)
            return [(self.storage, p, self.pool_layers[p.name], None, "") for p in first] + [
                (self.storage2, p, self.pool_layers[p.name], None, "") for p in second
            ]
        return [(self.storage, p, self.pool_layers[p.name], None, "") for p in plans]

    def _put_plan(self, storage, plan: _PoolRangePlan, layers, part, suffix) -> bool:
        entry = self.pools[plan.name]
        # From the pool layout, never from the ranges below: see
        # _object_sizes_per_page.
        per_page = self._part_sizes[(plan.name, part)] if part else _object_sizes_per_page(entry)
        if len(per_page) != plan.entries_per_page:
            raise ValueError(
                f"UMBP pool {plan.name} declares {len(per_page)} object "
                f"sizes per page but yields {plan.entries_per_page} objects."
            )
        object_sizes = [
            per_page[index % plan.entries_per_page] for index in range(len(plan.keys))
        ]
        meta = self._layer_group_ranges(plan, layers, part)
        if meta is None:
            raise ValueError(f"UMBP pool {plan.name} covers none of layers {layers}.")
        ptrs, sizes, offsets = meta
        keys = [key + suffix for key in plan.keys] if suffix else plan.keys

        # An object's ranges must tile it exactly, so a chunk boundary may
        # fall between objects but never inside one.
        step = self._entries_per_call(sizes)
        for start in range(0, len(keys), step):
            end = start + step
            chunk_keys = keys[start:end]
            results = list(
                storage.client.batch_put_ranges_from_ptr(
                    chunk_keys,
                    object_sizes[start:end],
                    ptrs[start:end],
                    sizes[start:end],
                    offsets[start:end],
                )
            )
            if len(results) != len(chunk_keys) or not all(results):
                logger.warning(
                    "UMBP offload failed: pool=%s part=%s object_range=[%d,%d) "
                    "success=%d/%d returned=%d",
                    plan.name,
                    part,
                    start,
                    min(end, len(keys)),
                    sum(bool(value) for value in results),
                    len(chunk_keys),
                    len(results),
                )
                return False
        return True

    def _freeze_gc_once(self) -> None:
        if self._gc_frozen:
            return
        freeze_gc("UMBP direct linker")
        self._gc_frozen = True

    def num_completed_offloads(self) -> int:
        # The tree agrees on the drain count across ranks before calling pop.
        return self._offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self._offload_results.get_nowait()

    def reset(self) -> None:
        self._pending.clear()
        self._load_queue.join()
        self._offload_queue.join()
        while True:
            try:
                self._offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self._completed_loads.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        for thread, queue in (
            (self._offload_thread, self._offload_queue),
            (self._load_thread, self._load_queue),
        ):
            if thread.is_alive():
                queue.put(None)
                thread.join()
        if self._standalone_process_mode and self._registered:
            # StandaloneProcess deregistration is client-wide; one call tears
            # down every registered region. Keep the GPU tensors alive until
            # the synchronous RPC has completed successfully.
            self.storage.client.deregister_memory(self._registered[0][0])
        if self._stage_arena is not None:
            self._stage_exec.shutdown(wait=True)
            self._stage_reads.shutdown(wait=True)
            logger.info("UMBPSTAGE final %s", dict(self._stage_stats))
        if self.storage2 is not None:
            self._split_executor.shutdown(wait=True)
            self._exists_executor.shutdown(wait=True)
            if self._standalone_process_mode and self._registered:
                self.storage2.client.deregister_memory(self._registered[0][0])
            self.storage2.close()
        if getattr(self, "_stage_handle", None) is not None:
            if self._stage_pinned:
                torch.cuda.cudart().cudaHostUnregister(self._stage_ptr)
            self._stage_arena = None
            self._stage_alloc.free(self._stage_handle)
            self._stage_handle = None
        logger.info("UMBP direct linker stats: %s", self._stats)
        self.storage.close()
        self._closed = True
