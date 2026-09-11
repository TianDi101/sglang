from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import Future
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
from sglang.srt.mem_cache.utils import hash_str_to_int64
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

# Budget by ranges because pool layouts attach different counts per object;
# 8192 stays below gRPC's default message limit.
RANGES_PER_CALL = int(os.getenv("UMBP_RANGES_PER_CALL", "8192"))

# Split replicated KV reads across the attention TP group: every rank reads a
# disjoint window of the pages and the group all-gathers the rest. Opt-in.
SPLIT_LOAD = os.getenv("UMBP_LOAD_SPLIT", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "off",
}
# Below this many agreed pages the exchange costs more than the reads it saves.
SPLIT_MIN_PAGES = int(os.getenv("UMBP_LOAD_SPLIT_MIN_PAGES", "16"))


# Stable across ranks because the enum is a closed set defined in one place.
_POOL_IDS = {name: index for index, name in enumerate(PoolName)}


def _is_verified_split_kvcache(kvcache: Any) -> bool:
    """Whether this KV cache is known to hold identical KV on every TP rank."""
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    return isinstance(kvcache, (DSATokenToKVPool, DeepSeekV4TokenToKVPool))


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

    def __init__(self, num_layers: int, on_group_ready=None):
        self.num_layers = num_layers
        self._producer_index = -1
        self.consumer_index = -1
        self._futures: dict[int, list[Future]] = {}
        # Called on the forward thread once a layer's reads have landed, which
        # is where the split exchange belongs: every rank reaches it in layer
        # order, so the collective is issued in the same order on all of them.
        self._on_group_ready = on_group_ready

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
            futures[threshold].result()
            if self._on_group_ready is not None:
                self._on_group_ready(index, threshold)
        except BaseException as error:
            raise RuntimeError("UMBP layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self._futures.pop(index, None)

    def reset(self) -> None:
        self._producer_index = -1
        self.consumer_index = -1
        self._futures.clear()


@dataclass
class _PoolRangePlan:
    """Object keys and locations for one pool load.

    A split plan carries only this rank's window in ``keys``/``locations``;
    ``all_locations`` keeps this rank's rows for every agreed page so the
    exchange can scatter the other ranks' windows into them.
    """

    name: PoolName
    keys: list[str]
    locations: list[int]
    entries_per_page: int
    all_locations: list[int] | None = None
    windows: tuple[tuple[int, int], ...] = ()

    @property
    def split(self) -> bool:
        return bool(self.windows)


def _split_windows(num_pages: int, tp_size: int) -> tuple[tuple[int, int], ...]:
    """Return contiguous equal-width page windows, one per rank."""
    width = -(-num_pages // tp_size)
    return tuple(
        (min(rank * width, num_pages), min((rank + 1) * width, num_pages))
        for rank in range(tp_size)
    )


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
        self._tp_size = server_args.tp_size
        self._split_load = bool(SPLIT_LOAD and self._tp_size > 1)

        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        if self._split_load:
            self._validate_split_scope(kvcache, server_args)
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
        self._tp_rank = tp_rank
        self._split_pg = None
        self._split_sync_group = None
        self._split_world = 0
        self._split_state: dict[int, dict] = {}
        self._split_send: torch.Tensor | None = None
        self._split_recv: torch.Tensor | None = None
        self._split_status: torch.Tensor | None = None
        if self._split_load:
            if not distributed:
                raise ValueError("UMBP_LOAD_SPLIT needs an initialized process group.")
            self._split_process_group()
            self._split_sync_group = self._resolve_split_sync_group(params)
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
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
        if self._split_load:
            self._split_status = torch.empty(
                1,
                dtype=torch.int64,
                device=next(iter(self.pools.values())).components[0][0].device,
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
            rank_suffix = f"tp{tp_rank}_cp{params.attn_cp_rank}_pp{params.pp_rank}"
            self.storage.mla_suffix = rank_suffix
            self.storage.mha_suffix = rank_suffix
            self._register_buffers()
            # Report the mode rather than asserting it in the text: the line is
            # what every acceptance check greps to prove the linker attached at
            # all, and it used to say "standalone_process" unconditionally, so
            # it could not have shown an embedded run for what it was.
            logger.info(
                "UMBPDirectLinker topology=%s+%s ranged_io=yes",
                mode.name,
                self.backend_mode.name if self.backend_mode is not None else None,
            )
        except BaseException:
            self.storage.close()
            raise

        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            on_group_ready=self._exchange_ready_groups if self._split_load else None,
        )
        if self._split_load:
            logger.info(
                "UMBP split load enabled: ranks=%d, min_pages=%d, layer_group=%d",
                self._split_world,
                SPLIT_MIN_PAGES,
                self.layer_group,
            )
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

    def _object_keys(self, transfer: PoolTransfer) -> list[str]:
        keys, _ = self._object_keys_for_pages(list(transfer.keys or []), transfer)
        return keys

    def _page_exists(self, page_keys: list[str], transfer: PoolTransfer) -> list[bool]:
        entry = self.pools[transfer.name]
        objects_per_page = 1 if entry.packed else len(entry.components)
        max_objects = CHUNK_PAGES * self.num_layers
        pages_per_call = max(1, max_objects // objects_per_page)

        page_exists = []
        for start in range(0, len(page_keys), pages_per_call):
            chunk_pages = page_keys[start : start + pages_per_call]
            object_keys, _ = self._object_keys_for_pages(chunk_pages, transfer)
            exists = list(self.storage.client.batch_exists(object_keys))
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
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys or [])
        if not page_keys:
            return []

        valid_pages = list(range(1, len(page_keys) + 1))
        for transfer in expanded:
            # Probe only as far as the surviving boundary: no hit policy reads
            # past its own end offset, so pages beyond the longest candidate
            # cannot change the answer. This runs synchronously inside the
            # scheduler's prefill batch build, and DP-attention ranks are
            # lockstep, so every extra key is stall charged to all of them.
            page_exists = self._page_exists(page_keys[: valid_pages[-1]], transfer)
            valid_pages = self._apply_hit_policy(valid_pages, page_exists, transfer)
            if not valid_pages:
                break

        self._stats["lookup"] += 1
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
        pending = self._pending
        common_pages = None
        if self._split_load:
            # Join the agreement before the emptiness check: a rank that has
            # nothing to load still has to reach the collective, or the ranks
            # that do would wait for a peer that never arrives. Contributing an
            # empty page set simply empties the intersection, which turns the
            # split off for everyone -- the safe direction.
            common_pages = self._agree_common_pages(
                self._page_hashes_by_pool(self._group_transfers(list(pending.values())))
            )
        if not pending:
            return -1
        self._freeze_gc_once()
        rids = list(pending)
        plans = self._build_load_plans(
            list(pending.values()), common_pages=common_pages
        )
        ready_event = device_module.Event()
        ready_event.record()
        counter_index = self.layer_done_counter.update_producer()
        if any(plan.split for plan in plans):
            self._split_state[counter_index] = {
                "plans": plans,
                "groups": self._layer_groups(),
                "exchanged": -1,
                "failure": None,
                "rows": {},
            }
        self._load_queue.put((counter_index, rids, plans, ready_event))
        self._pending = {}
        self._stats["load"] += len(pending)
        return counter_index

    # ---- split load: agree the page set before cutting the windows ----

    @staticmethod
    def _group_transfers(
        request_transfers: list[list[PoolTransfer]],
    ) -> dict[PoolName, list[PoolTransfer]]:
        grouped: dict[PoolName, list[PoolTransfer]] = {}
        for transfers in request_transfers:
            for transfer in transfers:
                grouped.setdefault(transfer.name, []).append(transfer)
        return grouped

    @staticmethod
    def _page_hashes_by_pool(
        grouped: dict[PoolName, list[PoolTransfer]],
    ) -> dict[PoolName, list[int]]:
        """Rank-agnostic page identities, in the order the plan will name them.

        Object keys carry this rank's own ``tp{n}_cp{n}_pp{n}`` suffix, so they
        cannot be compared across ranks; the page hashes they are derived from
        can. Must walk ``grouped`` exactly as ``_build_load_plans`` does so
        position *i* here is page *i* there.
        """
        page_hashes: dict[PoolName, list[int]] = {}
        for name, transfers in grouped.items():
            if name not in _POOL_IDS:
                continue
            hashes: list[int] = []
            for transfer in transfers:
                hashes.extend(
                    hash_str_to_int64(page_key) for page_key in transfer.keys or ()
                )
            page_hashes[name] = hashes
        return page_hashes

    def _agree_common_pages(
        self, page_hashes: dict[PoolName, list[int]]
    ) -> dict[PoolName, list[int]]:
        """Return, per pool, the pages every rank in the split group has.

        The load's page set is settled after the tree insert has deduplicated
        whatever each rank already held, so it is not safe to assume the ranks
        agree on it. Gather the sets, intersect them, and split only the pages
        common to all -- every rank derives the answer from the same gathered
        matrix, so they cut identical windows or none at all.
        """
        group = self._split_sync_group
        world = self._split_world
        local: list[int] = [len(page_hashes)]
        for name in sorted(page_hashes, key=_POOL_IDS.__getitem__):
            hashes = page_hashes[name]
            local.append(_POOL_IDS[name])
            local.append(len(hashes))
            local.extend(hashes)

        # Ranks carry different page counts, so agree on a width before the
        # gather; the vectors are self-describing and ignore the padding.
        width = torch.tensor([len(local)], dtype=torch.int64)
        torch.distributed.all_reduce(
            width, op=torch.distributed.ReduceOp.MAX, group=group
        )
        padded = torch.zeros(int(width.item()), dtype=torch.int64)
        padded[: len(local)] = torch.tensor(local, dtype=torch.int64)
        gathered = [torch.empty_like(padded) for _ in range(world)]
        torch.distributed.all_gather(gathered, padded, group=group)

        common = self._intersect_page_vectors(
            [self._parse_page_vector(vector.tolist()) for vector in gathered]
        )
        return {
            name: common[_POOL_IDS[name]]
            for name in page_hashes
            if _POOL_IDS[name] in common
        }

    @staticmethod
    def _parse_page_vector(values: list[int]) -> dict[int, list[int]] | None:
        """Decode one rank's ``[pools, (pool, count, pages...)...]`` vector."""
        if not values or values[0] < 0:
            return None
        pools: dict[int, list[int]] = {}
        position = 1
        for _ in range(values[0]):
            if position + 2 > len(values):
                return None
            pool_id, count = values[position], values[position + 1]
            position += 2
            if count < 0 or position + count > len(values) or pool_id in pools:
                return None
            pools[pool_id] = values[position : position + count]
            position += count
        return pools

    @staticmethod
    def _intersect_page_vectors(
        per_rank: list[dict[int, list[int]] | None],
    ) -> dict[int, list[int]]:
        """Pages present on every rank, ordered as the group's rank 0 has them.

        A pool whose pages are not unique on some rank is dropped: a page named
        twice has no single row to scatter into. Every rank sees the same
        gathered matrix, so every rank drops the same pools.
        """
        if not per_rank or any(pools is None for pools in per_rank):
            return {}
        common: dict[int, list[int]] = {}
        for pool_id, pages in per_rank[0].items():
            if len(set(pages)) != len(pages):
                continue
            others = []
            for pools in per_rank[1:]:
                theirs = pools.get(pool_id)
                if theirs is None or len(set(theirs)) != len(theirs):
                    others = None
                    break
                others.append(set(theirs))
            if others is None:
                continue
            shared = [page for page in pages if all(page in other for other in others)]
            if shared:
                common[pool_id] = shared
        return common

    def _build_load_plans(
        self,
        request_transfers: list[list[PoolTransfer]],
        *,
        common_pages: dict[PoolName, list[int]] | None = None,
        materialize_indices: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> list[_PoolRangePlan]:
        """Build a batch plan shared by load and offload.

        ``common_pages`` splits the pages every rank agreed on; offload passes
        none, so it always gets whole plans.
        """
        grouped = self._group_transfers(request_transfers)

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
            # Only the split path needs page identities; offload skips the hash.
            page_hashes: list[int] = []
            for transfer in transfers:
                page_keys = list(transfer.keys or [])
                if common_pages is not None:
                    page_hashes.extend(
                        hash_str_to_int64(page_key) for page_key in page_keys
                    )
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
            if common_pages is not None and len(page_hashes) != len(locations):
                raise ValueError(
                    f"UMBP pool {name} page-hash mismatch: "
                    f"hashes={len(page_hashes)} rows={len(locations)}."
                )
            agreed = (common_pages or {}).get(name) or []
            if self._split_world > 1 and len(agreed) >= SPLIT_MIN_PAGES:
                plans.extend(
                    self._split_pool_plans(
                        name=name,
                        keys=keys,
                        locations=locations,
                        page_hashes=page_hashes,
                        entries_per_page=entries_per_page,
                        agreed=agreed,
                    )
                )
            else:
                plans.append(_PoolRangePlan(name, keys, locations, entries_per_page))

        # A split plan can hold no keys at all: with fewer agreed pages than
        # ranks the trailing windows are empty, and that rank contributes only
        # the exchange.
        if not plans or not (plans[0].keys or plans[0].split):
            raise ValueError("Layer-wise UMBP load has no object keys.")
        return plans

    def _split_pool_plans(
        self,
        *,
        name: PoolName,
        keys: list[str],
        locations: list[int],
        page_hashes: list[int],
        entries_per_page: int,
        agreed: list[int],
    ) -> list[_PoolRangePlan]:
        """Cut the agreed pages into per-rank windows, keeping the rest whole.

        The agreed pages are ordered by the group's rank 0, so window *r* names
        the same pages on every rank; the rows it writes into stay this rank's
        own. Pages this rank holds that the group did not agree on are loaded
        in full, exactly as they would be without the split.
        """
        index_of: dict[int, int] = {}
        for index, page in enumerate(page_hashes):
            index_of.setdefault(page, index)
        selected = [index_of[page] for page in agreed if page in index_of]
        # The intersection only keeps pages this rank reported, with duplicates
        # already excluded, so every agreed page must resolve here.
        assert len(selected) == len(agreed), (
            f"UMBP pool {name} agreed on {len(agreed)} pages but resolved "
            f"{len(selected)} of them locally."
        )

        def objects(indices: list[int]) -> list[str]:
            return [
                key
                for index in indices
                for key in keys[
                    index * entries_per_page : (index + 1) * entries_per_page
                ]
            ]

        split_locations = [locations[index] for index in selected]
        windows = _split_windows(len(split_locations), self._split_world)
        start, end = windows[self._tp_rank]
        plans = [
            _PoolRangePlan(
                name,
                objects(selected[start:end]),
                split_locations[start:end],
                entries_per_page,
                all_locations=split_locations,
                windows=windows,
            )
        ]
        taken = set(selected)
        rest = [index for index in range(len(locations)) if index not in taken]
        if rest:
            plans.append(
                _PoolRangePlan(
                    name,
                    objects(rest),
                    [locations[index] for index in rest],
                    entries_per_page,
                )
            )
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
                    self._run_layer_wise_batch(counter_index, plans, ready_event)
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

    def _range_items(self, plan: _PoolRangePlan, layers: list[int]):
        """(base_ptr, row_stride, size, offset) per emitted range, in wire order.

        Grouped by layer, and within a layer by component. Every object emits
        this same tuple sequence; the only thing that varies across objects is
        the pointer, by ``row * row_stride``. That invariant is what the
        vectorized builder rests on.
        """
        entry = self.pools[plan.name]
        items: list[list[tuple[int, int, int, int]]] = []
        for logical_layer in layers:
            buffer_index = entry.layer_mapping.get(logical_layer)
            if buffer_index is None:
                continue
            items.append(
                [
                    (*component[buffer_index], offsets[buffer_index])
                    for component, offsets in zip(
                        entry.buffer_meta, entry._component_offsets
                    )
                ]
            )
        return items

    def _layer_group_ranges(self, plan: _PoolRangePlan, layers: list[int]):
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
        items = self._range_items(plan, layers)
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
        self, counter_index: int, plans: list[_PoolRangePlan], ready_event: object
    ) -> None:
        released = 0
        try:
            ready_event.synchronize()
            by_layer: dict[int, list[_PoolRangePlan]] = defaultdict(list)
            for plan in plans:
                for logical_layer in self.pool_layers[plan.name]:
                    by_layer[logical_layer].append(plan)

            for group in self._layer_groups():
                for plan in self._plans_covering(by_layer, group):
                    meta = self._layer_group_ranges(plan, group)
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    step = self._entries_per_call(sizes)
                    for start in range(0, len(plan.keys), step):
                        end = start + step
                        chunk_keys = plan.keys[start:end]
                        results = list(
                            self.storage.client.batch_get_ranges_into_ptr(
                                chunk_keys,
                                ptrs[start:end],
                                sizes[start:end],
                                offsets[start:end],
                            )
                        )
                        if len(results) != len(chunk_keys) or not all(results):
                            where = (
                                f"layer={group[0]}"
                                if len(group) == 1
                                else f"layers={group[0]}..{group[-1]}"
                            )
                            raise RuntimeError(
                                f"UMBP get failed for pool={plan.name}, {where}: "
                                f"success={sum(bool(value) for value in results)}/"
                                f"{len(chunk_keys)}."
                            )
                # Only now is every layer in the group readable, so they are
                # released together. A group wider than 1 trades overlap
                # granularity for fewer times each object is named on the wire.
                for logical_layer in group:
                    self.layer_done_counter.complete(counter_index, logical_layer)
                released = group[-1] + 1
        except BaseException as error:
            state = self._split_state.get(counter_index)
            if state is None:
                self.layer_done_counter.fail(counter_index, error)
            else:
                # The forward pass has to reach the next group exchange, where a
                # reduction tells every rank to give up together. Failing the
                # futures here instead would stop this rank at the wait while the
                # others went on to a collective it never joins.
                state["failure"] = error
                for logical_layer in range(released, self.num_layers):
                    self.layer_done_counter.complete(counter_index, logical_layer)
            logger.exception("UMBP layer-wise load batch failed")

    def _layer_groups(self) -> list[list[int]]:
        return [
            list(range(start, min(start + self.layer_group, self.num_layers)))
            for start in range(0, self.num_layers, self.layer_group)
        ]

    # ---- split load: exchange the shares the other ranks read ----

    @staticmethod
    def _validate_split_scope(kvcache, server_args) -> None:
        if not _is_verified_split_kvcache(kvcache):
            raise ValueError(
                "UMBP_LOAD_SPLIT is supported only for verified DSA and "
                "DeepSeek-V4 KV caches."
            )

        unsupported = []
        if server_args.attn_cp_size > 1:
            unsupported.append("attention CP")
        if server_args.enable_dp_attention:
            unsupported.append("DP attention")
        if server_args.pp_size > 1:
            unsupported.append("pipeline parallelism")
        if unsupported:
            raise ValueError(
                "UMBP_LOAD_SPLIT does not support " + ", ".join(unsupported) + "."
            )

    def _split_process_group(self):
        """Return the device communicator used by the split exchange."""
        if self._split_pg is None:
            from sglang.srt.distributed.parallel_state import get_attn_tp_group

            group = get_attn_tp_group().device_group
            rank = torch.distributed.get_rank(group=group)
            world = torch.distributed.get_world_size(group=group)
            if rank != self._tp_rank or world != self._tp_size:
                raise RuntimeError(
                    "UMBP split load requires the device attention-TP group to "
                    "match the connector TP keyspace: "
                    f"group=({rank}/{world}), connector=({self._tp_rank}/{self._tp_size})."
                )
            self._split_pg = group
            self._split_world = world
        return self._split_pg

    def _resolve_split_sync_group(self, params: CacheInitParams):
        """Return the CPU cache group the page-set agreement reduces over.

        The agreement runs on the scheduler thread while the exchange runs on
        the forward thread. Keeping it on the cache group -- the same host-side
        group every other lockstep reduction in this cache uses -- keeps the two
        off one communicator, where their order across ranks would depend on how
        the threads interleaved.
        """
        for group in (params.attn_tp_cache_group, params.tp_cache_group):
            if (
                group is not None
                and torch.distributed.get_world_size(group=group) == self._split_world
                and torch.distributed.get_rank(group=group) == self._tp_rank
            ):
                return group
        raise RuntimeError(
            "UMBP split load found no cache group matching the split group "
            f"({self._tp_rank}/{self._split_world})."
        )

    def _split_pool_geometry(self, plan: _PoolRangePlan, logical_layer: int):
        """Return tensors and row geometry for one pool layer."""
        entry = self.pools[plan.name]
        buffer_index = entry.layer_mapping.get(logical_layer)
        if buffer_index is None:
            return None
        geometry = []
        for component, meta in zip(entry.components, entry.buffer_meta):
            _, row_stride, size = meta[buffer_index]
            if row_stride <= 0 or size % row_stride:
                raise ValueError(
                    f"UMBP pool {plan.name} layer {logical_layer} has a row size "
                    f"{size} that is not a multiple of its stride {row_stride}."
                )
            geometry.append((component[buffer_index], row_stride, size // row_stride))
        spans = {span for _, _, span in geometry}
        if len(spans) != 1:
            raise ValueError(
                f"UMBP pool {plan.name} components disagree on rows per page: "
                f"{sorted(spans)}."
            )
        return geometry

    def _split_layout(self, plans: list[_PoolRangePlan], group: list[int]):
        layout = []
        offset = 0
        for plan in plans:
            width = max(end - start for start, end in plan.windows)
            for logical_layer in group:
                geometry = self._split_pool_geometry(plan, logical_layer)
                if geometry is None:
                    continue
                for tensor, row_stride, row_span in geometry:
                    span = width * row_span * row_stride
                    layout.append((plan, tensor, row_span, offset))
                    offset += -(-span // 256) * 256
        return layout, offset

    def _split_rows(
        self, plan: _PoolRangePlan, rank: int, row_span: int, device, cache
    ):
        key = (id(plan), rank)
        rows = cache.get(key)
        if rows is None:
            start, end = plan.windows[rank]
            assert plan.all_locations is not None
            pages = torch.tensor(
                plan.all_locations[start:end], dtype=torch.int64, device=device
            )
            if row_span > 1:
                pages = (
                    pages[:, None]
                    + torch.arange(row_span, dtype=torch.int64, device=device)
                ).reshape(-1)
            rows = pages
            cache[key] = rows
        return rows

    @staticmethod
    def _split_view(buffer: torch.Tensor, offset: int, rows: int, like: torch.Tensor):
        nbytes = rows * like.stride(0) * like.element_size()
        return (
            buffer[offset : offset + nbytes]
            .view(like.dtype)
            .view(rows, *like.shape[1:])
        )

    def _exchange_ready_groups(self, counter_index: int, logical_layer: int) -> None:
        state = self._split_state.get(counter_index)
        if state is None:
            return
        target = logical_layer // self.layer_group
        try:
            while state["exchanged"] < target:
                state["exchanged"] += 1
                self._exchange_group(state, state["exchanged"])
        finally:
            if target >= len(state["groups"]) - 1:
                self._split_state.pop(counter_index, None)

    def _prepare_split_group(self, state, plans, group, device):
        layout, slot_bytes = self._split_layout(plans, group)
        if not layout:
            return layout, slot_bytes

        self._split_reserve(slot_bytes, device)
        for plan, _, row_span, _ in layout:
            for rank in range(self._split_world):
                self._split_rows(plan, rank, row_span, device, state["rows"])

        for plan, tensor, row_span, offset in layout:
            rows = state["rows"][(id(plan), self._tp_rank)]
            if rows.numel():
                torch.index_select(
                    tensor,
                    0,
                    rows,
                    out=self._split_view(
                        self._split_send, offset, rows.numel(), tensor
                    ),
                )
        return layout, slot_bytes

    def _exchange_group(self, state: dict, group_index: int) -> None:
        plans = [plan for plan in state["plans"] if plan.split]
        if not plans:
            return
        group = state["groups"][group_index]
        process_group = self._split_process_group()
        world = self._split_world
        device = self.pools[plans[0].name].components[0][0].device

        failure = state["failure"]
        layout: list = []
        slot_bytes = 0
        if failure is None:
            try:
                layout, slot_bytes = self._prepare_split_group(
                    state, plans, group, device
                )
            except BaseException as error:
                logger.exception("UMBP split load could not prepare its share")
                failure = error

        agreed = self._split_status
        assert agreed is not None
        agreed.fill_(0 if failure is not None else 1)
        torch.distributed.all_reduce(
            agreed, op=torch.distributed.ReduceOp.MIN, group=process_group
        )
        if not agreed.item():
            raise RuntimeError(
                "UMBP split load failed, or could not prepare its share, on at "
                "least one rank"
            ) from failure
        if not layout:
            return

        torch.distributed.all_gather_into_tensor(
            self._split_recv[: slot_bytes * world],
            self._split_send[:slot_bytes],
            group=process_group,
        )

        for rank in range(world):
            if rank == self._tp_rank:
                continue
            base = rank * slot_bytes
            for plan, tensor, row_span, offset in layout:
                rows = state["rows"][(id(plan), rank)]
                if not rows.numel():
                    continue
                tensor.index_copy_(
                    0,
                    rows,
                    self._split_view(
                        self._split_recv, base + offset, rows.numel(), tensor
                    ),
                )

    def _split_reserve(self, slot_bytes: int, device) -> None:
        needed = slot_bytes * self._split_world
        if self._split_recv is not None and self._split_recv.numel() >= needed:
            return
        send = torch.empty(slot_bytes, dtype=torch.uint8, device=device)
        recv = torch.empty(needed, dtype=torch.uint8, device=device)
        self._split_send, self._split_recv = send, recv
        logger.info(
            "UMBP split load staging: ranks=%d, %.1f MiB send + %.1f MiB recv per GPU",
            self._split_world,
            slot_bytes / (1 << 20),
            needed / (1 << 20),
        )

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
        for plan in plans:
            entry = self.pools[plan.name]
            # From the pool layout, never from the ranges below: see
            # _object_sizes_per_page.
            per_page = _object_sizes_per_page(entry)
            if len(per_page) != plan.entries_per_page:
                raise ValueError(
                    f"UMBP pool {plan.name} declares {len(per_page)} object "
                    f"sizes per page but yields {plan.entries_per_page} objects."
                )
            object_sizes = [
                per_page[index % plan.entries_per_page]
                for index in range(len(plan.keys))
            ]
            ptrs, sizes, offsets = self._all_layer_ranges(plan)

            # Preserve page order so the tier can collapse a layer into a strided copy.

            # An object's ranges must tile it exactly, so a chunk boundary may
            # fall between objects but never inside one.
            step = self._entries_per_call(sizes)
            for start in range(0, len(plan.keys), step):
                end = start + step
                chunk_keys = plan.keys[start:end]
                results = list(
                    self.storage.client.batch_put_ranges_from_ptr(
                        chunk_keys,
                        object_sizes[start:end],
                        ptrs[start:end],
                        sizes[start:end],
                        offsets[start:end],
                    )
                )
                if len(results) != len(chunk_keys) or not all(results):
                    logger.warning(
                        "UMBP offload failed: pool=%s object_range=[%d,%d) "
                        "success=%d/%d returned=%d",
                        plan.name,
                        start,
                        min(end, len(plan.keys)),
                        sum(bool(value) for value in results),
                        len(chunk_keys),
                        len(results),
                    )
                    return False

        self._stats["offload"] += len(tasks)
        self._stats["offload_batches"] += 1
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
        self._split_state.clear()
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
        logger.info("UMBP direct linker stats: %s", self._stats)
        self.storage.close()
        self._closed = True
