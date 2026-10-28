"""Capacity-aware replay that turns candidate prefixes into actual cache hits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import heapq
from typing import Any, Mapping, Sequence

from ssu_sim.cache.config import CacheReplayConfig
from ssu_sim.cache.radix import (
    CacheShardSnapshot,
    CheckpointAttachment,
    RadixLeafLRUCache,
)
from ssu_sim.models.attention import PrefixCachePolicy
from ssu_sim.models.llm_model import ModelProfile, PrefillComputeContext
from ssu_sim.models.execution import validate_execution_combination, require_evaluated_full_hit
from ssu_sim.models.state import (
    CheckpointGroup,
    LayerKind,
    StateComponent,
    StatePersistencePolicy,
    checkpoint_boundaries,
)
from ssu_sim.prefix.block_hasher import (
    CanonicalRequestPath,
    SharedPrefixBlockIdMemo,
    build_request_block_ids_with_shared_memo,
)
from ssu_sim.workload.request import Request


@dataclass(frozen=True)
class PrefixRecoveryFunnel:
    """Configured, object-shareable, resident, and recoverable boundaries."""

    seq_len: int
    configured_prefix_tokens: int
    shareable_prefix_tokens: int
    resident_prefix_tokens: int
    recovery_boundary_tokens: int
    shareability_source: str = "complete_shared_blocks"
    loss_causes: tuple[str, ...] = ()
    resident_cache_bytes: int = 0
    resident_token_bytes: int = 0
    resident_checkpoint_bytes: int = 0

    def __post_init__(self) -> None:
        values = (
            self.recovery_boundary_tokens,
            self.resident_prefix_tokens,
            self.shareable_prefix_tokens,
            self.configured_prefix_tokens,
            self.seq_len,
        )
        if any(value < 0 for value in values) or tuple(sorted(values)) != values:
            raise ValueError(
                "recovery funnel must satisfy 0 <= recovery <= resident <= "
                "shareable <= configured <= seq_len"
            )
        if min(
            self.resident_cache_bytes,
            self.resident_token_bytes,
            self.resident_checkpoint_bytes,
        ) < 0:
            raise ValueError("resident recovery byte counts must not be negative")

    @property
    def replay_tokens(self) -> int:
        return self.resident_prefix_tokens - self.recovery_boundary_tokens

    @property
    def suffix_tokens(self) -> int:
        return self.seq_len - self.resident_prefix_tokens


@dataclass(frozen=True)
class CacheReplaySummary:
    config: CacheReplayConfig
    total_requests: int
    measured_requests: int
    warmup_requests: int
    bootstrap_requests: int
    write_miss_blocks: bool
    input_tokens: int
    configured_prefix_tokens: int
    shareable_prefix_tokens: int
    resident_prefix_tokens: int
    cacheable_blocks: int
    candidate_blocks: int
    hit_blocks: int
    cacheable_bytes: int
    candidate_bytes: int
    hit_bytes: int
    compulsory_miss_requests: int
    capacity_miss_requests: int
    inserted_count: int
    inserted_bytes: int
    deduplicated_insert_count: int
    deduplicated_insert_bytes: int
    rejected_count: int
    rejected_bytes: int
    eviction_count: int
    eviction_bytes: int
    self_eviction_count: int
    self_eviction_bytes: int
    initial_snapshots: tuple[CacheShardSnapshot, ...]
    peak_used_bytes_by_shard: tuple[int, ...]
    peak_evictable_bytes_by_shard: tuple[int, ...]
    peak_protected_bytes_by_shard: tuple[int, ...]
    peak_total_used_bytes: int
    peak_total_evictable_bytes: int
    peak_total_protected_bytes: int
    peak_total_checkpoint_attachment_bytes: int
    final_snapshots: tuple[CacheShardSnapshot, ...]
    checkpoint_admitted_count: int = 0
    checkpoint_admitted_bytes: int = 0
    checkpoint_deduplicated_count: int = 0
    checkpoint_deduplicated_bytes: int = 0
    checkpoint_rejected_count: int = 0
    checkpoint_rejected_bytes: int = 0
    phase_counters: Mapping[str, _Counters] = field(default_factory=dict)
    lifecycle_snapshots: Mapping[str, tuple[CacheShardSnapshot, ...]] = field(
        default_factory=dict
    )

    @property
    def configured_prefix_ratio(self) -> float:
        return self.configured_prefix_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def shareable_prefix_ratio(self) -> float:
        return self.shareable_prefix_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def candidate_prefix_ratio(self) -> float:
        return self.shareable_prefix_ratio

    @property
    def actual_prefix_ratio(self) -> float:
        return self.resident_prefix_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def actual_block_hit_rate(self) -> float:
        return self.hit_blocks / self.cacheable_blocks if self.cacheable_blocks else 0.0

    @property
    def candidate_block_hit_rate(self) -> float:
        return self.candidate_blocks / self.cacheable_blocks if self.cacheable_blocks else 0.0

    @property
    def candidate_byte_hit_rate(self) -> float:
        return self.candidate_bytes / self.cacheable_bytes if self.cacheable_bytes else 0.0

    @property
    def actual_byte_hit_rate(self) -> float:
        return self.hit_bytes / self.cacheable_bytes if self.cacheable_bytes else 0.0

    @property
    def inserted_blocks(self) -> int:
        return self.inserted_count

    @property
    def deduplicated_insert_blocks(self) -> int:
        return self.deduplicated_insert_count

    @property
    def rejected_blocks(self) -> int:
        return self.rejected_count

    @property
    def evicted_blocks(self) -> int:
        return self.eviction_count

    @property
    def self_evicted_blocks(self) -> int:
        return self.self_eviction_count

    def to_dict(self) -> dict[str, object]:
        initial_used = sum(item.used_bytes for item in self.initial_snapshots)
        initial_protected = sum(item.protected_bytes for item in self.initial_snapshots)
        initial_evictable = sum(item.evictable_bytes for item in self.initial_snapshots)
        final_used = sum(item.used_bytes for item in self.final_snapshots)
        final_protected = sum(item.protected_bytes for item in self.final_snapshots)
        final_evictable = sum(item.evictable_bytes for item in self.final_snapshots)
        shards = []
        for index, final in enumerate(self.final_snapshots):
            payload = final.to_dict()
            payload.update(
                {
                    "initial_used_bytes": self.initial_snapshots[index].used_bytes,
                    "initial_evictable_bytes": self.initial_snapshots[index].evictable_bytes,
                    "initial_protected_bytes": self.initial_snapshots[index].protected_bytes,
                    "peak_used_bytes": self.peak_used_bytes_by_shard[index],
                    "peak_evictable_bytes": self.peak_evictable_bytes_by_shard[index],
                    "peak_protected_bytes": self.peak_protected_bytes_by_shard[index],
                }
            )
            shards.append(payload)
        return {
            "semantics": {
                "model": "object-identity block-radix replay with byte-weighted leaf-LRU",
                "hit": "longest resident object prefix, capped by the shareable prefix",
                "eviction": "oldest unlocked leaf first; ancestors become candidates recursively",
                "locking": "matched paths are protected until restore/prefill completion",
                "eviction_bytes": "logical serialized cache capacity released, not SSD delete or GC IO",
                "admission": "stored objects follow the model partial-tail policy",
            },
            "config": self.config.to_dict(),
            "requests": {
                "total": self.total_requests,
                "bootstrap": self.bootstrap_requests,
                "warmup": self.warmup_requests,
                "measured": self.measured_requests,
                "normal": self.warmup_requests + self.measured_requests,
            },
            "phases": {
                name: _counter_payload(counter)
                for name, counter in self.phase_counters.items()
            },
            "lifecycle": {
                **{
                    name: _snapshot_totals(snapshots)
                    for name, snapshots in self.lifecycle_snapshots.items()
                },
                "measured_peak": {
                    "used_bytes": self.peak_total_used_bytes,
                    "protected_bytes": self.peak_total_protected_bytes,
                    "evictable_bytes": self.peak_total_evictable_bytes,
                    "checkpoint_attachment_bytes": (
                        self.peak_total_checkpoint_attachment_bytes
                    ),
                },
            },
            "hit_rate": {
                "configured_prefix": self.configured_prefix_ratio,
                "shareable_prefix": self.shareable_prefix_ratio,
                "resident_prefix": self.actual_prefix_ratio,
                "candidate_block": self.candidate_block_hit_rate,
                "actual_block": self.actual_block_hit_rate,
                "candidate_byte": self.candidate_byte_hit_rate,
                "actual_byte": self.actual_byte_hit_rate,
                "input_tokens": self.input_tokens,
                "configured_prefix_tokens": self.configured_prefix_tokens,
                "shareable_prefix_tokens": self.shareable_prefix_tokens,
                "resident_prefix_tokens": self.resident_prefix_tokens,
                "cacheable_blocks": self.cacheable_blocks,
                "candidate_blocks": self.candidate_blocks,
                "hit_blocks": self.hit_blocks,
                "cacheable_bytes": self.cacheable_bytes,
                "candidate_bytes": self.candidate_bytes,
                "hit_bytes": self.hit_bytes,
            },
            "misses": {
                "compulsory_request_count": self.compulsory_miss_requests,
                "capacity_request_count": self.capacity_miss_requests,
            },
            "admission": {
                "write_miss_blocks": self.write_miss_blocks,
                "inserted_count": self.inserted_count,
                "inserted_blocks": self.inserted_blocks,
                "inserted_bytes": self.inserted_bytes,
                "deduplicated_insert_count": self.deduplicated_insert_count,
                "deduplicated_insert_blocks": self.deduplicated_insert_blocks,
                "deduplicated_insert_bytes": self.deduplicated_insert_bytes,
                "rejected_count": self.rejected_count,
                "rejected_blocks": self.rejected_blocks,
                "rejected_bytes": self.rejected_bytes,
                "checkpoint_admitted_count": self.checkpoint_admitted_count,
                "checkpoint_admitted_bytes": self.checkpoint_admitted_bytes,
                "checkpoint_deduplicated_count": self.checkpoint_deduplicated_count,
                "checkpoint_deduplicated_bytes": self.checkpoint_deduplicated_bytes,
                "checkpoint_rejected_count": self.checkpoint_rejected_count,
                "checkpoint_rejected_bytes": self.checkpoint_rejected_bytes,
            },
            "eviction": {
                "count": self.eviction_count,
                "blocks": self.evicted_blocks,
                "bytes": self.eviction_bytes,
                "self_eviction_count": self.self_eviction_count,
                "self_eviction_blocks": self.self_evicted_blocks,
                "self_eviction_bytes": self.self_eviction_bytes,
            },
            "capacity": {
                "total_capacity_bytes": self.config.total_capacity_bytes,
                "initial_used_bytes": initial_used,
                "initial_evictable_bytes": initial_evictable,
                "initial_protected_bytes": initial_protected,
                "peak_used_bytes": self.peak_total_used_bytes,
                "peak_evictable_bytes": self.peak_total_evictable_bytes,
                "peak_protected_bytes": self.peak_total_protected_bytes,
                "final_used_bytes": final_used,
                "final_evictable_bytes": final_evictable,
                "final_protected_bytes": final_protected,
                "shards": shards,
            },
            "tree": {
                "node_count": sum(item.node_count for item in self.final_snapshots),
                "leaf_count": sum(item.leaf_count for item in self.final_snapshots),
                "max_depth": max((item.max_depth for item in self.final_snapshots), default=0),
            },
        }


@dataclass(frozen=True)
class CacheReplayResult:
    requests: tuple[Request, ...]
    summary: CacheReplaySummary
    write_block_indices_by_request: dict[str, tuple[int, ...]]
    recovery_by_request: dict[str, PrefixRecoveryFunnel]


@dataclass(frozen=True)
class _PendingCompletion:
    completion_time_us: float
    sequence: int
    request_id: str
    shard: int
    cache_path: tuple[str, ...]
    block_sizes: tuple[int, ...]
    matched_nodes: tuple[Any, ...]
    write_start_index: int
    phase: str
    model: ModelProfile
    checkpoint_start_tokens: int
    end_tokens: int
    exact_checkpoint_tokens: int | None = None
    decode_cache_path: tuple[str, ...] = ()
    decode_block_sizes: tuple[int, ...] = ()
    decode_end_tokens: int | None = None


@dataclass
class _Counters:
    input_tokens: int = 0
    configured_prefix_tokens: int = 0
    shareable_prefix_tokens: int = 0
    resident_prefix_tokens: int = 0
    recovery_boundary_tokens: int = 0
    cacheable_blocks: int = 0
    candidate_blocks: int = 0
    hit_blocks: int = 0
    cacheable_bytes: int = 0
    candidate_bytes: int = 0
    hit_bytes: int = 0
    compulsory_miss_requests: int = 0
    capacity_miss_requests: int = 0
    inserted_count: int = 0
    inserted_bytes: int = 0
    deduplicated_insert_count: int = 0
    deduplicated_insert_bytes: int = 0
    rejected_count: int = 0
    rejected_bytes: int = 0
    eviction_count: int = 0
    eviction_bytes: int = 0
    self_eviction_count: int = 0
    self_eviction_bytes: int = 0
    checkpoint_admitted_count: int = 0
    checkpoint_admitted_bytes: int = 0
    checkpoint_deduplicated_count: int = 0
    checkpoint_deduplicated_bytes: int = 0
    checkpoint_rejected_count: int = 0
    checkpoint_rejected_bytes: int = 0


def _counter_payload(counter: _Counters) -> dict[str, object]:
    configured_ratio = (
        counter.configured_prefix_tokens / counter.input_tokens
        if counter.input_tokens
        else 0.0
    )
    shareable_ratio = (
        counter.shareable_prefix_tokens / counter.input_tokens
        if counter.input_tokens
        else 0.0
    )
    resident_ratio = (
        counter.resident_prefix_tokens / counter.input_tokens
        if counter.input_tokens
        else 0.0
    )
    recovery_ratio = (
        counter.recovery_boundary_tokens / counter.input_tokens
        if counter.input_tokens
        else 0.0
    )
    return {
        "hit_rate": {
            "configured_prefix": configured_ratio,
            "shareable_prefix": shareable_ratio,
            "resident_prefix": resident_ratio,
            "recovery_boundary": recovery_ratio,
            "candidate_block": (
                counter.candidate_blocks / counter.cacheable_blocks
                if counter.cacheable_blocks
                else 0.0
            ),
            "actual_block": (
                counter.hit_blocks / counter.cacheable_blocks
                if counter.cacheable_blocks
                else 0.0
            ),
            "candidate_byte": (
                counter.candidate_bytes / counter.cacheable_bytes
                if counter.cacheable_bytes
                else 0.0
            ),
            "actual_byte": (
                counter.hit_bytes / counter.cacheable_bytes
                if counter.cacheable_bytes
                else 0.0
            ),
            "input_tokens": counter.input_tokens,
            "configured_prefix_tokens": counter.configured_prefix_tokens,
            "shareable_prefix_tokens": counter.shareable_prefix_tokens,
            "resident_prefix_tokens": counter.resident_prefix_tokens,
            "recovery_boundary_tokens": counter.recovery_boundary_tokens,
            "replay_tokens": (
                counter.resident_prefix_tokens - counter.recovery_boundary_tokens
            ),
            "cacheable_blocks": counter.cacheable_blocks,
            "candidate_blocks": counter.candidate_blocks,
            "hit_blocks": counter.hit_blocks,
            "cacheable_bytes": counter.cacheable_bytes,
            "candidate_bytes": counter.candidate_bytes,
            "hit_bytes": counter.hit_bytes,
        },
        "misses": {
            "compulsory_request_count": counter.compulsory_miss_requests,
            "capacity_request_count": counter.capacity_miss_requests,
        },
        "admission": {
            "inserted_count": counter.inserted_count,
            "inserted_bytes": counter.inserted_bytes,
            "deduplicated_count": counter.deduplicated_insert_count,
            "deduplicated_bytes": counter.deduplicated_insert_bytes,
            "rejected_count": counter.rejected_count,
            "rejected_bytes": counter.rejected_bytes,
            "checkpoint_admitted_count": counter.checkpoint_admitted_count,
            "checkpoint_admitted_bytes": counter.checkpoint_admitted_bytes,
            "checkpoint_deduplicated_count": counter.checkpoint_deduplicated_count,
            "checkpoint_deduplicated_bytes": counter.checkpoint_deduplicated_bytes,
            "checkpoint_rejected_count": counter.checkpoint_rejected_count,
            "checkpoint_rejected_bytes": counter.checkpoint_rejected_bytes,
        },
        "eviction": {
            "count": counter.eviction_count,
            "bytes": counter.eviction_bytes,
            "self_eviction_count": counter.self_eviction_count,
            "self_eviction_bytes": counter.self_eviction_bytes,
        },
    }


def _snapshot_totals(
    snapshots: Sequence[CacheShardSnapshot],
) -> dict[str, int]:
    return {
        "used_bytes": sum(item.used_bytes for item in snapshots),
        "protected_bytes": sum(item.protected_bytes for item in snapshots),
        "evictable_bytes": sum(item.evictable_bytes for item in snapshots),
        "checkpoint_attachment_bytes": sum(
            item.attachment_bytes for item in snapshots
        ),
    }


def replay_prefix_cache(
    requests: Sequence[Request],
    models: Mapping[str, ModelProfile],
    *,
    block_size_tokens: int,
    config: CacheReplayConfig,
    write_miss_blocks: bool = True,
    prefix_cache_policy: PrefixCachePolicy | None = None,
    canonical_paths: Mapping[str, CanonicalRequestPath] | None = None,
    decode_state_writes: bool = False,
) -> CacheReplayResult:
    """Replay a cold or warmed block-radix cache and return effective requests."""

    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    policy = prefix_cache_policy or PrefixCachePolicy()
    for model in models.values():
        validate_execution_combination(model, policy)
    ordered = sorted(requests, key=lambda item: (item.arrival_time_us, item.request_id))
    request_ids = [request.request_id for request in ordered]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("cache replay requires unique request_id values")
    bootstrap_requests = sum(request.is_bootstrap for request in ordered)
    normal_request_count = len(ordered) - bootstrap_requests
    warmup_requests = int(normal_request_count * config.warmup_fraction)
    cache = RadixLeafLRUCache(config)
    phase_counters = {
        "bootstrap": _Counters(),
        "warmup": _Counters(),
        "measured": _Counters(),
    }
    counters = phase_counters["measured"]
    cold_initial_snapshots = tuple(
        CacheShardSnapshot(
            shard=index,
            capacity_bytes=capacity,
            used_bytes=0,
            evictable_bytes=0,
            protected_bytes=0,
            node_count=0,
            leaf_count=0,
            max_depth=0,
        )
        for index, capacity in enumerate(config.capacity_bytes_by_shard)
    )
    bootstrap_drained_snapshots: tuple[CacheShardSnapshot, ...] | None = None
    pending: list[tuple[float, int, _PendingCompletion]] = []
    emitted: list[Request] = []
    size_cache: dict[tuple[str, int, str], tuple[int, ...]] = {}
    shared_prefix_id_cache: SharedPrefixBlockIdMemo = {}
    measurement_started = False
    initial_snapshots: tuple[CacheShardSnapshot, ...] | None = None
    peak_used_by_shard = [0] * config.shard_count
    peak_evictable_by_shard = [0] * config.shard_count
    peak_protected_by_shard = [0] * config.shard_count
    peak_total_used = 0
    peak_total_evictable = 0
    peak_total_protected = 0
    peak_total_checkpoint_attachment = 0
    write_block_indices_by_request: dict[str, tuple[int, ...]] = {}
    recovery_by_request: dict[str, PrefixRecoveryFunnel] = {}
    sequence = 0
    normal_request_index = 0
    bootstrap_drained = False

    def observe() -> None:
        nonlocal peak_total_used, peak_total_evictable, peak_total_protected
        nonlocal peak_total_checkpoint_attachment
        if not measurement_started:
            return
        used_by_shard = cache.used_bytes_by_shard
        protected_by_shard = cache.protected_bytes_by_shard
        evictable_by_shard = tuple(
            used - protected
            for used, protected in zip(used_by_shard, protected_by_shard)
        )
        for index, used in enumerate(used_by_shard):
            peak_used_by_shard[index] = max(
                peak_used_by_shard[index], used
            )
            peak_evictable_by_shard[index] = max(
                peak_evictable_by_shard[index], evictable_by_shard[index]
            )
            peak_protected_by_shard[index] = max(
                peak_protected_by_shard[index], protected_by_shard[index]
            )
        total_used = sum(used_by_shard)
        peak_total_used = max(peak_total_used, total_used)
        peak_total_evictable = max(
            peak_total_evictable, sum(evictable_by_shard)
        )
        peak_total_protected = max(
            peak_total_protected, sum(protected_by_shard)
        )
        peak_total_checkpoint_attachment = max(
            peak_total_checkpoint_attachment,
            sum(cache.attachment_bytes_by_shard),
        )

    def finish(item: _PendingCompletion) -> None:
        phase_counter = phase_counters[item.phase]
        if write_miss_blocks and item.write_start_index < len(item.cache_path):
            inserted = cache.insert_path(
                item.cache_path,
                item.block_sizes,
                shard=item.shard,
                write_start_index=item.write_start_index,
            )
        else:
            inserted = None
        if item.phase == "measured":
            write_block_indices_by_request[item.request_id] = (
                tuple(node.depth - 1 for node in inserted.new_nodes)
                if inserted is not None
                else ()
            )
        new_nodes = list(inserted.new_nodes if inserted is not None else ())
        insert_results = [inserted] if inserted is not None else []

        if write_miss_blocks and item.cache_path:
            resident_input = cache.match_path(
                item.cache_path,
                shard=item.shard,
                lock=False,
            ).matched_nodes
            _admit_checkpoint_interval(
                cache,
                resident_input,
                model=item.model,
                policy=policy,
                prefix_tokens=item.checkpoint_start_tokens,
                end_tokens=item.end_tokens,
                block_size_tokens=block_size_tokens,
                measured=True,
                counters=phase_counter,
                exact_boundary_tokens=item.exact_checkpoint_tokens,
            )

        if (
            write_miss_blocks
            and decode_state_writes
            and item.decode_cache_path
            and item.decode_block_sizes
        ):
            common = _common_prefix_length(item.cache_path, item.decode_cache_path)
            decoded = cache.insert_path(
                item.decode_cache_path,
                item.decode_block_sizes,
                shard=item.shard,
                write_start_index=common,
            )
            insert_results.append(decoded)
            new_nodes.extend(decoded.new_nodes)
            resident_output = cache.match_path(
                item.decode_cache_path,
                shard=item.shard,
                lock=False,
            ).matched_nodes
            _admit_checkpoint_interval(
                cache,
                resident_output,
                model=item.model,
                policy=policy,
                prefix_tokens=item.end_tokens,
                end_tokens=item.decode_end_tokens or item.end_tokens,
                block_size_tokens=block_size_tokens,
                measured=True,
                counters=phase_counter,
            )
        cache.release(item.matched_nodes)
        evicted = cache.enforce_capacity(
            item.shard,
            new_nodes=new_nodes,
        )
        for result in insert_results:
            phase_counter.inserted_count += result.inserted_count
            phase_counter.inserted_bytes += result.inserted_bytes
            phase_counter.deduplicated_insert_count += result.deduplicated_count
            phase_counter.deduplicated_insert_bytes += result.deduplicated_bytes
            phase_counter.rejected_count += result.rejected_count
            phase_counter.rejected_bytes += result.rejected_bytes
        phase_counter.eviction_count += evicted.eviction_count
        phase_counter.eviction_bytes += evicted.eviction_bytes
        phase_counter.self_eviction_count += evicted.self_eviction_count
        phase_counter.self_eviction_bytes += evicted.self_eviction_bytes
        observe()

    def finish_due(timestamp_us: float) -> None:
        while pending and pending[0][0] <= timestamp_us:
            _time, _order, item = heapq.heappop(pending)
            finish(item)

    for request in ordered:
        if not request.is_bootstrap and not bootstrap_drained:
            while pending:
                _time, _order, item = heapq.heappop(pending)
                finish(item)
            bootstrap_drained = True
            bootstrap_drained_snapshots = (
                cache.snapshots() if bootstrap_requests else cold_initial_snapshots
            )
        finish_due(request.arrival_time_us)
        if request.is_bootstrap:
            phase = "bootstrap"
        elif normal_request_index < warmup_requests:
            phase = "warmup"
        else:
            phase = "measured"
        measured = phase == "measured"
        request_counters = phase_counters[phase]
        if not request.is_bootstrap:
            normal_request_index += 1
        if measured and not measurement_started:
            measurement_started = True
            initial_snapshots = cache.snapshots()
            observe()

        model = _model_for_request(request, models)
        _validate_request(request, model)
        trace_block_ids = _request_block_ids(
            request,
            block_size_tokens,
            shared_prefix_id_cache,
        )
        object_policy = getattr(model.attention, "prefix_granularity", None)
        stores_partial_tail = (
            object_policy is None
            or object_policy.partial_tail_storage == "sealed_padded"
        )
        cache_block_count = (
            _ceil_div(request.seq_len, block_size_tokens)
            if stores_partial_tail and request.seq_len
            else request.seq_len // block_size_tokens
        )
        cache_block_ids = trace_block_ids[:cache_block_count]
        cache_path = tuple(f"{request.model_id}:{value}" for value in cache_block_ids)
        block_sizes = _block_sizes(
            model,
            cache_block_count,
            block_size_tokens,
            policy,
            size_cache,
        )
        canonical = canonical_paths.get(request.request_id) if canonical_paths else None
        if canonical is not None:
            shareable_tokens = canonical.shareable_prefix_tokens
            shareability_source = canonical.shareability_source
            lookup_block_ids = canonical.lookup_block_ids
        elif request.block_ids:
            shareable_tokens = request.matched_prefix_len
            shareability_source = "explicit_object_ids"
            lookup_block_ids = request.restore_block_ids or trace_block_ids
        else:
            shareable_tokens = (
                request.matched_prefix_len // block_size_tokens * block_size_tokens
            )
            shareability_source = "complete_shared_blocks"
            lookup_block_ids = trace_block_ids
        if not stores_partial_tail:
            shareable_tokens = shareable_tokens // block_size_tokens * block_size_tokens
        candidate_blocks = min(
            (
                _ceil_div(shareable_tokens, block_size_tokens)
                if stores_partial_tail and shareable_tokens
                else shareable_tokens // block_size_tokens
            ),
            cache_block_count,
        )
        candidate_cache_path = tuple(
            f"{request.model_id}:{value}"
            for value in lookup_block_ids[:candidate_blocks]
        )
        shard = cache.shard_for_path(cache_path) if cache_path else (
            cache.shard_for_path(
                (f"{request.model_id}:{trace_block_ids[0]}",)
            )
            if trace_block_ids
            else None
        )
        if candidate_cache_path:
            match = cache.match_path(
                candidate_cache_path,
                shard=shard,
                lock=True,
            )
            hit_blocks = match.matched_blocks
            matched_nodes = match.matched_nodes
            missing_was_admitted = match.missing_was_admitted
        else:
            hit_blocks = 0
            matched_nodes = ()
            missing_was_admitted = False
        hit_tokens = min(hit_blocks * block_size_tokens, shareable_tokens)
        recovery_boundary, restored_checkpoint_keys = _actual_recovery_boundary(
            matched_nodes,
            model=model,
            policy=policy,
            matched_prefix_tokens=hit_tokens,
            seq_len=request.seq_len,
        )
        effective = replace(
            request,
            matched_prefix_len=hit_tokens,
            block_ids=trace_block_ids,
            restore_block_ids=tuple(lookup_block_ids),
            cache_shard=shard,
            recovery_boundary_tokens=recovery_boundary,
            restored_checkpoint_keys=restored_checkpoint_keys,
        )

        request_counters.input_tokens += request.seq_len
        request_counters.configured_prefix_tokens += request.matched_prefix_len
        request_counters.shareable_prefix_tokens += shareable_tokens
        request_counters.resident_prefix_tokens += hit_tokens
        request_counters.recovery_boundary_tokens += recovery_boundary
        request_counters.cacheable_blocks += cache_block_count
        request_counters.candidate_blocks += candidate_blocks
        request_counters.hit_blocks += hit_blocks
        request_counters.cacheable_bytes += sum(block_sizes)
        request_counters.candidate_bytes += sum(block_sizes[:candidate_blocks])
        request_counters.hit_bytes += sum(block_sizes[:hit_blocks])
        if hit_blocks < candidate_blocks:
            if missing_was_admitted:
                request_counters.capacity_miss_requests += 1
            else:
                request_counters.compulsory_miss_requests += 1

        if measured:
            emitted.append(effective)
            causes: list[str] = []
            if shareable_tokens < request.matched_prefix_len:
                causes.append("partial_tail_not_shareable")
            if hit_tokens < shareable_tokens:
                causes.append(
                    "capacity_or_cold_miss"
                    if missing_was_admitted
                    else "cold_start"
                )
            if recovery_boundary < hit_tokens:
                phase_plan = getattr(model.attention, "recovery_plan", None)
                if callable(phase_plan):
                    causes.extend(phase_plan(model, hit_tokens, request.seq_len, policy).causes)
                else:
                    causes.append("checkpoint_not_resident_or_too_sparse")
            recovery_by_request[request.request_id] = PrefixRecoveryFunnel(
                seq_len=request.seq_len,
                configured_prefix_tokens=request.matched_prefix_len,
                shareable_prefix_tokens=shareable_tokens,
                resident_prefix_tokens=hit_tokens,
                recovery_boundary_tokens=recovery_boundary,
                shareability_source=shareability_source,
                loss_causes=tuple(causes),
                resident_cache_bytes=sum(
                    node.resident_size_bytes for node in matched_nodes
                ),
                resident_token_bytes=sum(node.size_bytes for node in matched_nodes),
                resident_checkpoint_bytes=sum(
                    node.attachment_bytes for node in matched_nodes
                ),
            )

        path = canonical_paths.get(request.request_id) if canonical_paths is not None else None
        decode_cache_path: tuple[str, ...] = ()
        decode_block_sizes: tuple[int, ...] = ()
        if decode_state_writes and request.generation_len > 0 and path is not None:
            decode_cache_path = tuple(
                f"{request.model_id}:{value}" for value in path.output_block_ids
            )
            decode_block_sizes = _block_sizes(
                model,
                len(decode_cache_path),
                block_size_tokens,
                policy,
                size_cache,
            )
        if (cache_path or decode_cache_path) and (
            matched_nodes
            or (write_miss_blocks and hit_blocks < cache_block_count)
            or (write_miss_blocks and bool(decode_cache_path))
        ):
            write_start_index = min(
                _common_prefix_length(cache_path, candidate_cache_path),
                hit_blocks,
            )
            completion_time = request.arrival_time_us + _actual_prefill_time_us(
                effective,
                model,
                block_size_tokens,
                policy,
            )
            item = _PendingCompletion(
                completion_time_us=completion_time,
                sequence=sequence,
                request_id=request.request_id,
                shard=int(shard),
                cache_path=cache_path,
                block_sizes=block_sizes,
                matched_nodes=matched_nodes,
                write_start_index=write_start_index,
                phase=phase,
                model=model,
                checkpoint_start_tokens=recovery_boundary,
                end_tokens=request.seq_len,
                exact_checkpoint_tokens=(
                    request.seq_len
                    if request.is_bootstrap
                    else shareable_tokens
                ),
                decode_cache_path=decode_cache_path,
                decode_block_sizes=decode_block_sizes,
                decode_end_tokens=(request.output_end_token if decode_cache_path else None),
            )
            heapq.heappush(pending, (completion_time, sequence, item))
            sequence += 1
        observe()

    while pending:
        _time, _order, item = heapq.heappop(pending)
        finish(item)

    if initial_snapshots is None:
        initial_snapshots = cache.snapshots()
        measurement_started = True
        observe()
    cache.assert_consistent()
    final_snapshots = cache.snapshots()
    if bootstrap_drained_snapshots is None:
        bootstrap_drained_snapshots = (
            final_snapshots if bootstrap_requests else cold_initial_snapshots
        )
    summary = CacheReplaySummary(
        config=config,
        total_requests=len(ordered),
        measured_requests=len(emitted),
        warmup_requests=warmup_requests,
        bootstrap_requests=bootstrap_requests,
        write_miss_blocks=write_miss_blocks,
        input_tokens=counters.input_tokens,
        configured_prefix_tokens=counters.configured_prefix_tokens,
        shareable_prefix_tokens=counters.shareable_prefix_tokens,
        resident_prefix_tokens=counters.resident_prefix_tokens,
        cacheable_blocks=counters.cacheable_blocks,
        candidate_blocks=counters.candidate_blocks,
        hit_blocks=counters.hit_blocks,
        cacheable_bytes=counters.cacheable_bytes,
        candidate_bytes=counters.candidate_bytes,
        hit_bytes=counters.hit_bytes,
        compulsory_miss_requests=counters.compulsory_miss_requests,
        capacity_miss_requests=counters.capacity_miss_requests,
        inserted_count=counters.inserted_count,
        inserted_bytes=counters.inserted_bytes,
        deduplicated_insert_count=counters.deduplicated_insert_count,
        deduplicated_insert_bytes=counters.deduplicated_insert_bytes,
        rejected_count=counters.rejected_count,
        rejected_bytes=counters.rejected_bytes,
        eviction_count=counters.eviction_count,
        eviction_bytes=counters.eviction_bytes,
        self_eviction_count=counters.self_eviction_count,
        self_eviction_bytes=counters.self_eviction_bytes,
        initial_snapshots=initial_snapshots,
        peak_used_bytes_by_shard=tuple(peak_used_by_shard),
        peak_evictable_bytes_by_shard=tuple(peak_evictable_by_shard),
        peak_protected_bytes_by_shard=tuple(peak_protected_by_shard),
        peak_total_used_bytes=peak_total_used,
        peak_total_evictable_bytes=peak_total_evictable,
        peak_total_protected_bytes=peak_total_protected,
        peak_total_checkpoint_attachment_bytes=peak_total_checkpoint_attachment,
        final_snapshots=final_snapshots,
        checkpoint_admitted_count=counters.checkpoint_admitted_count,
        checkpoint_admitted_bytes=counters.checkpoint_admitted_bytes,
        checkpoint_deduplicated_count=counters.checkpoint_deduplicated_count,
        checkpoint_deduplicated_bytes=counters.checkpoint_deduplicated_bytes,
        checkpoint_rejected_count=counters.checkpoint_rejected_count,
        checkpoint_rejected_bytes=counters.checkpoint_rejected_bytes,
        phase_counters=phase_counters,
        lifecycle_snapshots={
            "cold_initial": cold_initial_snapshots,
            "bootstrap_drained": bootstrap_drained_snapshots,
            "measurement_start": initial_snapshots,
            "final_drained": final_snapshots,
        },
    )
    return CacheReplayResult(
        tuple(emitted),
        summary,
        write_block_indices_by_request,
        recovery_by_request,
    )


def _request_block_ids(
    request: Request,
    block_size_tokens: int,
    shared_prefix_memo: SharedPrefixBlockIdMemo | None = None,
) -> tuple[str, ...]:
    expected = _ceil_div(request.seq_len, block_size_tokens) if request.seq_len else 0
    if request.block_ids:
        if len(request.block_ids) != expected:
            raise ValueError(
                f"request {request.request_id} expected {expected} block_ids for "
                f"seq_len={request.seq_len}, found {len(request.block_ids)}"
            )
        if any(not value for value in request.block_ids):
            raise ValueError(f"request {request.request_id} contains an empty block_id")
        return tuple(request.block_ids)
    return build_request_block_ids_with_shared_memo(
        request.model_id,
        request.prefix_group_id,
        request.request_id,
        request.seq_len,
        request.matched_prefix_len,
        block_size_tokens,
        shared_prefix_memo if shared_prefix_memo is not None else {},
    )


def _block_sizes(
    model: ModelProfile,
    block_count: int,
    block_size_tokens: int,
    policy: PrefixCachePolicy,
    memo: dict[tuple[str, int, str], tuple[int, ...]],
) -> tuple[int, ...]:
    policy_key = _replay_policy_key(model, policy)
    key = (model.name, block_count, policy_key)
    cached = memo.get(key)
    if cached is not None:
        return cached
    block_bytes = _token_block_bytes(model, block_size_tokens, policy)
    if block_bytes <= 0:
        raise ValueError(f"model {model.name} produced non-positive token cache bytes")
    result = (block_bytes,) * block_count
    memo[key] = result
    return result


def _resolved_group_policy(
    group: CheckpointGroup,
    policy: PrefixCachePolicy,
) -> StatePersistencePolicy:
    candidates = [group.name, *(component.name for component in group.components)]
    overrides = [policy.state_policies[name] for name in candidates if name in policy.state_policies]
    if not overrides:
        return group.policy
    first = overrides[0]
    if any(value != first for value in overrides[1:]):
        raise ValueError(
            f"atomic checkpoint group {group.name} has conflicting component policies"
        )
    return first


def _checkpoint_groups_for_model(
    model: ModelProfile,
    policy: PrefixCachePolicy,
) -> tuple[CheckpointGroup, ...]:
    attention = model.attention
    groups: list[CheckpointGroup] = []
    effective_factory = getattr(attention, "effective_checkpoint_groups", None)
    if callable(effective_factory):
        groups.extend(effective_factory(model, policy))
    else:
        factory = getattr(attention, "checkpoint_groups", None)
        if callable(factory):
            groups.extend(factory(model))

    # A token-history default does not create a checkpoint group, but a
    # runtime snapshot override must still be able to materialize one.
    if "sliding_window_state" in policy.state_policies and not any(
        group.name == "sliding_window_state" for group in groups
    ):
        override = policy.state_policies["sliding_window_state"]
        for method_name in ("_sliding_checkpoint_group", "_sliding_group"):
            method = getattr(attention, method_name, None)
            if callable(method):
                generated = method(model, override)
                if generated is not None:
                    groups.append(generated)
                break

    if (
        getattr(attention, "name", "") == "deepseek_v4"
        and not callable(effective_factory)
    ):
        state_policy = policy.swa_state_policy()
        if state_policy.storage_form != "token_history":
            serialized_per_layer = (
                int(getattr(attention, "sliding_window"))
                * int(getattr(attention, "main_entry_bytes"))
            )
            groups.append(
                CheckpointGroup(
                    "swa",
                    (
                        StateComponent(
                            "swa_snapshot",
                            "swa_state",
                            serialized_per_layer,
                            8,
                            model.num_layers,
                            serialized_bytes_per_layer=serialized_per_layer,
                        ),
                    ),
                    state_policy,
                    (LayerKind.SWA,),
                )
            )

    resolved: list[CheckpointGroup] = []
    for group in groups:
        group_policy = _resolved_group_policy(group, policy)
        if group_policy.storage_form == "token_history":
            continue
        resolved.append(replace(group, policy=group_policy))
    return tuple(resolved)


def _token_block_bytes(
    model: ModelProfile,
    block_size_tokens: int,
    policy: PrefixCachePolicy,
) -> int:
    components = list(model.cache_components())
    attention = model.attention
    sliding_default = getattr(attention, "sliding_persistence", None)
    if isinstance(sliding_default, StatePersistencePolicy):
        sliding_policy = policy.state_policy("sliding_window_state", sliding_default)
        if sliding_policy.storage_form != "token_history":
            components = [
                component for component in components if component.category != "sliding_kv"
            ]
    result = sum(
        component.bytes_per_block(model, block_size_tokens)
        for component in components
    )
    if getattr(attention, "name", "") == "deepseek_v4":
        if policy.swa_state_policy().storage_form == "token_history":
            result += (
                block_size_tokens
                * model.num_layers
                * int(getattr(attention, "main_entry_bytes"))
            )
    return result


def _replay_policy_key(model: ModelProfile, policy: PrefixCachePolicy) -> str:
    state = ",".join(
        f"{name}={value.storage_form}:{value.interval_tokens}:{value.retain_prompt_end}"
        for name, value in sorted(policy.state_policies.items())
    )
    return (
        f"{getattr(model.attention, 'name', '')}|{policy.swa_mode}|"
        f"{policy.swa_checkpoint_interval_tokens}|{state}|"
        f"draft_execution={policy.include_draft_execution}|"
        f"exact_resident={policy.assume_exact_at_resident}"
        f"|encoder={policy.encoder_state}|tail={policy.encoder_tail_policy}"
    )


def _actual_recovery_boundary(
    matched_nodes: Sequence[Any],
    *,
    model: ModelProfile,
    policy: PrefixCachePolicy,
    matched_prefix_tokens: int,
    seq_len: int,
) -> tuple[int, tuple[str, ...]]:
    phase_plan = getattr(model.attention, "recovery_plan", None)
    if callable(phase_plan):
        return phase_plan(model, matched_prefix_tokens, seq_len, policy).recovery_boundary_tokens, ()
    groups = _checkpoint_groups_for_model(model, policy)
    if not groups:
        return matched_prefix_tokens, ()
    boundary_sets: list[set[int]] = []
    for group in groups:
        if group.policy.storage_form == "none":
            boundary_sets.append(set())
            continue
        boundaries = {
            attachment.boundary_token
            for node in matched_nodes
            for attachment in node.attachments.values()
            if attachment.key.startswith(f"{group.name}@")
            and attachment.boundary_token <= matched_prefix_tokens
        }
        boundary_sets.append(boundaries)
    common = set.intersection(*boundary_sets) if boundary_sets else set()
    recovery = max(common, default=0)
    keys = (
        tuple(f"{group.name}@{recovery}" for group in groups)
        if recovery > 0
        else ()
    )
    return recovery, keys


def _admit_checkpoint_interval(
    cache: RadixLeafLRUCache,
    resident_nodes: Sequence[Any],
    *,
    model: ModelProfile,
    policy: PrefixCachePolicy,
    prefix_tokens: int,
    end_tokens: int,
    block_size_tokens: int,
    measured: bool,
    counters: _Counters,
    exact_boundary_tokens: int | None = None,
) -> None:
    if end_tokens <= prefix_tokens:
        return
    for group in _checkpoint_groups_for_model(model, policy):
        if group.policy.storage_form != "snapshot":
            continue
        if (
            policy.assume_exact_at_resident
            and group.policy == StatePersistencePolicy.terminal()
        ):
            boundaries = (
                (exact_boundary_tokens,)
                if exact_boundary_tokens is not None
                and prefix_tokens < exact_boundary_tokens <= end_tokens
                else ()
            )
        else:
            boundaries = checkpoint_boundaries(
                group.policy,
                prefix_tokens=prefix_tokens,
                end_tokens=end_tokens,
                block_size_tokens=block_size_tokens,
            )
        extents = tuple(
            (component.name, component.serialized_bytes_per_checkpoint())
            for component in group.components
        )
        for boundary in boundaries:
            owner_index = _ceil_div(boundary, block_size_tokens) - 1
            size_bytes = group.serialized_bytes_per_checkpoint
            if not 0 <= owner_index < len(resident_nodes):
                if measured:
                    counters.checkpoint_rejected_count += 1
                    counters.checkpoint_rejected_bytes += size_bytes
                continue
            result = cache.attach_checkpoint(
                resident_nodes[owner_index],
                CheckpointAttachment(
                    key=f"{group.name}@{boundary}",
                    boundary_token=boundary,
                    size_bytes=size_bytes,
                    component_extents=extents,
                ),
            )
            if not measured:
                continue
            if result.admitted:
                counters.checkpoint_admitted_count += 1
                counters.checkpoint_admitted_bytes += result.attachment_bytes
                counters.inserted_bytes += result.attachment_bytes
            elif result.deduplicated:
                counters.checkpoint_deduplicated_count += 1
                counters.checkpoint_deduplicated_bytes += result.attachment_bytes
                counters.deduplicated_insert_bytes += result.attachment_bytes
            elif result.rejected:
                counters.checkpoint_rejected_count += 1
                counters.checkpoint_rejected_bytes += result.attachment_bytes
                counters.rejected_bytes += result.attachment_bytes
            counters.eviction_count += result.eviction.eviction_count
            counters.eviction_bytes += result.eviction.eviction_bytes
            counters.self_eviction_count += result.eviction.self_eviction_count
            counters.self_eviction_bytes += result.eviction.self_eviction_bytes


def _common_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def _actual_prefill_time_us(
    request: Request,
    model: ModelProfile,
    block_size_tokens: int,
    policy: PrefixCachePolicy,
) -> float:
    read_plan = model.attention.map_hit_to_reads(
        model=model,
        hit_tokens=request.matched_prefix_len,
        block_size_tokens=block_size_tokens,
        seq_len=request.seq_len,
        decode_len=request.generation_len,
        policy=policy,
        request_key=request.request_id,
    )
    context = PrefillComputeContext(
        seq_len=request.seq_len,
        matched_prefix_len=request.matched_prefix_len,
        generation_len=request.generation_len,
        encoder_state=policy.encoder_state,
        encoder_tail_policy=policy.encoder_tail_policy,
    )
    recovery_boundary = (
        request.recovery_boundary_tokens
        if request.recovery_boundary_tokens is not None
        else request.matched_prefix_len - read_plan.restore_recompute_tokens
    )
    if read_plan.reusable_prefix_tokens is not None:
        recovery_boundary = min(recovery_boundary, read_plan.reusable_prefix_tokens)
    breakdown = model.compute.cache_compute_breakdown(
        model,
        context,
        restore_recompute_tokens=request.matched_prefix_len - recovery_boundary,
    )
    require_evaluated_full_hit(breakdown, model_id=model.name, request_id=request.request_id)
    timing = model.compute.cache_timing_estimates(breakdown)["cache_path"]
    return timing.time_us if timing.time_us is not None else timing.normalized_reference_us


def _model_for_request(
    request: Request, models: Mapping[str, ModelProfile]
) -> ModelProfile:
    try:
        return models[request.model_id]
    except KeyError as exc:
        raise ValueError(f"unknown model_id: {request.model_id}") from exc


def _validate_request(request: Request, model: ModelProfile) -> None:
    if request.seq_len < 0 or request.generation_len < 0:
        raise ValueError(f"request {request.request_id} has a negative sequence length")
    if not 0 <= request.matched_prefix_len <= request.seq_len:
        raise ValueError(
            f"request {request.request_id} matched_prefix_len must be within seq_len"
        )
    if model.max_context_tokens is not None and request.seq_len > model.max_context_tokens:
        raise ValueError(
            f"request {request.request_id} seq_len exceeds "
            f"{model.name} max_context_tokens={model.max_context_tokens}"
        )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
