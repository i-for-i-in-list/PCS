"""Shared policies and state descriptions for hybrid attention caches."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Mapping

from ssu_sim.utils.units import ceil_div


ENCODER_SCENARIO_VALUES = {
    "encoder_state": ("unavailable", "available"),
    "encoder_tail_policy": ("recompute", "retain"),
}


def normalize_encoder_scenarios(values: Mapping[str, object]) -> dict[str, list[str]]:
    """Canonical set order keeps single/both selection and dedup deterministic."""
    result = {}
    for name, allowed in ENCODER_SCENARIO_VALUES.items():
        raw = values.get(name, [allowed[0]])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)) or not raw or any(value not in allowed for value in raw):
            raise ValueError(f"{name} must select one or more of {', '.join(allowed)}")
        result[name] = [value for value in allowed if value in raw]
    return result


class LayerKind(str, Enum):
    FULL_ATTENTION = "full_attention"
    SLIDING_ATTENTION = "sliding_attention"
    MLA = "mla"
    KDA = "kda"
    LINEAR_ATTENTION = "linear_attention"
    SCONV = "sconv"
    MSA = "msa"
    DSA = "dsa"
    SWA = "swa"
    CSA = "csa"
    HCA = "hca"


@dataclass(frozen=True)
class PrefixGranularityPolicy:
    """Physical storage policy for a non-aligned prefix tail.

    Logical prefix matching and checkpoint recovery are deliberately handled
    elsewhere.  A sealed tail is reusable only when requests name the same
    immutable object (for example through explicit lineage).
    """

    partial_tail_storage: str = "sealed_padded"

    def __post_init__(self) -> None:
        if self.partial_tail_storage not in {
            "sealed_padded",
            "complete_blocks_only",
        }:
            raise ValueError(
                "partial_tail_storage must be sealed_padded or complete_blocks_only"
            )

    @classmethod
    def sealed_padded(cls) -> "PrefixGranularityPolicy":
        return cls()

    # Compatibility spelling used by the first v2 implementation draft.
    sealed_padded_tail = sealed_padded

    @classmethod
    def full_shared_blocks_only(cls) -> "PrefixGranularityPolicy":
        return cls(partial_tail_storage="complete_blocks_only")

    @classmethod
    def rewrite_partial_tail(cls) -> "PrefixGranularityPolicy":
        return cls.sealed_padded()

    @classmethod
    def checkpoint_boundary(cls) -> "PrefixGranularityPolicy":
        """Compatibility constructor for profiles created before recovery v3."""

        return cls.sealed_padded()


@dataclass(frozen=True)
class StatePersistencePolicy:
    """Persistence form for state that is not token-linear KV.

    ``fixed_interval`` always has an optional final prompt-end checkpoint in
    addition to regular interval boundaries. This is why a prompt of ``I+1``
    tokens has two checkpoints when ``retain_prompt_end`` is true.
    """

    storage_form: str = "snapshot"
    interval_tokens: int | None = None
    retain_prompt_end: bool = True

    def __post_init__(self) -> None:
        # Accept the names used by the early implementation draft, but expose
        # one stable public representation in reports and configuration.
        aliases = {"fixed_interval": "snapshot", "terminal_only": "snapshot"}
        canonical = aliases.get(self.storage_form, self.storage_form)
        object.__setattr__(self, "storage_form", canonical)
        if canonical not in {"token_history", "snapshot", "none"}:
            raise ValueError(
                "storage_form must be token_history, snapshot, or none"
            )
        if canonical == "snapshot":
            if self.interval_tokens is not None and self.interval_tokens <= 0:
                raise ValueError("snapshot interval_tokens must be positive when provided")
        elif self.interval_tokens is not None:
            raise ValueError("interval_tokens is only valid for snapshot persistence")
        if canonical == "none" and self.retain_prompt_end:
            raise ValueError("none persistence cannot retain a prompt-end checkpoint")
        if canonical == "token_history" and self.retain_prompt_end:
            raise ValueError("token_history does not use prompt-end snapshots")

    @classmethod
    def token_history(cls) -> "StatePersistencePolicy":
        return cls("token_history", None, False)

    @classmethod
    def fixed_interval(
        cls, interval_tokens: int, *, retain_prompt_end: bool = True
    ) -> "StatePersistencePolicy":
        return cls("snapshot", interval_tokens, retain_prompt_end)

    @classmethod
    def per_block(
        cls, block_size_tokens: int, *, retain_prompt_end: bool = True
    ) -> "StatePersistencePolicy":
        return cls.fixed_interval(
            block_size_tokens,
            retain_prompt_end=retain_prompt_end,
        )

    @classmethod
    def periodic(
        cls, interval_tokens: int, *, retain_prompt_end: bool = True
    ) -> "StatePersistencePolicy":
        return cls.fixed_interval(
            interval_tokens,
            retain_prompt_end=retain_prompt_end,
        )

    @classmethod
    def terminal(cls) -> "StatePersistencePolicy":
        return cls("snapshot", None, True)

    @classmethod
    def none(cls) -> "StatePersistencePolicy":
        return cls("none", None, False)

    @property
    def snapshot_cadence(self) -> str | None:
        """Compatibility view for callers migrating from the draft API."""

        if self.storage_form == "snapshot" and self.interval_tokens is not None:
            return "periodic"
        if self.storage_form == "snapshot":
            return "terminal"
        return None


@dataclass(frozen=True)
class BlockSizeConstraints:
    """Explicit block-size contract advertised by an attention profile."""

    minimum: int = 1
    multiple: int = 1
    native: int | None = None

    def __post_init__(self) -> None:
        if self.minimum <= 0 or self.multiple <= 0:
            raise ValueError("block-size minimum and multiple must be positive")
        if self.native is not None:
            if self.native < self.minimum or self.native % self.multiple:
                raise ValueError(
                    "native block size must satisfy the minimum and multiple constraints"
                )

    def validate(self, block_size_tokens: int) -> None:
        if block_size_tokens < self.minimum or block_size_tokens % self.multiple:
            raise ValueError(
                f"block_size_tokens must be >= {self.minimum} and a multiple of {self.multiple}"
            )


@dataclass(frozen=True)
class StateComponent:
    """One fixed-size state tensor used actively or stored in checkpoints."""

    name: str
    category: str
    elements_per_layer: int
    dtype_bits: int
    layer_count: int
    lifecycle: str = "checkpointable"
    scale_group_size: int | None = None
    scale_bytes: int = 0
    serialized_bytes_per_layer: int | None = None
    padding_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("state component name must not be empty")
        if self.category not in {
            "linear_state",
            "conv_state",
            "recurrent_state",
            "kda_state",
            "sconv_state",
            "swa_state",
            "compressor_state",
            "hidden_state",
        }:
            raise ValueError(f"unsupported state component category: {self.category}")
        if self.lifecycle not in {"active_only", "checkpointable"}:
            raise ValueError("state lifecycle must be active_only or checkpointable")
        for value, label in (
            (self.elements_per_layer, "elements_per_layer"),
            (self.dtype_bits, "dtype_bits"),
            (self.layer_count, "layer_count"),
        ):
            if value <= 0:
                raise ValueError(f"state component {label} must be positive")
        if self.scale_bytes < 0 or self.padding_bytes < 0:
            raise ValueError("state scale and padding bytes must not be negative")
        if (self.scale_group_size is None) != (self.scale_bytes == 0):
            raise ValueError(
                "scale_group_size and a positive scale_bytes must be provided together"
            )
        if self.scale_group_size is not None and self.scale_group_size <= 0:
            raise ValueError("scale_group_size must be positive")
        if self.serialized_bytes_per_layer is not None and self.serialized_bytes_per_layer <= 0:
            raise ValueError("serialized_bytes_per_layer must be positive")

    def nominal_bytes_per_checkpoint(self) -> int:
        return self.layer_count * ceil_div(self.elements_per_layer * self.dtype_bits, 8)

    def serialized_bytes_per_checkpoint(self) -> int:
        if self.serialized_bytes_per_layer is not None:
            return self.layer_count * self.serialized_bytes_per_layer
        per_layer = ceil_div(self.elements_per_layer * self.dtype_bits, 8)
        if self.scale_group_size is not None:
            per_layer += (
                ceil_div(self.elements_per_layer, self.scale_group_size)
                * self.scale_bytes
            )
        return self.layer_count * (per_layer + self.padding_bytes)


@dataclass(frozen=True)
class CheckpointGroup:
    """State tensors captured and restored atomically at one boundary."""

    name: str
    components: tuple[StateComponent, ...]
    policy: StatePersistencePolicy = StatePersistencePolicy()
    layer_kinds: tuple[LayerKind, ...] = ()
    atomic_admission: bool = True
    atomic_restore: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("checkpoint group name must not be empty")
        if not self.components:
            raise ValueError("checkpoint group must contain at least one state component")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("checkpoint group state component names must be unique")
        if any(component.lifecycle != "checkpointable" for component in self.components):
            raise ValueError("checkpoint groups may only contain checkpointable state")
        if self.policy.storage_form == "token_history":
            raise ValueError("checkpoint groups require snapshot or none persistence")

    @property
    def nominal_bytes_per_checkpoint(self) -> int:
        return sum(item.nominal_bytes_per_checkpoint() for item in self.components)

    @property
    def serialized_bytes_per_checkpoint(self) -> int:
        return sum(item.serialized_bytes_per_checkpoint() for item in self.components)


@dataclass(frozen=True)
class ExecutionPhase:
    """A bounded, ordered piece of model work; not a general task graph."""

    name: str
    token_start: int
    token_end: int
    layer_start: int
    layer_end: int
    read_components: tuple[str, ...] = ()
    write_components: tuple[str, ...] = ()
    approximate: bool = False

    def __post_init__(self) -> None:
        if not self.name or not 0 <= self.token_start <= self.token_end:
            raise ValueError("phase requires a name and an ordered token interval")
        if not 0 <= self.layer_start <= self.layer_end:
            raise ValueError("phase requires an ordered layer interval")

    @property
    def tokens(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class PhaseTiming:
    name: str
    compute_us: float
    read_components: tuple[str, ...] = ()
    flops: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or any(not math.isfinite(v) or v < 0 for v in (self.compute_us, self.flops)):
            raise ValueError("phase timing must be named, finite and non-negative")


@dataclass(frozen=True)
class RestoreReadGroup:
    """Disjoint restored objects aggregated by component, charged only once."""

    component: str
    payload_bytes: float
    logical_ops: float = 0.0
    chunked_ios: float | None = None

    def __post_init__(self) -> None:
        values = (self.payload_bytes, self.logical_ops, self.chunked_ios or 0.0)
        if not self.component or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("restore group must be named, finite and non-negative")


def ordered_restore_groups(
    phases: tuple[PhaseTiming, ...], groups: tuple[RestoreReadGroup, ...]
) -> tuple[tuple[int, RestoreReadGroup], ...]:
    first_use: dict[str, int] = {}
    for index, phase in enumerate(phases):
        for component in phase.read_components:
            first_use.setdefault(component, index)
    if len({g.component for g in groups}) != len(groups):
        raise ValueError("restore groups must be disjoint by component")
    result = []
    for group in groups:
        if group.payload_bytes <= 0:
            continue
        if group.component not in first_use:
            raise ValueError(f"restored component has no consumer: {group.component}")
        result.append((first_use[group.component], group))
    return tuple(sorted(result, key=lambda item: (item[0], item[1].component)))


def shared_read_ready_times(
    phases: tuple[PhaseTiming, ...], groups: tuple[RestoreReadGroup, ...], bandwidth_GBps: float
) -> dict[str, float]:
    """One shared read channel per isolated request, earliest consumers first."""
    if not math.isfinite(bandwidth_GBps) or bandwidth_GBps <= 0:
        raise ValueError("read bandwidth must be finite and positive")
    total = 0.0
    ready: dict[str, float] = {}
    for _, group in ordered_restore_groups(phases, groups):
        total += group.payload_bytes
        ready[group.component] = total / (bandwidth_GBps * 1000.0)
    return ready


def phase_chain_time_us(
    phases: tuple[PhaseTiming, ...],
    read_ready_us: Mapping[str, float] | None = None,
    *,
    overhead_us: float = 0.0,
    first_token_us: float = 0.0,
) -> float:
    """None explicitly requests the zero-storage-latency lower bound.

    Fixed overhead is added once outside the chain, retaining the existing
    two-phase PCS timing convention. A supplied completion map must be complete.
    """
    elapsed = 0.0
    for phase in phases:
        ready = 0.0
        if read_ready_us is not None:
            for component in phase.read_components:
                if component not in read_ready_us:
                    raise ValueError(f"missing restore completion: {component}")
                value = read_ready_us[component]
                if not math.isfinite(value) or value < 0:
                    raise ValueError("restore completion must be finite and non-negative")
                ready = max(ready, value)
        elapsed = max(elapsed, ready) + phase.compute_us
    return overhead_us + elapsed + first_token_us


def phase_read_deadlines(
    phases: tuple[PhaseTiming, ...],
    groups: tuple[RestoreReadGroup, ...],
    target_us: float,
    *,
    overhead_us: float = 0.0,
    first_token_us: float = 0.0,
) -> tuple[dict[str, float | None], ...]:
    """Disjoint read groups with deadlines; equal deadlines are coalesced."""
    by_deadline: dict[float, dict[str, float | None]] = {}
    suffix_compute = [0.0] * len(phases)
    running = 0.0
    for i in range(len(phases) - 1, -1, -1):
        running += phases[i].compute_us
        suffix_compute[i] = running
    for index, group in ordered_restore_groups(phases, groups):
        deadline = target_us - overhead_us - first_token_us - suffix_compute[index]
        bucket = by_deadline.setdefault(deadline, {
            "deadline_us": deadline, "payload_bytes": 0.0,
            "logical_ops": 0.0, "chunked_ios": 0.0,
        })
        bucket["payload_bytes"] += group.payload_bytes
        bucket["logical_ops"] += group.logical_ops
        if group.chunked_ios is None:
            bucket["chunked_ios"] = None
        elif bucket["chunked_ios"] is not None:
            bucket["chunked_ios"] += group.chunked_ios
    return tuple(by_deadline[key] for key in sorted(by_deadline))


def phase_bandwidth_threshold(
    phases: tuple[PhaseTiming, ...], groups: tuple[RestoreReadGroup, ...],
    no_cache_us: float, *, mode: str = "dependency", overhead_us: float = 0.0,
    first_token_us: float = 0.0,
) -> tuple[float | None, str]:
    total_bytes = sum(group.payload_bytes for group in groups)
    compute_us = sum(phase.compute_us for phase in phases)
    if mode == "optimistic":
        if total_bytes <= 0:
            return 0.0, "no_restore_required"
        window = compute_us - overhead_us
        return ((total_bytes / (1000.0 * window), "finite") if window > 0
                else (None, "no_overlap_window"))
    if compute_us + overhead_us + first_token_us >= no_cache_us - 1e-9:
        return None, "no_compute_saving"
    if total_bytes <= 0:
        return 0.0, "no_restore_required"
    cumulative = required = 0.0
    for group in phase_read_deadlines(phases, groups, no_cache_us,
                                     overhead_us=overhead_us, first_token_us=first_token_us):
        cumulative += group["payload_bytes"]
        if group["deadline_us"] <= 0:
            return None, "no_replay_slack"
        required = max(required, cumulative / (1000.0 * group["deadline_us"]))
    return required, "finite"


@dataclass(frozen=True)
class RestoreExecutionPlan:
    """Restore bounds plus optional ordered execution phases and state sources."""

    matched_prefix_tokens: int
    recovery_boundary_tokens: int
    replay_end_tokens: int
    execution_scope: str = "full_forward"
    causes: tuple[str, ...] = ()
    restored_objects: tuple[str, ...] = ()
    blocking_checkpoint_bytes: int = 0
    phases: tuple[ExecutionPhase, ...] = ()
    state_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.recovery_boundary_tokens <= self.replay_end_tokens:
            raise ValueError("recovery boundary must be within the replay interval")
        if self.replay_end_tokens > self.matched_prefix_tokens:
            raise ValueError("replay_end_tokens must not exceed matched_prefix_tokens")
        scope = (
            "proven_component_local"
            if self.execution_scope == "component_scoped"
            else self.execution_scope
        )
        object.__setattr__(self, "execution_scope", scope)
        if scope not in {"full_forward", "proven_component_local", "phased"}:
            raise ValueError(
                "execution_scope must be full_forward, proven_component_local or phased"
            )
        if self.blocking_checkpoint_bytes < 0:
            raise ValueError("blocking_checkpoint_bytes must not be negative")

    @property
    def reusable_prefix_tokens(self) -> int:
        return self.matched_prefix_tokens

    @property
    def restore_recompute_tokens(self) -> int:
        return self.replay_end_tokens - self.recovery_boundary_tokens

    @property
    def total_component_recompute_tokens(self) -> int:
        # Full-forward replay executes once from the latest common boundary.
        return self.restore_recompute_tokens


@dataclass(frozen=True)
class SparseAccessPolicy:
    """Sparse restore, locality, storage-layout, and reuse assumptions."""

    restore_mode: str = "full_restore"
    locality_scenario: str = "window_global"
    reuse_scope: str = "request_union"
    storage_layout: str = "all_kv_heads_together"
    query_granularity_tokens: int = 1
    seed: int = 0
    global_blocks: int = 0
    local_blocks: int = 0

    def __post_init__(self) -> None:
        if self.restore_mode not in {"full_restore", "sparse_on_demand"}:
            raise ValueError("restore_mode must be full_restore or sparse_on_demand")
        aliases = {"clustered": "clustered_lower", "dispersed": "dispersed_upper"}
        scenario = aliases.get(self.locality_scenario, self.locality_scenario)
        object.__setattr__(self, "locality_scenario", scenario)
        if scenario not in {
            "clustered_lower",
            "window_global",
            "seeded_uniform",
            "dispersed_upper",
        }:
            raise ValueError(
                "locality_scenario must be clustered_lower, window_global, "
                "seeded_uniform, or dispersed_upper"
            )
        if self.reuse_scope not in {"request_union", "per_query_no_reuse"}:
            raise ValueError("reuse_scope must be request_union or per_query_no_reuse")
        if self.storage_layout not in {
            "all_kv_heads_together",
            "per_gqa_group",
        }:
            raise ValueError(
                "storage_layout must be all_kv_heads_together or per_gqa_group"
            )
        if self.query_granularity_tokens <= 0:
            raise ValueError("query_granularity_tokens must be positive")
        if self.global_blocks < 0 or self.local_blocks < 0:
            raise ValueError("global_blocks and local_blocks must not be negative")

    @property
    def selection_mode(self) -> str:
        return self.locality_scenario

    def selected_block_indices(
        self,
        *,
        logical_blocks: int,
        selected_blocks: int,
        request_key: str = "",
    ) -> tuple[int, ...]:
        if logical_blocks < 0 or selected_blocks < 0:
            raise ValueError("logical_blocks and selected_blocks must not be negative")
        count = min(logical_blocks, selected_blocks)
        if count == 0:
            return ()
        if self.locality_scenario == "clustered_lower":
            return tuple(range(max(logical_blocks - count, 0), logical_blocks))
        if self.locality_scenario == "window_global":
            chosen = set(range(min(self.global_blocks, logical_blocks)))
            local = min(self.local_blocks, logical_blocks)
            chosen.update(range(logical_blocks - local, logical_blocks))
            if len(chosen) < count:
                for index in range(logical_blocks - 1, -1, -1):
                    chosen.add(index)
                    if len(chosen) == count:
                        break
            return tuple(sorted(chosen)[:count])
        if self.locality_scenario == "dispersed_upper":
            if count == 1:
                return (logical_blocks - 1,)
            selected = {
                round(index * (logical_blocks - 1) / (count - 1))
                for index in range(count)
            }
            # Rounding is unique for count <= logical_blocks, but keep the
            # invariant explicit for alternative numeric implementations.
            if len(selected) != count:
                for index in range(logical_blocks):
                    selected.add(index)
                    if len(selected) == count:
                        break
            return tuple(sorted(selected))

        ranked = sorted(
            range(logical_blocks),
            key=lambda index: _sparse_rank(self.seed, request_key, index),
        )
        return tuple(sorted(ranked[:count]))

    def query_multiplier(self, query_tokens: int) -> int:
        if query_tokens < 0:
            raise ValueError("query token count must not be negative")
        queries = ceil_div(query_tokens, self.query_granularity_tokens) if query_tokens else 0
        return queries if self.reuse_scope == "per_query_no_reuse" else int(queries > 0)


def checkpoint_boundaries(
    policy: StatePersistencePolicy,
    *,
    prefix_tokens: int,
    end_tokens: int,
    block_size_tokens: int,
) -> tuple[int, ...]:
    """Return newly materialized state boundaries in ``(prefix, end]``.

    ``block_size_tokens`` is validated but intentionally does not constrain
    checkpoint alignment; exact boundaries are stored as checkpoint sidecars
    on the radix node that owns the corresponding token range.
    """

    if prefix_tokens < 0 or end_tokens < prefix_tokens:
        raise ValueError("checkpoint token interval is invalid")
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    if policy.storage_form in {"none", "token_history"} or end_tokens == prefix_tokens:
        return ()
    if policy.storage_form != "snapshot":
        raise ValueError(f"unsupported checkpoint storage form: {policy.storage_form}")
    if policy.interval_tokens is None:
        return (end_tokens,) if policy.retain_prompt_end else ()
    interval = policy.interval_tokens
    first = prefix_tokens // interval + 1
    last = end_tokens // interval
    result = [index * interval for index in range(first, last + 1)]
    if policy.retain_prompt_end and (not result or result[-1] != end_tokens):
        result.append(end_tokens)
    return tuple(result)


def checkpoint_tokens_for_prefix(
    policy: StatePersistencePolicy,
    *,
    prefix_tokens: int,
    block_size_tokens: int,
) -> tuple[int, ...]:
    return checkpoint_boundaries(
        policy,
        prefix_tokens=0,
        end_tokens=prefix_tokens,
        block_size_tokens=block_size_tokens,
    )


def nearest_checkpoint_token(
    policy: StatePersistencePolicy,
    *,
    target_tokens: int,
    block_size_tokens: int,
) -> int:
    if target_tokens < 0:
        raise ValueError("target_tokens must not be negative")
    boundaries = checkpoint_tokens_for_prefix(
        policy,
        prefix_tokens=target_tokens,
        block_size_tokens=block_size_tokens,
    )
    return boundaries[-1] if boundaries else 0


def checkpoint_count(
    policy: StatePersistencePolicy,
    *,
    prefix_tokens: int,
    block_size_tokens: int,
) -> int:
    return len(
        checkpoint_tokens_for_prefix(
            policy,
            prefix_tokens=prefix_tokens,
            block_size_tokens=block_size_tokens,
        )
    )


def _sparse_rank(seed: int, request_key: str, index: int) -> bytes:
    payload = f"{seed}|{request_key}|{index}".encode("utf-8")
    return hashlib.sha256(payload).digest()
