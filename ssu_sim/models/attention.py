"""Attention-specific cache-state and IO mapping models."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm
from typing import Mapping, Protocol
import warnings

from ssu_sim.models.footprint import (
    ActiveExecutionFootprint,
    CacheFootprintReport,
    CalculatorReferenceFootprint,
    ComponentFootprint,
    PrefixStoreFootprint,
)
from ssu_sim.models.state import (
    BlockSizeConstraints,
    CheckpointGroup,
    ExecutionPhase,
    LayerKind,
    PrefixGranularityPolicy,
    RestoreExecutionPlan,
    SparseAccessPolicy,
    StateComponent,
    StatePersistencePolicy,
    checkpoint_boundaries,
    checkpoint_count,
    checkpoint_tokens_for_prefix,
    nearest_checkpoint_token,
)
from ssu_sim.utils.units import ceil_div


class ModelLike(Protocol):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int


@dataclass(frozen=True)
class PrefixCachePolicy:
    """On-disk policy for state that cannot use ordinary prefix blocks."""

    # ``None`` means that the selected model's own persistence policy applies.
    # The legacy full/periodic/zero modes remain explicit overrides.
    swa_mode: str | None = None
    swa_checkpoint_interval_tokens: int = 2048
    state_policies: Mapping[str, StatePersistencePolicy] = field(default_factory=dict)
    sparse_access_policy: SparseAccessPolicy | None = None
    include_draft_execution: bool = False
    assume_exact_at_resident: bool = False
    encoder_state: str = "unavailable"
    encoder_tail_policy: str = "recompute"

    def __post_init__(self) -> None:
        if self.encoder_state not in {"available", "unavailable"}:
            raise ValueError("encoder_state must be available or unavailable")
        if self.encoder_tail_policy not in {"recompute", "retain"}:
            raise ValueError("encoder_tail_policy must be recompute or retain")
        if self.swa_mode not in {None, "full", "periodic", "zero"}:
            raise ValueError("swa_mode must be model default, full, periodic, or zero")
        if self.swa_checkpoint_interval_tokens <= 0:
            raise ValueError("swa_checkpoint_interval_tokens must be positive")
        if any(not name for name in self.state_policies):
            raise ValueError("state policy group names must not be empty")
        if any(
            not isinstance(value, StatePersistencePolicy)
            for value in self.state_policies.values()
        ):
            raise TypeError("state_policies values must be StatePersistencePolicy")
        object.__setattr__(self, "state_policies", dict(self.state_policies))

    def state_policy(
        self,
        group_name: str,
        default: StatePersistencePolicy,
    ) -> StatePersistencePolicy:
        return self.state_policies.get(group_name, default)

    def swa_state_policy(
        self,
        default: StatePersistencePolicy | None = None,
    ) -> StatePersistencePolicy:
        override = self.state_policies.get("swa") or self.state_policies.get(
            "sliding_window_state"
        )
        if override is not None:
            return override
        if self.swa_mode is None:
            return default or StatePersistencePolicy.periodic(
                self.swa_checkpoint_interval_tokens,
                retain_prompt_end=False,
            )
        if self.swa_mode == "full":
            return StatePersistencePolicy.token_history()
        if self.swa_mode == "periodic":
            return StatePersistencePolicy.periodic(
                self.swa_checkpoint_interval_tokens,
                retain_prompt_end=False,
            )
        return StatePersistencePolicy.none()


@dataclass(frozen=True)
class CacheComponent:
    """One independently sized part of the serialized attention cache."""

    name: str
    category: str
    elements_per_token_per_layer: int
    dtype_bits: int
    scale_group_size: int | None = None
    scale_bytes: int = 0
    layer_count: int | None = None
    compression_ratio: int = 1
    serialized_bytes_per_entry: int | None = None
    nominal_bytes_per_entry: int | None = None
    padding_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cache component name must not be empty")
        if self.category not in {
            "main_kv",
            "latent",
            "rope_key",
            "indexer",
            "full_kv",
            "sliding_kv",
            "mla_kv",
            "compressed_kv",
            "draft_kv",
            "draft_indexer",
        }:
            raise ValueError(f"unsupported cache component category: {self.category}")
        if self.elements_per_token_per_layer <= 0:
            raise ValueError("cache component elements must be positive")
        if self.dtype_bits <= 0:
            raise ValueError("cache component dtype_bits must be positive")
        if self.scale_bytes < 0:
            raise ValueError("scale_bytes must not be negative")
        if (self.scale_group_size is None) != (self.scale_bytes == 0):
            raise ValueError(
                "scale_group_size and a positive scale_bytes must be provided together"
            )
        if self.scale_group_size is not None and self.scale_group_size <= 0:
            raise ValueError("scale_group_size must be positive")
        if self.layer_count is not None and self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if self.serialized_bytes_per_entry is not None and self.serialized_bytes_per_entry <= 0:
            raise ValueError("serialized_bytes_per_entry must be positive")
        if self.nominal_bytes_per_entry is not None and self.nominal_bytes_per_entry <= 0:
            raise ValueError("nominal_bytes_per_entry must be positive")
        if self.padding_bytes < 0:
            raise ValueError("padding_bytes must not be negative")

    def bytes_per_token_per_layer(self) -> int:
        if self.serialized_bytes_per_entry is not None:
            return self.serialized_bytes_per_entry
        payload = ceil_div(self.elements_per_token_per_layer * self.dtype_bits, 8)
        if self.scale_group_size is None:
            return payload + self.padding_bytes
        scales = ceil_div(self.elements_per_token_per_layer, self.scale_group_size)
        return payload + scales * self.scale_bytes + self.padding_bytes

    def nominal_bytes_per_token_per_layer(self) -> int:
        """Tensor payload without serialization scale or padding overhead."""

        if self.nominal_bytes_per_entry is not None:
            return self.nominal_bytes_per_entry
        return ceil_div(self.elements_per_token_per_layer * self.dtype_bits, 8)

    def bytes_per_token(self, model: ModelLike) -> int | float:
        layers = self.layer_count if self.layer_count is not None else model.num_layers
        numerator = layers * self.bytes_per_token_per_layer()
        if numerator % self.compression_ratio == 0:
            return numerator // self.compression_ratio
        return numerator / self.compression_ratio

    def bytes_per_block(self, model: ModelLike, block_size_tokens: int) -> int:
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if block_size_tokens % self.compression_ratio != 0:
            raise ValueError(
                "block_size_tokens must be divisible by component compression_ratio"
            )
        layers = self.layer_count if self.layer_count is not None else model.num_layers
        entries = block_size_tokens // self.compression_ratio
        return layers * entries * self.bytes_per_token_per_layer()

    def report_dict(self, model: ModelLike, block_size_tokens: int) -> dict[str, object]:
        bytes_per_token = self.bytes_per_token(model)
        bytes_per_block = self.bytes_per_block(model, block_size_tokens)
        return {
            "category": self.category,
            "elements_per_token_per_layer": self.elements_per_token_per_layer,
            "elements_per_entry_per_layer": self.elements_per_token_per_layer,
            "dtype_bits": self.dtype_bits,
            "scale_group_size": self.scale_group_size,
            "scale_bytes": self.scale_bytes,
            "layer_count": self.layer_count if self.layer_count is not None else model.num_layers,
            "compression_ratio": self.compression_ratio,
            "serialized_bytes_per_entry": self.bytes_per_token_per_layer(),
            "nominal_bytes_per_entry": self.nominal_bytes_per_token_per_layer(),
            "padding_bytes": self.padding_bytes,
            "bytes_per_token": bytes_per_token,
            "bytes_per_block": bytes_per_block,
            "bytes_per_native_token": bytes_per_token,
            "bytes_per_native_block": bytes_per_block,
        }


@dataclass(frozen=True)
class AttentionIOStream:
    """An IO stream for one cache category.

    ``physical_blocks`` may be fractional for an expected sparse-access trace.
    ``full_block_weight`` and ``last_block_weight`` compactly encode the
    expected number of accesses to each cached block for optional trace output.
    """

    component: str
    logical_blocks: int
    physical_blocks: float
    payload_bytes: float
    io_count: float
    block_bytes: int
    lower_bound_blocks: float
    upper_bound_blocks: float
    full_block_weight: float = 1.0
    last_block_weight: float = 1.0
    access_pattern: str = "sequential"
    trace_token_count: int | None = None
    trace_block_indices: tuple[int, ...] | None = None
    trace_checkpoint_tokens: tuple[int, ...] | None = None
    selection_source: str | None = None
    query_reuse_scope: str | None = None
    consumer_phase: str = "suffix"
    criticality: str = "blocking"

    def __post_init__(self) -> None:
        if self.consumer_phase not in {"replay", "suffix", "optional"}:
            raise ValueError("consumer_phase must be replay, suffix, or optional")
        if self.criticality not in {"blocking", "non_blocking"}:
            raise ValueError("criticality must be blocking or non_blocking")


@dataclass(frozen=True)
class AttentionReadPlan:
    logical_blocks: int
    streams: tuple[AttentionIOStream, ...]
    materialization_us: float = 0.0
    reusable_prefix_tokens: int | None = None
    restore_recompute_tokens: int = 0
    restore_execution_plan: RestoreExecutionPlan | None = None
    # Portion of the legacy time proxy that estimates latent-to-K/V compute.
    # Routing and other separately configured overhead remain in materialization_us.
    compute_materialization_us: float = 0.0

    @property
    def physical_blocks(self) -> float:
        return sum(stream.physical_blocks for stream in self.streams)

    @property
    def payload_bytes(self) -> float:
        return sum(stream.payload_bytes for stream in self.streams)

    @property
    def replay_gate_bytes(self) -> float:
        return sum(
            stream.payload_bytes
            for stream in self.streams
            if stream.criticality == "blocking" and stream.consumer_phase == "replay"
        )

    @property
    def suffix_gate_bytes(self) -> float:
        return sum(
            stream.payload_bytes
            for stream in self.streams
            if stream.criticality == "blocking"
            and stream.consumer_phase in {"replay", "suffix"}
        )

    @property
    def optional_restore_bytes(self) -> float:
        return sum(
            stream.payload_bytes
            for stream in self.streams
            if stream.criticality == "non_blocking" or stream.consumer_phase == "optional"
        )


@dataclass(frozen=True)
class AttentionWritePlan:
    logical_blocks: int
    streams: tuple[AttentionIOStream, ...]

    @property
    def payload_bytes(self) -> float:
        return sum(stream.payload_bytes for stream in self.streams)


@dataclass(frozen=True)
class AttentionCapacityPlan:
    logical_blocks: int
    bytes_by_component: dict[str, int]
    checkpoint_count: int = 0

    @property
    def payload_bytes(self) -> int:
        return sum(self.bytes_by_component.values())


class AttentionProfile(Protocol):
    name: str

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        ...

    def cache_bytes_per_token(self, model: ModelLike) -> int | float:
        ...

    def cache_bytes_per_block(self, model: ModelLike, block_size_tokens: int) -> int:
        ...

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        ...

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionWritePlan:
        ...

    def map_prefix_to_capacity(
        self,
        model: ModelLike,
        prefix_tokens: int,
        block_size_tokens: int,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionCapacityPlan:
        ...

    def cache_footprint(
        self,
        model: ModelLike,
        *,
        seq_len: int,
        block_size_tokens: int,
        sequence_count: int = 1,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> CacheFootprintReport:
        ...

    def block_size_constraints(self) -> BlockSizeConstraints:
        ...


class _ComponentProfileMixin:
    def cache_bytes_per_token(self, model: ModelLike) -> int | float:
        return sum(component.bytes_per_token(model) for component in self.cache_components(model))

    def cache_bytes_per_block(self, model: ModelLike, block_size_tokens: int) -> int:
        return sum(
            component.bytes_per_block(model, block_size_tokens)
            for component in self.cache_components(model)
        )

    def block_size_constraints(self) -> BlockSizeConstraints:
        ratios = tuple(
            component.compression_ratio
            for component in getattr(self, "_constraint_components", ())
        )
        return BlockSizeConstraints(multiple=lcm(*ratios) if ratios else 1)

    def _full_read_plan(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        *,
        materialization_us: float = 0.0,
        compute_materialization_us: float = 0.0,
    ) -> AttentionReadPlan:
        indices, reusable_tokens = _prefix_block_indices(
            hit_tokens,
            block_size_tokens,
            getattr(
                self,
                "prefix_granularity",
                PrefixGranularityPolicy.sealed_padded(),
            ),
        )
        logical_blocks = len(indices)
        streams = _full_streams(
            self.cache_components(model), model, logical_blocks, block_size_tokens, "sequential"
        )
        return AttentionReadPlan(
            logical_blocks,
            streams,
            materialization_us,
            reusable_prefix_tokens=reusable_tokens,
            compute_materialization_us=compute_materialization_us,
        )

    def _full_write_plan(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        *,
        prefix_tokens: int = 0,
    ) -> AttentionWritePlan:
        indices = _write_block_indices(
            prefix_tokens,
            miss_tokens,
            block_size_tokens,
            getattr(
                self,
                "prefix_granularity",
                PrefixGranularityPolicy.sealed_padded(),
            ),
        )
        streams = _component_streams_for_indices(
            self.cache_components(model),
            model,
            indices,
            block_size_tokens,
            access_pattern="sequential",
        )
        return AttentionWritePlan(len(indices), streams)

    def map_prefix_to_capacity(
        self,
        model: ModelLike,
        prefix_tokens: int,
        block_size_tokens: int,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionCapacityPlan:
        indices, _ = _prefix_block_indices(
            prefix_tokens,
            block_size_tokens,
            getattr(
                self,
                "prefix_granularity",
                PrefixGranularityPolicy.sealed_padded(),
            ),
        )
        logical_blocks = len(indices)
        return AttentionCapacityPlan(
            logical_blocks,
            {
                component.name: logical_blocks
                * component.bytes_per_block(model, block_size_tokens)
                for component in self.cache_components(model)
            },
        )

    def cache_footprint(
        self,
        model: ModelLike,
        *,
        seq_len: int,
        block_size_tokens: int,
        sequence_count: int = 1,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> CacheFootprintReport:
        del policy
        return _token_state_footprint(
            model=model,
            token_components=self.cache_components(model),
            checkpoint_groups=(),
            seq_len=seq_len,
            block_size_tokens=block_size_tokens,
            sequence_count=sequence_count,
            prefix_granularity=getattr(
                self,
                "prefix_granularity",
                PrefixGranularityPolicy.sealed_padded(),
            ),
            policy_override=None,
            formula_id=getattr(
                self, "calculator_formula_id", getattr(self, "name", type(self).__name__)
            ),
            calculator_settings=getattr(self, "calculator_settings", {}),
            calculator_snapshot_sha256=getattr(
                self, "calculator_snapshot_sha256", "unverified"
            ),
            calculator_semantic_class=getattr(
                self, "calculator_semantic_class", "token_payload"
            ),
        )


@dataclass(frozen=True)
class DenseKVAttentionProfile(_ComponentProfileMixin):
    """MHA, GQA, or MQA with independently sized key and value states."""

    key_head_dim: int
    value_head_dim: int
    key_dtype_bits: int
    value_dtype_bits: int
    key_scale_group_size: int | None = None
    key_scale_bytes: int = 0
    value_scale_group_size: int | None = None
    value_scale_bytes: int = 0
    draft_layer_count: int = 0
    include_draft_cache: bool = False
    name: str = "gqa"
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.sealed_padded
    )
    calculator_formula_id: str = "standard_gqa"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "token_payload"

    def __post_init__(self) -> None:
        if self.draft_layer_count < 0:
            raise ValueError("draft_layer_count must not be negative")

    def _effective_draft_layer_count(self) -> int:
        if not self.include_draft_cache:
            return 0
        if self.draft_layer_count:
            return self.draft_layer_count
        nextn = int(self.calculator_settings.get("num_nextn_predict_layers", 0) or 0)
        modules = int(self.calculator_settings.get("num_mtp_modules", 0) or 0)
        layers_per_module = int(
            self.calculator_settings.get("mtp_transformer_layers", 1) or 1
        )
        return nextn or modules * layers_per_module

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        self._validate_heads(model)
        components: list[CacheComponent] = [
            CacheComponent(
                "key",
                "main_kv",
                model.num_kv_heads * self.key_head_dim,
                self.key_dtype_bits,
                self.key_scale_group_size,
                self.key_scale_bytes,
            ),
            CacheComponent(
                "value",
                "main_kv",
                model.num_kv_heads * self.value_head_dim,
                self.value_dtype_bits,
                self.value_scale_group_size,
                self.value_scale_bytes,
            ),
        ]
        draft_layers = self._effective_draft_layer_count()
        if draft_layers:
            components.extend(
                (
                    CacheComponent(
                        "draft_key",
                        "draft_kv",
                        model.num_kv_heads * self.key_head_dim,
                        self.key_dtype_bits,
                        self.key_scale_group_size,
                        self.key_scale_bytes,
                        layer_count=draft_layers,
                    ),
                    CacheComponent(
                        "draft_value",
                        "draft_kv",
                        model.num_kv_heads * self.value_head_dim,
                        self.value_dtype_bits,
                        self.value_scale_group_size,
                        self.value_scale_bytes,
                        layer_count=draft_layers,
                    ),
                )
            )
        return tuple(components)

    def _validate_heads(self, model: ModelLike) -> None:
        if model.num_q_heads <= 0 or model.num_kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.name == "mha" and model.num_kv_heads != model.num_q_heads:
            raise ValueError("MHA requires num_kv_heads == num_q_heads")
        if self.name == "gqa" and not (
            1 < model.num_kv_heads < model.num_q_heads
            and model.num_q_heads % model.num_kv_heads == 0
        ):
            raise ValueError(
                "GQA requires 1 < num_kv_heads < num_q_heads and exact head grouping"
            )
        if self.name == "mqa" and model.num_kv_heads != 1:
            raise ValueError("MQA requires num_kv_heads == 1")

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        del request_key
        return self._full_read_plan(model, hit_tokens, block_size_tokens)

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionWritePlan:
        return self._full_write_plan(
            model,
            miss_tokens,
            block_size_tokens,
            prefix_tokens=prefix_tokens,
        )


@dataclass(frozen=True)
class MLAAttentionProfile(_ComponentProfileMixin):
    """DeepSeek-style latent KV plus the decoupled RoPE key state."""

    latent_dim: int
    rope_key_dim: int
    latent_dtype_bits: int
    rope_key_dtype_bits: int
    latent_scale_group_size: int | None = None
    latent_scale_bytes: int = 0
    rope_key_scale_group_size: int | None = None
    rope_key_scale_bytes: int = 0
    materialization_us_per_token: float = 0.0
    draft_layer_count: int = 0
    include_draft_cache: bool = False
    name: str = "mla"
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.sealed_padded
    )
    calculator_formula_id: str = "mla"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "token_payload"

    def __post_init__(self) -> None:
        if self.materialization_us_per_token < 0:
            raise ValueError("materialization_us_per_token must not be negative")
        if self.draft_layer_count < 0:
            raise ValueError("draft_layer_count must not be negative")

    def _effective_draft_layer_count(self) -> int:
        if not self.include_draft_cache:
            return 0
        return self.draft_layer_count or int(
            self.calculator_settings.get("num_nextn_predict_layers", 0) or 0
        )

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        components: list[CacheComponent] = [
            CacheComponent(
                "kv_latent",
                "main_kv",
                self.latent_dim,
                self.latent_dtype_bits,
                self.latent_scale_group_size,
                self.latent_scale_bytes,
            ),
            CacheComponent(
                "rope_key",
                "main_kv",
                self.rope_key_dim,
                self.rope_key_dtype_bits,
                self.rope_key_scale_group_size,
                self.rope_key_scale_bytes,
            ),
        ]
        draft_layers = self._effective_draft_layer_count()
        if draft_layers:
            components.extend(
                (
                    CacheComponent(
                        "draft_kv_latent",
                        "draft_kv",
                        self.latent_dim,
                        self.latent_dtype_bits,
                        self.latent_scale_group_size,
                        self.latent_scale_bytes,
                        layer_count=draft_layers,
                    ),
                    CacheComponent(
                        "draft_rope_key",
                        "draft_kv",
                        self.rope_key_dim,
                        self.rope_key_dtype_bits,
                        self.rope_key_scale_group_size,
                        self.rope_key_scale_bytes,
                        layer_count=draft_layers,
                    ),
                )
            )
        return tuple(components)

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        del request_key
        return self._full_read_plan(
            model,
            hit_tokens,
            block_size_tokens,
            materialization_us=hit_tokens * self.materialization_us_per_token,
            compute_materialization_us=hit_tokens * self.materialization_us_per_token,
        )

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionWritePlan:
        return self._full_write_plan(
            model,
            miss_tokens,
            block_size_tokens,
            prefix_tokens=prefix_tokens,
        )


@dataclass(frozen=True)
class DSAAttentionProfile(_ComponentProfileMixin):
    """DeepSeek Sparse Attention built on MLA plus a lightning indexer."""

    latent_dim: int
    rope_key_dim: int
    index_key_dim: int
    latent_dtype_bits: int
    rope_key_dtype_bits: int
    index_dtype_bits: int
    index_topk: int
    read_mode: str
    block_mapping: str = "window_global"
    window_size: int = 2048
    global_tokens: int = 128
    latent_scale_group_size: int | None = None
    latent_scale_bytes: int = 0
    rope_key_scale_group_size: int | None = None
    rope_key_scale_bytes: int = 0
    index_scale_group_size: int | None = None
    index_scale_bytes: int = 0
    routing_us_per_query: float = 0.0
    indexer_layer_count: int | None = None
    draft_layer_count: int = 0
    draft_indexer_layer_count: int = 0
    include_draft_cache: bool = False
    sparse_access_policy: SparseAccessPolicy | None = None
    name: str = "dsa"
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.sealed_padded
    )
    calculator_formula_id: str = "dsa_mla"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "token_payload"

    def __post_init__(self) -> None:
        if self.index_topk <= 0:
            raise ValueError("DSA index_topk must be positive")
        if self.read_mode not in {"full_restore", "sparse_on_demand"}:
            raise ValueError("DSA read_mode must be full_restore or sparse_on_demand")
        if self.block_mapping != "window_global":
            raise ValueError("DSA block_mapping must be window_global")
        if self.window_size <= 0:
            raise ValueError("DSA window_size must be positive")
        if self.global_tokens < 0:
            raise ValueError("DSA global_tokens must not be negative")
        if self.routing_us_per_query < 0:
            raise ValueError("routing_us_per_query must not be negative")
        if self.indexer_layer_count is not None and self.indexer_layer_count <= 0:
            raise ValueError("DSA indexer_layer_count must be positive when provided")
        if self.draft_layer_count < 0 or self.draft_indexer_layer_count < 0:
            raise ValueError("DSA draft layer counts must not be negative")

    def _effective_draft_layer_count(self) -> int:
        if not self.include_draft_cache:
            return 0
        return self.draft_layer_count or int(
            self.calculator_settings.get("num_nextn_predict_layers", 0) or 0
        )

    def _effective_draft_indexer_layer_count(self) -> int:
        if not self.include_draft_cache:
            return 0
        return self.draft_indexer_layer_count or int(
            self.calculator_settings.get("draft_indexer_layers", 0) or 0
        ) or self._effective_draft_layer_count()

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        components: list[CacheComponent] = [
            CacheComponent(
                "kv_latent",
                "main_kv",
                self.latent_dim,
                self.latent_dtype_bits,
                self.latent_scale_group_size,
                self.latent_scale_bytes,
            ),
            CacheComponent(
                "rope_key",
                "main_kv",
                self.rope_key_dim,
                self.rope_key_dtype_bits,
                self.rope_key_scale_group_size,
                self.rope_key_scale_bytes,
            ),
            CacheComponent(
                "indexer_key",
                "indexer",
                self.index_key_dim,
                self.index_dtype_bits,
                self.index_scale_group_size,
                self.index_scale_bytes,
                layer_count=self.indexer_layer_count,
            ),
        ]
        draft_layers = self._effective_draft_layer_count()
        if draft_layers:
            components.extend(
                (
                    CacheComponent(
                        "draft_kv_latent",
                        "draft_kv",
                        self.latent_dim,
                        self.latent_dtype_bits,
                        self.latent_scale_group_size,
                        self.latent_scale_bytes,
                        layer_count=draft_layers,
                    ),
                    CacheComponent(
                        "draft_rope_key",
                        "draft_kv",
                        self.rope_key_dim,
                        self.rope_key_dtype_bits,
                        self.rope_key_scale_group_size,
                        self.rope_key_scale_bytes,
                        layer_count=draft_layers,
                    ),
                )
            )
        draft_indexer_layers = self._effective_draft_indexer_layer_count()
        if draft_indexer_layers:
            components.append(
                CacheComponent(
                    "draft_indexer_key",
                    "draft_indexer",
                    self.index_key_dim,
                    self.index_dtype_bits,
                    self.index_scale_group_size,
                    self.index_scale_bytes,
                    layer_count=draft_indexer_layers,
                )
            )
        return tuple(components)

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        access_policy = self._effective_sparse_policy(policy, block_size_tokens)
        if access_policy.restore_mode == "full_restore":
            query_count = max(seq_len - hit_tokens, 0) + max(decode_len, 0)
            return self._full_read_plan(
                model,
                hit_tokens,
                block_size_tokens,
                materialization_us=(
                    query_count * self.routing_us_per_query if hit_tokens > 0 else 0.0
                ),
            )
        if hit_tokens <= 0:
            return AttentionReadPlan(0, (), 0.0)
        query_count = max(seq_len - hit_tokens, 0) + max(decode_len, 0)
        logical_blocks = ceil_div(hit_tokens, block_size_tokens)
        if query_count == 0:
            return AttentionReadPlan(logical_blocks, (), 0.0)

        main_block_bytes = _category_block_bytes(
            self.cache_components(model), model, block_size_tokens, "main_kv"
        )
        index_block_bytes = _category_block_bytes(
            self.cache_components(model), model, block_size_tokens, "indexer"
        )
        if access_policy.locality_scenario == "window_global":
            selected_indices = _window_global_sparse_block_indices(
                hit_tokens,
                block_size_tokens,
                self.window_size,
                self.global_tokens,
            )
        else:
            lower = min(logical_blocks, ceil_div(self.index_topk, block_size_tokens))
            upper = min(logical_blocks, self.index_topk)
            selected_count = (
                lower
                if access_policy.locality_scenario == "clustered_lower"
                else upper
            )
            selected_indices = access_policy.selected_block_indices(
                logical_blocks=logical_blocks,
                selected_blocks=selected_count,
                request_key=(
                    f"{request_key}:dsa:{hit_tokens}:{seq_len}:{decode_len}"
                ),
            )
        repetition = access_policy.query_multiplier(query_count)
        trace_indices = selected_indices * repetition
        main_blocks = float(len(trace_indices))
        analytical_weight = (
            len(selected_indices) / logical_blocks if logical_blocks else 0.0
        )
        if access_policy.locality_scenario == "window_global":
            lower_bound = upper_bound = main_blocks
        else:
            lower_bound = float(
                min(logical_blocks, ceil_div(self.index_topk, block_size_tokens))
                * repetition
            )
            upper_bound = float(min(logical_blocks, self.index_topk) * repetition)
        main_stream = AttentionIOStream(
            component="main_kv",
            logical_blocks=logical_blocks,
            physical_blocks=main_blocks,
            payload_bytes=main_blocks * main_block_bytes,
            io_count=main_blocks,
            block_bytes=main_block_bytes,
            lower_bound_blocks=lower_bound,
            upper_bound_blocks=upper_bound,
            full_block_weight=analytical_weight,
            last_block_weight=analytical_weight,
            access_pattern="random",
            trace_block_indices=trace_indices,
            selection_source=access_policy.locality_scenario,
            query_reuse_scope=access_policy.reuse_scope,
        )
        index_ops = float(logical_blocks)
        index_stream = AttentionIOStream(
            component="indexer",
            logical_blocks=logical_blocks,
            physical_blocks=index_ops,
            payload_bytes=index_ops * index_block_bytes,
            io_count=index_ops,
            block_bytes=index_block_bytes,
            lower_bound_blocks=index_ops,
            upper_bound_blocks=index_ops,
            full_block_weight=1.0,
            last_block_weight=1.0,
            access_pattern="sequential",
            trace_block_indices=tuple(range(logical_blocks)),
            selection_source="full_index_scan",
            query_reuse_scope="request_union",
        )
        components = self.cache_components(model)
        draft_kv_components = tuple(
            component for component in components if component.category == "draft_kv"
        )
        draft_indexer_components = tuple(
            component
            for component in components
            if component.category == "draft_indexer"
        )
        draft_streams = _component_streams_for_indices(
            draft_kv_components,
            model,
            trace_indices,
            block_size_tokens,
            access_pattern="random",
            logical_context_blocks=logical_blocks,
            selection_source=access_policy.locality_scenario,
            query_reuse_scope=access_policy.reuse_scope,
        )
        draft_indexer_streams = _component_streams_for_indices(
            draft_indexer_components,
            model,
            tuple(range(logical_blocks)),
            block_size_tokens,
            access_pattern="sequential",
            logical_context_blocks=logical_blocks,
            selection_source="full_index_scan",
            query_reuse_scope="request_union",
        )
        return AttentionReadPlan(
            logical_blocks,
            (main_stream, index_stream, *draft_streams, *draft_indexer_streams),
            query_count * self.routing_us_per_query,
        )

    def _effective_sparse_policy(
        self,
        policy: PrefixCachePolicy | None,
        block_size_tokens: int,
    ) -> SparseAccessPolicy:
        if policy is not None and policy.sparse_access_policy is not None:
            return policy.sparse_access_policy
        if self.sparse_access_policy is not None:
            return self.sparse_access_policy
        return SparseAccessPolicy(
            restore_mode=self.read_mode,
            locality_scenario=self.block_mapping,
            reuse_scope="request_union",
            storage_layout="all_kv_heads_together",
            global_blocks=ceil_div(self.global_tokens, block_size_tokens),
            local_blocks=ceil_div(self.window_size, block_size_tokens),
        )

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionWritePlan:
        return self._full_write_plan(
            model,
            miss_tokens,
            block_size_tokens,
            prefix_tokens=prefix_tokens,
        )


@dataclass(frozen=True)
class DeepSeekV41AttentionProfile(_ComponentProfileMixin):
    """CED/CSA2: four global owners and execution-only local state.

    Global entries describe effective data. Prefix objects use the existing
    sealed/padded block layout and immutable partial-object versions.
    """

    head_dim: int = 512
    index_head_dim: int = 128
    sliding_window: int = 128
    main_dtype_bits: int = 4
    indexer_dtype_bits: int = 4
    swa_dtype_bits: int = 8
    hc_mult: int = 4
    include_draft_cache: bool = False
    draft_layer_count: int = 3
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20
    kv_owner_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    prefix_granularity: PrefixGranularityPolicy = PrefixGranularityPolicy.sealed_padded()
    calculator_formula_id: str = "deepseek_v41_csa2"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "hybrid"
    name: str = field(init=False, default="deepseek_v41")

    def __post_init__(self) -> None:
        for name in ("compress_ratios", "kv_owner_layer_ids", "index_source_layer_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.compress_ratios != (0, 0) + (2,) * 18 + (1,) * 20:
            raise ValueError("V4.1 requires the 40-layer CED compression schedule")
        if self.kv_owner_layer_ids != (2, 8, 14, 20) or self.index_source_layer_ids != (2, 8, 14, 20, 24, 28, 32, 36):
            raise ValueError("V4.1 requires its four KV owners and eight index-producing layers")
        if (self.main_dtype_bits, self.indexer_dtype_bits, self.swa_dtype_bits) not in {(4, 4, 8), (16, 16, 16)}:
            raise ValueError("V4.1 supports released FP4/FP4/FP8 or BF16/BF16/BF16 cache layouts")
        if self.head_dim != 512 or self.index_head_dim != 128 or self.sliding_window != 128 or self.hc_mult != 4:
            raise ValueError("V4.1 uses head_dim=512, index_head_dim=128, window=128 and hc_mult=4")
        if self.draft_layer_count != 3:
            raise ValueError("V4.1 DSpark has three separate draft layers")

    @property
    def main_entry_bytes(self) -> int:
        return self.head_dim * self.main_dtype_bits // 8 + (self.head_dim // 16 if self.main_dtype_bits == 4 else 0)

    @property
    def indexer_entry_bytes(self) -> int:
        return self.index_head_dim * self.indexer_dtype_bits // 8 + (self.index_head_dim // 32 if self.indexer_dtype_bits == 4 else 0)

    @property
    def swa_entry_bytes(self) -> int:
        return self.head_dim * self.swa_dtype_bits // 8 + (self.head_dim // 32 if self.swa_dtype_bits == 8 else 0)

    def block_size_constraints(self) -> BlockSizeConstraints:
        return BlockSizeConstraints(minimum=2, multiple=2)

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        if model.num_layers != 40 or model.num_kv_heads != 1:
            raise ValueError("V4.1 requires 40 backbone layers and one shared KV head")
        components = []
        for owner in self.kv_owner_layer_ids:
            side = "encoder" if owner < 20 else "decoder"
            ratio = 2 if owner < 20 else 1
            for kind, dim, bits, entry in (
                ("main", self.head_dim, self.main_dtype_bits, self.main_entry_bytes),
                ("indexer", self.index_head_dim, self.indexer_dtype_bits, self.indexer_entry_bytes),
            ):
                components.append(CacheComponent(
                    f"{side}_{owner}_{kind}", "compressed_kv" if kind == "main" else "indexer",
                    dim, bits, layer_count=1, compression_ratio=ratio,
                    serialized_bytes_per_entry=entry,
                ))
        return tuple(components)

    def global_bytes(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        return (self.main_entry_bytes + self.indexer_entry_bytes) * (3 * (tokens // 2) + tokens)

    def generated_global_entries(self, prefix_tokens: int, new_tokens: int) -> dict[str, int]:
        if prefix_tokens < 0 or new_tokens < 0:
            raise ValueError("prefix and new token counts must be non-negative")
        return {
            "encoder": 3 * ((prefix_tokens + new_tokens) // 2 - prefix_tokens // 2),
            "decoder": new_tokens,
        }

    def generated_global_bytes(self, prefix_tokens: int, new_tokens: int) -> int:
        entries = self.generated_global_entries(prefix_tokens, new_tokens)
        return sum(entries.values()) * (self.main_entry_bytes + self.indexer_entry_bytes)

    def state_components(self, model: ModelLike) -> tuple[StateComponent, ...]:
        return (
            StateComponent("encoder_swa", "swa_state", self.sliding_window * self.head_dim,
                           self.swa_dtype_bits, 20, lifecycle="active_only",
                           serialized_bytes_per_layer=self.sliding_window * self.swa_entry_bytes),
            StateComponent("decoder_swa", "swa_state", self.sliding_window * self.head_dim,
                           self.swa_dtype_bits, 20, lifecycle="active_only",
                           serialized_bytes_per_layer=self.sliding_window * self.swa_entry_bytes),
            StateComponent("compression_tail", "compressor_state", 2 * 2 * self.head_dim,
                           32, 3, lifecycle="active_only"),
        )

    def recovery_plan(self, model: ModelLike, hit_tokens: int, seq_len: int,
                      policy: PrefixCachePolicy | None = None) -> RestoreExecutionPlan:
        if not 0 <= hit_tokens <= seq_len:
            raise ValueError("V4.1 hit must be within the prompt")
        policy = policy or PrefixCachePolicy()
        old_start = max(0, seq_len - self.sliding_window)
        needs_old = old_start < hit_tokens
        missing_state = hit_tokens > 0 and policy.encoder_state == "unavailable"
        missing_outputs = needs_old and policy.encoder_tail_policy == "recompute"
        replay = min(hit_tokens, self.sliding_window) if missing_state or missing_outputs else 0
        components = self.cache_components(model)
        encoder = tuple(c.name for c in components if c.name.startswith("encoder")) if hit_tokens else ()
        decoder = tuple(c.name for c in components if c.name.startswith("decoder")) if hit_tokens else ()
        writes = ("encoder_swa", "compression_tail") if missing_state else ()
        phases = (
            ExecutionPhase("encoder_restore", hit_tokens - replay, hit_tokens, 0, 20,
                           encoder if replay else (), writes + (("encoder_outputs",) if replay else ()), bool(replay)),
            ExecutionPhase("encoder_suffix", hit_tokens, seq_len, 0, 20,
                           () if replay else encoder,
                           ("encoder_global", "encoder_swa", "compression_tail", "encoder_outputs")),
            ExecutionPhase("decoder_global_projection", hit_tokens, seq_len, 20, 21,
                           (), ("decoder_global",)),
            ExecutionPhase("decoder_tail", max(0, seq_len - self.sliding_window), seq_len, 20, 40,
                           decoder, ("decoder_swa",)),
        )
        causes = tuple(name for name, present in (
            ("encoder_state_missing", missing_state), ("encoder_outputs_missing", missing_outputs)
        ) if present)
        return RestoreExecutionPlan(
            hit_tokens, hit_tokens - replay, hit_tokens, execution_scope="phased", causes=causes,
            phases=phases, state_sources={
                "encoder_state": "cold_start" if not hit_tokens else ("bounded_replay" if missing_state else "assumed_ready"),
                "encoder_outputs": ("suffix" if not needs_old else
                                    "encoder_replay" if replay else "assumed_retained_snapshot"),
                "encoder_tail_policy": policy.encoder_tail_policy,
            },
        )

    def map_hit_to_reads(self, model: ModelLike, hit_tokens: int, block_size_tokens: int,
                        seq_len: int, decode_len: int, policy: PrefixCachePolicy | None = None,
                        request_key: str = "") -> AttentionReadPlan:
        self.block_size_constraints().validate(block_size_tokens)
        if policy and policy.sparse_access_policy and policy.sparse_access_policy.restore_mode != "full_restore":
            raise ValueError("V4.1 prefix IO currently supports full_restore only")
        plan = self.recovery_plan(model, hit_tokens, seq_len, policy)
        indices, _ = _prefix_block_indices(hit_tokens, block_size_tokens, self.prefix_granularity)
        from dataclasses import replace
        streams = tuple(replace(stream, consumer_phase=(
            "replay" if stream.component.startswith("encoder") and plan.restore_recompute_tokens else "suffix"
        )) for stream in self._streams(model, indices, block_size_tokens))
        return AttentionReadPlan(len(indices), streams, reusable_prefix_tokens=hit_tokens,
                                 restore_recompute_tokens=plan.restore_recompute_tokens,
                                 restore_execution_plan=plan)

    def map_miss_to_writes(self, model: ModelLike, miss_tokens: int, block_size_tokens: int,
                          prefix_tokens: int = 0, policy: PrefixCachePolicy | None = None) -> AttentionWritePlan:
        self.block_size_constraints().validate(block_size_tokens)
        indices = _write_block_indices(prefix_tokens, miss_tokens, block_size_tokens, self.prefix_granularity)
        return AttentionWritePlan(len(indices), self._streams(model, indices, block_size_tokens))

    def _streams(self, model, indices, block_size_tokens):
        count = len(indices)
        if not count:
            return ()
        return tuple(AttentionIOStream(
            c.name, count, float(count), count * c.bytes_per_block(model, block_size_tokens),
            float(count), c.bytes_per_block(model, block_size_tokens), float(count), float(count),
            trace_block_indices=indices,
        ) for c in self.cache_components(model))

    def state_footprints(self, model: ModelLike, tokens: int,
                         policy: PrefixCachePolicy | None = None) -> dict[str, ComponentFootprint]:
        policy = policy or PrefixCachePolicy()
        t = min(tokens, self.sliding_window)
        result = {
            side + "_swa": ComponentFootprint(20 * t * self.head_dim * self.swa_dtype_bits // 8,
                20 * t * self.swa_entry_bytes, 20 * self.sliding_window * self.swa_entry_bytes if tokens else 0)
            for side in ("encoder", "decoder")
        }
        tail = 3 * 2 * self.head_dim * 4 * (tokens % 2)
        result["compression_tail"] = ComponentFootprint(tail, tail, 3 * 2 * 2 * self.head_dim * 4 if tokens else 0)
        if policy.encoder_tail_policy == "retain":
            boundary = t * (self.hc_mult * model.d_model * 2 + self.hc_mult * 4)
            result["encoder_outputs"] = ComponentFootprint(boundary, boundary, boundary)
        if self.include_draft_cache:
            payload = self.draft_layer_count * t * self.swa_entry_bytes
            result["draft_swa"] = ComponentFootprint(
                self.draft_layer_count * t * self.head_dim * self.swa_dtype_bits // 8,
                payload, self.draft_layer_count * self.sliding_window * self.swa_entry_bytes if tokens else 0)
        return result

    def resident_state_bytes(self, model: ModelLike, seq_len: int) -> int:
        return self.global_bytes(seq_len) + sum(int(c.serialized_bytes) for c in self.state_footprints(model, seq_len).values())

    def cache_layout(self, model: ModelLike, block_size_tokens: int) -> dict[str, object]:
        self.block_size_constraints().validate(block_size_tokens)
        return {"type": "deepseek_v41_csa2", "global_owner_layers": self.kv_owner_layer_ids,
                "index_source_layers": self.index_source_layer_ids,
                "main_entry_bytes": self.main_entry_bytes, "indexer_entry_bytes": self.indexer_entry_bytes,
                "swa_entry_bytes": self.swa_entry_bytes,
                "components": {c.name: c.report_dict(model, block_size_tokens) for c in self.cache_components(model)}}

    def cache_footprint(self, model: ModelLike, *, seq_len: int, block_size_tokens: int,
                        sequence_count: int = 1, policy=None) -> CacheFootprintReport:
        from ssu_sim.models.calculator_reference import calculate_reference_footprint
        self.block_size_constraints().validate(block_size_tokens)
        policy = policy if isinstance(policy, PrefixCachePolicy) else PrefixCachePolicy()
        active = {}
        prefix = {}
        capacity = self.map_prefix_to_capacity(model, seq_len, block_size_tokens, policy)
        for c in self.cache_components(model):
            entries = seq_len // c.compression_ratio
            nominal = entries * c.nominal_bytes_per_token_per_layer() * sequence_count
            serialized = entries * c.bytes_per_token_per_layer() * sequence_count
            active[c.name] = ComponentFootprint(nominal, serialized, serialized)
            stored = capacity.bytes_by_component[c.name] * sequence_count
            prefix[c.name] = ComponentFootprint(nominal, stored, stored)
        for name, value in self.state_footprints(model, seq_len, policy).items():
            active[name] = ComponentFootprint(value.nominal_payload_bytes * sequence_count,
                int(value.serialized_bytes) * sequence_count, value.reserved_bytes * sequence_count)
        reference = calculate_reference_footprint(model, tokens=seq_len, sequence_count=sequence_count,
            precision_bytes=0.5, indexer_precision_bytes=0.5, include_state=True,
            include_draft=self.include_draft_cache, encoder_tail_policy=policy.encoder_tail_policy)
        return CacheFootprintReport(reference,
            ActiveExecutionFootprint(sum(x.nominal_payload_bytes for x in active.values()),
                sum(int(x.serialized_bytes) for x in active.values()), sum(x.reserved_bytes for x in active.values()), active),
            PrefixStoreFootprint(sum(x.nominal_payload_bytes for x in prefix.values()),
                sum(int(x.serialized_bytes) for x in prefix.values()), capacity.logical_blocks * len(prefix) * sequence_count,
                0, prefix))


@dataclass(frozen=True)
class DeepSeekV4HybridAttentionProfile(_ComponentProfileMixin):
    """Official DeepSeek-V4 hybrid SWA/CSA/HCA cache serialization.

    External schedules accept the website values ``0/4/128`` and the legacy
    repository sentinel ``1`` for SWA. Internally every layer is represented
    by :class:`LayerKind`; numeric ratios remain only as a compatibility view
    for the compute/report adapters.
    """

    variant: str
    compress_ratios: tuple[int | LayerKind, ...]
    main_entry_bytes: int = 584
    indexer_entry_bytes: int = 66
    main_nope_elements: int = 448
    main_rope_elements: int = 64
    main_scale_bytes: int = 7
    main_padding_bytes: int = 1
    indexer_elements: int = 128
    index_topk: int = 1024
    sliding_window: int = 128
    native_block_size_tokens: int = 256
    draft_compress_ratios: tuple[int | LayerKind, ...] = ()
    include_draft_cache: bool = False
    swa_persistence: StatePersistencePolicy = field(
        default_factory=lambda: StatePersistencePolicy.periodic(
            2048, retain_prompt_end=False
        )
    )
    swa_persistence_origin: str = "engine_fallback"
    calculator_formula_id: str = "deepseek_v4_hybrid"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "hybrid"
    name: str = "deepseek_v4"
    layer_kinds: tuple[LayerKind, ...] = field(init=False, repr=False)
    draft_layer_kinds: tuple[LayerKind, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = self.variant.lower()
        if normalized not in {"pro", "flash"}:
            raise ValueError("DeepSeek V4 variant must be pro or flash")
        object.__setattr__(self, "variant", normalized)
        if not self.compress_ratios:
            raise ValueError("DeepSeek V4 compress_ratios must not be empty")
        kinds: list[LayerKind] = []
        legacy_swa = False
        for value in self.compress_ratios:
            if isinstance(value, LayerKind):
                if value not in {LayerKind.SWA, LayerKind.CSA, LayerKind.HCA}:
                    raise ValueError("DeepSeek V4 layer kinds must be SWA, CSA, or HCA")
                kinds.append(value)
                continue
            if value in {0, 1}:
                kinds.append(LayerKind.SWA)
                legacy_swa = legacy_swa or value == 1
            elif value == 4:
                kinds.append(LayerKind.CSA)
            elif value == 128:
                kinds.append(LayerKind.HCA)
            else:
                raise ValueError(
                    "DeepSeek V4 compress_ratios may contain only 0/1, 4, and 128"
                )
        if legacy_swa:
            warnings.warn(
                "DeepSeek V4 SWA sentinel 1 is deprecated; use website sentinel 0",
                DeprecationWarning,
                stacklevel=2,
            )
        object.__setattr__(self, "layer_kinds", tuple(kinds))
        object.__setattr__(
            self,
            "compress_ratios",
            tuple(
                0 if kind is LayerKind.SWA else 4 if kind is LayerKind.CSA else 128
                for kind in kinds
            ),
        )
        raw_draft_ratios: tuple[int | LayerKind, ...] = self.draft_compress_ratios
        if not raw_draft_ratios:
            calculator_ratios = self.calculator_settings.get("compress_ratios", ())
            if isinstance(calculator_ratios, (list, tuple)):
                raw_draft_ratios = tuple(calculator_ratios[len(kinds) :])
        draft_kinds: list[LayerKind] = []
        draft_legacy_swa = False
        for value in raw_draft_ratios:
            if isinstance(value, LayerKind):
                if value not in {LayerKind.SWA, LayerKind.CSA, LayerKind.HCA}:
                    raise ValueError("DeepSeek V4 draft layer kinds must be SWA, CSA, or HCA")
                draft_kinds.append(value)
            elif value in {0, 1}:
                draft_kinds.append(LayerKind.SWA)
                draft_legacy_swa = draft_legacy_swa or value == 1
            elif value == 4:
                draft_kinds.append(LayerKind.CSA)
            elif value == 128:
                draft_kinds.append(LayerKind.HCA)
            else:
                raise ValueError(
                    "DeepSeek V4 draft_compress_ratios may contain only 0/1, 4, and 128"
                )
        if draft_legacy_swa:
            warnings.warn(
                "DeepSeek V4 draft SWA sentinel 1 is deprecated; use website sentinel 0",
                DeprecationWarning,
                stacklevel=2,
            )
        object.__setattr__(self, "draft_layer_kinds", tuple(draft_kinds))
        object.__setattr__(
            self,
            "draft_compress_ratios",
            tuple(
                0 if kind is LayerKind.SWA else 4 if kind is LayerKind.CSA else 128
                for kind in draft_kinds
            ),
        )
        for value, label in (
            (self.main_entry_bytes, "main_entry_bytes"),
            (self.indexer_entry_bytes, "indexer_entry_bytes"),
            (self.main_nope_elements, "main_nope_elements"),
            (self.main_rope_elements, "main_rope_elements"),
            (self.indexer_elements, "indexer_elements"),
            (self.index_topk, "index_topk"),
            (self.sliding_window, "sliding_window"),
            (self.native_block_size_tokens, "native_block_size_tokens"),
        ):
            if value <= 0:
                raise ValueError(f"DeepSeek V4 {label} must be positive")
        if self.main_scale_bytes < 0 or self.main_padding_bytes < 0:
            raise ValueError("DeepSeek V4 scale and padding bytes must not be negative")
        if (
            self.swa_persistence.storage_form == "snapshot"
            and self.swa_persistence.interval_tokens is not None
            and self.swa_persistence.interval_tokens % self.sliding_window
        ):
            raise ValueError(
                "DeepSeek V4 SWA checkpoint interval must be a multiple of sliding_window"
            )
        if self.native_block_size_tokens % self.sliding_window != 0:
            raise ValueError("DeepSeek V4 native block must be divisible by sliding_window")

    @property
    def csa_layer_count(self) -> int:
        return self.layer_kinds.count(LayerKind.CSA)

    @property
    def hca_layer_count(self) -> int:
        return self.layer_kinds.count(LayerKind.HCA)

    @property
    def pure_swa_layer_count(self) -> int:
        return self.layer_kinds.count(LayerKind.SWA)

    @property
    def draft_layer_count(self) -> int:
        return len(self.draft_layer_kinds) if self.include_draft_cache else 0

    @property
    def draft_csa_layer_count(self) -> int:
        return (
            self.draft_layer_kinds.count(LayerKind.CSA)
            if self.include_draft_cache
            else 0
        )

    @property
    def draft_hca_layer_count(self) -> int:
        return (
            self.draft_layer_kinds.count(LayerKind.HCA)
            if self.include_draft_cache
            else 0
        )

    @property
    def draft_swa_layer_count(self) -> int:
        # Every V4 draft layer retains the live SWA history in addition to any
        # CSA/HCA compressed representation, mirroring the main schedule.
        return self.draft_layer_count

    def block_size_constraints(self) -> BlockSizeConstraints:
        return BlockSizeConstraints(
            minimum=self.sliding_window,
            multiple=self.sliding_window,
            native=self.native_block_size_tokens,
        )

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        self._validate_model(model)
        components: list[CacheComponent] = []
        if self.csa_layer_count:
            components.extend(
                (
                    CacheComponent(
                        "csa_main",
                        "main_kv",
                        self.main_nope_elements + self.main_rope_elements,
                        8,
                        layer_count=self.csa_layer_count,
                        compression_ratio=4,
                        serialized_bytes_per_entry=self.main_entry_bytes,
                        nominal_bytes_per_entry=(
                            self.main_nope_elements + 2 * self.main_rope_elements
                        ),
                        padding_bytes=self.main_padding_bytes,
                    ),
                    CacheComponent(
                        "csa_indexer",
                        "indexer",
                        self.indexer_elements,
                        4,
                        layer_count=self.csa_layer_count,
                        compression_ratio=4,
                        serialized_bytes_per_entry=self.indexer_entry_bytes,
                    ),
                )
            )
        if self.hca_layer_count:
            components.append(
                CacheComponent(
                    "hca_main",
                    "main_kv",
                    self.main_nope_elements + self.main_rope_elements,
                    8,
                    layer_count=self.hca_layer_count,
                    compression_ratio=128,
                    serialized_bytes_per_entry=self.main_entry_bytes,
                    nominal_bytes_per_entry=(
                        self.main_nope_elements + 2 * self.main_rope_elements
                    ),
                    padding_bytes=self.main_padding_bytes,
                )
            )
        if self.draft_csa_layer_count:
            components.extend(
                (
                    CacheComponent(
                        "draft_csa_main",
                        "draft_kv",
                        self.main_nope_elements + self.main_rope_elements,
                        8,
                        layer_count=self.draft_csa_layer_count,
                        compression_ratio=4,
                        serialized_bytes_per_entry=self.main_entry_bytes,
                        nominal_bytes_per_entry=(
                            self.main_nope_elements + 2 * self.main_rope_elements
                        ),
                        padding_bytes=self.main_padding_bytes,
                    ),
                    CacheComponent(
                        "draft_csa_indexer",
                        "draft_indexer",
                        self.indexer_elements,
                        4,
                        layer_count=self.draft_csa_layer_count,
                        compression_ratio=4,
                        serialized_bytes_per_entry=self.indexer_entry_bytes,
                    ),
                )
            )
        if self.draft_hca_layer_count:
            components.append(
                CacheComponent(
                    "draft_hca_main",
                    "draft_kv",
                    self.main_nope_elements + self.main_rope_elements,
                    8,
                    layer_count=self.draft_hca_layer_count,
                    compression_ratio=128,
                    serialized_bytes_per_entry=self.main_entry_bytes,
                    nominal_bytes_per_entry=(
                        self.main_nope_elements + 2 * self.main_rope_elements
                    ),
                    padding_bytes=self.main_padding_bytes,
                )
            )
        return tuple(components)

    def effective_checkpoint_groups(
        self,
        model: ModelLike,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> tuple[CheckpointGroup, ...]:
        """Expose periodic/terminal SWA snapshots as replay sidecar objects."""

        self._validate_model(model)
        if isinstance(policy, PrefixCachePolicy):
            state_policy = policy.swa_state_policy(self.swa_persistence)
        elif isinstance(policy, StatePersistencePolicy):
            state_policy = policy
        else:
            state_policy = self.swa_persistence
        if state_policy.storage_form == "token_history":
            return ()
        components = [
            StateComponent(
                "swa_window",
                "swa_state",
                self.sliding_window
                * (self.main_nope_elements + 2 * self.main_rope_elements),
                8,
                model.num_layers,
                serialized_bytes_per_layer=(
                    self.sliding_window * self.main_entry_bytes
                ),
            )
        ]
        if self.draft_layer_count:
            components.append(
                StateComponent(
                    "draft_swa_window",
                    "swa_state",
                    self.sliding_window
                    * (self.main_nope_elements + 2 * self.main_rope_elements),
                    8,
                    self.draft_layer_count,
                    serialized_bytes_per_layer=(
                        self.sliding_window * self.main_entry_bytes
                    ),
                )
            )
        return (
            CheckpointGroup(
                "swa",
                tuple(components),
                state_policy,
                (LayerKind.SWA,),
            ),
        )

    def checkpoint_groups(self, model: ModelLike) -> tuple[CheckpointGroup, ...]:
        return self.effective_checkpoint_groups(model)

    def state_components(self, model: ModelLike) -> tuple[StateComponent, ...]:
        policy = StatePersistencePolicy.terminal()
        return self.effective_checkpoint_groups(model, policy)[0].components

    def cache_bytes_per_block(self, model: ModelLike, block_size_tokens: int) -> int:
        self._validate_block_size(block_size_tokens)
        return super().cache_bytes_per_block(model, block_size_tokens)

    def cache_layout(self, model: ModelLike, block_size_tokens: int) -> dict[str, object]:
        self._validate_block_size(block_size_tokens)
        self._validate_model(model)
        components = {
            component.name: component.report_dict(model, block_size_tokens)
            for component in self.cache_components(model)
        }
        components["swa"] = {
            "category": "main_kv",
            "layer_count": model.num_layers,
            "compression_ratio": 1,
            "sliding_window_tokens": self.sliding_window,
            "serialized_bytes_per_entry": self.main_entry_bytes,
            "storage": "policy_dependent",
        }
        if self.draft_layer_count:
            components["draft_swa"] = {
                "category": "draft_kv",
                "layer_count": self.draft_layer_count,
                "compression_ratio": 1,
                "sliding_window_tokens": self.sliding_window,
                "serialized_bytes_per_entry": self.main_entry_bytes,
                "storage": "policy_dependent",
            }
        return {
            "type": "deepseek_v4_hybrid",
            "variant": self.variant,
            "native_block_size_tokens": self.native_block_size_tokens,
            "compress_ratios": list(self.compress_ratios),
            "layer_kinds": [kind.value for kind in self.layer_kinds],
            "layer_counts": {
                "swa": self.pure_swa_layer_count,
                "csa": self.csa_layer_count,
                "hca": self.hca_layer_count,
            },
            "include_draft_cache": self.include_draft_cache,
            "draft_compress_ratios": list(self.draft_compress_ratios),
            "draft_layer_kinds": [kind.value for kind in self.draft_layer_kinds],
            "draft_layer_counts": {
                "swa": (
                    self.draft_layer_kinds.count(LayerKind.SWA)
                    if self.include_draft_cache
                    else 0
                ),
                "csa": self.draft_csa_layer_count,
                "hca": self.draft_hca_layer_count,
                "live_swa_state": self.draft_swa_layer_count,
            },
            "main_entry": {
                "nope_fp8_bytes": self.main_nope_elements,
                "rope_bf16_elements": self.main_rope_elements,
                "scale_bytes": self.main_scale_bytes,
                "padding_bytes": self.main_padding_bytes,
                "serialized_bytes": self.main_entry_bytes,
            },
            "csa_indexer_entry": {
                "mxfp4_elements": self.indexer_elements,
                "topk": self.index_topk,
                "serialized_bytes": self.indexer_entry_bytes,
                "allocator_page_bytes_excluded": 132,
            },
            "components": components,
        }

    def resident_state_bytes(self, model: ModelLike, seq_len: int) -> int:
        """Accelerator-resident compressed state plus the live SWA window."""

        if seq_len < 0:
            raise ValueError("seq_len must not be negative")
        self._validate_model(model)
        compressed = 0
        for component in self.cache_components(model):
            entries = ceil_div(seq_len, component.compression_ratio) if seq_len else 0
            layers = component.layer_count or model.num_layers
            compressed += entries * layers * component.bytes_per_token_per_layer()
        swa_layers = model.num_layers + self.draft_layer_count
        swa = min(seq_len, self.sliding_window) * swa_layers * self.main_entry_bytes
        return compressed + swa

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        del seq_len, decode_len, request_key
        self._validate_request(hit_tokens, block_size_tokens)
        policy = policy or PrefixCachePolicy()
        self._validate_policy(policy)
        state_policy = policy.swa_state_policy(self.swa_persistence)
        reusable_tokens = (hit_tokens // block_size_tokens) * block_size_tokens
        logical_blocks = reusable_tokens // block_size_tokens
        streams = list(
            _component_streams(
                self.cache_components(model),
                model,
                logical_blocks,
                block_size_tokens,
                access_pattern="sequential",
            )
        )

        checkpoint_tokens = 0
        if state_policy.storage_form == "token_history" and reusable_tokens > 0:
            checkpoint_tokens = reusable_tokens
            restored_tokens = min(self.sliding_window, reusable_tokens)
            streams.append(
                _single_extent_stream(
                    "swa",
                    restored_tokens * model.num_layers * self.main_entry_bytes,
                    trace_block_index=max(ceil_div(reusable_tokens, block_size_tokens) - 1, 0),
                    trace_token_count=reusable_tokens,
                )
            )
            if self.draft_layer_count:
                streams.append(
                    _single_extent_stream(
                        "draft_swa",
                        restored_tokens
                        * self.draft_layer_count
                        * self.main_entry_bytes,
                        trace_block_index=max(
                            ceil_div(reusable_tokens, block_size_tokens) - 1, 0
                        ),
                        trace_token_count=reusable_tokens,
                    )
                )
        elif state_policy.storage_form == "snapshot" and reusable_tokens > 0:
            checkpoint_tokens = nearest_checkpoint_token(
                state_policy,
                target_tokens=reusable_tokens,
                block_size_tokens=block_size_tokens,
            )
            if checkpoint_tokens > 0:
                snapshot_bytes = (
                    self.sliding_window * model.num_layers * self.main_entry_bytes
                )
                streams.append(
                    _single_extent_stream(
                        "swa",
                        snapshot_bytes,
                        trace_block_index=ceil_div(
                            checkpoint_tokens, block_size_tokens
                        )
                        - 1,
                    )
                )
                if self.draft_layer_count:
                    streams.append(
                        _single_extent_stream(
                            "draft_swa",
                            self.sliding_window
                            * self.draft_layer_count
                            * self.main_entry_bytes,
                            trace_block_index=ceil_div(
                                checkpoint_tokens, block_size_tokens
                            )
                            - 1,
                        )
                    )

        recovery_boundary = min(reusable_tokens, checkpoint_tokens)
        restore_plan = RestoreExecutionPlan(
            matched_prefix_tokens=hit_tokens,
            recovery_boundary_tokens=recovery_boundary,
            replay_end_tokens=hit_tokens,
            causes=(
                ()
                if recovery_boundary == hit_tokens
                else (
                    "swa_state_not_persisted"
                    if state_policy.storage_form == "none"
                    else "checkpoint_or_token_tail",
                )
            ),
            restored_objects=(
                (("swa", "draft_swa") if self.draft_layer_count else ("swa",))
                if checkpoint_tokens > 0
                else ()
            ),
            blocking_checkpoint_bytes=sum(
                int(stream.payload_bytes)
                for stream in streams
                if stream.component in {"swa", "draft_swa"}
            ),
        )
        return AttentionReadPlan(
            logical_blocks=logical_blocks,
            streams=tuple(streams),
            reusable_prefix_tokens=reusable_tokens,
            restore_recompute_tokens=restore_plan.restore_recompute_tokens,
            restore_execution_plan=restore_plan,
        )

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionWritePlan:
        self._validate_request(miss_tokens, block_size_tokens)
        if prefix_tokens < 0:
            raise ValueError("prefix_tokens must not be negative")
        policy = policy or PrefixCachePolicy()
        self._validate_policy(policy)
        state_policy = policy.swa_state_policy(self.swa_persistence)
        old_blocks = prefix_tokens // block_size_tokens
        end_tokens = prefix_tokens + miss_tokens
        new_blocks = end_tokens // block_size_tokens
        completed_blocks = max(new_blocks - old_blocks, 0)
        indices = tuple(range(old_blocks, new_blocks))
        streams = list(
            _component_streams(
                self.cache_components(model),
                model,
                completed_blocks,
                block_size_tokens,
                access_pattern="sequential",
                trace_block_indices=indices,
            )
        )

        if state_policy.storage_form == "token_history" and miss_tokens:
            streams.extend(
                _token_range_streams(
                    component="swa",
                    start_token=prefix_tokens,
                    end_token=end_tokens,
                    block_size_tokens=block_size_tokens,
                    bytes_per_token=model.num_layers * self.main_entry_bytes,
                )
            )
            if self.draft_layer_count:
                streams.extend(
                    _token_range_streams(
                        component="draft_swa",
                        start_token=prefix_tokens,
                        end_token=end_tokens,
                        block_size_tokens=block_size_tokens,
                        bytes_per_token=(
                            self.draft_layer_count * self.main_entry_bytes
                        ),
                    )
                )
        elif state_policy.storage_form == "snapshot":
            boundaries = checkpoint_boundaries(
                state_policy,
                prefix_tokens=prefix_tokens,
                end_tokens=end_tokens,
                block_size_tokens=block_size_tokens,
            )
            if boundaries:
                checkpoint_indices = tuple(
                    max(ceil_div(checkpoint, block_size_tokens) - 1, 0)
                    for checkpoint in boundaries
                )
                streams.append(
                    _uniform_stream(
                        component="swa",
                        block_count=len(boundaries),
                        block_bytes=(
                            self.sliding_window
                            * model.num_layers
                            * self.main_entry_bytes
                        ),
                        trace_block_indices=checkpoint_indices,
                        trace_checkpoint_tokens=boundaries,
                    )
                )
                if self.draft_layer_count:
                    streams.append(
                        _uniform_stream(
                            component="draft_swa",
                            block_count=len(boundaries),
                            block_bytes=(
                                self.sliding_window
                                * self.draft_layer_count
                                * self.main_entry_bytes
                            ),
                            trace_block_indices=checkpoint_indices,
                            trace_checkpoint_tokens=boundaries,
                        )
                    )
        return AttentionWritePlan(completed_blocks, tuple(streams))

    def map_prefix_to_capacity(
        self,
        model: ModelLike,
        prefix_tokens: int,
        block_size_tokens: int,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionCapacityPlan:
        self._validate_request(prefix_tokens, block_size_tokens)
        policy = policy or PrefixCachePolicy()
        self._validate_policy(policy)
        state_policy = policy.swa_state_policy(self.swa_persistence)
        logical_blocks = prefix_tokens // block_size_tokens
        reusable_tokens = logical_blocks * block_size_tokens
        result = {
            component.name: logical_blocks
            * component.bytes_per_block(model, block_size_tokens)
            for component in self.cache_components(model)
        }
        checkpoint_total = 0
        if state_policy.storage_form == "token_history":
            result["swa"] = prefix_tokens * model.num_layers * self.main_entry_bytes
            if self.draft_layer_count:
                result["draft_swa"] = (
                    prefix_tokens * self.draft_layer_count * self.main_entry_bytes
                )
        elif state_policy.storage_form == "snapshot":
            checkpoint_total = checkpoint_count(
                state_policy,
                prefix_tokens=reusable_tokens,
                block_size_tokens=block_size_tokens,
            )
            result["swa"] = (
                checkpoint_total
                * self.sliding_window
                * model.num_layers
                * self.main_entry_bytes
            )
            if self.draft_layer_count:
                result["draft_swa"] = (
                    checkpoint_total
                    * self.sliding_window
                    * self.draft_layer_count
                    * self.main_entry_bytes
                )
        else:
            result["swa"] = 0
            if self.draft_layer_count:
                result["draft_swa"] = 0
        return AttentionCapacityPlan(logical_blocks, result, checkpoint_total)

    def cache_footprint(
        self,
        model: ModelLike,
        *,
        seq_len: int,
        block_size_tokens: int,
        sequence_count: int = 1,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> CacheFootprintReport:
        _validate_footprint_query(seq_len, block_size_tokens, sequence_count)
        self._validate_block_size(block_size_tokens)
        self._validate_model(model)
        if isinstance(policy, PrefixCachePolicy):
            state_policy = policy.swa_state_policy(self.swa_persistence)
        elif isinstance(policy, StatePersistencePolicy):
            state_policy = policy
        else:
            state_policy = self.swa_persistence

        calculator_components = {
            "csa_main": (
                self.csa_layer_count
                * (seq_len // 4)
                * (self.main_nope_elements + self.main_rope_elements)
                * sequence_count
            ),
            "csa_indexer": (
                self.csa_layer_count
                * (seq_len // 4)
                * ceil_div(self.indexer_elements * 4, 8)
                * sequence_count
            ),
            "hca_main": (
                self.hca_layer_count
                * (seq_len // 128)
                * (self.main_nope_elements + self.main_rope_elements)
                * sequence_count
            ),
            "swa": (
                model.num_layers
                * self.sliding_window
                * (self.main_nope_elements + self.main_rope_elements)
                * sequence_count
                if seq_len
                else 0
            ),
        }
        if self.draft_csa_layer_count:
            calculator_components["draft_csa_main"] = (
                self.draft_csa_layer_count
                * (seq_len // 4)
                * (self.main_nope_elements + self.main_rope_elements)
                * sequence_count
            )
            calculator_components["draft_csa_indexer"] = (
                self.draft_csa_layer_count
                * (seq_len // 4)
                * ceil_div(self.indexer_elements * 4, 8)
                * sequence_count
            )
        if self.draft_hca_layer_count:
            calculator_components["draft_hca_main"] = (
                self.draft_hca_layer_count
                * (seq_len // 128)
                * (self.main_nope_elements + self.main_rope_elements)
                * sequence_count
            )
        if self.draft_swa_layer_count:
            calculator_components["draft_swa"] = (
                self.draft_swa_layer_count
                * self.sliding_window
                * (self.main_nope_elements + self.main_rope_elements)
                * sequence_count
                if seq_len
                else 0
            )
        active_components: dict[str, ComponentFootprint] = {}
        for component in self.cache_components(model):
            layers = component.layer_count or model.num_layers
            entries = ceil_div(seq_len, component.compression_ratio) if seq_len else 0
            nominal = (
                entries
                * layers
                * component.nominal_bytes_per_token_per_layer()
                * sequence_count
            )
            serialized = (
                entries
                * layers
                * component.bytes_per_token_per_layer()
                * sequence_count
            )
            active_components[component.name] = ComponentFootprint(
                nominal, serialized, serialized
            )
        live_swa_tokens = min(seq_len, self.sliding_window)
        active_swa = live_swa_tokens * model.num_layers * self.main_entry_bytes * sequence_count
        reserved_swa = (
            self.sliding_window * model.num_layers * self.main_entry_bytes * sequence_count
            if seq_len
            else 0
        )
        active_components["swa"] = ComponentFootprint(
            live_swa_tokens
            * model.num_layers
            * (self.main_nope_elements + 2 * self.main_rope_elements)
            * sequence_count,
            active_swa,
            reserved_swa,
        )
        if self.draft_swa_layer_count:
            active_draft_swa = (
                live_swa_tokens
                * self.draft_swa_layer_count
                * self.main_entry_bytes
                * sequence_count
            )
            reserved_draft_swa = (
                self.sliding_window
                * self.draft_swa_layer_count
                * self.main_entry_bytes
                * sequence_count
                if seq_len
                else 0
            )
            active_components["draft_swa"] = ComponentFootprint(
                live_swa_tokens
                * self.draft_swa_layer_count
                * (self.main_nope_elements + 2 * self.main_rope_elements)
                * sequence_count,
                active_draft_swa,
                reserved_draft_swa,
            )

        logical_blocks = seq_len // block_size_tokens
        prefix_components: dict[str, ComponentFootprint] = {}
        object_count = 0
        for component in self.cache_components(model):
            serialized = (
                logical_blocks
                * component.bytes_per_block(model, block_size_tokens)
                * sequence_count
            )
            native_tokens = logical_blocks * block_size_tokens
            layers = component.layer_count or model.num_layers
            entries = native_tokens // component.compression_ratio
            nominal = (
                entries
                * layers
                * component.nominal_bytes_per_token_per_layer()
                * sequence_count
            )
            prefix_components[component.name] = ComponentFootprint(
                nominal, serialized, serialized
            )
            object_count += logical_blocks * sequence_count

        checkpoint_total = 0
        if state_policy.storage_form == "token_history":
            serialized = seq_len * model.num_layers * self.main_entry_bytes * sequence_count
            prefix_components["swa"] = ComponentFootprint(
                seq_len
                * model.num_layers
                * (self.main_nope_elements + 2 * self.main_rope_elements)
                * sequence_count,
                serialized,
                serialized,
            )
            object_count += (ceil_div(seq_len, block_size_tokens) if seq_len else 0) * sequence_count
            if self.draft_swa_layer_count:
                draft_serialized = (
                    seq_len
                    * self.draft_swa_layer_count
                    * self.main_entry_bytes
                    * sequence_count
                )
                prefix_components["draft_swa"] = ComponentFootprint(
                    seq_len
                    * self.draft_swa_layer_count
                    * (self.main_nope_elements + 2 * self.main_rope_elements)
                    * sequence_count,
                    draft_serialized,
                    draft_serialized,
                )
                object_count += (
                    ceil_div(seq_len, block_size_tokens) if seq_len else 0
                ) * sequence_count
        elif state_policy.storage_form == "snapshot":
            checkpoint_total = checkpoint_count(
                state_policy,
                prefix_tokens=logical_blocks * block_size_tokens,
                block_size_tokens=block_size_tokens,
            ) * sequence_count
            serialized = (
                checkpoint_total
                * self.sliding_window
                * model.num_layers
                * self.main_entry_bytes
            )
            prefix_components["swa"] = ComponentFootprint(
                checkpoint_total
                * self.sliding_window
                * model.num_layers
                * (self.main_nope_elements + 2 * self.main_rope_elements),
                serialized,
                serialized,
            )
            if self.draft_swa_layer_count:
                draft_serialized = (
                    checkpoint_total
                    * self.sliding_window
                    * self.draft_swa_layer_count
                    * self.main_entry_bytes
                )
                prefix_components["draft_swa"] = ComponentFootprint(
                    checkpoint_total
                    * self.sliding_window
                    * self.draft_swa_layer_count
                    * (self.main_nope_elements + 2 * self.main_rope_elements),
                    draft_serialized,
                    draft_serialized,
                )
            object_count += checkpoint_total
        else:
            prefix_components["swa"] = ComponentFootprint(0, 0, 0)
            if self.draft_swa_layer_count:
                prefix_components["draft_swa"] = ComponentFootprint(0, 0, 0)

        return CacheFootprintReport(
            CalculatorReferenceFootprint(
                sum(calculator_components.values()),
                self.calculator_formula_id,
                {
                    **dict(self.calculator_settings),
                    "seq_len": seq_len,
                    "sequence_count": sequence_count,
                    "main_precision_bits": 8,
                    "indexer_precision_bits": 4,
                },
                self.calculator_snapshot_sha256,
                self.calculator_semantic_class,
                calculator_components,
            ),
            _active_footprint(active_components),
            _prefix_footprint(
                prefix_components,
                object_count=object_count,
                checkpoint_count_value=checkpoint_total,
            ),
        )

    def _validate_model(self, model: ModelLike) -> None:
        if len(self.compress_ratios) != model.num_layers:
            raise ValueError(
                "DeepSeek V4 compress_ratios length must equal model.num_layers"
            )
        if model.num_kv_heads != 1:
            raise ValueError("DeepSeek V4 requires shared-KV MQA with num_kv_heads=1")
        if self.variant == "pro":
            expected = (LayerKind.HCA, LayerKind.HCA) + tuple(
                LayerKind.CSA if layer % 2 == 0 else LayerKind.HCA
                for layer in range(2, 61)
            )
            if model.num_layers != 61 or self.layer_kinds != expected:
                raise ValueError("DeepSeek V4-Pro requires the official 30-CSA/31-HCA schedule")
        else:
            expected = (LayerKind.SWA, LayerKind.SWA) + tuple(
                LayerKind.CSA if layer % 2 == 0 else LayerKind.HCA
                for layer in range(2, 43)
            )
            if model.num_layers != 43 or self.layer_kinds != expected:
                raise ValueError(
                    "DeepSeek V4-Flash requires the official 2-SWA/21-CSA/20-HCA schedule"
                )

    def _validate_block_size(self, block_size_tokens: int) -> None:
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if block_size_tokens % self.sliding_window != 0:
            raise ValueError("DeepSeek V4 block_size_tokens must be a multiple of 128")

    def _validate_policy(self, policy: PrefixCachePolicy) -> None:
        state_policy = policy.swa_state_policy(self.swa_persistence)
        if (
            state_policy.storage_form == "snapshot"
            and state_policy.interval_tokens is not None
            and state_policy.interval_tokens % self.sliding_window != 0
        ):
            raise ValueError("DeepSeek V4 SWA checkpoint interval must be a multiple of 128")

    def _validate_request(self, tokens: int, block_size_tokens: int) -> None:
        if tokens < 0:
            raise ValueError("token count must not be negative")
        self._validate_block_size(block_size_tokens)


class _StatefulTokenProfileMixin(_ComponentProfileMixin):
    """Shared token-cache plus checkpoint-state behavior for hybrid profiles."""

    prefix_granularity: PrefixGranularityPolicy
    calculator_formula_id: str
    calculator_snapshot_sha256: str
    calculator_settings: Mapping[str, object]
    calculator_semantic_class: str

    def checkpoint_groups(self, model: ModelLike) -> tuple[CheckpointGroup, ...]:
        raise NotImplementedError

    def state_components(self, model: ModelLike) -> tuple[StateComponent, ...]:
        return tuple(
            component
            for group in self.checkpoint_groups(model)
            for component in group.components
        )

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        del seq_len, decode_len, request_key
        _validate_token_request(hit_tokens, block_size_tokens)
        block_indices, reusable_tokens = _prefix_block_indices(
            hit_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        streams = list(
            _component_streams_for_indices(
                self.cache_components(model),
                model,
                block_indices,
                block_size_tokens,
                access_pattern="sequential",
            )
        )
        state_streams, restore_plan = _checkpoint_read_plan(
            self.checkpoint_groups(model),
            target_tokens=hit_tokens,
            reusable_prefix_tokens=reusable_tokens,
            block_size_tokens=block_size_tokens,
            policy_override=policy,
        )
        streams.extend(state_streams)
        return AttentionReadPlan(
            logical_blocks=len(block_indices),
            streams=tuple(streams),
            reusable_prefix_tokens=reusable_tokens,
            restore_recompute_tokens=restore_plan.restore_recompute_tokens,
            restore_execution_plan=restore_plan,
        )

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> AttentionWritePlan:
        _validate_token_interval(prefix_tokens, miss_tokens, block_size_tokens)
        indices = _write_block_indices(
            prefix_tokens,
            miss_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        streams = list(
            _component_streams_for_indices(
                self.cache_components(model),
                model,
                indices,
                block_size_tokens,
                access_pattern="sequential",
            )
        )
        streams.extend(
            _checkpoint_write_streams(
                self.checkpoint_groups(model),
                prefix_tokens=prefix_tokens,
                end_tokens=prefix_tokens + miss_tokens,
                block_size_tokens=block_size_tokens,
                policy_override=policy,
            )
        )
        return AttentionWritePlan(len(indices), tuple(streams))

    def map_prefix_to_capacity(
        self,
        model: ModelLike,
        prefix_tokens: int,
        block_size_tokens: int,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> AttentionCapacityPlan:
        _validate_token_request(prefix_tokens, block_size_tokens)
        block_indices, _reusable = _prefix_block_indices(
            prefix_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        result = {
            component.name: len(block_indices)
            * component.bytes_per_block(model, block_size_tokens)
            for component in self.cache_components(model)
        }
        state_bytes, checkpoints = _checkpoint_capacity(
            self.checkpoint_groups(model),
            prefix_tokens=prefix_tokens,
            block_size_tokens=block_size_tokens,
            policy_override=policy,
        )
        result.update(state_bytes)
        return AttentionCapacityPlan(len(block_indices), result, checkpoints)

    def cache_footprint(
        self,
        model: ModelLike,
        *,
        seq_len: int,
        block_size_tokens: int,
        sequence_count: int = 1,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> CacheFootprintReport:
        return _token_state_footprint(
            model=model,
            token_components=self.cache_components(model),
            checkpoint_groups=self.checkpoint_groups(model),
            seq_len=seq_len,
            block_size_tokens=block_size_tokens,
            sequence_count=sequence_count,
            prefix_granularity=self.prefix_granularity,
            policy_override=policy,
            formula_id=self.calculator_formula_id,
            calculator_settings=self.calculator_settings,
            calculator_snapshot_sha256=self.calculator_snapshot_sha256,
            calculator_semantic_class=self.calculator_semantic_class,
        )


@dataclass(frozen=True)
class QwenLinearFullHybridAttentionProfile(_StatefulTokenProfileMixin):
    """Qwen hybrid cache: full-attention KV plus linear recurrent state."""

    full_attention_layers: int
    linear_attention_layers: int
    key_head_dim: int
    value_head_dim: int
    key_dtype_bits: int
    value_dtype_bits: int
    conv_state_elements_per_layer: int
    recurrent_state_elements_per_layer: int
    conv_state_dtype_bits: int = 16
    recurrent_state_dtype_bits: int = 32
    state_persistence: StatePersistencePolicy = field(
        default_factory=StatePersistencePolicy.terminal
    )
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.checkpoint_boundary
    )
    calculator_formula_id: str = "qwen_linear_full_hybrid"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "active_sequence_state"
    name: str = "qwen_linear_full_hybrid"

    def __post_init__(self) -> None:
        _validate_positive_fields(
            self,
            (
                "full_attention_layers",
                "linear_attention_layers",
                "key_head_dim",
                "value_head_dim",
                "key_dtype_bits",
                "value_dtype_bits",
                "conv_state_elements_per_layer",
                "recurrent_state_elements_per_layer",
                "conv_state_dtype_bits",
                "recurrent_state_dtype_bits",
            ),
        )
        if self.state_persistence.storage_form == "token_history":
            raise ValueError(
                "linear recurrent state must use snapshot or none persistence"
            )

    def _validate_model(self, model: ModelLike) -> None:
        if self.full_attention_layers + self.linear_attention_layers != model.num_layers:
            raise ValueError("Qwen hybrid layer counts must equal model.num_layers")
        if model.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        self._validate_model(model)
        return _kv_components(
            prefix="full",
            category="full_kv",
            layer_count=self.full_attention_layers,
            kv_heads=model.num_kv_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            key_dtype_bits=self.key_dtype_bits,
            value_dtype_bits=self.value_dtype_bits,
        )

    def checkpoint_groups(self, model: ModelLike) -> tuple[CheckpointGroup, ...]:
        self._validate_model(model)
        return (
            CheckpointGroup(
                "linear_state",
                (
                    StateComponent(
                        "linear_conv_state",
                        "conv_state",
                        self.conv_state_elements_per_layer,
                        self.conv_state_dtype_bits,
                        self.linear_attention_layers,
                    ),
                    StateComponent(
                        "linear_recurrent_state",
                        "recurrent_state",
                        self.recurrent_state_elements_per_layer,
                        self.recurrent_state_dtype_bits,
                        self.linear_attention_layers,
                    ),
                ),
                self.state_persistence,
                (LayerKind.LINEAR_ATTENTION,),
            ),
        )


@dataclass(frozen=True)
class KDAHybridAttentionProfile(_StatefulTokenProfileMixin):
    """Kimi-style MLA token cache plus KDA convolution/recurrent state."""

    mla_layers: int
    kda_layers: int
    latent_dim: int
    rope_key_dim: int
    latent_dtype_bits: int
    rope_key_dtype_bits: int
    kda_conv_state_elements_per_layer: int
    kda_recurrent_state_elements_per_layer: int
    kda_conv_state_dtype_bits: int = 16
    kda_recurrent_state_dtype_bits: int = 32
    state_persistence: StatePersistencePolicy = field(
        default_factory=StatePersistencePolicy.terminal
    )
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.checkpoint_boundary
    )
    calculator_formula_id: str = "kimi_kda_mla_hybrid"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "hybrid"
    name: str = "kimi_kda_mla_hybrid"

    def __post_init__(self) -> None:
        _validate_positive_fields(
            self,
            (
                "mla_layers",
                "kda_layers",
                "latent_dim",
                "rope_key_dim",
                "latent_dtype_bits",
                "rope_key_dtype_bits",
                "kda_conv_state_elements_per_layer",
                "kda_recurrent_state_elements_per_layer",
                "kda_conv_state_dtype_bits",
                "kda_recurrent_state_dtype_bits",
            ),
        )
        if self.state_persistence.storage_form == "token_history":
            raise ValueError(
                "KDA state must use snapshot or none persistence"
            )

    def _validate_model(self, model: ModelLike) -> None:
        if self.mla_layers + self.kda_layers != model.num_layers:
            raise ValueError("KDA/MLA layer counts must equal model.num_layers")

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        self._validate_model(model)
        return (
            CacheComponent(
                "mla_kv_latent",
                "mla_kv",
                self.latent_dim,
                self.latent_dtype_bits,
                layer_count=self.mla_layers,
            ),
            CacheComponent(
                "mla_rope_key",
                "mla_kv",
                self.rope_key_dim,
                self.rope_key_dtype_bits,
                layer_count=self.mla_layers,
            ),
        )

    def checkpoint_groups(self, model: ModelLike) -> tuple[CheckpointGroup, ...]:
        self._validate_model(model)
        return (
            CheckpointGroup(
                "kda_state",
                (
                    StateComponent(
                        "kda_conv_state",
                        "kda_state",
                        self.kda_conv_state_elements_per_layer,
                        self.kda_conv_state_dtype_bits,
                        self.kda_layers,
                    ),
                    StateComponent(
                        "kda_recurrent_state",
                        "kda_state",
                        self.kda_recurrent_state_elements_per_layer,
                        self.kda_recurrent_state_dtype_bits,
                        self.kda_layers,
                    ),
                ),
                self.state_persistence,
                (LayerKind.KDA,),
            ),
        )


# Explicit alias used by catalog code that includes the MLA part in the class name.
KimiKDAMLAAttentionProfile = KDAHybridAttentionProfile


@dataclass(frozen=True)
class MiniMaxMSAAttentionProfile(_ComponentProfileMixin):
    """MiniMax sparse attention with full index scan and top-k block KV reads."""

    key_head_dim: int
    value_head_dim: int
    key_dtype_bits: int
    value_dtype_bits: int
    indexer_dim: int
    indexer_dtype_bits: int
    topk_blocks: int
    full_attention_layers: int | None = None
    sparse_attention_layers: int | None = None
    sparse_access_policy: SparseAccessPolicy = field(default_factory=SparseAccessPolicy)
    # Compatibility alias for early callers. New configurations should set
    # ``sparse_access_policy.restore_mode`` so all sparse-read assumptions live
    # in one public policy object.
    read_mode: str | None = None
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.sealed_padded
    )
    calculator_formula_id: str = "minimax_msa"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "token_payload"
    name: str = "minimax_msa"

    def __post_init__(self) -> None:
        _validate_positive_fields(
            self,
            (
                "key_head_dim",
                "value_head_dim",
                "key_dtype_bits",
                "value_dtype_bits",
                "indexer_dim",
                "indexer_dtype_bits",
                "topk_blocks",
            ),
        )
        if self.read_mode is not None and self.read_mode not in {
            "full_restore",
            "sparse_on_demand",
        }:
            raise ValueError("MSA read_mode must be full_restore or sparse_on_demand")

    def _layer_counts(self, model: ModelLike) -> tuple[int, int]:
        if self.full_attention_layers is None and self.sparse_attention_layers is None:
            return 0, model.num_layers
        if self.full_attention_layers is None or self.sparse_attention_layers is None:
            raise ValueError(
                "MSA full_attention_layers and sparse_attention_layers must be set together"
            )
        if self.full_attention_layers < 0 or self.sparse_attention_layers <= 0:
            raise ValueError(
                "MSA full_attention_layers must be non-negative and sparse_attention_layers positive"
            )
        if self.full_attention_layers + self.sparse_attention_layers != model.num_layers:
            raise ValueError("MSA attention layer counts must equal model.num_layers")
        return self.full_attention_layers, self.sparse_attention_layers

    def _effective_sparse_policy(
        self, policy: PrefixCachePolicy | None
    ) -> SparseAccessPolicy:
        if policy is not None and policy.sparse_access_policy is not None:
            return policy.sparse_access_policy
        if self.read_mode is None:
            return self.sparse_access_policy
        return SparseAccessPolicy(
            restore_mode=self.read_mode,
            locality_scenario=self.sparse_access_policy.locality_scenario,
            reuse_scope=self.sparse_access_policy.reuse_scope,
            storage_layout=self.sparse_access_policy.storage_layout,
            query_granularity_tokens=(
                self.sparse_access_policy.query_granularity_tokens
            ),
            seed=self.sparse_access_policy.seed,
            global_blocks=self.sparse_access_policy.global_blocks,
            local_blocks=self.sparse_access_policy.local_blocks,
        )

    def _component_groups(
        self, model: ModelLike
    ) -> tuple[
        tuple[CacheComponent, ...],
        tuple[CacheComponent, ...],
        tuple[CacheComponent, ...],
    ]:
        if model.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        full_layers, sparse_layers = self._layer_counts(model)
        full = (
            _kv_components(
                prefix="msa_full",
                category="main_kv",
                layer_count=full_layers,
                kv_heads=model.num_kv_heads,
                key_head_dim=self.key_head_dim,
                value_head_dim=self.value_head_dim,
                key_dtype_bits=self.key_dtype_bits,
                value_dtype_bits=self.value_dtype_bits,
            )
            if full_layers
            else ()
        )
        sparse = _kv_components(
            prefix="msa_sparse",
            category="main_kv",
            layer_count=sparse_layers,
            kv_heads=model.num_kv_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            key_dtype_bits=self.key_dtype_bits,
            value_dtype_bits=self.value_dtype_bits,
        )
        indexer = (
            CacheComponent(
                "msa_indexer",
                "indexer",
                self.indexer_dim,
                self.indexer_dtype_bits,
                layer_count=sparse_layers,
            ),
        )
        return full, sparse, indexer

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        full, sparse, indexer = self._component_groups(model)
        return (*full, *sparse, *indexer)

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        _validate_token_request(hit_tokens, block_size_tokens)
        access_policy = self._effective_sparse_policy(policy)
        block_indices, reusable_tokens = _prefix_block_indices(
            hit_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        if not block_indices:
            return AttentionReadPlan(0, (), reusable_prefix_tokens=reusable_tokens)
        components = self.cache_components(model)
        if access_policy.restore_mode == "full_restore":
            return AttentionReadPlan(
                len(block_indices),
                _component_streams_for_indices(
                    components,
                    model,
                    block_indices,
                    block_size_tokens,
                    access_pattern="sequential",
                ),
                reusable_prefix_tokens=reusable_tokens,
            )

        full_components, sparse_components, index_components = self._component_groups(model)
        query_count = max(seq_len - hit_tokens, 0) + max(decode_len, 0)
        if query_count == 0:
            return AttentionReadPlan(
                len(block_indices),
                (),
                reusable_prefix_tokens=reusable_tokens,
            )
        selected_count = min(self.topk_blocks, len(block_indices))
        query_chunks = ceil_div(
            query_count, access_policy.query_granularity_tokens
        )
        realized_queries = 1 if access_policy.reuse_scope == "request_union" else query_chunks
        group_count = model.num_kv_heads
        selections: dict[tuple[int, int], tuple[int, ...]] = {}
        for query_chunk in range(realized_queries):
            for gqa_group in range(group_count):
                relative = access_policy.selected_block_indices(
                    logical_blocks=len(block_indices),
                    selected_blocks=selected_count,
                    request_key=(
                        f"{request_key}:msa:{hit_tokens}:{seq_len}:{decode_len}:"
                        f"q{query_chunk}:g{gqa_group}"
                    ),
                )
                selections[(query_chunk, gqa_group)] = tuple(
                    block_indices[index] for index in relative
                )
        full_streams = _component_streams_for_indices(
            full_components,
            model,
            block_indices,
            block_size_tokens,
            access_pattern="sequential",
        )
        if access_policy.storage_layout == "all_kv_heads_together":
            if access_policy.reuse_scope == "request_union":
                selected_indices = tuple(
                    sorted(
                        set().union(
                            *(set(indices) for indices in selections.values())
                        )
                    )
                )
            else:
                selected_indices = tuple(
                    index
                    for query_chunk in range(realized_queries)
                    for index in sorted(
                        set().union(
                            *(
                                set(selections[(query_chunk, group)])
                                for group in range(group_count)
                            )
                        )
                    )
                )
            main_streams = _component_streams_for_indices(
                sparse_components,
                model,
                selected_indices,
                block_size_tokens,
                access_pattern="random",
                logical_context_blocks=len(block_indices),
                selection_source=access_policy.selection_mode,
                query_reuse_scope=access_policy.reuse_scope,
            )
        else:
            main_streams = _per_gqa_group_streams(
                sparse_components,
                model,
                selections,
                block_size_tokens,
                group_count=group_count,
                logical_context_blocks=len(block_indices),
                selection_source=access_policy.selection_mode,
                query_reuse_scope=access_policy.reuse_scope,
            )
        index_streams = _component_streams_for_indices(
            index_components,
            model,
            block_indices,
            block_size_tokens,
            access_pattern="sequential",
        )
        return AttentionReadPlan(
            len(block_indices),
            (*full_streams, *main_streams, *index_streams),
            reusable_prefix_tokens=reusable_tokens,
        )

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionWritePlan:
        _validate_token_interval(prefix_tokens, miss_tokens, block_size_tokens)
        indices = _write_block_indices(
            prefix_tokens,
            miss_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        return AttentionWritePlan(
            len(indices),
            _component_streams_for_indices(
                self.cache_components(model),
                model,
                indices,
                block_size_tokens,
                access_pattern="sequential",
            ),
        )

    def map_prefix_to_capacity(
        self,
        model: ModelLike,
        prefix_tokens: int,
        block_size_tokens: int,
        policy: PrefixCachePolicy | None = None,
    ) -> AttentionCapacityPlan:
        _validate_token_request(prefix_tokens, block_size_tokens)
        indices, _ = _prefix_block_indices(
            prefix_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        return AttentionCapacityPlan(
            len(indices),
            {
                item.name: len(indices) * item.bytes_per_block(model, block_size_tokens)
                for item in self.cache_components(model)
            },
        )

    def cache_footprint(
        self,
        model: ModelLike,
        *,
        seq_len: int,
        block_size_tokens: int,
        sequence_count: int = 1,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> CacheFootprintReport:
        return _token_state_footprint(
            model=model,
            token_components=self.cache_components(model),
            checkpoint_groups=(),
            seq_len=seq_len,
            block_size_tokens=block_size_tokens,
            sequence_count=sequence_count,
            prefix_granularity=self.prefix_granularity,
            policy_override=None,
            formula_id=self.calculator_formula_id,
            calculator_settings=self.calculator_settings,
            calculator_snapshot_sha256=self.calculator_snapshot_sha256,
            calculator_semantic_class=self.calculator_semantic_class,
        )


@dataclass(frozen=True)
class MixedFullSlidingGQAAttentionProfile(_ComponentProfileMixin):
    """Layer-mixed GQA with full-context and rolling-window KV states."""

    full_attention_layers: int
    sliding_attention_layers: int
    sliding_window: int
    key_head_dim: int
    value_head_dim: int
    key_dtype_bits: int
    value_dtype_bits: int
    full_num_kv_heads: int | None = None
    sliding_num_kv_heads: int | None = None
    full_key_head_dim: int | None = None
    full_value_head_dim: int | None = None
    sliding_key_head_dim: int | None = None
    sliding_value_head_dim: int | None = None
    sliding_persistence: StatePersistencePolicy = field(
        default_factory=StatePersistencePolicy.token_history
    )
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.sealed_padded
    )
    calculator_formula_id: str = "mixed_full_sliding_gqa"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "resident_window"
    name: str = "mixed_full_sliding_gqa"

    def __post_init__(self) -> None:
        _validate_layer_partition(
            (self.full_attention_layers, self.sliding_attention_layers),
            "mixed full/sliding",
        )
        _validate_positive_fields(
            self,
            (
                "sliding_window",
                "key_head_dim",
                "value_head_dim",
                "key_dtype_bits",
                "value_dtype_bits",
            ),
        )
        for name in (
            "full_num_kv_heads",
            "sliding_num_kv_heads",
            "full_key_head_dim",
            "full_value_head_dim",
            "sliding_key_head_dim",
            "sliding_value_head_dim",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

    def _validate_model(self, model: ModelLike) -> None:
        stored_layers = self.full_attention_layers + self.sliding_attention_layers
        if stored_layers > model.num_layers:
            raise ValueError(
                "mixed full/sliding stored layer counts must not exceed model.num_layers"
            )
        if model.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

    def _full_geometry(self, model: ModelLike) -> tuple[int, int, int]:
        return (
            self.full_num_kv_heads or model.num_kv_heads,
            self.full_key_head_dim or self.key_head_dim,
            self.full_value_head_dim or self.value_head_dim,
        )

    def _sliding_geometry(self, model: ModelLike) -> tuple[int, int, int]:
        return (
            self.sliding_num_kv_heads or model.num_kv_heads,
            self.sliding_key_head_dim or self.key_head_dim,
            self.sliding_value_head_dim or self.value_head_dim,
        )

    def _component_groups(
        self, model: ModelLike
    ) -> tuple[tuple[CacheComponent, ...], tuple[CacheComponent, ...]]:
        self._validate_model(model)
        full_heads, full_key_dim, full_value_dim = self._full_geometry(model)
        sliding_heads, sliding_key_dim, sliding_value_dim = self._sliding_geometry(model)
        full = (
            _kv_components(
                prefix="full",
                category="full_kv",
                layer_count=self.full_attention_layers,
                kv_heads=full_heads,
                key_head_dim=full_key_dim,
                value_head_dim=full_value_dim,
                key_dtype_bits=self.key_dtype_bits,
                value_dtype_bits=self.value_dtype_bits,
            )
            if self.full_attention_layers
            else ()
        )
        sliding = (
            _kv_components(
                prefix="sliding",
                category="sliding_kv",
                layer_count=self.sliding_attention_layers,
                kv_heads=sliding_heads,
                key_head_dim=sliding_key_dim,
                value_head_dim=sliding_value_dim,
                key_dtype_bits=self.key_dtype_bits,
                value_dtype_bits=self.value_dtype_bits,
            )
            if self.sliding_attention_layers
            else ()
        )
        return full, sliding

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        full, sliding = self._component_groups(model)
        return (*full, *sliding)

    def state_components(self, model: ModelLike) -> tuple[StateComponent, ...]:
        group = self._sliding_checkpoint_group(model, self.sliding_persistence)
        return group.components if group is not None else ()

    def effective_checkpoint_groups(
        self,
        model: ModelLike,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> tuple[CheckpointGroup, ...]:
        effective = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        group = self._sliding_checkpoint_group(model, effective)
        return (group,) if group is not None else ()

    def checkpoint_groups(self, model: ModelLike) -> tuple[CheckpointGroup, ...]:
        return self.effective_checkpoint_groups(model)

    def _sliding_checkpoint_group(
        self,
        model: ModelLike,
        policy: StatePersistencePolicy,
    ) -> CheckpointGroup | None:
        self._validate_model(model)
        if policy.storage_form == "token_history":
            return None
        if self.sliding_attention_layers == 0:
            return None
        sliding_heads, sliding_key_dim, sliding_value_dim = self._sliding_geometry(model)
        return CheckpointGroup(
            "sliding_window_state",
            (
                StateComponent(
                    "sliding_key_window",
                    "swa_state",
                    self.sliding_window * sliding_heads * sliding_key_dim,
                    self.key_dtype_bits,
                    self.sliding_attention_layers,
                ),
                StateComponent(
                    "sliding_value_window",
                    "swa_state",
                    self.sliding_window * sliding_heads * sliding_value_dim,
                    self.value_dtype_bits,
                    self.sliding_attention_layers,
                ),
            ),
            policy,
            (LayerKind.SLIDING_ATTENTION,),
        )

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        del seq_len, decode_len, request_key
        _validate_token_request(hit_tokens, block_size_tokens)
        full_components, sliding_components = self._component_groups(model)
        block_indices, reusable_tokens = _prefix_block_indices(
            hit_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        streams = list(
            _component_streams_for_indices(
                full_components,
                model,
                block_indices,
                block_size_tokens,
                access_pattern="sequential",
            )
        )
        effective_policy = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        if effective_policy.storage_form == "token_history":
            window_indices = _window_block_indices(
                reusable_tokens,
                block_size_tokens,
                self.sliding_window,
            )
            streams.extend(
                _component_streams_for_indices(
                    sliding_components,
                    model,
                    window_indices,
                    block_size_tokens,
                    access_pattern="sequential",
                    logical_context_blocks=len(block_indices),
                )
            )
            restore_plan = RestoreExecutionPlan(
                matched_prefix_tokens=hit_tokens,
                recovery_boundary_tokens=reusable_tokens,
                replay_end_tokens=hit_tokens,
                causes=("token_cache_tail",) if reusable_tokens < hit_tokens else (),
                restored_objects=("sliding_token_history",),
            )
        else:
            group = self._sliding_checkpoint_group(model, effective_policy)
            assert group is not None
            state_streams, restore_plan = _checkpoint_read_plan(
                (group,),
                target_tokens=hit_tokens,
                reusable_prefix_tokens=reusable_tokens,
                block_size_tokens=block_size_tokens,
                policy_override=policy,
            )
            streams.extend(state_streams)
        return AttentionReadPlan(
            len(block_indices),
            tuple(streams),
            reusable_prefix_tokens=reusable_tokens,
            restore_recompute_tokens=restore_plan.restore_recompute_tokens,
            restore_execution_plan=restore_plan,
        )

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> AttentionWritePlan:
        _validate_token_interval(prefix_tokens, miss_tokens, block_size_tokens)
        full_components, sliding_components = self._component_groups(model)
        indices = _write_block_indices(
            prefix_tokens,
            miss_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        streams = list(
            _component_streams_for_indices(
                full_components,
                model,
                indices,
                block_size_tokens,
                access_pattern="sequential",
            )
        )
        effective_policy = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        if effective_policy.storage_form == "token_history":
            streams.extend(
                _component_streams_for_indices(
                    sliding_components,
                    model,
                    indices,
                    block_size_tokens,
                    access_pattern="sequential",
                )
            )
        else:
            group = self._sliding_checkpoint_group(model, effective_policy)
            assert group is not None
            streams.extend(
                _checkpoint_write_streams(
                    (group,),
                    prefix_tokens=prefix_tokens,
                    end_tokens=prefix_tokens + miss_tokens,
                    block_size_tokens=block_size_tokens,
                    policy_override=policy,
                )
            )
        return AttentionWritePlan(len(indices), tuple(streams))

    def map_prefix_to_capacity(
        self,
        model: ModelLike,
        prefix_tokens: int,
        block_size_tokens: int,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> AttentionCapacityPlan:
        _validate_token_request(prefix_tokens, block_size_tokens)
        full_components, sliding_components = self._component_groups(model)
        indices, _ = _prefix_block_indices(
            prefix_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        result = {
            item.name: len(indices) * item.bytes_per_block(model, block_size_tokens)
            for item in full_components
        }
        effective_policy = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        checkpoints = 0
        if effective_policy.storage_form == "token_history":
            result.update(
                {
                    item.name: len(indices)
                    * item.bytes_per_block(model, block_size_tokens)
                    for item in sliding_components
                }
            )
        else:
            group = self._sliding_checkpoint_group(model, effective_policy)
            assert group is not None
            state_bytes, checkpoints = _checkpoint_capacity(
                (group,),
                prefix_tokens=prefix_tokens,
                block_size_tokens=block_size_tokens,
                policy_override=policy,
            )
            result.update(state_bytes)
        return AttentionCapacityPlan(len(indices), result, checkpoints)

    def cache_footprint(
        self,
        model: ModelLike,
        *,
        seq_len: int,
        block_size_tokens: int,
        sequence_count: int = 1,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> CacheFootprintReport:
        full, sliding = self._component_groups(model)
        effective_sliding = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        return _windowed_footprint(
            model=model,
            full_components=full,
            sliding_components=sliding,
            sliding_window=self.sliding_window,
            checkpoint_groups=(),
            seq_len=seq_len,
            block_size_tokens=block_size_tokens,
            sequence_count=sequence_count,
            prefix_granularity=self.prefix_granularity,
            sliding_policy=effective_sliding,
            policy_override=policy,
            formula_id=self.calculator_formula_id,
            calculator_settings=self.calculator_settings,
            calculator_snapshot_sha256=self.calculator_snapshot_sha256,
            calculator_semantic_class=self.calculator_semantic_class,
        )


@dataclass(frozen=True)
class InklingHybridAttentionProfile(_ComponentProfileMixin):
    """Inkling global/SWA KV plus SConv state and optional draft KV."""

    global_attention_layers: int
    sliding_attention_layers: int
    sconv_layers: int
    sliding_window: int
    key_head_dim: int
    value_head_dim: int
    key_dtype_bits: int
    value_dtype_bits: int
    sconv_state_elements_per_layer: int
    global_num_kv_heads: int | None = None
    sliding_num_kv_heads: int | None = None
    global_key_head_dim: int | None = None
    global_value_head_dim: int | None = None
    sliding_key_head_dim: int | None = None
    sliding_value_head_dim: int | None = None
    sconv_state_dtype_bits: int = 16
    draft_attention_layers: int = 0
    include_draft_cache: bool = False
    sliding_persistence: StatePersistencePolicy = field(
        default_factory=StatePersistencePolicy.token_history
    )
    sconv_persistence: StatePersistencePolicy = field(
        default_factory=StatePersistencePolicy.terminal
    )
    prefix_granularity: PrefixGranularityPolicy = field(
        default_factory=PrefixGranularityPolicy.checkpoint_boundary
    )
    calculator_formula_id: str = "inkling_hybrid"
    calculator_snapshot_sha256: str = "unverified"
    calculator_settings: Mapping[str, object] = field(default_factory=dict)
    calculator_semantic_class: str = "hybrid"
    name: str = "inkling_hybrid"

    def __post_init__(self) -> None:
        _validate_layer_partition(
            (self.global_attention_layers, self.sliding_attention_layers),
            "Inkling attention",
        )
        if self.sconv_layers <= 0:
            raise ValueError("Inkling sconv_layers must be positive")
        if self.draft_attention_layers < 0:
            raise ValueError("draft_attention_layers must not be negative")
        _validate_positive_fields(
            self,
            (
                "sliding_window",
                "key_head_dim",
                "value_head_dim",
                "key_dtype_bits",
                "value_dtype_bits",
                "sconv_state_elements_per_layer",
                "sconv_state_dtype_bits",
            ),
        )
        for name in (
            "global_num_kv_heads",
            "sliding_num_kv_heads",
            "global_key_head_dim",
            "global_value_head_dim",
            "sliding_key_head_dim",
            "sliding_value_head_dim",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.sconv_persistence.storage_form == "token_history":
            raise ValueError(
                "SConv state must use snapshot or none persistence"
            )

    def _validate_model(self, model: ModelLike) -> None:
        attention_layers = self.global_attention_layers + self.sliding_attention_layers
        if attention_layers > model.num_layers:
            raise ValueError(
                "Inkling global/sliding layer counts must not exceed model.num_layers"
            )
        if self.sconv_layers > model.num_layers:
            raise ValueError("Inkling sconv_layers must not exceed model.num_layers")
        if model.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

    def _global_geometry(self, model: ModelLike) -> tuple[int, int, int]:
        return (
            self.global_num_kv_heads or model.num_kv_heads,
            self.global_key_head_dim or self.key_head_dim,
            self.global_value_head_dim or self.value_head_dim,
        )

    def _sliding_geometry(self, model: ModelLike) -> tuple[int, int, int]:
        return (
            self.sliding_num_kv_heads or model.num_kv_heads,
            self.sliding_key_head_dim or self.key_head_dim,
            self.sliding_value_head_dim or self.value_head_dim,
        )

    def _component_groups(
        self, model: ModelLike
    ) -> tuple[
        tuple[CacheComponent, ...],
        tuple[CacheComponent, ...],
        tuple[CacheComponent, ...],
    ]:
        self._validate_model(model)
        global_heads, global_key_dim, global_value_dim = self._global_geometry(model)
        sliding_heads, sliding_key_dim, sliding_value_dim = self._sliding_geometry(model)
        global_components = (
            _kv_components(
                prefix="global",
                category="full_kv",
                layer_count=self.global_attention_layers,
                kv_heads=global_heads,
                key_head_dim=global_key_dim,
                value_head_dim=global_value_dim,
                key_dtype_bits=self.key_dtype_bits,
                value_dtype_bits=self.value_dtype_bits,
            )
            if self.global_attention_layers
            else ()
        )
        sliding_components = (
            _kv_components(
                prefix="sliding",
                category="sliding_kv",
                layer_count=self.sliding_attention_layers,
                kv_heads=sliding_heads,
                key_head_dim=sliding_key_dim,
                value_head_dim=sliding_value_dim,
                key_dtype_bits=self.key_dtype_bits,
                value_dtype_bits=self.value_dtype_bits,
            )
            if self.sliding_attention_layers
            else ()
        )
        draft_components = (
            _kv_components(
                prefix="draft",
                category="draft_kv",
                layer_count=self.draft_attention_layers,
                kv_heads=global_heads,
                key_head_dim=global_key_dim,
                value_head_dim=global_value_dim,
                key_dtype_bits=self.key_dtype_bits,
                value_dtype_bits=self.value_dtype_bits,
            )
            if self.include_draft_cache and self.draft_attention_layers
            else ()
        )
        return global_components, sliding_components, draft_components

    def cache_components(self, model: ModelLike) -> tuple[CacheComponent, ...]:
        global_components, sliding_components, draft_components = self._component_groups(model)
        return (*global_components, *sliding_components, *draft_components)

    def _sconv_group(self, model: ModelLike) -> CheckpointGroup:
        self._validate_model(model)
        return CheckpointGroup(
            "sconv_state",
            (
                StateComponent(
                    "sconv_state",
                    "sconv_state",
                    self.sconv_state_elements_per_layer,
                    self.sconv_state_dtype_bits,
                    self.sconv_layers,
                ),
            ),
            self.sconv_persistence,
            (LayerKind.SCONV,),
        )

    def _sliding_group(
        self, model: ModelLike, policy: StatePersistencePolicy
    ) -> CheckpointGroup | None:
        if policy.storage_form == "token_history":
            return None
        if self.sliding_attention_layers == 0:
            return None
        sliding_heads, sliding_key_dim, sliding_value_dim = self._sliding_geometry(model)
        return CheckpointGroup(
            "sliding_window_state",
            (
                StateComponent(
                    "sliding_key_window",
                    "swa_state",
                    self.sliding_window * sliding_heads * sliding_key_dim,
                    self.key_dtype_bits,
                    self.sliding_attention_layers,
                ),
                StateComponent(
                    "sliding_value_window",
                    "swa_state",
                    self.sliding_window * sliding_heads * sliding_value_dim,
                    self.value_dtype_bits,
                    self.sliding_attention_layers,
                ),
            ),
            policy,
            (LayerKind.SLIDING_ATTENTION,),
        )

    def checkpoint_groups(self, model: ModelLike) -> tuple[CheckpointGroup, ...]:
        groups: list[CheckpointGroup] = []
        sliding = self._sliding_group(model, self.sliding_persistence)
        if sliding is not None:
            groups.append(sliding)
        groups.append(self._sconv_group(model))
        return tuple(groups)

    def effective_checkpoint_groups(
        self,
        model: ModelLike,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> tuple[CheckpointGroup, ...]:
        groups: list[CheckpointGroup] = []
        effective_sliding = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        sliding = self._sliding_group(model, effective_sliding)
        if sliding is not None:
            groups.append(sliding)
        sconv = self._sconv_group(model)
        effective_sconv = _effective_state_policy(
            policy, "sconv_state", sconv.policy
        )
        groups.append(
            CheckpointGroup(
                sconv.name,
                sconv.components,
                effective_sconv,
                sconv.layer_kinds,
                sconv.atomic_admission,
                sconv.atomic_restore,
            )
        )
        return tuple(groups)

    def state_components(self, model: ModelLike) -> tuple[StateComponent, ...]:
        return tuple(item for group in self.checkpoint_groups(model) for item in group.components)

    def map_hit_to_reads(
        self,
        model: ModelLike,
        hit_tokens: int,
        block_size_tokens: int,
        seq_len: int,
        decode_len: int,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
        request_key: str = "",
    ) -> AttentionReadPlan:
        del seq_len, decode_len, request_key
        _validate_token_request(hit_tokens, block_size_tokens)
        global_components, sliding_components, draft_components = self._component_groups(model)
        indices, reusable_tokens = _prefix_block_indices(
            hit_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        streams = list(
            _component_streams_for_indices(
                (*global_components, *draft_components),
                model,
                indices,
                block_size_tokens,
                access_pattern="sequential",
            )
        )
        effective_sliding = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        checkpoint_groups: list[CheckpointGroup] = [self._sconv_group(model)]
        if effective_sliding.storage_form == "token_history":
            streams.extend(
                _component_streams_for_indices(
                    sliding_components,
                    model,
                    _window_block_indices(
                        reusable_tokens, block_size_tokens, self.sliding_window
                    ),
                    block_size_tokens,
                    access_pattern="sequential",
                    logical_context_blocks=len(indices),
                )
            )
        else:
            sliding_group = self._sliding_group(model, effective_sliding)
            assert sliding_group is not None
            checkpoint_groups.append(sliding_group)
        state_streams, restore_plan = _checkpoint_read_plan(
            tuple(checkpoint_groups),
            target_tokens=hit_tokens,
            reusable_prefix_tokens=reusable_tokens,
            block_size_tokens=block_size_tokens,
            policy_override=policy,
        )
        streams.extend(state_streams)
        return AttentionReadPlan(
            len(indices),
            tuple(streams),
            reusable_prefix_tokens=reusable_tokens,
            restore_recompute_tokens=restore_plan.restore_recompute_tokens,
            restore_execution_plan=restore_plan,
        )

    def map_miss_to_writes(
        self,
        model: ModelLike,
        miss_tokens: int,
        block_size_tokens: int,
        prefix_tokens: int = 0,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> AttentionWritePlan:
        _validate_token_interval(prefix_tokens, miss_tokens, block_size_tokens)
        global_components, sliding_components, draft_components = self._component_groups(model)
        indices = _write_block_indices(
            prefix_tokens,
            miss_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        streams = list(
            _component_streams_for_indices(
                (*global_components, *draft_components),
                model,
                indices,
                block_size_tokens,
                access_pattern="sequential",
            )
        )
        effective_sliding = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        checkpoint_groups: list[CheckpointGroup] = [self._sconv_group(model)]
        if effective_sliding.storage_form == "token_history":
            streams.extend(
                _component_streams_for_indices(
                    sliding_components,
                    model,
                    indices,
                    block_size_tokens,
                    access_pattern="sequential",
                )
            )
        else:
            sliding_group = self._sliding_group(model, effective_sliding)
            assert sliding_group is not None
            checkpoint_groups.append(sliding_group)
        streams.extend(
            _checkpoint_write_streams(
                tuple(checkpoint_groups),
                prefix_tokens=prefix_tokens,
                end_tokens=prefix_tokens + miss_tokens,
                block_size_tokens=block_size_tokens,
                policy_override=policy,
            )
        )
        return AttentionWritePlan(len(indices), tuple(streams))

    def map_prefix_to_capacity(
        self,
        model: ModelLike,
        prefix_tokens: int,
        block_size_tokens: int,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> AttentionCapacityPlan:
        _validate_token_request(prefix_tokens, block_size_tokens)
        global_components, sliding_components, draft_components = self._component_groups(model)
        indices, _ = _prefix_block_indices(
            prefix_tokens,
            block_size_tokens,
            self.prefix_granularity,
        )
        result = {
            item.name: len(indices) * item.bytes_per_block(model, block_size_tokens)
            for item in (*global_components, *draft_components)
        }
        effective_sliding = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        checkpoint_groups: list[CheckpointGroup] = [self._sconv_group(model)]
        if effective_sliding.storage_form == "token_history":
            result.update(
                {
                    item.name: len(indices)
                    * item.bytes_per_block(model, block_size_tokens)
                    for item in sliding_components
                }
            )
        else:
            sliding_group = self._sliding_group(model, effective_sliding)
            assert sliding_group is not None
            checkpoint_groups.append(sliding_group)
        state_bytes, checkpoints = _checkpoint_capacity(
            tuple(checkpoint_groups),
            prefix_tokens=prefix_tokens,
            block_size_tokens=block_size_tokens,
            policy_override=policy,
        )
        result.update(state_bytes)
        return AttentionCapacityPlan(len(indices), result, checkpoints)

    def cache_footprint(
        self,
        model: ModelLike,
        *,
        seq_len: int,
        block_size_tokens: int,
        sequence_count: int = 1,
        policy: PrefixCachePolicy | StatePersistencePolicy | None = None,
    ) -> CacheFootprintReport:
        global_components, sliding_components, draft_components = self._component_groups(model)
        effective_sliding = _effective_state_policy(
            policy, "sliding_window_state", self.sliding_persistence
        )
        return _windowed_footprint(
            model=model,
            full_components=(*global_components, *draft_components),
            sliding_components=sliding_components,
            sliding_window=self.sliding_window,
            checkpoint_groups=(self._sconv_group(model),),
            seq_len=seq_len,
            block_size_tokens=block_size_tokens,
            sequence_count=sequence_count,
            prefix_granularity=self.prefix_granularity,
            sliding_policy=effective_sliding,
            policy_override=policy,
            formula_id=self.calculator_formula_id,
            calculator_settings={
                **dict(self.calculator_settings),
                "include_draft_cache": self.include_draft_cache,
            },
            calculator_snapshot_sha256=self.calculator_snapshot_sha256,
            calculator_semantic_class=self.calculator_semantic_class,
        )


def _window_global_sparse_block_count(
    hit_tokens: int,
    block_size_tokens: int,
    window_size: int,
    global_tokens: int,
) -> int:
    return len(
        _window_global_sparse_block_indices(
            hit_tokens,
            block_size_tokens,
            window_size,
            global_tokens,
        )
    )


def _window_global_sparse_block_indices(
    hit_tokens: int,
    block_size_tokens: int,
    window_size: int,
    global_tokens: int,
) -> tuple[int, ...]:
    if hit_tokens <= 0:
        return ()
    selected_blocks: set[int] = set()
    _add_token_range_blocks(
        selected_blocks,
        start=0,
        end=min(global_tokens, hit_tokens),
        block_size_tokens=block_size_tokens,
    )
    _add_token_range_blocks(
        selected_blocks,
        start=max(0, hit_tokens - window_size),
        end=hit_tokens,
        block_size_tokens=block_size_tokens,
    )
    return tuple(sorted(selected_blocks))


def _component_streams(
    components: tuple[CacheComponent, ...],
    model: ModelLike,
    logical_blocks: int,
    block_size_tokens: int,
    *,
    access_pattern: str,
    trace_block_indices: tuple[int, ...] | None = None,
) -> tuple[AttentionIOStream, ...]:
    if logical_blocks == 0:
        return ()
    streams: list[AttentionIOStream] = []
    for component in components:
        block_bytes = component.bytes_per_block(model, block_size_tokens)
        blocks = float(logical_blocks)
        streams.append(
            AttentionIOStream(
                component=component.name,
                logical_blocks=logical_blocks,
                physical_blocks=blocks,
                payload_bytes=logical_blocks * block_bytes,
                io_count=blocks,
                block_bytes=block_bytes,
                lower_bound_blocks=blocks,
                upper_bound_blocks=blocks,
                access_pattern=access_pattern,
                trace_block_indices=trace_block_indices,
            )
        )
    return tuple(streams)


def _uniform_stream(
    *,
    component: str,
    block_count: int,
    block_bytes: int,
    trace_block_indices: tuple[int, ...] | None = None,
    trace_checkpoint_tokens: tuple[int, ...] | None = None,
) -> AttentionIOStream:
    blocks = float(block_count)
    return AttentionIOStream(
        component=component,
        logical_blocks=block_count,
        physical_blocks=blocks,
        payload_bytes=block_count * block_bytes,
        io_count=blocks,
        block_bytes=block_bytes,
        lower_bound_blocks=blocks,
        upper_bound_blocks=blocks,
        access_pattern="sequential",
        trace_block_indices=trace_block_indices,
        trace_checkpoint_tokens=trace_checkpoint_tokens,
    )


def _single_extent_stream(
    component: str,
    payload_bytes: int,
    *,
    trace_block_index: int,
    trace_token_count: int | None = None,
) -> AttentionIOStream:
    return AttentionIOStream(
        component=component,
        logical_blocks=1,
        physical_blocks=1.0,
        payload_bytes=payload_bytes,
        io_count=1.0,
        block_bytes=payload_bytes,
        lower_bound_blocks=1.0,
        upper_bound_blocks=1.0,
        access_pattern="sequential",
        trace_token_count=trace_token_count,
        trace_block_indices=(trace_block_index,),
    )


def _token_range_streams(
    *,
    component: str,
    start_token: int,
    end_token: int,
    block_size_tokens: int,
    bytes_per_token: int,
) -> tuple[AttentionIOStream, ...]:
    """Compact exact block extents for a possibly non-aligned token range."""

    if end_token <= start_token:
        return ()
    extents: list[tuple[int, int]] = []
    cursor = start_token
    while cursor < end_token and cursor % block_size_tokens:
        block_end = min(end_token, ceil_div(cursor, block_size_tokens) * block_size_tokens)
        extents.append((cursor // block_size_tokens, block_end - cursor))
        cursor = block_end
    full_end = end_token - (end_token % block_size_tokens)
    if full_end > cursor:
        full_indices = tuple(range(cursor // block_size_tokens, full_end // block_size_tokens))
        extents.extend((index, block_size_tokens) for index in full_indices)
        cursor = full_end
    if cursor < end_token:
        extents.append((cursor // block_size_tokens, end_token - cursor))

    grouped: dict[int, list[int]] = {}
    for index, token_count in extents:
        grouped.setdefault(token_count, []).append(index)
    return tuple(
        _uniform_stream(
            component=component,
            block_count=len(indices),
            block_bytes=token_count * bytes_per_token,
            trace_block_indices=tuple(indices),
        )
        for token_count, indices in grouped.items()
    )


def _add_token_range_blocks(
    selected_blocks: set[int],
    *,
    start: int,
    end: int,
    block_size_tokens: int,
) -> None:
    if end <= start:
        return
    first_block = start // block_size_tokens
    last_block = (end - 1) // block_size_tokens
    selected_blocks.update(range(first_block, last_block + 1))


def _category_block_bytes(
    components: tuple[CacheComponent, ...],
    model: ModelLike,
    block_size_tokens: int,
    category: str,
) -> int:
    return sum(
        component.bytes_per_block(model, block_size_tokens)
        for component in components
        if component.category == category
    )


def _full_streams(
    components: tuple[CacheComponent, ...],
    model: ModelLike,
    logical_blocks: int,
    block_size_tokens: int,
    access_pattern: str,
) -> tuple[AttentionIOStream, ...]:
    if logical_blocks == 0:
        return ()
    streams: list[AttentionIOStream] = []
    for category in dict.fromkeys(component.category for component in components):
        block_bytes = _category_block_bytes(components, model, block_size_tokens, category)
        if block_bytes == 0:
            continue
        blocks = float(logical_blocks)
        streams.append(
            AttentionIOStream(
                component=category,
                logical_blocks=logical_blocks,
                physical_blocks=blocks,
                payload_bytes=blocks * block_bytes,
                io_count=blocks,
                block_bytes=block_bytes,
                lower_bound_blocks=blocks,
                upper_bound_blocks=blocks,
                access_pattern=access_pattern,
            )
        )
    return tuple(streams)


def _kv_components(
    *,
    prefix: str,
    category: str,
    layer_count: int,
    kv_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    key_dtype_bits: int,
    value_dtype_bits: int,
) -> tuple[CacheComponent, ...]:
    if layer_count < 0:
        raise ValueError("KV layer_count must not be negative")
    if layer_count == 0:
        return ()
    return (
        CacheComponent(
            f"{prefix}_key",
            category,
            kv_heads * key_head_dim,
            key_dtype_bits,
            layer_count=layer_count,
        ),
        CacheComponent(
            f"{prefix}_value",
            category,
            kv_heads * value_head_dim,
            value_dtype_bits,
            layer_count=layer_count,
        ),
    )


def _component_streams_for_indices(
    components: tuple[CacheComponent, ...],
    model: ModelLike,
    block_indices: tuple[int, ...],
    block_size_tokens: int,
    *,
    access_pattern: str,
    repetition: int = 1,
    logical_context_blocks: int | None = None,
    selection_source: str | None = None,
    query_reuse_scope: str | None = None,
) -> tuple[AttentionIOStream, ...]:
    if repetition < 0:
        raise ValueError("stream repetition must not be negative")
    if not block_indices or repetition == 0:
        return ()
    repeated_indices = block_indices * repetition
    physical_blocks = float(len(repeated_indices))
    logical_blocks = (
        logical_context_blocks
        if logical_context_blocks is not None
        else len(block_indices)
    )
    categories = tuple(dict.fromkeys(component.category for component in components))
    return tuple(
        AttentionIOStream(
            component=category,
            logical_blocks=logical_blocks,
            physical_blocks=physical_blocks,
            payload_bytes=physical_blocks
            * _category_block_bytes(
                components, model, block_size_tokens, category
            ),
            io_count=physical_blocks,
            block_bytes=_category_block_bytes(
                components, model, block_size_tokens, category
            ),
            lower_bound_blocks=physical_blocks,
            upper_bound_blocks=physical_blocks,
            access_pattern=access_pattern,
            trace_block_indices=repeated_indices,
            selection_source=selection_source,
            query_reuse_scope=query_reuse_scope,
        )
        for category in categories
    )


def _per_gqa_group_streams(
    components: tuple[CacheComponent, ...],
    model: ModelLike,
    selections: Mapping[tuple[int, int], tuple[int, ...]],
    block_size_tokens: int,
    *,
    group_count: int,
    logical_context_blocks: int,
    selection_source: str,
    query_reuse_scope: str,
) -> tuple[AttentionIOStream, ...]:
    """Realize a physical KV layout with one extent per shared-KV group."""

    if group_count <= 0:
        raise ValueError("group_count must be positive")
    streams: list[AttentionIOStream] = []
    categories = tuple(dict.fromkeys(component.category for component in components))
    for category in categories:
        aggregate_bytes = _category_block_bytes(
            components, model, block_size_tokens, category
        )
        if aggregate_bytes % group_count:
            raise ValueError(
                f"category {category} block bytes cannot be split across GQA groups"
            )
        group_block_bytes = aggregate_bytes // group_count
        for group in range(group_count):
            indices = tuple(
                index
                for (query_chunk, selected_group), selected in sorted(selections.items())
                if selected_group == group
                for index in selected
            )
            if not indices:
                continue
            blocks = float(len(indices))
            streams.append(
                AttentionIOStream(
                    component=f"{category}:g{group}",
                    logical_blocks=logical_context_blocks,
                    physical_blocks=blocks,
                    payload_bytes=blocks * group_block_bytes,
                    io_count=blocks,
                    block_bytes=group_block_bytes,
                    lower_bound_blocks=blocks,
                    upper_bound_blocks=blocks,
                    access_pattern="random",
                    trace_block_indices=indices,
                    selection_source=selection_source,
                    query_reuse_scope=query_reuse_scope,
                )
            )
    return tuple(streams)


def _prefix_block_indices(
    tokens: int,
    block_size_tokens: int,
    policy: PrefixGranularityPolicy,
) -> tuple[tuple[int, ...], int]:
    _validate_token_request(tokens, block_size_tokens)
    if policy.partial_tail_storage == "complete_blocks_only":
        block_count = tokens // block_size_tokens
        reusable_tokens = block_count * block_size_tokens
    else:
        block_count = ceil_div(tokens, block_size_tokens) if tokens else 0
        reusable_tokens = tokens
    return tuple(range(block_count)), reusable_tokens


def _write_block_indices(
    prefix_tokens: int,
    miss_tokens: int,
    block_size_tokens: int,
    policy: PrefixGranularityPolicy,
) -> tuple[int, ...]:
    _validate_token_interval(prefix_tokens, miss_tokens, block_size_tokens)
    if miss_tokens == 0:
        return ()
    end_tokens = prefix_tokens + miss_tokens
    if policy.partial_tail_storage == "complete_blocks_only":
        start = prefix_tokens // block_size_tokens
        end = end_tokens // block_size_tokens
    else:
        # A suffix beginning inside a sealed partial block rewrites that block.
        start = prefix_tokens // block_size_tokens
        end = ceil_div(end_tokens, block_size_tokens)
    return tuple(range(start, end))


def _window_block_indices(
    prefix_tokens: int,
    block_size_tokens: int,
    window_tokens: int,
) -> tuple[int, ...]:
    if prefix_tokens <= 0:
        return ()
    start_token = max(prefix_tokens - window_tokens, 0)
    first = start_token // block_size_tokens
    end = ceil_div(prefix_tokens, block_size_tokens)
    return tuple(range(first, end))


def _checkpoint_read_plan(
    groups: tuple[CheckpointGroup, ...],
    *,
    target_tokens: int,
    reusable_prefix_tokens: int,
    block_size_tokens: int,
    policy_override: PrefixCachePolicy | StatePersistencePolicy | None = None,
) -> tuple[tuple[AttentionIOStream, ...], RestoreExecutionPlan]:
    streams: list[AttentionIOStream] = []
    checkpoint_sets: dict[str, set[int]] = {}
    effective_policies: dict[str, StatePersistencePolicy] = {}
    blocking_bytes = 0
    for group in groups:
        policy = _effective_state_policy(policy_override, group.name, group.policy)
        effective_policies[group.name] = policy
        if policy.storage_form == "token_history":
            raise ValueError("checkpoint state cannot use token_history persistence")
        checkpoint_sets[group.name] = set(
            checkpoint_tokens_for_prefix(
                policy,
                prefix_tokens=reusable_prefix_tokens,
                block_size_tokens=block_size_tokens,
            )
        )

    common_boundaries: set[int] = {0}
    if groups:
        common_boundaries.update(
            set.intersection(
                *(checkpoint_sets[group.name] for group in groups)
            )
        )
    recovery_boundary = max(common_boundaries)
    restored: list[str] = []
    if recovery_boundary > 0:
        for group in groups:
            snapshot_bytes = group.serialized_bytes_per_checkpoint
            streams.append(
                _single_extent_stream(
                    group.name,
                    snapshot_bytes,
                    trace_block_index=max(
                        ceil_div(recovery_boundary, block_size_tokens) - 1,
                        0,
                    ),
                    trace_token_count=recovery_boundary,
                )
            )
            restored.append(group.name)
            blocking_bytes += snapshot_bytes

    causes: list[str] = []
    if recovery_boundary < target_tokens:
        if reusable_prefix_tokens < target_tokens:
            causes.append("token_cache_tail")
        for group in groups:
            group_policy = effective_policies[group.name]
            if group_policy.storage_form == "none":
                causes.append(f"{group.name}:state_not_persisted")
            elif target_tokens not in checkpoint_sets[group.name]:
                causes.append(f"{group.name}:checkpoint_tail")

    return (
        tuple(streams),
        RestoreExecutionPlan(
            matched_prefix_tokens=target_tokens,
            recovery_boundary_tokens=recovery_boundary,
            replay_end_tokens=target_tokens,
            causes=tuple(dict.fromkeys(causes)),
            restored_objects=tuple(restored),
            blocking_checkpoint_bytes=blocking_bytes,
        ),
    )


def _checkpoint_write_streams(
    groups: tuple[CheckpointGroup, ...],
    *,
    prefix_tokens: int,
    end_tokens: int,
    block_size_tokens: int,
    policy_override: PrefixCachePolicy | StatePersistencePolicy | None = None,
) -> tuple[AttentionIOStream, ...]:
    streams: list[AttentionIOStream] = []
    for group in groups:
        policy = _effective_state_policy(policy_override, group.name, group.policy)
        if policy.storage_form == "token_history":
            raise ValueError("checkpoint state cannot use token_history persistence")
        boundaries = checkpoint_boundaries(
            policy,
            prefix_tokens=prefix_tokens,
            end_tokens=end_tokens,
            block_size_tokens=block_size_tokens,
        )
        if not boundaries:
            continue
        indices = tuple(max(ceil_div(token, block_size_tokens) - 1, 0) for token in boundaries)
        streams.append(
            _uniform_stream(
                component=group.name,
                block_count=len(boundaries),
                block_bytes=group.serialized_bytes_per_checkpoint,
                trace_block_indices=indices,
                trace_checkpoint_tokens=boundaries,
            )
        )
    return tuple(streams)


def _checkpoint_capacity(
    groups: tuple[CheckpointGroup, ...],
    *,
    prefix_tokens: int,
    block_size_tokens: int,
    policy_override: PrefixCachePolicy | StatePersistencePolicy | None = None,
) -> tuple[dict[str, int], int]:
    result: dict[str, int] = {}
    total_count = 0
    for group in groups:
        policy = _effective_state_policy(policy_override, group.name, group.policy)
        if policy.storage_form == "token_history":
            raise ValueError("checkpoint state cannot use token_history persistence")
        count = checkpoint_count(
            policy,
            prefix_tokens=prefix_tokens,
            block_size_tokens=block_size_tokens,
        )
        result[group.name] = count * group.serialized_bytes_per_checkpoint
        total_count += count
    return result, total_count


def _merge_restore_plans(
    reusable_prefix_tokens: int,
    plans: list[RestoreExecutionPlan],
) -> RestoreExecutionPlan:
    if not plans:
        return RestoreExecutionPlan(
            matched_prefix_tokens=reusable_prefix_tokens,
            recovery_boundary_tokens=reusable_prefix_tokens,
            replay_end_tokens=reusable_prefix_tokens,
        )
    target = max(plan.matched_prefix_tokens for plan in plans)
    recovery = min(plan.recovery_boundary_tokens for plan in plans)
    return RestoreExecutionPlan(
        matched_prefix_tokens=target,
        recovery_boundary_tokens=recovery,
        replay_end_tokens=target,
        causes=tuple(
            dict.fromkeys(cause for plan in plans for cause in plan.causes)
        ),
        restored_objects=tuple(
            dict.fromkeys(item for plan in plans for item in plan.restored_objects)
        ),
        blocking_checkpoint_bytes=sum(
            plan.blocking_checkpoint_bytes for plan in plans
        ),
    )


def _effective_state_policy(
    policy: PrefixCachePolicy | StatePersistencePolicy | None,
    group_name: str,
    default: StatePersistencePolicy,
) -> StatePersistencePolicy:
    if isinstance(policy, StatePersistencePolicy):
        return policy
    if isinstance(policy, PrefixCachePolicy):
        return policy.state_policy(group_name, default)
    return default


def _token_state_footprint(
    *,
    model: ModelLike,
    token_components: tuple[CacheComponent, ...],
    checkpoint_groups: tuple[CheckpointGroup, ...],
    seq_len: int,
    block_size_tokens: int,
    sequence_count: int,
    prefix_granularity: PrefixGranularityPolicy,
    policy_override: PrefixCachePolicy | StatePersistencePolicy | None,
    formula_id: str,
    calculator_settings: Mapping[str, object],
    calculator_snapshot_sha256: str,
    calculator_semantic_class: str,
) -> CacheFootprintReport:
    _validate_footprint_query(seq_len, block_size_tokens, sequence_count)
    block_indices, reusable_tokens = _prefix_block_indices(
        seq_len,
        block_size_tokens,
        prefix_granularity,
    )
    active_by_component: dict[str, ComponentFootprint] = {}
    prefix_by_component: dict[str, ComponentFootprint] = {}
    calculator_by_component: dict[str, int] = {}
    prefix_objects = 0

    for component in token_components:
        layers = component.layer_count or model.num_layers
        nominal = (
            seq_len
            * layers
            * component.nominal_bytes_per_token_per_layer()
            * sequence_count
        )
        serialized = (
            seq_len
            * layers
            * component.bytes_per_token_per_layer()
            * sequence_count
        )
        active_by_component[component.name] = ComponentFootprint(
            nominal, serialized, serialized
        )
        calculator_by_component[component.name] = _calculator_token_bytes(
            component,
            model,
            tokens=seq_len,
            sequence_count=sequence_count,
            calculator_settings=calculator_settings,
            snapshot_sha256=calculator_snapshot_sha256,
        )
        prefix_nominal = (
            reusable_tokens
            * layers
            * component.nominal_bytes_per_token_per_layer()
            * sequence_count
        )
        prefix_serialized = (
            len(block_indices)
            * component.bytes_per_block(model, block_size_tokens)
            * sequence_count
        )
        prefix_by_component[component.name] = ComponentFootprint(
            prefix_nominal, prefix_serialized, prefix_serialized
        )
        prefix_objects += len(block_indices) * sequence_count

    total_checkpoints = 0
    for group in checkpoint_groups:
        active_nominal = group.nominal_bytes_per_checkpoint * sequence_count
        active_serialized = group.serialized_bytes_per_checkpoint * sequence_count
        active_by_component[group.name] = ComponentFootprint(
            active_nominal, active_serialized, active_serialized
        )
        effective_policy = _effective_state_policy(
            policy_override, group.name, group.policy
        )
        count = checkpoint_count(
            effective_policy,
            prefix_tokens=seq_len,
            block_size_tokens=block_size_tokens,
        ) * sequence_count
        prefix_nominal = count * group.nominal_bytes_per_checkpoint
        prefix_serialized = count * group.serialized_bytes_per_checkpoint
        prefix_by_component[group.name] = ComponentFootprint(
            prefix_nominal, prefix_serialized, prefix_serialized
        )
        calculator_count = (
            count
            if calculator_semantic_class in {"hybrid", "retained_checkpoints"}
            else sequence_count
        )
        calculator_by_component[group.name] = (
            calculator_count * group.nominal_bytes_per_checkpoint
        )
        total_checkpoints += count
        prefix_objects += count

    active = _active_footprint(active_by_component)
    prefix = _prefix_footprint(
        prefix_by_component,
        object_count=prefix_objects,
        checkpoint_count_value=total_checkpoints,
    )
    return CacheFootprintReport(
        CalculatorReferenceFootprint(
            total_bytes=sum(calculator_by_component.values()),
            formula_id=formula_id,
            settings={
                **dict(calculator_settings),
                "seq_len": seq_len,
                "sequence_count": sequence_count,
            },
            snapshot_sha256=calculator_snapshot_sha256,
            semantic_class=calculator_semantic_class,
            by_component=calculator_by_component,
        ),
        active,
        prefix,
    )


def _windowed_footprint(
    *,
    model: ModelLike,
    full_components: tuple[CacheComponent, ...],
    sliding_components: tuple[CacheComponent, ...],
    sliding_window: int,
    checkpoint_groups: tuple[CheckpointGroup, ...],
    seq_len: int,
    block_size_tokens: int,
    sequence_count: int,
    prefix_granularity: PrefixGranularityPolicy,
    sliding_policy: StatePersistencePolicy,
    policy_override: PrefixCachePolicy | StatePersistencePolicy | None,
    formula_id: str,
    calculator_settings: Mapping[str, object],
    calculator_snapshot_sha256: str,
    calculator_semantic_class: str,
) -> CacheFootprintReport:
    _validate_footprint_query(seq_len, block_size_tokens, sequence_count)
    indices, reusable_tokens = _prefix_block_indices(
        seq_len, block_size_tokens, prefix_granularity
    )
    active_by_component: dict[str, ComponentFootprint] = {}
    prefix_by_component: dict[str, ComponentFootprint] = {}
    calculator_by_component: dict[str, int] = {}
    object_count = 0
    checkpoint_total = 0

    def add_active(component: CacheComponent, used_tokens: int, reserved_tokens: int) -> None:
        layers = component.layer_count or model.num_layers
        nominal = (
            used_tokens
            * layers
            * component.nominal_bytes_per_token_per_layer()
            * sequence_count
        )
        serialized = (
            used_tokens
            * layers
            * component.bytes_per_token_per_layer()
            * sequence_count
        )
        reserved = (
            reserved_tokens
            * layers
            * component.bytes_per_token_per_layer()
            * sequence_count
        )
        active_by_component[component.name] = ComponentFootprint(
            nominal, serialized, reserved
        )
        calculator_by_component[component.name] = _calculator_token_bytes(
            component,
            model,
            tokens=used_tokens,
            sequence_count=sequence_count,
            calculator_settings=calculator_settings,
            snapshot_sha256=calculator_snapshot_sha256,
        )

    for component in full_components:
        add_active(component, seq_len, seq_len)
        layers = component.layer_count or model.num_layers
        nominal = (
            reusable_tokens
            * layers
            * component.nominal_bytes_per_token_per_layer()
            * sequence_count
        )
        serialized = (
            len(indices)
            * component.bytes_per_block(model, block_size_tokens)
            * sequence_count
        )
        prefix_by_component[component.name] = ComponentFootprint(
            nominal, serialized, serialized
        )
        object_count += len(indices) * sequence_count

    live_window_tokens = min(seq_len, sliding_window)
    reserved_window_tokens = sliding_window if seq_len else 0
    for component in sliding_components:
        add_active(component, live_window_tokens, reserved_window_tokens)

    all_checkpoint_groups = list(checkpoint_groups)
    if sliding_policy.storage_form == "token_history":
        for component in sliding_components:
            layers = component.layer_count or model.num_layers
            nominal = (
                reusable_tokens
                * layers
                * component.nominal_bytes_per_token_per_layer()
                * sequence_count
            )
            serialized = (
                len(indices)
                * component.bytes_per_block(model, block_size_tokens)
                * sequence_count
            )
            prefix_by_component[component.name] = ComponentFootprint(
                nominal, serialized, serialized
            )
            object_count += len(indices) * sequence_count
    else:
        sliding_states = tuple(
            StateComponent(
                f"{component.name}_window",
                "swa_state",
                sliding_window * component.elements_per_token_per_layer,
                component.dtype_bits,
                component.layer_count or model.num_layers,
                serialized_bytes_per_layer=(
                    sliding_window * component.bytes_per_token_per_layer()
                ),
            )
            for component in sliding_components
        )
        if sliding_states:
            all_checkpoint_groups.append(
                CheckpointGroup(
                    "sliding_window_state",
                    sliding_states,
                    sliding_policy,
                    (LayerKind.SLIDING_ATTENTION,),
                )
            )

    for group in all_checkpoint_groups:
        active_nominal = group.nominal_bytes_per_checkpoint * sequence_count
        active_serialized = group.serialized_bytes_per_checkpoint * sequence_count
        # Sliding-window state is already represented by the live sliding KV.
        if group.name != "sliding_window_state":
            active_by_component[group.name] = ComponentFootprint(
                active_nominal, active_serialized, active_serialized
            )
        effective_group_policy = _effective_state_policy(
            policy_override, group.name, group.policy
        )
        count = checkpoint_count(
            effective_group_policy,
            prefix_tokens=seq_len,
            block_size_tokens=block_size_tokens,
        ) * sequence_count
        nominal = count * group.nominal_bytes_per_checkpoint
        serialized = count * group.serialized_bytes_per_checkpoint
        prefix_by_component[group.name] = ComponentFootprint(
            nominal, serialized, serialized
        )
        if group.name != "sliding_window_state":
            calculator_by_component[group.name] = (
                nominal
                if calculator_semantic_class in {"hybrid", "retained_checkpoints"}
                else active_nominal
            )
        checkpoint_total += count
        object_count += count

    return CacheFootprintReport(
        CalculatorReferenceFootprint(
            total_bytes=sum(calculator_by_component.values()),
            formula_id=formula_id,
            settings={
                **dict(calculator_settings),
                "seq_len": seq_len,
                "sequence_count": sequence_count,
            },
            snapshot_sha256=calculator_snapshot_sha256,
            semantic_class=calculator_semantic_class,
            by_component=calculator_by_component,
        ),
        _active_footprint(active_by_component),
        _prefix_footprint(
            prefix_by_component,
            object_count=object_count,
            checkpoint_count_value=checkpoint_total,
        ),
    )


def _active_footprint(
    components: Mapping[str, ComponentFootprint]
) -> ActiveExecutionFootprint:
    return ActiveExecutionFootprint(
        nominal_payload_bytes=sum(item.nominal_payload_bytes for item in components.values()),
        serialized_bytes=_sum_optional_serialized(components),
        reserved_bytes=sum(item.reserved_bytes for item in components.values()),
        by_component=dict(components),
    )


def _prefix_footprint(
    components: Mapping[str, ComponentFootprint],
    *,
    object_count: int,
    checkpoint_count_value: int,
) -> PrefixStoreFootprint:
    return PrefixStoreFootprint(
        nominal_payload_bytes=sum(item.nominal_payload_bytes for item in components.values()),
        serialized_bytes=_sum_optional_serialized(components),
        object_count=object_count,
        checkpoint_count=checkpoint_count_value,
        by_component=dict(components),
    )


def _sum_optional_serialized(
    components: Mapping[str, ComponentFootprint],
) -> int | None:
    values = [item.serialized_bytes for item in components.values()]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _calculator_token_bytes(
    component: CacheComponent,
    model: ModelLike,
    *,
    tokens: int,
    sequence_count: int,
    calculator_settings: Mapping[str, object],
    snapshot_sha256: str,
) -> int:
    """External calculator payload, independent of runtime serialization."""

    bits = component.dtype_bits
    if calculator_settings:
        precision_key = (
            "indexer_fixed_precision_id"
            if component.category in {"indexer", "draft_indexer"}
            and "indexer_fixed_precision_id" in calculator_settings
            else "default_precision_id"
        )
        precision = str(calculator_settings.get(precision_key, "bf16_fp16"))
        bits = {
            "bf16_fp16": 16,
            "bf16": 16,
            "fp16": 16,
            "fp8_int8": 8,
            "fp8": 8,
            "int8": 8,
            "fp4_int4": 4,
            "fp4": 4,
            "int4": 4,
        }.get(precision, 16)
    layers = component.layer_count or model.num_layers
    entries = ceil_div(tokens, component.compression_ratio) if tokens else 0
    return (
        entries
        * layers
        * ceil_div(component.elements_per_token_per_layer * bits, 8)
        * sequence_count
    )


def _validate_token_request(tokens: int, block_size_tokens: int) -> None:
    if tokens < 0:
        raise ValueError("token count must not be negative")
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")


def _validate_token_interval(
    prefix_tokens: int, miss_tokens: int, block_size_tokens: int
) -> None:
    if prefix_tokens < 0 or miss_tokens < 0:
        raise ValueError("prefix and miss token counts must not be negative")
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")


def _validate_footprint_query(
    seq_len: int, block_size_tokens: int, sequence_count: int
) -> None:
    _validate_token_request(seq_len, block_size_tokens)
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")


def _validate_positive_fields(value: object, names: tuple[str, ...]) -> None:
    for name in names:
        if int(getattr(value, name)) <= 0:
            raise ValueError(f"{name} must be positive")


def _validate_layer_partition(values: tuple[int, ...], label: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f"{label} layer counts must not be negative")
    if sum(values) <= 0:
        raise ValueError(f"{label} must contain at least one layer")
