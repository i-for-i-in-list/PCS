"""LLM model profile."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import wraps
import math
from typing import Any, Mapping, Protocol, runtime_checkable

from ssu_sim.models.attention import AttentionProfile, CacheComponent, PrefixCachePolicy
from ssu_sim.models.state import RestoreExecutionPlan
from ssu_sim.models.deepseek_compute import v4_head, v4_segment, v41_segment
from ssu_sim.models.compute_counts import causal_pairs, token_count, delta_rule_work, softmax_work, block_sparse_pairs
from ssu_sim.models.execution import require_evaluated_full_hit
from ssu_sim.models.metadata import (
    CatalogMetadata,
    ModelEvidence,
    ModelIdentity,
)


class ComputeModelLike(Protocol):
    d_model: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    attention: AttentionProfile


@dataclass(frozen=True)
class OperationClassBreakdown:
    """FLOPs grouped by execution class instead of model-specific component.

    These classes are deliberately coarse.  They provide a stable boundary for
    runtimes that have different effective rates for GEMMs, attention kernels,
    sparse gathers, indexers, convolutions, and recurrent scans.  Component
    breakdowns (for example ``csa_core``) remain available separately for
    backwards-compatible reports.
    """

    dense_gemm: float = 0
    dense_attention: float = 0
    sparse_attention: float = 0
    indexer: float = 0
    convolution: float = 0
    recurrent: float = 0
    moe: float = 0
    routing: float = 0
    compression: float = 0
    elementwise: float = 0
    unclassified: float = 0
    special_operation_counts: dict[str, int] = field(default_factory=dict)
    counting_omissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"operation-class FLOPs for {name} must be finite and non-negative"
                )

    @property
    def total_flops(self) -> float:
        return sum(self.as_dict().values())

    def as_dict(self) -> dict[str, float]:
        return {
            "dense_gemm": self.dense_gemm,
            "dense_attention": self.dense_attention,
            "sparse_attention": self.sparse_attention,
            "indexer": self.indexer,
            "convolution": self.convolution,
            "recurrent": self.recurrent,
            "moe": self.moe,
            "routing": self.routing,
            "compression": self.compression,
            "elementwise": self.elementwise,
            "unclassified": self.unclassified,
        }

    def __add__(self, other: object) -> "OperationClassBreakdown":
        if not isinstance(other, OperationClassBreakdown):
            return NotImplemented
        left = self.as_dict()
        right = other.as_dict()
        return OperationClassBreakdown(
            **{name: left[name] + right[name] for name in left},
            special_operation_counts={name: self.special_operation_counts.get(name, 0) + other.special_operation_counts.get(name, 0)
                                      for name in self.special_operation_counts.keys() | other.special_operation_counts.keys()},
            counting_omissions=tuple(sorted(set(self.counting_omissions + other.counting_omissions))),
        )


@dataclass(frozen=True)
class TimingEstimate:
    """Timing evidence for an operation-class FLOP breakdown.

    ``normalized_reference_us`` is always the legacy analytical FLOP-equivalent
    reference derived from ``effective_prefill_tflops``.  It is not a calibrated
    runtime measurement.  ``calibrated_us`` is populated only in
    ``component_calibrated`` mode and only when every operation class with
    non-zero work has an explicit throughput.

    ``time_us`` is the value selected by ``timing_mode``.  It is deliberately
    ``None`` for an incomplete component calibration; callers must not silently
    substitute the normalized reference in that case.
    """

    timing_mode: str
    evidence: str
    total_flops: float
    time_us: float | None
    normalized_reference_us: float
    calibrated_us: float | None
    is_calibrated: bool
    missing_operation_classes: tuple[str, ...] = ()
    operation_class_time_us: dict[str, float] = field(default_factory=dict)
    omissions: tuple[str, ...] = ()


@runtime_checkable
class FFNProfile(Protocol):
    """Feed-forward FLOP model composable with any attention formula."""

    layer_count: int | None

    def operation_breakdown(
        self,
        *,
        model: ComputeModelLike,
        q_len: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        ...


@dataclass(frozen=True)
class PrefillComputeContext:
    """Request shape needed to compute prefix-aware prefill FLOPs."""

    seq_len: int
    matched_prefix_len: int
    generation_len: int = 0
    batch_size: int = 1
    encoder_state: str = "unavailable"
    encoder_tail_policy: str = "recompute"

    def __post_init__(self) -> None:
        for name in ("seq_len", "matched_prefix_len", "generation_len", "batch_size"):
            object.__setattr__(self, name, token_count(getattr(self, name), name))
        if self.seq_len < 0:
            raise ValueError("seq_len must not be negative")
        if self.generation_len < 0:
            raise ValueError("generation_len must not be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.matched_prefix_len <= self.seq_len:
            raise ValueError("matched_prefix_len must be within seq_len")

    @property
    def miss_len(self) -> int:
        return self.seq_len - self.matched_prefix_len


@dataclass(frozen=True)
class CacheComputeBreakdown:
    """Auditable no-cache and cache-path prefill work for one request."""

    no_cache_flops: float
    cache_path_flops: float
    miss_suffix_flops: float
    compression_tail_flops: float
    swa_restore_flops: float
    restore_recompute_flops: float
    saved_flops: float
    restore_recompute_tokens: int
    cache_path_by_component: dict[str, float]
    no_cache_by_component: dict[str, float]
    # ``saved_flops`` is the historical non-negative field.  New consumers
    # should use ``net_saved_flops`` so restore work that exceeds the no-cache
    # path remains visible instead of being silently clamped to zero.
    net_saved_flops: float = 0
    cache_path_by_operation_class: dict[str, float] = field(default_factory=dict)
    no_cache_by_operation_class: dict[str, float] = field(default_factory=dict)
    phase_flops: dict[str, float] = field(default_factory=dict)
    phase_by_component: dict[str, dict[str, float]] = field(default_factory=dict)
    phase_by_operation_class: dict[str, dict[str, float]] = field(default_factory=dict)
    execution_plan: RestoreExecutionPlan | None = None
    suffix_body_flops: float = 0
    mandatory_conversion_flops: float = 0
    # Subtotal already included in restore_recompute_flops; not an extra phase.
    restore_conversion_flops: float = 0
    finalization_flops: float = 0
    output_head_flops: float = 0
    suffix_by_operation_class: dict[str, float] = field(default_factory=dict)
    restore_by_operation_class: dict[str, float] = field(default_factory=dict)
    conversion_by_operation_class: dict[str, float] = field(default_factory=dict)
    finalization_by_operation_class: dict[str, float] = field(default_factory=dict)
    output_head_by_operation_class: dict[str, float] = field(default_factory=dict)
    special_operation_counts: dict[str, int] = field(default_factory=dict)
    counting_omissions: tuple[str, ...] = ()
    algorithm_id: str = ""
    full_hit_finalization_status: str = "not_applicable"


@dataclass(frozen=True)
class DenseFFNProfile:
    """Dense Transformer FFN FLOPs for a declared set of layers."""

    intermediate_size: int
    layer_count: int | None = None
    gated: bool = True
    input_dim: int | None = None
    activation: str = "silu"
    activation_parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_positive(self.intermediate_size, "ffn.intermediate_size")
        if self.layer_count is not None:
            _validate_positive(self.layer_count, "ffn.layer_count")

    def operation_breakdown(
        self,
        *,
        model: ComputeModelLike,
        q_len: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        if q_len <= 0:
            return OperationClassBreakdown()
        layers = self.layer_count if self.layer_count is not None else model.num_layers
        projections = 3 if self.gated else 2
        base = OperationClassBreakdown(
            dense_gemm=(
                2
                * batch_size
                * q_len
                * (self.input_dim or model.d_model)
                * self.intermediate_size
                * projections
                * layers
            )
        )
        count = batch_size * q_len * self.intermediate_size * layers
        return base + _activation_work(self.activation, count, self.activation_parameters) + OperationClassBreakdown(
            elementwise=count if self.gated else 0)

    def flops(self, *, model: ComputeModelLike, q_len: float, batch_size: int) -> float:
        return self.operation_breakdown(
            model=model, q_len=q_len, batch_size=batch_size
        ).total_flops


@dataclass(frozen=True)
class MoEFFNProfile:
    """Routed/shared expert FFN FLOPs for a declared set of layers."""

    n_experts: int
    experts_per_token: int
    expert_intermediate_size: int
    layer_count: int | None = None
    router_included: bool = True
    shared_experts: int = 0
    expert_input_dim: int | None = None
    shared_intermediate_size: int | None = None
    shared_input_dim: int | None = None
    router_input_dim: int | None = None
    router_output_dim: int | None = None
    shared_gate: bool = False
    latent_dim: int | None = None
    activation: str = "silu"
    router_activation: str = "unspecified"
    activation_parameters: Mapping[str, Any] | None = None
    normalize_topk_weights: bool = False
    router_group_count: int = 1
    shared_intermediate_weighting: bool = False
    shared_separate_outputs: bool = False
    router_normalization: str = "standard"
    router_activation_on_selected: bool = False
    router_score_bias: bool = False
    router_group_top2: bool = False
    router_sum_epsilon: bool = False
    router_weight_scale: bool = False
    routed_output_scale: bool = False

    def __post_init__(self) -> None:
        _validate_positive(self.n_experts, "moe.n_experts")
        _validate_positive(self.experts_per_token, "moe.experts_per_token")
        _validate_positive(
            self.expert_intermediate_size, "moe.expert_intermediate_size"
        )
        if self.layer_count is not None:
            _validate_positive(self.layer_count, "moe.layer_count")
        if self.shared_experts < 0:
            raise ValueError("moe.shared_experts must not be negative")
        for name in ("expert_input_dim", "shared_intermediate_size", "shared_input_dim", "router_input_dim", "router_output_dim", "latent_dim"):
            if getattr(self, name) is not None:
                _validate_positive(getattr(self, name), "moe." + name)
        if self.latent_dim and self.expert_input_dim and self.latent_dim != self.expert_input_dim:
            raise ValueError("latent_dim and expert_input_dim must agree")
        if self.experts_per_token > self.n_experts:
            raise ValueError("experts_per_token must not exceed n_experts")

    def operation_breakdown(
        self,
        *,
        model: ComputeModelLike,
        q_len: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        if q_len <= 0:
            return OperationClassBreakdown()
        layers = self.layer_count if self.layer_count is not None else model.num_layers
        routing = (
            2
            * batch_size
            * q_len
            * (self.router_input_dim or model.d_model)
            * (self.router_output_dim or self.n_experts)
            * layers
            if self.router_included
            else 0
        )
        expert_dim = self.latent_dim or self.expert_input_dim or model.d_model
        expert = (
            batch_size
            * q_len
            * self.experts_per_token
            * 2
            * 3
            * expert_dim
            * self.expert_intermediate_size
            * layers
        )
        expert += (6 * batch_size * q_len * self.shared_experts
                   * (self.shared_input_dim or model.d_model)
                   * (self.shared_intermediate_size or self.expert_intermediate_size) * layers)
        projection = 4 * batch_size * q_len * model.d_model * self.latent_dim * layers if self.latent_dim else 0
        if self.shared_gate:
            projection += 2 * batch_size * q_len * model.d_model * layers
        activations = batch_size * q_len * layers * (self.experts_per_token * self.expert_intermediate_size
            + self.shared_experts * (self.shared_intermediate_size or self.expert_intermediate_size))
        output_mix = batch_size * q_len * layers * (
            max(2 * self.experts_per_token - 1, 0) * expert_dim
            + (model.d_model if self.shared_experts else 0))
        result = OperationClassBreakdown(moe=expert, routing=routing, dense_gemm=projection,
            elementwise=activations + output_mix) + _activation_work(self.activation, activations, self.activation_parameters)
        if self.shared_intermediate_weighting:
            result += OperationClassBreakdown(elementwise=batch_size * q_len * layers * self.shared_experts
                * (self.shared_intermediate_size or self.expert_intermediate_size))
        if self.shared_separate_outputs:
            result += OperationClassBreakdown(elementwise=batch_size * q_len * layers
                * max(0, self.shared_experts - 1) * (self.shared_input_dim or model.d_model))
        if self.routed_output_scale:
            result += OperationClassBreakdown(elementwise=batch_size * q_len * layers * model.d_model)
        if self.router_included:
            rows = batch_size * q_len * layers
            width = self.router_output_dim or self.n_experts
            activation_width = self.experts_per_token if self.router_activation_on_selected else width
            if self.router_activation == "softmax":
                result = result + _softmax_operations(rows, rows * activation_width)
            elif self.router_activation == "sigmoid":
                result = result + _activation_work("sigmoid", rows * activation_width)
            else:
                result = result + OperationClassBreakdown(counting_omissions=("router_activation_and_selection_arithmetic_not_fully_specified",))
            result += OperationClassBreakdown(special_operation_counts={"topk_select": rows})
            if self.normalize_topk_weights:
                result += OperationClassBreakdown(elementwise=rows * (2 * self.experts_per_token - 1 + int(self.router_sum_epsilon)))
            if self.router_score_bias:
                result += OperationClassBreakdown(elementwise=rows * width)
            if self.router_weight_scale:
                result += OperationClassBreakdown(elementwise=rows * self.experts_per_token)
            if self.router_group_top2:
                # Source noaux_tc: sum the top TWO scores in each group,
                # select groups, then select individual experts. Comparisons
                # remain separately named kernel-dependent operations.
                result += OperationClassBreakdown(elementwise=rows * self.router_group_count,
                    special_operation_counts={"group_top2_select": rows * self.router_group_count,
                                              "topk_group_select": rows})
            elif self.router_group_count != 1:
                result += OperationClassBreakdown(counting_omissions=("grouped_router_selection_schedule",))
            if self.router_normalization == "inkling_selected_logsigmoid":
                chosen = self.experts_per_token + self.shared_experts
                # Correction bias for routed choices; stable logsumexp then
                # exponentiation and the two routing scale multiplications.
                result += OperationClassBreakdown(elementwise=rows * (self.n_experts + 5 * chosen),
                    special_operation_counts={"logsigmoid": rows * chosen, "log": rows,
                        "exp": 2 * rows * chosen, "max_compare": rows * (chosen - 1)})
        if self.shared_gate:
            result = result + _activation_work("sigmoid", batch_size * q_len * layers)
            result = result + OperationClassBreakdown(elementwise=batch_size * q_len * layers * model.d_model)
        return result

    def flops(self, *, model: ComputeModelLike, q_len: float, batch_size: int) -> float:
        return self.operation_breakdown(
            model=model, q_len=q_len, batch_size=batch_size
        ).total_flops


@dataclass(frozen=True)
class ParallelFFNProfile:
    """Branches executing on the same layers; layers are not added together."""

    branches: tuple[FFNProfile, ...]
    layer_count: int | None = None

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("parallel FFN requires branches")
        if self.layer_count is not None:
            _validate_positive(self.layer_count, "parallel.layer_count")
        for branch in self.branches:
            if branch.layer_count is not None and self.layer_count is not None and branch.layer_count != self.layer_count:
                raise ValueError("parallel FFN branches must cover the same layers")

    def operation_breakdown(self, *, model: ComputeModelLike, q_len: float, batch_size: int) -> OperationClassBreakdown:
        result = OperationClassBreakdown()
        for branch in self.branches:
            if branch.layer_count is None and self.layer_count is not None:
                branch = replace(branch, layer_count=self.layer_count)
            result = result + branch.operation_breakdown(model=model, q_len=q_len, batch_size=batch_size)
        return result

    def flops(self, *, model: ComputeModelLike, q_len: float, batch_size: int) -> float:
        return self.operation_breakdown(model=model, q_len=q_len, batch_size=batch_size).total_flops


@dataclass(frozen=True)
class LayeredFFNProfile:
    """A layer-exact composition of dense and/or expert FFN segments."""

    segments: tuple[FFNProfile, ...]
    layer_count: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("layered FFN must contain at least one segment")
        for segment in self.segments:
            if segment.layer_count is None:
                raise ValueError("each layered FFN segment must declare layer_count")

    def operation_breakdown(
        self,
        *,
        model: ComputeModelLike,
        q_len: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        declared_layers = sum(segment.layer_count or 0 for segment in self.segments)
        if declared_layers != model.num_layers:
            raise ValueError(
                "layered FFN segment layer counts must equal model.num_layers"
            )
        result = OperationClassBreakdown()
        for segment in self.segments:
            result = result + segment.operation_breakdown(
                model=model, q_len=q_len, batch_size=batch_size
            )
        return result

    def flops(self, *, model: ComputeModelLike, q_len: float, batch_size: int) -> float:
        return self.operation_breakdown(
            model=model, q_len=q_len, batch_size=batch_size
        ).total_flops


@dataclass(frozen=True)
class MoEProfile:
    """Legacy MoE schema, now including its leading dense FFN layers.

    ``dense_intermediate_size`` is optional for old YAML compatibility.  When
    omitted, the expert width is used as an explicit compatibility fallback;
    canonical model configs should supply the model's real dense width.
    """

    n_experts: int
    experts_per_token: int
    expert_intermediate_size: int
    router_included: bool = True
    first_dense_layers: int = 0
    shared_experts: int = 0
    dense_intermediate_size: int | None = None
    dense_gated: bool = True
    layer_count: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        _validate_positive(self.n_experts, "moe.n_experts")
        _validate_positive(self.experts_per_token, "moe.experts_per_token")
        _validate_positive(
            self.expert_intermediate_size, "moe.expert_intermediate_size"
        )
        if self.first_dense_layers < 0:
            raise ValueError("moe.first_dense_layers must not be negative")
        if self.shared_experts < 0:
            raise ValueError("moe.shared_experts must not be negative")
        if self.dense_intermediate_size is not None:
            _validate_positive(
                self.dense_intermediate_size, "moe.dense_intermediate_size"
            )

    def operation_breakdown(
        self,
        *,
        model: ComputeModelLike,
        q_len: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        dense_layers = min(self.first_dense_layers, model.num_layers)
        routed_layers = max(model.num_layers - dense_layers, 0)
        result = OperationClassBreakdown()
        if dense_layers:
            result = result + DenseFFNProfile(
                intermediate_size=(
                    self.dense_intermediate_size or self.expert_intermediate_size
                ),
                layer_count=dense_layers,
                gated=self.dense_gated,
            ).operation_breakdown(model=model, q_len=q_len, batch_size=batch_size)
        if routed_layers:
            result = result + MoEFFNProfile(
                n_experts=self.n_experts,
                experts_per_token=self.experts_per_token,
                expert_intermediate_size=self.expert_intermediate_size,
                layer_count=routed_layers,
                router_included=self.router_included,
                shared_experts=self.shared_experts,
            ).operation_breakdown(model=model, q_len=q_len, batch_size=batch_size)
        return result

    def flops(self, *, model: ComputeModelLike, q_len: float, batch_size: int) -> float:
        return self.operation_breakdown(
            model=model, q_len=q_len, batch_size=batch_size
        ).total_flops


@runtime_checkable
class PrefillFormula(Protocol):
    """Unified analytical prefill interface for all attention families."""

    type: str

    def operation_breakdown(
        self,
        model: ComputeModelLike,
        *,
        q_len: float,
        start_context: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        ...

    def saved_recompute_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        ...

    def actual_prefill_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        ...

    def cache_compute_breakdown(
        self,
        model: ComputeModelLike,
        context: PrefillComputeContext,
        *,
        restore_recompute_tokens: int = 0,
    ) -> CacheComputeBreakdown:
        ...


class _OperationFormulaMixin:
    """Default prefix/replay accounting for operation-aware formulas."""

    def saved_recompute_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        return self.operation_breakdown(
            model,
            q_len=context.matched_prefix_len,
            start_context=0,
            batch_size=context.batch_size,
        ).total_flops

    def actual_prefill_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        return self.suffix_operation_breakdown(model, context).total_flops

    def mandatory_conversion_breakdown(self, model, context):
        return OperationClassBreakdown()

    def suffix_operation_breakdown(self, model, context):
        return self.operation_breakdown(
            model,
            q_len=context.miss_len,
            start_context=context.matched_prefix_len,
            batch_size=context.batch_size,
        ) + self.mandatory_conversion_breakdown(model, context)

    def cache_compute_breakdown(
        self,
        model: ComputeModelLike,
        context: PrefillComputeContext,
        *,
        restore_recompute_tokens: int = 0,
    ) -> CacheComputeBreakdown:
        if not 0 <= restore_recompute_tokens <= context.matched_prefix_len:
            raise ValueError("restore_recompute_tokens must be within matched prefix")
        no_cache = self.operation_breakdown(
            model,
            q_len=context.seq_len,
            start_context=0,
            batch_size=context.batch_size,
        )
        effective_cached = context.matched_prefix_len - restore_recompute_tokens
        recompute = self.operation_breakdown(
            model,
            q_len=restore_recompute_tokens,
            start_context=effective_cached,
            batch_size=context.batch_size,
        )
        suffix = self.operation_breakdown(
            model,
            q_len=context.miss_len,
            start_context=context.matched_prefix_len,
            batch_size=context.batch_size,
        )
        conversion_context = replace(context, matched_prefix_len=effective_cached)
        conversion = self.mandatory_conversion_breakdown(model, conversion_context)
        restore_conversion = 0
        if context.miss_len == 0:
            # Conversion needed only to replay missing state belongs to restore.
            recompute = recompute + conversion
            restore_conversion = conversion.total_flops
            conversion = OperationClassBreakdown()
        required_suffix = suffix + conversion
        cache_path = recompute + required_suffix
        net_saved = no_cache.total_flops - cache_path.total_flops
        return CacheComputeBreakdown(
            no_cache_flops=no_cache.total_flops,
            cache_path_flops=cache_path.total_flops,
            miss_suffix_flops=required_suffix.total_flops,
            compression_tail_flops=recompute.compression,
            swa_restore_flops=0.0,
            restore_recompute_flops=recompute.total_flops,
            saved_flops=max(net_saved, 0),
            restore_recompute_tokens=restore_recompute_tokens,
            cache_path_by_component={"suffix_body": suffix.total_flops, "mandatory_conversion": conversion.total_flops,
                                     "state_restoration": recompute.total_flops},
            no_cache_by_component={},
            net_saved_flops=net_saved,
            cache_path_by_operation_class=cache_path.as_dict(),
            no_cache_by_operation_class=no_cache.as_dict(),
            suffix_body_flops=suffix.total_flops,
            mandatory_conversion_flops=conversion.total_flops,
            restore_conversion_flops=restore_conversion,
            suffix_by_operation_class=required_suffix.as_dict(),
            restore_by_operation_class=recompute.as_dict(),
            conversion_by_operation_class=conversion.as_dict(),
            algorithm_id=getattr(self, "algorithm_id", "causal_mac2_materialized"),
            special_operation_counts=cache_path.special_operation_counts,
            counting_omissions=tuple(sorted(set(cache_path.counting_omissions
                + getattr(self, "counting_omissions", ())))),
        )


def _ffn_breakdown(
    *,
    model: ComputeModelLike,
    q_len: float,
    batch_size: int,
    ffn: FFNProfile | None,
    moe: MoEProfile | None,
) -> OperationClassBreakdown:
    if ffn is not None and moe is not None:
        raise ValueError("configure either ffn or legacy moe, not both")
    profile = ffn if ffn is not None else moe
    if profile is None:
        return OperationClassBreakdown()
    return profile.operation_breakdown(
        model=model, q_len=q_len, batch_size=batch_size
    )


def _with_draft_execution(
    base: OperationClassBreakdown,
    *,
    model: ComputeModelLike,
    draft_execution_mode: str,
    draft_layer_count: int,
) -> OperationClassBreakdown:
    """Add an explicit average-layer analytical draft scenario.

    Calculator fields establish the number of draft Transformer layers but do
    not expose kernel-specific throughput.  The extra work is therefore kept
    in ``unclassified`` rather than being presented as calibrated attention or
    MoE work.
    """

    if draft_execution_mode not in {"disabled", "prefill"}:
        raise ValueError("draft_execution_mode must be disabled or prefill")
    if draft_layer_count < 0:
        raise ValueError("draft_layer_count must not be negative")
    if draft_execution_mode == "disabled" or draft_layer_count == 0:
        return base
    return base + OperationClassBreakdown(
        unclassified=base.total_flops * draft_layer_count / model.num_layers
    )


def _activation_work(kind: str, count: int, parameters=None) -> OperationClassBreakdown:
    count = token_count(count, "activation elements")
    if not count:
        return OperationClassBreakdown()
    if kind in {"silu", "swish", "sigmoid"}:
        return OperationClassBreakdown(elementwise=count * (4 if kind != "sigmoid" else 3),
                                       special_operation_counts={"exp": count})
    if kind in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
        return OperationClassBreakdown(elementwise=8 * count, special_operation_counts={"tanh": count})
    if kind == "situ":
        if parameters is None or "linear_beta" not in parameters:
            return OperationClassBreakdown(counting_omissions=("situ_linear_beta_not_declared",))
        extra = parameters["linear_beta"] is not None
        return OperationClassBreakdown(elementwise=(6 + 2 * extra) * count,
            special_operation_counts={"exp": count, "tanh": count * (1 + extra)})
    if kind == "swigluoai":
        return OperationClassBreakdown(elementwise=6 * count,
            special_operation_counts={"exp": count, "clamp_compare": 3 * count})
    if kind == "gelu":
        return OperationClassBreakdown(elementwise=4 * count, special_operation_counts={"erf": count})
    if kind in {"relu", "relu2", "squared_relu"}:
        return OperationClassBreakdown(elementwise=count if kind != "relu" else 0,
                                       special_operation_counts={"relu_compare": count})
    return OperationClassBreakdown(counting_omissions=(f"activation:{kind}",))


def _softmax_operations(rows: int, entries: int) -> OperationClassBreakdown:
    if not rows:
        return OperationClassBreakdown()
    arithmetic, special = softmax_work(rows, entries)
    return OperationClassBreakdown(elementwise=arithmetic, special_operation_counts=special)


def _scalar_profile_work(profile, q_len, batch_size):
    if profile is None:
        return OperationClassBreakdown(counting_omissions=("norm_rope_bias_schedule_unverified",))
    rows = batch_size * token_count(q_len)
    result = OperationClassBreakdown(counting_omissions=tuple(profile.get("omissions", ())))
    for norm in profile.get("norms", ()):
        n = rows * norm["layers"] * norm.get("rows_per_token", 1)
        width = norm["width"]
        affine = norm.get("affine", True)
        if norm["kind"] == "rms":
            arithmetic = n * ((4 if affine else 3) * width + 1)
        elif norm["kind"] == "layer":
            arithmetic = n * ((7 if affine else 5) * width + 1)
            if affine and not norm.get("bias", True):
                arithmetic -= n * width
        else:
            result += OperationClassBreakdown(counting_omissions=("norm:" + norm["kind"],))
            continue
        result += OperationClassBreakdown(elementwise=arithmetic, special_operation_counts={"rsqrt": n})
    for rope in profile.get("rope", ()):
        result += OperationClassBreakdown(elementwise=3 * rows * rope["layers"] * rope["heads"] * rope["dim"])
    for bias in profile.get("bias", ()):
        result += OperationClassBreakdown(elementwise=rows * bias["layers"] * bias["width"])
    for op in profile.get("elementwise", ()):
        result += OperationClassBreakdown(elementwise=rows * op["layers"] * op["width"] * op.get("rows_per_token", 1))
    return result


def _attention_residual_work(model, q_len, batch_size, block_size):
    """Kimi's source-level attention residual stack, including its final mix."""
    if not block_size or not q_len:
        return OperationClassBreakdown()
    rows, d, blocks = batch_size * q_len, model.d_model, 0
    result = OperationClassBreakdown()
    def mix(entries):
        # Affine-free RMS, scores and weighted value contraction. The source
        # multiplies the two learned score weights once per invocation.
        return (OperationClassBreakdown(dense_gemm=4 * rows * entries * d,
                    elementwise=rows * entries * (3 * d + 1) + d,
                    special_operation_counts={"rsqrt": rows * entries})
                + _softmax_operations(rows, rows * entries))
    for layer in range(model.num_layers):
        if blocks:
            result += mix(blocks + 1)
        if layer % block_size == 0:
            blocks += 1
        else:
            result += OperationClassBreakdown(elementwise=rows * d)
        result += mix(blocks + 1)
        result += OperationClassBreakdown(elementwise=rows * d)
    return result + mix(blocks + 1)


def _scalar_profiled(method):
    @wraps(method)
    def counted(self, model, *, q_len, start_context, batch_size):
        result = method(self, model, q_len=q_len, start_context=start_context, batch_size=batch_size)
        return result + _scalar_profile_work(self.scalar_profile, q_len, batch_size)
    return counted


@dataclass(frozen=True)
class GQAPrefillFormula(_OperationFormulaMixin):
    """Materialized GQA, exact valid causal pairs, MAC=2."""
    moe: MoEProfile | None = None
    type: str = "gqa"
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None
    query_gate: bool = False

    def __post_init__(self):
        if self.type not in {"mha", "gqa", "mqa"}:
            raise ValueError("prefill formula type must be mha, gqa, or mqa")
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    @_scalar_profiled
    def operation_breakdown(self, model, *, q_len, start_context, batch_size):
        q = token_count(q_len)
        return _standard_attention_breakdown(
            model=model, q_len=q, pairs=causal_pairs(start_context, q),
            batch_size=batch_size, layer_count=model.num_layers,
            num_q_heads=model.num_q_heads, num_kv_heads=model.num_kv_heads,
            key_head_dim=model.attention.key_head_dim,
            value_head_dim=model.attention.value_head_dim, query_gate=self.query_gate,
        ) + _ffn_breakdown(model=model, q_len=q, batch_size=batch_size, ffn=self.ffn, moe=self.moe)


@dataclass(frozen=True)
class MLAPrefillFormula(_OperationFormulaMixin):
    """Expanded-KV prefill with required latent-prefix materialization."""
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    moe: MoEProfile | None = None
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None
    mla_algorithm: str = "materialized"
    restored_cache_representation: str = "latent"
    output_gate: bool = False
    type: str = field(init=False, default="mla")

    @property
    def algorithm_id(self) -> str:
        return f"{self.type}_{self.mla_algorithm}_{self.restored_cache_representation}"

    def __post_init__(self):
        for name in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            _validate_positive(getattr(self, name), name)
        if self.mla_algorithm not in {"materialized", "absorbed"}:
            raise ValueError("mla_algorithm must be materialized or absorbed")
        if self.restored_cache_representation not in {"latent", "expanded"}:
            raise ValueError("unknown restored MLA representation")
        if self.mla_algorithm == "absorbed" and self.restored_cache_representation != "latent":
            raise ValueError("absorbed MLA requires latent prefix state")
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    def mandatory_conversion_breakdown(self, model, context):
        if context.miss_len == 0 or self.mla_algorithm == "absorbed" or self.restored_cache_representation == "expanded":
            return OperationClassBreakdown()
        return OperationClassBreakdown(dense_gemm=(
            2 * context.batch_size * context.matched_prefix_len * self.kv_lora_rank
            * model.num_q_heads * (self.qk_nope_head_dim + self.v_head_dim) * model.num_layers))

    @_scalar_profiled
    def operation_breakdown(self, model, *, q_len, start_context, batch_size):
        q = token_count(q_len)
        return _mla_attention_breakdown(
            model=model, q_len=q, pairs=causal_pairs(start_context, q), batch_size=batch_size,
            layer_count=model.num_layers, q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank, qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim, value_head_dim=self.v_head_dim,
            algorithm=self.mla_algorithm, output_gate=self.output_gate,
        ) + _ffn_breakdown(model=model, q_len=q, batch_size=batch_size, ffn=self.ffn, moe=self.moe)


@dataclass(frozen=True)
class DSAPrefillFormula(MLAPrefillFormula):
    """Full causal index scoring followed by exact capped selected attention."""
    window_size: int = 2048
    global_tokens: int = 128
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_topk: int | None = None
    indexer_compute_layers: int | None = None
    index_q_input_dim: int | None = None
    index_k_input_dim: int | None = None
    index_projections: bool = True
    index_score_scale: bool = False
    index_hadamard: bool = False
    type: str = field(init=False, default="dsa")

    def __post_init__(self):
        super().__post_init__()
        _validate_positive(self.window_size, "window_size")
        if self.global_tokens < 0:
            raise ValueError("global_tokens must not be negative")
        for name in ("index_n_heads", "index_head_dim", "index_topk", "indexer_compute_layers",
                     "index_q_input_dim", "index_k_input_dim"):
            if getattr(self, name) is not None:
                _validate_positive(getattr(self, name), name)
        if (self.index_n_heads is None) != (self.index_head_dim is None):
            raise ValueError("index_n_heads and index_head_dim must be provided together")

    @_scalar_profiled
    def operation_breakdown(self, model, *, q_len, start_context, batch_size):
        q = token_count(q_len)
        full_pairs = causal_pairs(start_context, q)
        cap = self.index_topk if self.index_topk is not None else self.window_size + self.global_tokens
        selected = causal_pairs(start_context, q, cap=cap)
        main = _mla_attention_breakdown(
            model=model, q_len=q, pairs=selected, batch_size=batch_size,
            layer_count=model.num_layers, q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank, qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim, value_head_dim=self.v_head_dim,
            algorithm=self.mla_algorithm, output_gate=self.output_gate, sparse=True)
        ih = self.index_n_heads or model.num_q_heads
        di = self.index_head_dim or self.qk_rope_head_dim
        layers = self.indexer_compute_layers if self.indexer_compute_layers is not None else model.num_layers
        projections = 0
        if self.index_projections:
            projections = 2 * batch_size * q * layers * (
                (self.index_q_input_dim or self.q_lora_rank) * ih * di
                + (self.index_k_input_dim or model.d_model) * di + model.d_model * ih)
        # Dot products plus per-head weights and reduction. ReLU comparisons
        # and top-k selections are named special work, not floating MACs.
        score = batch_size * layers * full_pairs * (2 * ih * di + ih + max(ih - 1, 0)
                                                   + (ih if self.index_score_scale else 0))
        rotation = 0
        if self.index_hadamard:
            if di <= 0 or di & (di-1):
                raise ValueError("Hadamard index dimension must be a power of two")
            # A fast Hadamard has log2(D) butterfly stages, D add/sub each,
            # followed by D scale multiplications, for each Q head and new K.
            rotation = batch_size * layers * q * (ih+1) * di * di.bit_length()
        return main + OperationClassBreakdown(dense_gemm=projections, indexer=score,
            elementwise=rotation,
            special_operation_counts={"relu_compare": batch_size * layers * full_pairs * ih,
                                      "topk_select": batch_size * layers * q,
                                      **({"quantized_index_elements": batch_size*layers*q*(ih+1)*di}
                                         if self.index_hadamard else {})},
            counting_omissions=(("indexer_quantization_rounding_and_scale_kernel_arithmetic",)
                                if self.index_hadamard else ())
        ) + _ffn_breakdown(
            model=model, q_len=q, batch_size=batch_size, ffn=self.ffn, moe=self.moe)


def _causal_entry_sum(end: int, ratio: int = 1, cap: int | None = None) -> int:
    """Sum visible floor(i / ratio) entries for positions 1..end, optionally capped."""
    if end <= 0:
        return 0
    limit = end if cap is None else min(end, ratio * cap - 1)
    quotient, remainder = divmod(limit, ratio)
    result = ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)
    return result + (end - limit) * (cap or 0)


@dataclass(frozen=True)
class DeepSeekV41PrefillFormula:
    """Analytical production CED schedule, not the reference engine's dense loops.

    MACs count as two FLOPs. Index selection/comparisons, normalization,
    quantization and copies remain timing omissions; no measured speed claim.
    """

    head_dim: int = 512
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_positions: int = 16384
    candidate_block_size: int = 8
    n_routed_experts: int = 384
    experts_per_token: int = 6
    n_shared_experts: int = 1
    expert_intermediate_size: int = 2304
    hc_mult: int = 4
    sliding_window: int = 128
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_ngram_orders: int = 3
    rope_head_dim: int = 64
    hc_sinkhorn_iters: int = 20
    vocab_size: int = 129280
    type: str = field(init=False, default="deepseek_v41")

    def __post_init__(self) -> None:
        for name in ("head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "index_n_heads",
                     "index_head_dim", "index_topk", "candidate_positions", "n_routed_experts",
                     "experts_per_token", "n_shared_experts", "expert_intermediate_size", "hc_mult",
                     "sliding_window", "engram_n_heads", "engram_head_dim", "engram_ngram_orders",
                     "rope_head_dim", "hc_sinkhorn_iters", "vocab_size", "candidate_block_size"):
            _validate_positive(getattr(self, name), name)
        if self.candidate_positions % self.candidate_block_size:
            raise ValueError("candidate_positions must contain a whole number of candidate blocks")

    def _segment(self, model, start, end, *, encoder, swa_start,
                 generate_global=False, compression_tail=False, batch_size=1):
        ledger = v41_segment(model, self, start, end, encoder=encoder, swa_start=swa_start,
                             generate_global=generate_global, compression_tail=compression_tail,
                             batch=batch_size)
        return ledger.components, OperationClassBreakdown(
            **ledger.operations, special_operation_counts=ledger.special,
            counting_omissions=self._counting_omissions())

    def _counting_omissions(self) -> tuple[str, ...]:
        return (
            "production CED schedule differs from the dense reference engine loops",
            "bounded encoder replay is an approximate state-recovery scenario",
            "quantization rounding/saturation arithmetic is a separate special operation",
            "candidate selection/top-k comparisons depend on the selected kernel",
            "hardware padding/fusion and Engram host transfers are not FLOPs",
        )

    def _phase_work(self, model: ComputeModelLike, context: PrefillComputeContext):
        policy = PrefixCachePolicy(encoder_state=context.encoder_state,
                                   encoder_tail_policy=context.encoder_tail_policy)
        plan = model.attention.recovery_plan(model, context.matched_prefix_len, context.seq_len, policy)
        h, n, b = context.matched_prefix_len, context.seq_len, context.batch_size
        r = plan.restore_recompute_tokens
        missing_state = h > 0 and context.encoder_state == "unavailable"
        work = {}
        work["encoder_restore"] = self._segment(
            model, h-r, h, encoder=True, swa_start=h-r,
            compression_tail=missing_state, batch_size=b)
        work["encoder_suffix"] = self._segment(
            model, h, n, encoder=True, swa_start=h-r if missing_state else 0,
            generate_global=True, batch_size=b)
        count, d, dim = n-h, model.d_model, self.head_dim
        projection = b * count * 2 * dim * (d + self.index_head_dim)
        mixing = b * count * (2 * self.hc_mult - 1) * d
        norms = b * count * (4 * dim + 1 + 4 * self.index_head_dim + 1)
        input_norm = b * count * (4*d+1)
        rope = b * count * 6 * self.rope_head_dim
        work["decoder_global_projection"] = (
            {"decoder_global_projection": projection, "mhc_first_input_mix": mixing,
             "decoder_global_normalization": norms, "decoder_global_rope": rope,
             "decoder_input_normalization": input_norm},
            OperationClassBreakdown(dense_gemm=projection, elementwise=mixing+norms+rope+input_norm,
                                    special_operation_counts={"rsqrt": 3*b*count,
                                        "quantized_cache_elements": b*count*(dim+self.index_head_dim)}))
        tail_start = max(0, n-self.sliding_window)
        parts, ops = self._segment(model, tail_start, n, encoder=False,
                                   swa_start=tail_start, batch_size=b)
        # The first decoder input of new tail positions was already collapsed
        # for their global KV. It has one owner and is not computed twice.
        reused_mix = b * min(count,self.sliding_window) * (2*self.hc_mult-1) * d
        parts["mhc_elementwise"] = parts.get("mhc_elementwise",0)-reused_mix
        reused_norm = b * min(count,self.sliding_window) * (4*d+1)
        parts["normalization"] = parts.get("normalization",0)-reused_norm
        values = ops.as_dict()
        values["elementwise"] -= reused_mix+reused_norm
        head_projection = b * 2 * d * self.vocab_size if n else 0
        head_scalar = b * ((2*self.hc_mult-1)*d + 4*d+1) if n else 0
        parts["output_head"] = head_projection+head_scalar
        values["dense_gemm"] += head_projection
        values["elementwise"] += head_scalar
        specials = dict(ops.special_operation_counts)
        specials["rsqrt"] = specials.get("rsqrt",0)+(b if n else 0)-b*min(count,self.sliding_window)
        work["decoder_tail"] = (parts, OperationClassBreakdown(
            **values, special_operation_counts=specials, counting_omissions=self._counting_omissions()))
        return plan, work

    def cache_compute_breakdown(self, model: ComputeModelLike, context: PrefillComputeContext,
                                *, restore_recompute_tokens: int = 0) -> CacheComputeBreakdown:
        plan, work = self._phase_work(model, context)
        _, cold = self._phase_work(model, PrefillComputeContext(
            context.seq_len, 0, context.generation_len, context.batch_size))
        def totals(values):
            parts = {}
            ops = OperationClassBreakdown()
            for breakdown, operation in values.values():
                for key,value in breakdown.items():
                    parts[key] = parts.get(key,0)+value
                ops = ops+operation
            return parts,ops
        parts,ops = totals(work)
        cold_parts,cold_ops = totals(cold)
        restore = work["encoder_restore"][1]
        body = work["encoder_suffix"][1]+work["decoder_global_projection"][1]
        tail = work["decoder_tail"][1]
        d,b,n = model.d_model,context.batch_size,context.seq_len
        head = OperationClassBreakdown(
            dense_gemm=b*2*d*self.vocab_size if n else 0,
            elementwise=b*((2*self.hc_mult-1)*d+4*d+1) if n else 0,
            special_operation_counts={"rsqrt": b if n else 0})
        final_values = {k:tail.as_dict()[k]-head.as_dict()[k] for k in tail.as_dict()}
        final_special = {k:v-head.special_operation_counts.get(k,0)
                         for k,v in tail.special_operation_counts.items()}
        final = OperationClassBreakdown(**final_values, special_operation_counts=final_special)
        suffix = body+final if context.miss_len else OperationClassBreakdown()
        net = cold_ops.total_flops-ops.total_flops
        return CacheComputeBreakdown(
            no_cache_flops=cold_ops.total_flops, cache_path_flops=ops.total_flops,
            miss_suffix_flops=suffix.total_flops,
            compression_tail_flops=parts.get("compression_tail",0),
            swa_restore_flops=work["encoder_restore"][0].get("swa",0),
            restore_recompute_flops=restore.total_flops, saved_flops=max(net,0),
            restore_recompute_tokens=plan.restore_recompute_tokens,
            cache_path_by_component=parts, no_cache_by_component=cold_parts, net_saved_flops=net,
            cache_path_by_operation_class=ops.as_dict(), no_cache_by_operation_class=cold_ops.as_dict(),
            phase_flops={name:operation.total_flops for name,(_,operation) in work.items()},
            phase_by_component={name:values for name,(values,_) in work.items()},
            phase_by_operation_class={name:operation.as_dict() for name,(_,operation) in work.items()},
            execution_plan=plan, suffix_body_flops=body.total_flops,
            finalization_flops=final.total_flops, output_head_flops=head.total_flops,
            suffix_by_operation_class=suffix.as_dict(), restore_by_operation_class=restore.as_dict(),
            finalization_by_operation_class=final.as_dict(), output_head_by_operation_class=head.as_dict(),
            special_operation_counts=ops.special_operation_counts,
            counting_omissions=self._counting_omissions(),
            algorithm_id="deepseek_v41_production_ced_bounded128",
        )

    def actual_prefill_flops(self, model, context) -> float:
        return self.cache_compute_breakdown(model, context).cache_path_flops

    def saved_recompute_flops(self, model, context) -> float:
        return self.cache_compute_breakdown(model, context).saved_flops

    def operation_breakdown(self, model, *, q_len: float, start_context: float, batch_size: int):
        context = PrefillComputeContext(int(start_context + q_len), int(start_context), batch_size=batch_size)
        return OperationClassBreakdown(**self.cache_compute_breakdown(model, context).cache_path_by_operation_class)


@dataclass(frozen=True)
class DeepSeekV4PrefillFormula:
    """Analytical DeepSeek-V4 prefill model for hybrid SWA/CSA/HCA layers."""

    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_routed_experts: int
    experts_per_token: int
    n_shared_experts: int
    expert_intermediate_size: int
    hash_routing_layers: int = 3
    mhc_multiplier: int = 4
    sliding_window: int = 128
    hc_sinkhorn_iters: int = 20
    vocab_size: int = 129280
    type: str = field(init=False, default="deepseek_v4")

    def __post_init__(self) -> None:
        for name in (
            "head_dim",
            "rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "n_routed_experts",
            "experts_per_token",
            "n_shared_experts",
            "expert_intermediate_size",
            "mhc_multiplier",
            "sliding_window",
            "hc_sinkhorn_iters",
            "vocab_size",
        ):
            _validate_positive(getattr(self, name), name)
        if self.hash_routing_layers < 0:
            raise ValueError("hash_routing_layers must not be negative")
        if self.rope_head_dim >= self.head_dim:
            raise ValueError("rope_head_dim must be smaller than head_dim")

    def saved_recompute_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        return self.cache_compute_breakdown(model, context).saved_flops

    def actual_prefill_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        return self.cache_compute_breakdown(model, context).cache_path_flops

    def operation_breakdown(
        self, model, *, q_len, start_context, batch_size,
    ) -> OperationClassBreakdown:
        ledger = v4_segment(model, self, token_count(start_context),
                            token_count(start_context) + token_count(q_len), batch_size)
        return OperationClassBreakdown(
            **ledger.operations, special_operation_counts=ledger.special,
            counting_omissions=self._counting_omissions())

    def _counting_omissions(self) -> tuple[str, ...]:
        return (
            "quantization rounding/saturation arithmetic is a separate special operation",
            "top-k comparisons depend on the selected kernel",
            "hardware padding/fusion is outside logical arithmetic",
            "compressor checkpoint state is assumed available at a reusable boundary",
        )

    def cache_compute_breakdown(
        self, model, context, *, restore_recompute_tokens: int = 0,
    ) -> CacheComputeBreakdown:
        if not 0 <= restore_recompute_tokens <= context.matched_prefix_len:
            raise ValueError("restore_recompute_tokens must be within matched prefix")
        n, h, r, batch = context.seq_len, context.matched_prefix_len, restore_recompute_tokens, context.batch_size
        cold = v4_segment(model, self, 0, n, batch)
        restore = v4_segment(model, self, h-r, h, batch)
        suffix = v4_segment(model, self, h, n, batch)
        def operations(ledger):
            return OperationClassBreakdown(**ledger.operations, special_operation_counts=ledger.special,
                                           counting_omissions=self._counting_omissions())
        cold_ops, restore_ops, suffix_ops = operations(cold), operations(restore), operations(suffix)
        head_ledger = v4_head(model, self, batch)
        cold_head = operations(head_ledger) if n else OperationClassBreakdown()
        head = cold_head if context.miss_len or r else OperationClassBreakdown()
        cold_ops = cold_ops + cold_head
        cache_ops = restore_ops + suffix_ops + head
        names = cold.components.keys() | restore.components.keys() | suffix.components.keys()
        cache_parts = {key: restore.components.get(key,0)+suffix.components.get(key,0) for key in names}
        cold.components["output_head"] = cold_head.total_flops
        cache_parts["output_head"] = head.total_flops
        omissions = self._counting_omissions()
        if n and not context.miss_len and not r:
            omissions += ("full_hit_finalization_requires_hidden_or_logits",)
        net = cold_ops.total_flops-cache_ops.total_flops
        return CacheComputeBreakdown(
            no_cache_flops=cold_ops.total_flops, cache_path_flops=cache_ops.total_flops,
            miss_suffix_flops=suffix_ops.total_flops,
            compression_tail_flops=restore.components.get("compression",0)+restore.components.get("compression_elementwise",0),
            swa_restore_flops=restore.components.get("swa",0),
            restore_recompute_flops=restore_ops.total_flops, saved_flops=max(net,0),
            restore_recompute_tokens=r, cache_path_by_component=cache_parts,
            no_cache_by_component=cold.components, net_saved_flops=net,
            cache_path_by_operation_class=cache_ops.as_dict(),
            no_cache_by_operation_class=cold_ops.as_dict(),
            suffix_body_flops=suffix_ops.total_flops,
            output_head_flops=head.total_flops,
            output_head_by_operation_class=head.as_dict(),
            suffix_by_operation_class=suffix_ops.as_dict(),
            restore_by_operation_class=restore_ops.as_dict(),
            special_operation_counts=cache_ops.special_operation_counts,
            counting_omissions=omissions,
            algorithm_id="deepseek_v4_official_logical_causal_swa_csa_hca",
        )

    def _segment_breakdown(self, model, *, q_len, start_context, batch_size) -> dict[str, int]:
        return v4_segment(model, self, token_count(start_context),
                          token_count(start_context)+token_count(q_len), batch_size).components

def _standard_attention_breakdown(
    *, model, q_len, batch_size, layer_count, num_q_heads, num_kv_heads,
    key_head_dim, value_head_dim, pairs=None, avg_kv_len=None, sparse=False,
    kv_projection_layers=None, key_equals_value=False, query_gate=False, softmax_sink_entries=0,
):
    q = token_count(q_len)
    if not q or not layer_count:
        return OperationClassBreakdown()
    if pairs is None:
        raise ValueError("attention requires an explicit integer pair count")
    kv_layers = layer_count if kv_projection_layers is None else kv_projection_layers
    if not 0 <= kv_layers <= layer_count:
        raise ValueError("KV producer layers must be within executed layers")
    if key_equals_value and key_head_dim != value_head_dim:
        raise ValueError("shared K/V projection requires equal head dimensions")
    q_dim, o_dim = num_q_heads * key_head_dim, num_q_heads * value_head_dim
    kv_dim = num_kv_heads * (key_head_dim + (0 if key_equals_value else value_head_dim))
    projection = 2 * batch_size * q * model.d_model * (
        (q_dim * (2 if query_gate else 1) + o_dim) * layer_count + kv_dim * kv_layers)
    core = 2 * batch_size * pairs * num_q_heads * (key_head_dim + value_head_dim) * layer_count
    rows, entries = batch_size * q * num_q_heads * layer_count, batch_size * pairs * num_q_heads * layer_count
    return OperationClassBreakdown(
        dense_gemm=projection, sparse_attention=core if sparse else 0,
        dense_attention=0 if sparse else core,
        elementwise=entries + (batch_size * q * o_dim * layer_count if query_gate else 0)
    ) + _softmax_operations(rows, entries + rows * softmax_sink_entries) + (
        _activation_work("sigmoid", batch_size * q * o_dim * layer_count) if query_gate else OperationClassBreakdown())


def _mla_attention_breakdown(
    *, model, q_len, batch_size, layer_count, q_lora_rank, kv_lora_rank,
    qk_nope_head_dim, qk_rope_head_dim, value_head_dim, pairs=None,
    avg_kv_len=None, algorithm="materialized", output_gate=False, sparse=False,
):
    q = token_count(q_len)
    if not q or not layer_count:
        return OperationClassBreakdown()
    if pairs is None:
        raise ValueError("MLA requires an explicit integer pair count")
    h, d, r = model.num_q_heads, model.d_model, kv_lora_rank
    projection = 2 * batch_size * q * (
        d * (q_lora_rank + r + qk_rope_head_dim)
        + q_lora_rank * h * (qk_nope_head_dim + qk_rope_head_dim)
        + h * value_head_dim * d)
    if algorithm == "materialized":
        projection += 2 * batch_size * q * r * h * (qk_nope_head_dim + value_head_dim)
        core = 2 * batch_size * pairs * h * (qk_nope_head_dim + qk_rope_head_dim + value_head_dim)
    elif algorithm == "absorbed":
        projection += 2 * batch_size * q * h * r * (qk_nope_head_dim + value_head_dim)
        core = 2 * batch_size * pairs * h * (2 * r + qk_rope_head_dim)
    else:
        raise ValueError("unknown MLA algorithm")
    if output_gate:
        projection += 2 * batch_size * q * d * h * value_head_dim
    rows, entries = batch_size * q * h * layer_count, batch_size * pairs * h * layer_count
    return OperationClassBreakdown(
        dense_gemm=projection * layer_count,
        dense_attention=0 if sparse else core * layer_count,
        sparse_attention=core * layer_count if sparse else 0,
        elementwise=entries + (batch_size * q * h * value_head_dim * layer_count if output_gate else 0)
    ) + _softmax_operations(rows, entries) + (
        _activation_work("sigmoid", batch_size * q * h * value_head_dim * layer_count) if output_gate else OperationClassBreakdown())


@dataclass(frozen=True)
class AnalyticalFallbackPrefillFormula(_OperationFormulaMixin):
    """Explicit dense-equivalent fallback for a not-yet-specialized family.

    It is intentionally named ``fallback`` so reports cannot mistake this
    shape-based estimate for a kernel-calibrated implementation.
    """

    type: str
    layer_count: int | None = None
    projection_flops_per_token_per_layer: float | None = None
    attention_flops_per_query_key_per_layer: float | None = None
    context_cap: int | None = None
    sparse_attention: bool = False
    moe: MoEProfile | None = None
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("fallback prefill type must not be empty")
        if self.layer_count is not None:
            _validate_positive(self.layer_count, "fallback.layer_count")
        for value, name in (
            (
                self.projection_flops_per_token_per_layer,
                "projection_flops_per_token_per_layer",
            ),
            (
                self.attention_flops_per_query_key_per_layer,
                "attention_flops_per_query_key_per_layer",
            ),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"fallback.{name} must be finite and non-negative")
        if self.context_cap is not None:
            _validate_positive(self.context_cap, "fallback.context_cap")
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    @_scalar_profiled
    def operation_breakdown(
        self,
        model: ComputeModelLike,
        *,
        q_len: float,
        start_context: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        if q_len <= 0:
            return OperationClassBreakdown()
        layers = self.layer_count if self.layer_count is not None else model.num_layers
        projection_per_token = (
            self.projection_flops_per_token_per_layer
            if self.projection_flops_per_token_per_layer is not None
            else 8.0 * model.d_model * model.d_model
        )
        attention_per_pair = (
            self.attention_flops_per_query_key_per_layer
            if self.attention_flops_per_query_key_per_layer is not None
            else 4.0 * model.d_model
        )
        pairs = causal_pairs(start_context, q_len, cap=self.context_cap)
        projection = batch_size * q_len * projection_per_token * layers
        attention = (
            batch_size * pairs * attention_per_pair * layers
        )
        operations = OperationClassBreakdown(
            dense_gemm=projection,
            dense_attention=0.0 if self.sparse_attention else attention,
            sparse_attention=attention if self.sparse_attention else 0.0,
        )
        return operations + _ffn_breakdown(
            model=model,
            q_len=q_len,
            batch_size=batch_size,
            ffn=self.ffn,
            moe=self.moe,
        )


@dataclass(frozen=True)
class MixedFullSlidingPrefillFormula(_OperationFormulaMixin):
    """Analytical full/sliding GQA layer mix."""

    full_attention_layers: int
    sliding_attention_layers: int
    sliding_window: int
    full_key_head_dim: int
    full_value_head_dim: int
    sliding_key_head_dim: int
    sliding_value_head_dim: int
    full_num_kv_heads: int | None = None
    sliding_num_kv_heads: int | None = None
    sliding_num_q_heads: int | None = None
    full_kv_projection_layers: int | None = None
    sliding_kv_projection_layers: int | None = None
    full_key_equals_value: bool = False
    per_layer_input_dim: int = 0
    full_softmax_sink_entries: int = 0
    sliding_softmax_sink_entries: int = 0
    moe: MoEProfile | None = None
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None
    type: str = field(init=False, default="mixed_full_sliding_gqa")

    def __post_init__(self) -> None:
        for value, name in (
            (self.full_attention_layers, "full_attention_layers"),
            (self.sliding_attention_layers, "sliding_attention_layers"),
            (self.sliding_window, "sliding_window"),
            (self.full_key_head_dim, "full_key_head_dim"),
            (self.full_value_head_dim, "full_value_head_dim"),
            (self.sliding_key_head_dim, "sliding_key_head_dim"),
            (self.sliding_value_head_dim, "sliding_value_head_dim"),
        ):
            _validate_positive(value, name)
        for value, name in (
            (self.full_num_kv_heads, "full_num_kv_heads"),
            (self.sliding_num_kv_heads, "sliding_num_kv_heads"),
            (self.sliding_num_q_heads, "sliding_num_q_heads"),
        ):
            if value is not None:
                _validate_positive(value, name)
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    @_scalar_profiled
    def operation_breakdown(
        self,
        model: ComputeModelLike,
        *,
        q_len: float,
        start_context: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        q_len = token_count(q_len)
        full = _standard_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=causal_pairs(start_context, q_len),
            batch_size=batch_size,
            layer_count=self.full_attention_layers,
            num_q_heads=model.num_q_heads,
            num_kv_heads=self.full_num_kv_heads or model.num_kv_heads,
            key_head_dim=self.full_key_head_dim,
            value_head_dim=self.full_value_head_dim,
            kv_projection_layers=self.full_kv_projection_layers,
            key_equals_value=self.full_key_equals_value,
            softmax_sink_entries=self.full_softmax_sink_entries,
        )
        sliding = _standard_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=causal_pairs(start_context, q_len, cap=self.sliding_window),
            batch_size=batch_size,
            layer_count=self.sliding_attention_layers,
            num_q_heads=self.sliding_num_q_heads or model.num_q_heads,
            num_kv_heads=self.sliding_num_kv_heads or model.num_kv_heads,
            key_head_dim=self.sliding_key_head_dim,
            value_head_dim=self.sliding_value_head_dim,
            kv_projection_layers=self.sliding_kv_projection_layers,
            softmax_sink_entries=self.sliding_softmax_sink_entries,
        )
        # Gemma per-layer inputs: global D->L*P, then each layer D->P->D.
        ple = OperationClassBreakdown(dense_gemm=6 * batch_size * q_len * model.d_model
                                      * self.per_layer_input_dim * model.num_layers)
        return full + sliding + ple + _activation_work("gelu_pytorch_tanh",
            batch_size * q_len * self.per_layer_input_dim * model.num_layers) + _ffn_breakdown(
            model=model,
            q_len=q_len,
            batch_size=batch_size,
            ffn=self.ffn,
            moe=self.moe,
        )


@dataclass(frozen=True)
class MiniMaxMSAPrefillFormula(_OperationFormulaMixin):
    """MiniMax MSA: dense GQA layers plus block-sparse GQA layers."""

    full_attention_layers: int
    sparse_attention_layers: int
    key_head_dim: int
    value_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk_blocks: int
    index_block_size: int
    local_blocks: int = 1
    algorithm_id: str = "minimax_m3_valid_causal_blocks_with_local_boost"
    counting_omissions: tuple[str, ...] = ("executed_dense_masked_vs_sparse_kernel_work_not_equated",)
    moe: MoEProfile | None = None
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None
    type: str = field(init=False, default="minimax_msa")

    def __post_init__(self) -> None:
        for value, name in (
            (self.full_attention_layers, "full_attention_layers"),
            (self.sparse_attention_layers, "sparse_attention_layers"),
            (self.key_head_dim, "key_head_dim"),
            (self.value_head_dim, "value_head_dim"),
            (self.index_n_heads, "index_n_heads"),
            (self.index_head_dim, "index_head_dim"),
            (self.index_topk_blocks, "index_topk_blocks"),
            (self.index_block_size, "index_block_size"),
        ):
            _validate_positive(value, name)
        if self.local_blocks < 0:
            raise ValueError("local_blocks must not be negative")
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    @_scalar_profiled
    def operation_breakdown(
        self,
        model: ComputeModelLike,
        *,
        q_len: float,
        start_context: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        q_len = token_count(q_len)
        full = _standard_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=causal_pairs(start_context, q_len),
            batch_size=batch_size,
            layer_count=self.full_attention_layers,
            num_q_heads=model.num_q_heads,
            num_kv_heads=model.num_kv_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
        )
        if not 1 <= self.local_blocks <= self.index_topk_blocks:
            raise ValueError("exact MSA count requires current-block local boost within top-k slots")
        selected_pairs = block_sparse_pairs(start_context, q_len, block_size=self.index_block_size,
                                             topk_blocks=self.index_topk_blocks)
        sparse = _standard_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=selected_pairs,
            batch_size=batch_size,
            layer_count=self.sparse_attention_layers,
            num_q_heads=model.num_q_heads,
            num_kv_heads=model.num_kv_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            sparse=True,
        )
        indexer = (
            2
            * batch_size
            * causal_pairs(start_context, q_len)
            * self.index_n_heads
            * self.index_head_dim
            * self.sparse_attention_layers
        )
        index_projection = (
            2
            * batch_size
            * q_len
            * model.d_model
            * (self.index_n_heads * self.index_head_dim + self.index_head_dim)
            * self.sparse_attention_layers
        )
        return full + sparse + OperationClassBreakdown(
            dense_gemm=index_projection, indexer=indexer,
            special_operation_counts={
                "max_pool_compare": batch_size * self.index_n_heads * self.sparse_attention_layers * (
                    causal_pairs(start_context, q_len)
                    - causal_pairs(start_context + self.index_block_size - 1, q_len, ratio=self.index_block_size)),
                "topk_select": batch_size * self.index_n_heads * q_len * self.sparse_attention_layers},
        ) + _ffn_breakdown(
            model=model,
            q_len=q_len,
            batch_size=batch_size,
            ffn=self.ffn,
            moe=self.moe,
        )


@dataclass(frozen=True)
class QwenLinearHybridPrefillFormula(_OperationFormulaMixin):
    """Full GQA plus analytical Gated-DeltaNet linear-attention work."""

    full_attention_layers: int
    linear_attention_layers: int
    full_key_head_dim: int
    full_value_head_dim: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    chunk_size: int = 64
    algorithm_id: str = "qwen_torch_chunk64_unittriangular_or_cached_recurrence"
    counting_omissions: tuple[str, ...] = ("fused_kernel_schedule_not_verified",)
    moe: MoEProfile | None = None
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None
    type: str = field(init=False, default="qwen_linear_full_hybrid")

    def __post_init__(self) -> None:
        for value, name in (
            (self.full_attention_layers, "full_attention_layers"),
            (self.linear_attention_layers, "linear_attention_layers"),
            (self.full_key_head_dim, "full_key_head_dim"),
            (self.full_value_head_dim, "full_value_head_dim"),
            (self.linear_num_key_heads, "linear_num_key_heads"),
            (self.linear_key_head_dim, "linear_key_head_dim"),
            (self.linear_num_value_heads, "linear_num_value_heads"),
            (self.linear_value_head_dim, "linear_value_head_dim"),
            (self.linear_conv_kernel_dim, "linear_conv_kernel_dim"),
        ):
            _validate_positive(value, name)
        if self.chunk_size not in {32, 64}:
            raise ValueError("chunk_size must be 32 or 64")
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    @_scalar_profiled
    def operation_breakdown(
        self,
        model: ComputeModelLike,
        *,
        q_len: float,
        start_context: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        q_len = token_count(q_len)
        full = _standard_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=causal_pairs(start_context, q_len),
            batch_size=batch_size,
            layer_count=self.full_attention_layers,
            num_q_heads=model.num_q_heads,
            num_kv_heads=model.num_kv_heads,
            key_head_dim=self.full_key_head_dim,
            value_head_dim=self.full_value_head_dim,
            query_gate=True,
        )
        state_width = (
            2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
        )
        linear_projection = (
            2 * batch_size * q_len * model.d_model * state_width
            + 2
            * batch_size
            * q_len
            * self.linear_num_value_heads
            * self.linear_value_head_dim
            * model.d_model
        ) * self.linear_attention_layers
        convolution = (
            2
            * batch_size
            * q_len
            * self.linear_conv_kernel_dim
            * state_width
            * self.linear_attention_layers
        )
        # z (output gate), a (decay) and b (update gate) are independent projections.
        linear_projection += 2 * batch_size * q_len * model.d_model * (
            self.linear_num_value_heads * self.linear_value_head_dim + 2 * self.linear_num_value_heads
        ) * self.linear_attention_layers
        work = delta_rule_work(q_len, self.linear_key_head_dim, self.linear_value_head_dim,
                               cached=start_context > 0, chunk_size=self.chunk_size)
        scale = batch_size * self.linear_num_value_heads * self.linear_attention_layers
        recurrent = scale * work["contractions"]
        nonlinear = _activation_work("silu", batch_size * q_len * state_width * self.linear_attention_layers)
        nonlinear += _activation_work("silu", scale * q_len * self.linear_value_head_dim)
        nonlinear += _activation_work("sigmoid", scale * q_len)
        # Softplus(a + dt_bias), -exp(A_log) * softplus; stable implementation
        # of softplus and the head-wise exponent are separated from arithmetic.
        parameter_heads = self.linear_num_value_heads * self.linear_attention_layers if q_len else 0
        nonlinear += OperationClassBreakdown(elementwise=scale * q_len * (2 + self.linear_value_head_dim) + parameter_heads,
            special_operation_counts={"softplus": scale * q_len, "exp": parameter_heads})
        return full + OperationClassBreakdown(
            dense_gemm=linear_projection,
            convolution=convolution,
            recurrent=recurrent,
            elementwise=scale * work["elementwise"],
            special_operation_counts={"exp": scale * work["exp"], "rsqrt": scale * work["rsqrt"]},
        ) + nonlinear + _ffn_breakdown(
            model=model,
            q_len=q_len,
            batch_size=batch_size,
            ffn=self.ffn,
            moe=self.moe,
        )


@dataclass(frozen=True)
class KDAHybridPrefillFormula(_OperationFormulaMixin):
    """Kimi KDA/MLA hybrid analytical formula."""

    mla_attention_layers: int
    kda_attention_layers: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    value_head_dim: int
    kda_num_key_heads: int
    kda_key_head_dim: int
    kda_num_value_heads: int
    kda_value_head_dim: int
    kda_conv_kernel_size: int
    mla_output_gate: bool = True
    kda_gate_rank: int | None = None
    kda_full_rank_gate: bool = True
    gate_lower_bound: float | None = None
    attention_residual_block_size: int | None = None
    chunk_size: int = 64
    algorithm_id: str = "fla_cd61447_kda_chunk64_reference_or_cached_recurrence"
    counting_omissions: tuple[str, ...] = ("fla_version_assumed_not_pinned_by_model_release", "fused_kernel_schedule_not_verified")
    moe: MoEProfile | None = None
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None
    type: str = field(init=False, default="kimi_kda_mla_hybrid")

    def mandatory_conversion_breakdown(self, model, context):
        if not context.miss_len:
            return OperationClassBreakdown()
        return OperationClassBreakdown(dense_gemm=2 * context.batch_size * context.matched_prefix_len
            * self.kv_lora_rank * model.num_q_heads * (self.qk_nope_head_dim + self.value_head_dim)
            * self.mla_attention_layers)

    def __post_init__(self) -> None:
        for name in (
            "mla_attention_layers",
            "kda_attention_layers",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "value_head_dim",
            "kda_num_key_heads",
            "kda_key_head_dim",
            "kda_num_value_heads",
            "kda_value_head_dim",
            "kda_conv_kernel_size",
        ):
            _validate_positive(getattr(self, name), name)
        if self.chunk_size not in {32, 64}:
            raise ValueError("chunk_size must be 32 or 64")
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    @_scalar_profiled
    def operation_breakdown(
        self,
        model: ComputeModelLike,
        *,
        q_len: float,
        start_context: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        q_len = token_count(q_len)
        mla = _mla_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=causal_pairs(start_context, q_len),
            batch_size=batch_size,
            layer_count=self.mla_attention_layers,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            value_head_dim=self.value_head_dim,
            output_gate=self.mla_output_gate,
        )
        kda_key_width = self.kda_num_key_heads * self.kda_key_head_dim
        kda_value_width = self.kda_num_value_heads * self.kda_value_head_dim
        state_width = 2 * kda_key_width + kda_value_width
        projection = (
            2 * batch_size * q_len * model.d_model * state_width
            + 2 * batch_size * q_len * kda_value_width * model.d_model
        ) * self.kda_attention_layers
        convolution = (
            2
            * batch_size
            * q_len
            * self.kda_conv_kernel_size
            * state_width
            * self.kda_attention_layers
        )
        rank = self.kda_gate_rank or self.kda_key_head_dim
        # f is low-rank; the released Kimi K3 output gate is full-rank.
        output_gate_projection = (model.d_model * kda_value_width if self.kda_full_rank_gate
                                  else model.d_model * rank + rank * kda_value_width)
        projection += 2 * batch_size * q_len * (model.d_model * rank + rank * kda_value_width + output_gate_projection
                                               + model.d_model * self.kda_num_value_heads) * self.kda_attention_layers
        work = delta_rule_work(q_len, self.kda_key_head_dim, self.kda_value_head_dim,
                               cached=start_context > 0, chunk_size=self.chunk_size, channelwise=True)
        scale = batch_size * self.kda_num_value_heads * self.kda_attention_layers
        recurrent = scale * work["contractions"]
        nonlinear = _activation_work("silu", batch_size * q_len * state_width * self.kda_attention_layers)
        nonlinear += _activation_work("sigmoid", scale * q_len * (1 + self.kda_value_head_dim))
        gate_elements = scale * q_len * self.kda_key_head_dim
        parameter_heads = self.kda_num_value_heads * self.kda_attention_layers if q_len else 0
        gate_arithmetic = (3 * gate_elements if self.gate_lower_bound is not None
                           else 2 * gate_elements + parameter_heads)
        nonlinear += OperationClassBreakdown(elementwise=gate_arithmetic + scale * q_len * self.kda_value_head_dim,
            special_operation_counts={"exp": parameter_heads})
        if self.gate_lower_bound is not None:
            nonlinear += _activation_work("sigmoid", gate_elements)
        else:
            nonlinear += OperationClassBreakdown(special_operation_counts={"softplus": gate_elements})
        return mla + OperationClassBreakdown(
            dense_gemm=projection,
            convolution=convolution,
            recurrent=recurrent,
            elementwise=scale * work["elementwise"],
            special_operation_counts={"exp": scale * work["exp"], "rsqrt": scale * work["rsqrt"]},
        ) + nonlinear + _attention_residual_work(model, q_len, batch_size, self.attention_residual_block_size) + _ffn_breakdown(
            model=model,
            q_len=q_len,
            batch_size=batch_size,
            ffn=self.ffn,
            moe=self.moe,
        )


@dataclass(frozen=True)
class InklingHybridPrefillFormula(_OperationFormulaMixin):
    """Inkling full/SWA attention with analytical SConv work."""

    full_attention_layers: int
    sliding_attention_layers: int
    sliding_window: int
    hidden_size: int
    full_num_kv_heads: int
    full_head_dim: int
    sliding_num_kv_heads: int
    sliding_head_dim: int
    sconv_kernel_size: int
    relative_head_dim: int = 0
    relative_extent: int = 0
    log_scaling_n_floor: int | None = None
    moe: MoEProfile | None = None
    ffn: FFNProfile | None = None
    scalar_profile: Mapping[str, Any] | None = None
    type: str = field(init=False, default="inkling_hybrid")

    def __post_init__(self) -> None:
        for name in (
            "full_attention_layers",
            "sliding_attention_layers",
            "sliding_window",
            "hidden_size",
            "full_num_kv_heads",
            "full_head_dim",
            "sliding_num_kv_heads",
            "sliding_head_dim",
            "sconv_kernel_size",
        ):
            _validate_positive(getattr(self, name), name)
        if self.moe is not None and self.ffn is not None:
            raise ValueError("configure either ffn or legacy moe, not both")

    @_scalar_profiled
    def operation_breakdown(
        self,
        model: ComputeModelLike,
        *,
        q_len: float,
        start_context: float,
        batch_size: int,
    ) -> OperationClassBreakdown:
        q_len = token_count(q_len)
        full = _standard_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=causal_pairs(start_context, q_len),
            batch_size=batch_size,
            layer_count=self.full_attention_layers,
            num_q_heads=model.num_q_heads,
            num_kv_heads=self.full_num_kv_heads,
            key_head_dim=self.full_head_dim,
            value_head_dim=self.full_head_dim,
        )
        sliding = _standard_attention_breakdown(
            model=model,
            q_len=q_len,
            pairs=causal_pairs(start_context, q_len, cap=self.sliding_window),
            batch_size=batch_size,
            layer_count=self.sliding_attention_layers,
            num_q_heads=model.num_q_heads,
            num_kv_heads=self.sliding_num_kv_heads,
            key_head_dim=self.sliding_head_dim,
            value_head_dim=self.sliding_head_dim,
        )
        channels = (
            self.full_attention_layers
            * (2 * self.full_num_kv_heads * self.full_head_dim + 2 * self.hidden_size)
            + self.sliding_attention_layers
            * (
                2 * self.sliding_num_kv_heads * self.sliding_head_dim
                + 2 * self.hidden_size
            )
        )
        convolution = (
            2 * batch_size * q_len * self.sconv_kernel_size * channels
        )
        log_work = OperationClassBreakdown()
        if self.log_scaling_n_floor is not None:
            # Per-position division/clamp/log/scale/add, then Q and bias scale.
            log_work = OperationClassBreakdown(
                elementwise=self.full_attention_layers * (3 * q_len + batch_size * model.num_q_heads * (
                    q_len * self.full_head_dim + causal_pairs(start_context, q_len, cap=self.relative_extent))),
                special_operation_counts={"log": q_len * self.full_attention_layers,
                                          "clamp_compare": q_len * self.full_attention_layers})
        return full + sliding + OperationClassBreakdown(
            convolution=convolution,
            dense_gemm=2 * batch_size * q_len * model.num_q_heads * self.relative_head_dim * (
                model.d_model * model.num_layers + self.relative_extent * self.full_attention_layers
                + self.sliding_window * self.sliding_attention_layers),
            elementwise=batch_size * q_len * channels + (batch_size * model.num_q_heads * (
                self.full_attention_layers * causal_pairs(start_context, q_len, cap=self.relative_extent)
                + self.sliding_attention_layers * causal_pairs(start_context, q_len, cap=self.sliding_window)) if self.relative_head_dim else 0),
        ) + log_work + _ffn_breakdown(
            model=model,
            q_len=q_len,
            batch_size=batch_size,
            ffn=self.ffn,
            moe=self.moe,
        )


@dataclass(frozen=True)
class ComputeProfile:
    """Compute model used to derive prefix-aware prefill budgets."""

    effective_prefill_tflops: float
    prefill_formula: PrefillFormula
    scheduler_overhead_us: float = 0.0
    timing_mode: str = "normalized_reference"
    operation_class_tflops: dict[str, float] = field(default_factory=dict)
    timing_omissions: tuple[str, ...] = ()
    output_head_spec: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.effective_prefill_tflops <= 0:
            raise ValueError("compute.effective_prefill_tflops must be positive")
        if self.scheduler_overhead_us < 0:
            raise ValueError("compute.scheduler_overhead_us must not be negative")
        if self.timing_mode not in {
            "normalized_reference",
            "component_calibrated",
        }:
            raise ValueError(
                "compute.timing_mode must be normalized_reference or "
                "component_calibrated"
            )

        valid_classes = set(OperationClassBreakdown().as_dict())
        throughputs = dict(self.operation_class_tflops)
        for operation_class, throughput in throughputs.items():
            if operation_class not in valid_classes:
                raise ValueError(
                    "unknown operation class in compute.operation_class_tflops: "
                    f"{operation_class}"
                )
            if not math.isfinite(throughput) or throughput <= 0:
                raise ValueError(
                    "compute.operation_class_tflops values must be finite and positive"
                )

        omissions = tuple(self.timing_omissions)
        if any(not isinstance(item, str) or not item.strip() for item in omissions):
            raise ValueError("compute.timing_omissions entries must be non-empty strings")
        if len(set(omissions)) != len(omissions):
            raise ValueError("compute.timing_omissions entries must be unique")

        # Defensive copies keep the frozen profile stable if a caller later
        # mutates the dict/list passed to its constructor.
        object.__setattr__(self, "operation_class_tflops", throughputs)
        object.__setattr__(self, "timing_omissions", omissions)
        if self.output_head_spec is not None:
            _validate_positive(int(self.output_head_spec["vocab_size"]), "output_head_spec.vocab_size")

    def saved_recompute_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        return self.prefill_formula.saved_recompute_flops(model, context)

    def actual_prefill_flops(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        """Modeled cache work; consult full_hit_finalization_status for completeness."""
        return self.cache_compute_breakdown(model, context).cache_path_flops

    def saved_recompute_operation_breakdown(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> OperationClassBreakdown:
        method = getattr(self.prefill_formula, "operation_breakdown", None)
        if method is None:
            return OperationClassBreakdown(
                unclassified=max(self.saved_recompute_flops(model, context), 0.0)
            )
        return method(
            model,
            q_len=float(context.matched_prefix_len),
            start_context=0.0,
            batch_size=context.batch_size,
        )

    def actual_prefill_operation_breakdown(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> OperationClassBreakdown:
        breakdown = self.cache_compute_breakdown(model, context)
        values = breakdown.cache_path_by_operation_class or {"unclassified": breakdown.cache_path_flops}
        return OperationClassBreakdown(**values, special_operation_counts=breakdown.special_operation_counts,
                                       counting_omissions=breakdown.counting_omissions)

    def estimate_timing(
        self, operations: OperationClassBreakdown
    ) -> TimingEstimate:
        """Estimate time without conflating a reference scalar with calibration."""

        operation_flops = operations.as_dict()
        total_flops = operations.total_flops
        normalized_reference_us = self._flops_to_us(total_flops)

        if self.timing_mode == "normalized_reference":
            return TimingEstimate(
                timing_mode=self.timing_mode,
                evidence="analytical_normalized_reference",
                total_flops=total_flops,
                time_us=normalized_reference_us,
                normalized_reference_us=normalized_reference_us,
                calibrated_us=None,
                is_calibrated=False,
                omissions=self.timing_omissions,
            )

        nonzero_classes = tuple(
            name for name, flops in operation_flops.items() if flops > 0
        )
        missing_classes = tuple(
            name
            for name in nonzero_classes
            if name not in self.operation_class_tflops
        )
        class_times = {
            name: operation_flops[name]
            / (self.operation_class_tflops[name] * 1_000_000.0)
            for name in nonzero_classes
            if name in self.operation_class_tflops
        }
        if missing_classes:
            return TimingEstimate(
                timing_mode=self.timing_mode,
                evidence="component_calibration_incomplete",
                total_flops=total_flops,
                time_us=None,
                normalized_reference_us=normalized_reference_us,
                calibrated_us=None,
                is_calibrated=False,
                missing_operation_classes=missing_classes,
                operation_class_time_us=class_times,
                omissions=self.timing_omissions,
            )

        calibrated_us = sum(class_times.values())
        return TimingEstimate(
            timing_mode=self.timing_mode,
            evidence="component_calibrated",
            total_flops=total_flops,
            time_us=calibrated_us,
            normalized_reference_us=normalized_reference_us,
            calibrated_us=calibrated_us,
            is_calibrated=True,
            operation_class_time_us=class_times,
            omissions=self.timing_omissions,
        )

    def selected_timing_estimates(self, breakdowns: Mapping[str, OperationClassBreakdown]) -> dict[str, TimingEstimate]:
        """One request timebase. Incomplete calibration falls back as a group;

        missing classes and the uncalibrated label remain visible. The raw
        ``estimate_timing`` API still returns None for incomplete calibration.
        """
        estimates = {name: self.estimate_timing(ops) for name, ops in breakdowns.items()}
        missing = tuple(sorted({kind for value in estimates.values() for kind in value.missing_operation_classes}))
        if missing:
            estimates = {name: replace(value, time_us=value.normalized_reference_us,
                calibrated_us=None, is_calibrated=False,
                evidence="request_group_normalized_fallback_incomplete_calibration",
                missing_operation_classes=missing, operation_class_time_us={})
                for name, value in estimates.items()}
        return estimates

    def cache_timing_estimates(self, breakdown: CacheComputeBreakdown) -> dict[str, TimingEstimate]:
        """Use stored results only; never re-run a formula to reconstruct phases."""
        def ops(values, total):
            return OperationClassBreakdown(**values) if values else OperationClassBreakdown(unclassified=total)
        groups = {
            "cache_path": ops(breakdown.cache_path_by_operation_class, breakdown.cache_path_flops),
            "no_cache": ops(breakdown.no_cache_by_operation_class, breakdown.no_cache_flops),
            "suffix": ops(breakdown.suffix_by_operation_class, breakdown.miss_suffix_flops),
            "restore": ops(breakdown.restore_by_operation_class, breakdown.restore_recompute_flops),
            "conversion": ops(breakdown.conversion_by_operation_class, breakdown.mandatory_conversion_flops),
            "finalization": ops(breakdown.finalization_by_operation_class, breakdown.finalization_flops),
            "output_head": ops(breakdown.output_head_by_operation_class, breakdown.output_head_flops),
        }
        for name, total in breakdown.phase_flops.items():
            groups["phase:" + name] = ops(breakdown.phase_by_operation_class.get(name, {}), total)
        return self.selected_timing_estimates(groups)

    def saved_recompute_timing_estimate(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> TimingEstimate:
        return self.estimate_timing(
            self.saved_recompute_operation_breakdown(model, context)
        )

    def actual_prefill_timing_estimate(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> TimingEstimate:
        breakdown = self.cache_compute_breakdown(model, context)
        estimate = self.cache_timing_estimates(breakdown)["cache_path"]
        if breakdown.full_hit_finalization_status == "not_evaluated":
            return replace(estimate, time_us=None, calibrated_us=None, is_calibrated=False,
                           evidence="full_hit_finalization_unmodeled",
                           omissions=tuple(sorted(set(estimate.omissions + breakdown.counting_omissions))))
        return estimate

    def cache_compute_breakdown(
        self,
        model: ComputeModelLike,
        context: PrefillComputeContext,
        *,
        restore_recompute_tokens: int = 0,
    ) -> CacheComputeBreakdown:
        formula_method = getattr(self.prefill_formula, "cache_compute_breakdown", None)
        if formula_method is not None:
            result = formula_method(
                model,
                context,
                restore_recompute_tokens=restore_recompute_tokens,
            )
            return self._with_full_hit_status(context, self._with_output_head(model, context, result))
        if restore_recompute_tokens:
            raise ValueError("restore recompute is not supported by this legacy formula")
        no_cache_context = PrefillComputeContext(
            seq_len=context.seq_len,
            matched_prefix_len=0,
            generation_len=context.generation_len,
            batch_size=context.batch_size,
        )
        no_cache_flops = self.prefill_formula.actual_prefill_flops(
            model, no_cache_context
        )
        cache_path_flops = self.prefill_formula.actual_prefill_flops(model, context)
        saved_flops = self.prefill_formula.saved_recompute_flops(model, context)
        net_saved = no_cache_flops - cache_path_flops
        return self._with_full_hit_status(context, self._with_output_head(model, context, CacheComputeBreakdown(
            no_cache_flops=no_cache_flops,
            cache_path_flops=cache_path_flops,
            miss_suffix_flops=cache_path_flops,
            compression_tail_flops=0.0,
            swa_restore_flops=0.0,
            restore_recompute_flops=0.0,
            saved_flops=saved_flops,
            restore_recompute_tokens=0,
            cache_path_by_component={},
            no_cache_by_component={},
            net_saved_flops=net_saved,
            cache_path_by_operation_class={},
            no_cache_by_operation_class={},
        )))

    @staticmethod
    def _with_full_hit_status(context, result):
        if context.seq_len == 0 or context.miss_len > 0:
            return replace(result, full_hit_finalization_status="not_applicable")
        # A replay or a dedicated decoder tail must also supply its output head.
        # Positive FLOPs alone do not prove that final hidden/logits exist.
        evaluated = result.output_head_flops > 0 and not any(
            "full_hit_finalization_requires_hidden_or_logits" == item
            for item in result.counting_omissions)
        omissions = result.counting_omissions
        if not evaluated:
            omissions = tuple(sorted(set(omissions + ("full_hit_finalization_unmodeled",))))
        return replace(result, full_hit_finalization_status="evaluated" if evaluated else "not_evaluated",
                       counting_omissions=omissions)

    def _with_output_head(self, model, context, result):
        # CED/V4 own their final collapse and head schedule.
        if getattr(self.prefill_formula, "type", None) in {"deepseek_v4", "deepseek_v41"} or context.seq_len == 0:
            return result
        if self.output_head_spec is None:
            return replace(result, counting_omissions=tuple(sorted(set(result.counting_omissions
                + ("output_head_geometry_not_declared",)))))
        head = OperationClassBreakdown(dense_gemm=2 * context.batch_size * model.d_model
                                       * int(self.output_head_spec["vocab_size"]))
        norm = self.output_head_spec.get("final_norm")
        if norm:
            declared_norm = dict(norm, layers=1)
            head += _scalar_profile_work({"norms": [declared_norm]}, 1, context.batch_size)
        ready = context.miss_len > 0 or result.restore_recompute_tokens > 0
        cache_head = head if ready else OperationClassBreakdown()
        cache_ops = (OperationClassBreakdown(**result.cache_path_by_operation_class)
                     if result.cache_path_by_operation_class else OperationClassBreakdown(unclassified=result.cache_path_flops)) + cache_head
        cold_ops = (OperationClassBreakdown(**result.no_cache_by_operation_class)
                    if result.no_cache_by_operation_class else OperationClassBreakdown(unclassified=result.no_cache_flops)) + head
        components = dict(result.cache_path_by_component)
        if not components:
            components["body"] = result.cache_path_flops
        components["output_head"] = cache_head.total_flops
        cold_components = dict(result.no_cache_by_component) or {"body": result.no_cache_flops}
        cold_components["output_head"] = head.total_flops
        omissions = result.counting_omissions + (() if ready else ("full_hit_finalization_requires_hidden_or_logits",))
        net = cold_ops.total_flops - cache_ops.total_flops
        specials = dict(result.special_operation_counts)
        for name, value in cache_head.special_operation_counts.items():
            specials[name] = specials.get(name, 0) + value
        return replace(result, no_cache_flops=cold_ops.total_flops, cache_path_flops=cache_ops.total_flops,
            saved_flops=max(net, 0), net_saved_flops=net,
            output_head_flops=cache_head.total_flops, output_head_by_operation_class=cache_head.as_dict(),
            cache_path_by_operation_class=cache_ops.as_dict(), no_cache_by_operation_class=cold_ops.as_dict(),
            cache_path_by_component=components, no_cache_by_component=cold_components,
            special_operation_counts=specials, counting_omissions=tuple(sorted(set(omissions))))

    def flops_to_us(self, flops: float) -> float:
        return self._flops_to_us(flops)

    def saved_recompute_time_us(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        return self._flops_to_us(self.saved_recompute_flops(model, context))

    def actual_prefill_time_us(
        self, model: ComputeModelLike, context: PrefillComputeContext
    ) -> float:
        breakdown = self.cache_compute_breakdown(model, context)
        require_evaluated_full_hit(breakdown, model_id=getattr(model, "name", "model"))
        estimate = self.cache_timing_estimates(breakdown)["cache_path"]
        assert estimate.time_us is not None
        return estimate.time_us

    def _flops_to_us(self, flops: float) -> float:
        # 1 TFLOP/s = 1e12 FLOP/s = 1e6 FLOP/us.
        return flops / (self.effective_prefill_tflops * 1_000_000.0)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    d_model: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    attention: AttentionProfile
    compute: ComputeProfile
    decode_us_per_token: float = 0.0
    max_context_tokens: int | None = None
    identity: ModelIdentity | None = None
    evidence: ModelEvidence = field(default_factory=ModelEvidence)
    catalog: CatalogMetadata = field(default_factory=CatalogMetadata)
    model_config_sha256: str | None = None
    runtime_config_sha256: str | None = None
    calculator_snapshot_sha256: str | None = None
    formula_implementation_id: str | None = None
    draft_support: str = "none"
    calculator_reference: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.identity is None:
            object.__setattr__(
                self,
                "identity",
                ModelIdentity(
                    id=self.name,
                    display_name=self.name,
                    family="legacy",
                ),
            )
        elif self.name != self.identity.id:
            raise ValueError("ModelProfile.name must equal immutable identity.id")
        if self.decode_us_per_token < 0:
            raise ValueError("decode_us_per_token must not be negative")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.draft_support not in {
            "exact",
            "upper_bound",
            "cache_only",
            "unsupported",
            "none",
        }:
            raise ValueError(
                "draft_support must be exact, upper_bound, cache_only, unsupported, or none"
            )
        object.__setattr__(self, "calculator_reference", dict(self.calculator_reference))

    @property
    def model_id(self) -> str:
        assert self.identity is not None
        return self.identity.id

    @property
    def display_name(self) -> str:
        assert self.identity is not None
        return self.identity.display_name

    def cache_bytes_per_token(self) -> int | float:
        return self.attention.cache_bytes_per_token(self)

    def cache_bytes_per_block(self, block_size_tokens: int) -> int:
        return self.attention.cache_bytes_per_block(self, block_size_tokens)

    def cache_components(self) -> tuple[CacheComponent, ...]:
        return self.attention.cache_components(self)


def _validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
