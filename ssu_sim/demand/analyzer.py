"""Translate LLM prefix-cache workloads into storage demand."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

from ssu_sim.cache import (
    CacheReplayConfig,
    CacheReplaySummary,
    PrefixRecoveryFunnel,
    replay_prefix_cache,
)
from ssu_sim.cache.replay import _checkpoint_groups_for_model
from ssu_sim.demand.report import build_demand_report
from ssu_sim.models.llm_model import (
    ModelProfile,
    PrefillComputeContext,
)
from ssu_sim.models.attention import PrefixCachePolicy
from ssu_sim.models.execution import validate_execution_combination, require_evaluated_full_hit
from ssu_sim.models.state import RestoreExecutionPlan, PhaseTiming, RestoreReadGroup, phase_chain_time_us
from ssu_sim.prefix.block_hasher import (
    CanonicalRequestPath,
    build_block_ids,
    canonicalize_request_paths,
)
from ssu_sim.workload.request import Request


@dataclass(frozen=True)
class DemandOp:
    """One logical KV-block storage operation."""

    time_us: float
    request_id: str
    model_id: str
    op_type: str
    block_id: str
    payload_bytes: int
    component: str
    operation_weight: float
    access_pattern: str
    chunked_ios: int | None = None
    latency_budget_us: float | None = None
    cache_shard: int | None = None
    consumer_phase: str = "suffix"
    criticality: str = "blocking"


@dataclass(frozen=True)
class CapacitySummary:
    reusable_prefix_bytes: int
    cumulative_unique_write_bytes: int
    retain_all_upper_bound_bytes: int
    reusable_prefix_blocks: int
    cumulative_unique_write_blocks: int
    reusable_prefix_bytes_by_component: dict[str, int]
    cumulative_unique_write_bytes_by_component: dict[str, int]
    retain_all_upper_bound_bytes_by_component: dict[str, int]


@dataclass(frozen=True)
class DemandAnalysis:
    ops: list[DemandOp]
    capacity: CapacitySummary
    latency_budgets_us: list[float]
    ttft_jobs: list["TTFTJob"]
    report: dict[str, Any]
    cache_replay: CacheReplaySummary | None = None


def unevaluated_analysis(error) -> DemandAnalysis:
    """A stopped case has no complete trace, workload totals or timeline."""
    return DemandAnalysis(
        ops=[], capacity=CapacitySummary(0, 0, 0, 0, 0, {}, {}, {}),
        latency_budgets_us=[], ttft_jobs=[],
        report={"demand_report_schema_version": 3,
                "evaluation": {**error.report(), "aborted": True},
                "warnings": [str(error)]},
    )


@dataclass(frozen=True)
class _DemandEvent:
    """A compact per-request event used for aggregation and peak windows."""

    time_us: float
    op_type: str
    payload_bytes: float
    logical_block_ops: float
    block_ops_lower_bound: float
    block_ops_upper_bound: float
    chunked_ios: float | None
    io_size_bytes: int
    component: str
    access_pattern: str


@dataclass(frozen=True)
class TTFTJob:
    """Per-request first-token boundary model inputs and cache-aware estimates."""

    request_id: str
    arrival_time_us: float
    model_id: str
    restore_payload_bytes: float
    restore_logical_block_ops: float
    restore_chunked_ios: float | None
    saved_recompute_time_us: float
    actual_prefill_time_us: float
    cache_path_overhead_us: float
    first_token_decode_us: float
    no_cache_ttft_us: float
    ideal_cache_ttft_us: float
    hidden_restore_budget_us: float
    break_even_restore_budget_us: float
    no_cache_prefill_flops: float = 0.0
    cache_path_flops: float = 0.0
    miss_suffix_flops: float = 0.0
    suffix_body_flops: float = 0.0
    mandatory_conversion_flops: float = 0.0
    restore_conversion_flops: float = 0.0
    finalization_flops: float = 0.0
    output_head_flops: float = 0.0
    special_operation_counts: dict[str, int] = field(default_factory=dict)
    counting_omissions: tuple[str, ...] = ()
    compute_algorithm_id: str = ""
    full_hit_finalization_status: str = "not_applicable"
    flops_exact: dict[str, str | None] = field(default_factory=dict)
    restore_recompute_tokens: int = 0
    restore_recompute_flops: float = 0.0
    compression_tail_flops: float = 0.0
    swa_restore_flops: float = 0.0
    cache_path_flops_by_component: dict[str, float] = field(default_factory=dict)
    no_cache_flops_by_component: dict[str, float] = field(default_factory=dict)
    net_saved_flops: float = 0.0
    cache_path_flops_by_operation_class: dict[str, float] = field(default_factory=dict)
    no_cache_flops_by_operation_class: dict[str, float] = field(default_factory=dict)
    analytical_flop_equivalent_ttft_us: float = 0.0
    calibrated_ttft_us: float | None = None
    timing_validation: str = "uncalibrated"
    missing_operation_classes: tuple[str, ...] = ()
    timing_omissions: tuple[str, ...] = ()
    replay_compute_time_us: float = 0.0
    miss_suffix_compute_time_us: float = 0.0
    mandatory_conversion_time_us: float = 0.0
    finalization_time_us: float = 0.0
    output_head_time_us: float = 0.0
    blocking_checkpoint_components: tuple[str, ...] = ()
    optimistic_overlap_estimate_us: float = 0.0
    dependency_aware_estimate_us: float = 0.0
    no_compute_io_overlap_estimate_us: float = 0.0
    matched_prefix_tokens: int = 0
    recovery_boundary_tokens: int = 0
    restore_causes: tuple[str, ...] = ()
    restored_objects: tuple[str, ...] = ()
    blocking_checkpoint_bytes: int = 0
    configured_prefix_tokens: int = 0
    shareable_prefix_tokens: int = 0
    resident_prefix_tokens: int = 0
    shareability_source: str = "assumed_envelope"
    recovery_loss_causes: tuple[str, ...] = ()
    resident_cache_bytes: float = 0.0
    planned_restore_payload_bytes: float = 0.0
    replay_gate_payload_bytes: float = 0.0
    replay_gate_logical_block_ops: float = 0.0
    replay_gate_chunked_ios: float | None = None
    suffix_gate_payload_bytes: float = 0.0
    optional_restore_payload_bytes: float = 0.0
    stranded_resident_bytes: float = 0.0
    replay_gate_components: tuple[str, ...] = ()
    suffix_gate_components: tuple[str, ...] = ()
    seq_len: int = 0
    suffix_tokens: int = 0
    execution_plan: RestoreExecutionPlan | None = None
    phase_timings: tuple[PhaseTiming, ...] = ()
    restore_read_groups: tuple[RestoreReadGroup, ...] = ()
    generated_global_payload_bytes: int = 0
    generated_global_entries: dict[str, int] = field(default_factory=dict)


def analyze_bandwidth_envelope(
    model: ModelProfile,
    *,
    seq_len: int,
    prefix_ratio: float,
    block_size_tokens: int,
    max_io_bytes: int | None = None,
    prefix_cache_policy: PrefixCachePolicy | None = None,
) -> DemandAnalysis:
    """Analyze one assumed-resident prefix without a workload/cache lifecycle."""

    if not 0.0 <= prefix_ratio <= 1.0:
        raise ValueError("prefix_ratio must be in [0, 1]")
    request = Request(
        request_id="envelope-request",
        arrival_time_us=0.0,
        model_id=model.name,
        tenant_id="envelope",
        seq_len=seq_len,
        generation_len=1,
        prefix_group_id="assumed-resident-prefix",
        matched_prefix_len=int(seq_len * prefix_ratio),
    )
    result = analyze_workload(
        [request],
        {model.name: model},
        block_size_tokens=block_size_tokens,
        qps=1.0,
        max_io_bytes=max_io_bytes,
        write_miss_blocks=False,
        include_trace=False,
        prefix_cache_policy=prefix_cache_policy,
        cache_replay_config=None,
        decode_state_writes=False,
    )
    result.report["analysis_mode"] = "bandwidth_envelope"
    result.report["prefix_ratio_semantics"] = "assumed_resident_prefix_ratio"
    return result


def analyze_workload(
    requests: list[Request],
    models: dict[str, ModelProfile],
    *,
    block_size_tokens: int,
    qps: float,
    max_io_bytes: int | None = None,
    peak_windows_us: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000),
    write_miss_blocks: bool = True,
    include_trace: bool = False,
    prefix_cache_policy: PrefixCachePolicy | None = None,
    cache_replay_config: CacheReplayConfig | None = None,
    decode_state_writes: bool = False,
    decode_write_step_us: float | None = None,
) -> DemandAnalysis:
    """Build an offered-demand report and, when requested, a logical IO trace.

    ``include_trace`` defaults to false so large sequence-length sweeps retain
    only compact per-request events instead of materializing every KV block.
    """

    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    if qps <= 0:
        raise ValueError("qps must be positive")
    if max_io_bytes is not None and max_io_bytes <= 0:
        raise ValueError("max_io_bytes must be positive when provided")
    if not peak_windows_us or any(window <= 0 for window in peak_windows_us):
        raise ValueError("peak_windows_us must contain positive values")
    if decode_write_step_us is not None and decode_write_step_us <= 0:
        raise ValueError("decode_write_step_us must be positive when provided")
    prefix_cache_policy = prefix_cache_policy or PrefixCachePolicy()
    for model in models.values():
        validate_execution_combination(model, prefix_cache_policy)
    canonical_requests, canonical_paths = canonicalize_request_paths(
        requests,
        block_size_tokens,
        version_partial_prefix_models=frozenset(name for name, model in models.items()
            if callable(getattr(model.attention, "recovery_plan", None))),
    )
    requests = list(canonical_requests)
    prefix_cache_policy = prefix_cache_policy or PrefixCachePolicy()
    cache_replay_summary: CacheReplaySummary | None = None
    replay_write_indices: dict[str, tuple[int, ...]] | None = None
    recovery_by_request: dict[str, PrefixRecoveryFunnel] = {}
    if cache_replay_config is not None:
        replay_result = replay_prefix_cache(
            requests,
            models,
            block_size_tokens=block_size_tokens,
            config=cache_replay_config,
            write_miss_blocks=write_miss_blocks,
            prefix_cache_policy=prefix_cache_policy,
            canonical_paths=canonical_paths,
            decode_state_writes=decode_state_writes,
        )
        requests = list(replay_result.requests)
        cache_replay_summary = replay_result.summary
        replay_write_indices = replay_result.write_block_indices_by_request
        recovery_by_request = replay_result.recovery_by_request

    ops: list[DemandOp] = []
    events: list[_DemandEvent] = []
    ttft_jobs: list[TTFTJob] = []
    latency_budgets_us: list[float] = []
    reusable_groups: dict[tuple[str, str], int] = {}
    max_seq_len_by_model: dict[str, int] = {}
    written_block_count = 0
    written_bytes = 0
    written_bytes_by_component: dict[str, int] = {}
    decode_written_blocks = 0
    decode_written_bytes = 0
    decode_written_bytes_by_component: dict[str, int] = {}
    decode_partial_rewrite_count = 0
    phased_compute_cache = {}
    compute_warnings: set[str] = set()
    for model in models.values():
        evidence = model.evidence.compute_formula
        if evidence.status in {"assumed", "scenario_estimate", "synthetic_scenario"}:
            fields = ", ".join(str(item.get("field", "geometry")) for item in evidence.assumptions)
            compute_warnings.add(
                f"{model.display_name}: FLOPs use explicit model assumptions"
                f"{f' ({fields})' if fields else ''}; results are scenario estimates, not a verified model operation count."
            )

    for request in sorted(requests, key=lambda item: (item.arrival_time_us, item.request_id)):
        if request.seq_len < 0 or request.generation_len < 0:
            raise ValueError(f"request {request.request_id} has a negative sequence length")
        if not 0 <= request.matched_prefix_len <= request.seq_len:
            raise ValueError(
                f"request {request.request_id} matched_prefix_len must be within seq_len"
            )
        try:
            model = models[request.model_id]
        except KeyError as exc:
            raise ValueError(f"unknown model_id: {request.model_id}") from exc
        if (
            model.max_context_tokens is not None
            and request.seq_len > model.max_context_tokens
        ):
            raise ValueError(
                f"request {request.request_id} seq_len exceeds "
                f"{model.name} max_context_tokens={model.max_context_tokens}"
            )
        max_seq_len_by_model[request.model_id] = max(
            max_seq_len_by_model.get(request.model_id, 0), request.seq_len
        )
        compute_context = PrefillComputeContext(
            seq_len=request.seq_len,
            matched_prefix_len=request.matched_prefix_len,
            generation_len=request.generation_len,
            encoder_state=prefix_cache_policy.encoder_state,
            encoder_tail_policy=prefix_cache_policy.encoder_tail_policy,
        )
        resident_read_plan = model.attention.map_hit_to_reads(
            model=model,
            hit_tokens=request.matched_prefix_len,
            block_size_tokens=block_size_tokens,
            seq_len=request.seq_len,
            decode_len=request.generation_len,
            policy=prefix_cache_policy,
            request_key=request.request_id,
        )
        read_plan = _apply_actual_checkpoint_restore(
            resident_read_plan,
            request=request,
            model=model,
            block_size_tokens=block_size_tokens,
            policy=prefix_cache_policy,
        )
        recovery_boundary = (
            read_plan.restore_execution_plan.recovery_boundary_tokens
            if read_plan.restore_execution_plan is not None
            else request.matched_prefix_len
        )
        recovery = recovery_by_request.get(request.request_id)
        if recovery is None:
            checkpoint_names = {
                group.name
                for group in _checkpoint_groups_for_model(
                    model, prefix_cache_policy
                )
            }
            resident_capacity_plan = model.attention.map_prefix_to_capacity(
                model=model,
                prefix_tokens=request.matched_prefix_len,
                block_size_tokens=block_size_tokens,
                policy=prefix_cache_policy,
            )
            resident_checkpoint_bytes = int(
                sum(
                    value
                    for name, value in resident_capacity_plan.bytes_by_component.items()
                    if name in checkpoint_names
                )
            )
            resident_cache_bytes = int(resident_capacity_plan.payload_bytes)
            recovery = PrefixRecoveryFunnel(
                seq_len=request.seq_len,
                configured_prefix_tokens=request.matched_prefix_len,
                shareable_prefix_tokens=request.matched_prefix_len,
                resident_prefix_tokens=request.matched_prefix_len,
                recovery_boundary_tokens=recovery_boundary,
                shareability_source="assumed_envelope",
                loss_causes=(
                    read_plan.restore_execution_plan.causes
                    if read_plan.restore_execution_plan is not None
                    else ()
                ),
                resident_cache_bytes=resident_cache_bytes,
                resident_token_bytes=(
                    resident_cache_bytes - resident_checkpoint_bytes
                ),
                resident_checkpoint_bytes=resident_checkpoint_bytes,
            )
        compute_key = (model.model_id, compute_context, read_plan.restore_recompute_tokens)
        compute_breakdown = phased_compute_cache.get(compute_key)
        if compute_breakdown is None:
            compute_breakdown = model.compute.cache_compute_breakdown(
                model, compute_context, restore_recompute_tokens=read_plan.restore_recompute_tokens)
            if compute_breakdown.execution_plan is not None:
                phased_compute_cache[compute_key] = compute_breakdown
        actual_prefill_flops = compute_breakdown.cache_path_flops
        finalization_evaluated = compute_breakdown.full_hit_finalization_status != "not_evaluated"
        if not finalization_evaluated and decode_state_writes and request.generation_len > 0:
            require_evaluated_full_hit(compute_breakdown, model_id=model.name, request_id=request.request_id)
        timings = model.compute.cache_timing_estimates(compute_breakdown)
        cache_timing = timings["cache_path"]
        no_cache_timing = timings["no_cache"]
        actual_prefill_time_us = (
            cache_timing.time_us
            if cache_timing.time_us is not None
            else cache_timing.normalized_reference_us
        )
        no_cache_prefill_time_us = (
            no_cache_timing.time_us
            if no_cache_timing.time_us is not None
            else no_cache_timing.normalized_reference_us
        )
        # A signed difference in the selected timing basis, not the historical
        # theoretical work of the matched prefix (which excludes restoration).
        saved_recompute_time_us = no_cache_prefill_time_us - actual_prefill_time_us
        replay_timing = timings["restore"]
        miss_timing = timings["suffix"]
        replay_compute_time_us = _selected_time_us(replay_timing)
        miss_suffix_compute_time_us = _selected_time_us(miss_timing)
        phase_timings = ()
        restore_read_groups = tuple(RestoreReadGroup(
            stream.component, stream.payload_bytes, stream.physical_blocks,
            stream.physical_blocks * _chunk_count(stream.block_bytes, max_io_bytes)
            if max_io_bytes is not None else None,
        ) for stream in read_plan.streams if stream.payload_bytes > 0 and stream.criticality == "blocking")
        if compute_breakdown.execution_plan is not None:
            phase_timings = tuple(PhaseTiming(
                phase.name,
                _selected_time_us(timings[f"phase:{phase.name}"]),
                phase.read_components, compute_breakdown.phase_flops[phase.name],
            ) for phase in compute_breakdown.execution_plan.phases)
        else:
            # Adapt the existing replay/suffix dependency order to the shared
            # phase arithmetic. Do not rerun formula segments here: they omit
            # request-level conversion and finalization work.
            replay_components = tuple(sorted({s.component for s in read_plan.streams
                if s.payload_bytes > 0 and s.criticality == "blocking" and s.consumer_phase == "replay"}))
            suffix_components = tuple(sorted({s.component for s in read_plan.streams
                if s.payload_bytes > 0 and s.criticality == "blocking" and s.consumer_phase == "suffix"}))
            # Mandatory conversion remains classified as suffix-required work,
            # but replay also needs the expanded prefix. Schedule the shared
            # conversion once, before replay, instead of ordering by metric name.
            conversion_before_replay = (
                compute_breakdown.restore_recompute_tokens > 0
                and compute_breakdown.mandatory_conversion_flops > 0
            )
            conversion_time = (_selected_time_us(timings["conversion"])
                               if conversion_before_replay else 0.0)
            conversion_flops = (compute_breakdown.mandatory_conversion_flops
                                if conversion_before_replay else 0)
            phase_timings = ((PhaseTiming("mandatory_conversion", conversion_time,
                                         replay_components, conversion_flops),)
                             if conversion_before_replay else ())
            phase_timings += (
                PhaseTiming("replay", replay_compute_time_us, replay_components,
                            compute_breakdown.restore_recompute_flops),
                PhaseTiming("suffix", max(miss_suffix_compute_time_us - conversion_time, 0.0),
                            suffix_components, compute_breakdown.miss_suffix_flops - conversion_flops),
            )
            if compute_context.miss_len == 0 and compute_breakdown.finalization_flops > 0:
                phase_timings += (PhaseTiming("finalization", _selected_time_us(timings["finalization"]),
                                             (), compute_breakdown.finalization_flops),)
            if compute_breakdown.output_head_flops > 0:
                phase_timings += (PhaseTiming("output_head", _selected_time_us(timings["output_head"]),
                                             (), compute_breakdown.output_head_flops),)
        if not math.isclose(sum(p.compute_us for p in phase_timings), actual_prefill_time_us,
                            rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError("cache-path and phase times must use the same work and timing basis")
        materialization_us = read_plan.materialization_us
        if (compute_breakdown.mandatory_conversion_flops + compute_breakdown.restore_conversion_flops > 0
                and read_plan.compute_materialization_us > 0):
            materialization_us -= min(materialization_us, read_plan.compute_materialization_us)
            compute_warnings.add(
                f"{model.display_name}: explicit conversion FLOPs replace the legacy latent materialization time proxy; independent routing overhead is retained."
            )
        if cache_timing.missing_operation_classes:
            compute_warnings.add(
                f"{model.display_name}: incomplete component calibration; the entire request uses normalized reference timing (missing: {', '.join(cache_timing.missing_operation_classes)})."
            )
        if any("full_hit" in item and ("hidden" in item or "finalization" in item)
               for item in compute_breakdown.counting_omissions):
            compute_warnings.add(
                f"{model.display_name}: a full prefix hit has zero suffix compute, but complete first-token finalization is not evaluated because final hidden/logits or output-head work is not modeled."
            )
        has_restore_payload = read_plan.suffix_gate_bytes > 0
        cache_path_overhead_us = (
            materialization_us + model.compute.scheduler_overhead_us if has_restore_payload else 0.0
        )
        read_deadlines = _component_read_deadlines(
            phase_timings, no_cache_prefill_time_us, cache_path_overhead_us
        ) if finalization_evaluated else {}
        group_key = (request.model_id, request.prefix_group_id)
        previous_tokens = reusable_groups.get(group_key, 0)
        reusable_groups[group_key] = max(previous_tokens, request.matched_prefix_len)

        latency_budget_us: float | None = None
        if request.matched_prefix_len > 0 and finalization_evaluated:
            latency_budget_us = (
                saved_recompute_time_us
                - cache_path_overhead_us
            )
            latency_budgets_us.append(latency_budget_us)

        read_chunked_ios = 0.0 if max_io_bytes is not None else None
        blocking_chunked_ios = 0.0 if max_io_bytes is not None else None
        replay_gate_chunked_ios = 0.0 if max_io_bytes is not None else None
        blocking_logical_ops = 0.0
        replay_gate_logical_ops = 0.0
        for stream in read_plan.streams:
            if stream.physical_blocks <= 0:
                continue
            chunked_per_block = _chunk_count(stream.block_bytes, max_io_bytes)
            if read_chunked_ios is not None and chunked_per_block is not None:
                read_chunked_ios += stream.physical_blocks * chunked_per_block
                if stream.criticality == "blocking":
                    blocking_chunked_ios += stream.physical_blocks * chunked_per_block
            if stream.criticality == "blocking":
                blocking_logical_ops += stream.physical_blocks
                if stream.consumer_phase == "replay":
                    replay_gate_logical_ops += stream.physical_blocks
                    if (
                        replay_gate_chunked_ios is not None
                        and chunked_per_block is not None
                    ):
                        replay_gate_chunked_ios += (
                            stream.physical_blocks * chunked_per_block
                        )
            events.append(
                _DemandEvent(
                    time_us=request.arrival_time_us,
                    op_type="read",
                    payload_bytes=stream.payload_bytes,
                    logical_block_ops=stream.physical_blocks,
                    block_ops_lower_bound=stream.lower_bound_blocks,
                    block_ops_upper_bound=stream.upper_bound_blocks,
                    chunked_ios=(
                        stream.physical_blocks * chunked_per_block
                        if chunked_per_block is not None
                        else None
                    ),
                    io_size_bytes=stream.block_bytes,
                    component=stream.component,
                    access_pattern=stream.access_pattern,
                )
            )
            if include_trace:
                trace_token_count = (
                    stream.trace_token_count
                    if stream.trace_token_count is not None
                    else (
                        read_plan.reusable_prefix_tokens
                        if read_plan.reusable_prefix_tokens is not None
                        else request.matched_prefix_len
                    )
                )
                logical_hit_ids = _read_block_ids(
                    request,
                    trace_token_count,
                    block_size_tokens,
                )
                indexed_ids = (
                    [
                        (index, logical_hit_ids[index])
                        for index in stream.trace_block_indices
                        if 0 <= index < len(logical_hit_ids)
                    ]
                    if stream.trace_block_indices is not None
                    else list(enumerate(logical_hit_ids))
                )
                for trace_position, (index, block_id) in enumerate(indexed_ids):
                    operation_weight = (
                        1.0
                        if stream.trace_block_indices is not None
                        else (
                            stream.last_block_weight
                            if trace_position == len(indexed_ids) - 1
                            else stream.full_block_weight
                        )
                    )
                    if operation_weight <= 0:
                        continue
                    ops.append(
                        DemandOp(
                            time_us=request.arrival_time_us,
                            request_id=request.request_id,
                            model_id=request.model_id,
                            op_type="read",
                            block_id=f"{block_id}:{stream.component}",
                            payload_bytes=stream.block_bytes,
                            component=stream.component,
                            operation_weight=operation_weight,
                            access_pattern=stream.access_pattern,
                            chunked_ios=chunked_per_block,
                            latency_budget_us=(read_deadlines.get(stream.component)
                                               if stream.criticality == "blocking" else None),
                            cache_shard=request.cache_shard,
                            consumer_phase=stream.consumer_phase,
                            criticality=stream.criticality,
                        )
                    )

        first_token_decode_us = model.decode_us_per_token
        analytical_ttft_us = (
            cache_timing.normalized_reference_us
            + cache_path_overhead_us
            + first_token_decode_us
        )
        calibrated_ttft_us = (
            cache_timing.calibrated_us
            + cache_path_overhead_us
            + first_token_decode_us
            if cache_timing.calibrated_us is not None
            else None
        )
        optimistic_lower_us = (
            max(actual_prefill_time_us, cache_path_overhead_us)
            + first_token_decode_us
        )
        dependency_without_storage_us = (
            cache_path_overhead_us
            + replay_compute_time_us
            + miss_suffix_compute_time_us
            + first_token_decode_us
        )
        if phase_timings:
            dependency_without_storage_us = phase_chain_time_us(
                phase_timings, overhead_us=cache_path_overhead_us, first_token_us=first_token_decode_us)
        ttft_jobs.append(
            TTFTJob(
                request_id=request.request_id,
                arrival_time_us=request.arrival_time_us,
                model_id=request.model_id,
                restore_payload_bytes=read_plan.suffix_gate_bytes,
                restore_logical_block_ops=blocking_logical_ops,
                restore_chunked_ios=blocking_chunked_ios,
                saved_recompute_time_us=saved_recompute_time_us,
                actual_prefill_time_us=actual_prefill_time_us,
                cache_path_overhead_us=cache_path_overhead_us,
                first_token_decode_us=first_token_decode_us,
                no_cache_ttft_us=(
                    no_cache_prefill_time_us + first_token_decode_us
                ),
                ideal_cache_ttft_us=(
                    max(actual_prefill_time_us, cache_path_overhead_us)
                    + first_token_decode_us
                ),
                hidden_restore_budget_us=(
                    actual_prefill_time_us - cache_path_overhead_us
                ),
                break_even_restore_budget_us=(
                    saved_recompute_time_us - cache_path_overhead_us
                ),
                no_cache_prefill_flops=compute_breakdown.no_cache_flops,
                cache_path_flops=compute_breakdown.cache_path_flops,
                miss_suffix_flops=compute_breakdown.miss_suffix_flops,
                suffix_body_flops=compute_breakdown.suffix_body_flops,
                mandatory_conversion_flops=compute_breakdown.mandatory_conversion_flops,
                restore_conversion_flops=compute_breakdown.restore_conversion_flops,
                finalization_flops=compute_breakdown.finalization_flops,
                output_head_flops=compute_breakdown.output_head_flops,
                special_operation_counts=compute_breakdown.special_operation_counts,
                counting_omissions=compute_breakdown.counting_omissions,
                compute_algorithm_id=compute_breakdown.algorithm_id,
                full_hit_finalization_status=compute_breakdown.full_hit_finalization_status,
                flops_exact={
                    "no_cache_flops_exact": _exact_flop_count(compute_breakdown.no_cache_flops),
                    "cache_path_flops_exact": _exact_flop_count(compute_breakdown.cache_path_flops),
                    "suffix_flops_exact": _exact_flop_count(compute_breakdown.miss_suffix_flops),
                    "suffix_body_flops_exact": _exact_flop_count(compute_breakdown.suffix_body_flops),
                    "mandatory_conversion_flops_exact": _exact_flop_count(compute_breakdown.mandatory_conversion_flops),
                    "restore_conversion_flops_exact": _exact_flop_count(compute_breakdown.restore_conversion_flops),
                    "restore_flops_exact": _exact_flop_count(compute_breakdown.restore_recompute_flops),
                    "finalization_flops_exact": _exact_flop_count(compute_breakdown.finalization_flops),
                    "output_head_flops_exact": _exact_flop_count(compute_breakdown.output_head_flops),
                    "net_saved_flops_exact": _exact_flop_count(compute_breakdown.net_saved_flops),
                },
                restore_recompute_tokens=compute_breakdown.restore_recompute_tokens,
                restore_recompute_flops=compute_breakdown.restore_recompute_flops,
                compression_tail_flops=compute_breakdown.compression_tail_flops,
                swa_restore_flops=compute_breakdown.swa_restore_flops,
                cache_path_flops_by_component=compute_breakdown.cache_path_by_component,
                no_cache_flops_by_component=compute_breakdown.no_cache_by_component,
                net_saved_flops=compute_breakdown.net_saved_flops,
                cache_path_flops_by_operation_class=(
                    compute_breakdown.cache_path_by_operation_class
                ),
                no_cache_flops_by_operation_class=(
                    compute_breakdown.no_cache_by_operation_class
                ),
                analytical_flop_equivalent_ttft_us=analytical_ttft_us,
                calibrated_ttft_us=calibrated_ttft_us,
                timing_validation=cache_timing.evidence,
                missing_operation_classes=cache_timing.missing_operation_classes,
                timing_omissions=cache_timing.omissions,
                replay_compute_time_us=replay_compute_time_us,
                miss_suffix_compute_time_us=miss_suffix_compute_time_us,
                mandatory_conversion_time_us=_selected_time_us(timings["conversion"]),
                finalization_time_us=_selected_time_us(timings["finalization"]),
                output_head_time_us=_selected_time_us(timings["output_head"]),
                blocking_checkpoint_components=(
                    read_plan.restore_execution_plan.restored_objects
                    if read_plan.restore_execution_plan is not None
                    else ()
                ),
                optimistic_overlap_estimate_us=optimistic_lower_us,
                dependency_aware_estimate_us=dependency_without_storage_us,
                no_compute_io_overlap_estimate_us=dependency_without_storage_us,
                matched_prefix_tokens=recovery.resident_prefix_tokens,
                recovery_boundary_tokens=recovery.recovery_boundary_tokens,
                restore_causes=(
                    read_plan.restore_execution_plan.causes
                    if read_plan.restore_execution_plan is not None
                    else ()
                ),
                restored_objects=(
                    read_plan.restore_execution_plan.restored_objects
                    if read_plan.restore_execution_plan is not None
                    else ()
                ),
                blocking_checkpoint_bytes=(
                    read_plan.restore_execution_plan.blocking_checkpoint_bytes
                    if read_plan.restore_execution_plan is not None
                    else 0
                ),
                configured_prefix_tokens=recovery.configured_prefix_tokens,
                shareable_prefix_tokens=recovery.shareable_prefix_tokens,
                resident_prefix_tokens=recovery.resident_prefix_tokens,
                shareability_source=recovery.shareability_source,
                recovery_loss_causes=recovery.loss_causes,
                resident_cache_bytes=float(recovery.resident_cache_bytes),
                planned_restore_payload_bytes=read_plan.payload_bytes,
                replay_gate_payload_bytes=read_plan.replay_gate_bytes,
                replay_gate_logical_block_ops=replay_gate_logical_ops,
                replay_gate_chunked_ios=replay_gate_chunked_ios,
                suffix_gate_payload_bytes=read_plan.suffix_gate_bytes,
                optional_restore_payload_bytes=read_plan.optional_restore_bytes,
                stranded_resident_bytes=max(
                    float(recovery.resident_cache_bytes)
                    - read_plan.payload_bytes,
                    0.0,
                ),
                replay_gate_components=tuple(
                    sorted(
                        {
                            stream.component
                            for stream in read_plan.streams
                            if stream.criticality == "blocking"
                            and stream.consumer_phase == "replay"
                        }
                    )
                ),
                suffix_gate_components=tuple(
                    sorted(
                        {
                            stream.component
                            for stream in read_plan.streams
                            if stream.criticality == "blocking"
                            and stream.consumer_phase in {"replay", "suffix"}
                        }
                    )
                ),
                seq_len=recovery.seq_len,
                suffix_tokens=recovery.suffix_tokens,
                execution_plan=compute_breakdown.execution_plan,
                phase_timings=phase_timings,
                restore_read_groups=restore_read_groups,
                generated_global_entries=(model.attention.generated_global_entries(request.matched_prefix_len, request.miss_len)
                    if hasattr(model.attention, "generated_global_entries") else {}),
                generated_global_payload_bytes=(
                    model.attention.generated_global_bytes(request.matched_prefix_len, request.miss_len)
                    if hasattr(model.attention, "generated_global_bytes") else 0),
            )
        )

        if replay_write_indices is not None:
            admitted_indices = replay_write_indices.get(request.request_id, ())
            if admitted_indices:
                write_start_block = admitted_indices[0]
                write_end_block = admitted_indices[-1] + 1
                if admitted_indices != tuple(range(write_start_block, write_end_block)):
                    raise AssertionError(
                        "radix replay admissions must form one contiguous block range"
                    )
                write_prefix_tokens = write_start_block * block_size_tokens
                write_miss_tokens = (
                    write_end_block - write_start_block
                ) * block_size_tokens
            else:
                write_prefix_tokens = 0
                write_miss_tokens = 0
        else:
            write_prefix_tokens = request.matched_prefix_len
            write_miss_tokens = request.miss_len

        if write_miss_blocks and write_miss_tokens > 0:
            write_plan = model.attention.map_miss_to_writes(
                model=model,
                miss_tokens=write_miss_tokens,
                block_size_tokens=block_size_tokens,
                prefix_tokens=write_prefix_tokens,
                policy=prefix_cache_policy,
            )
            write_time_us = request.arrival_time_us + actual_prefill_time_us
            written_block_count += write_plan.logical_blocks
            written_bytes += int(write_plan.payload_bytes)
            cache_components = model.cache_components()
            component_names = {component.name for component in cache_components}
            for stream in write_plan.streams:
                if stream.component in component_names or stream.component == "swa":
                    component_payloads = {stream.component: int(stream.payload_bytes)}
                else:
                    component_payloads = {
                        component.name: write_plan.logical_blocks
                        * component.bytes_per_block(model, block_size_tokens)
                        for component in cache_components
                        if component.category == stream.component
                    }
                for component_name, payload_bytes in component_payloads.items():
                    key = f"{model.name}:{component_name}"
                    written_bytes_by_component[key] = (
                        written_bytes_by_component.get(key, 0) + payload_bytes
                    )
            for stream in write_plan.streams:
                chunked_per_block = _chunk_count(stream.block_bytes, max_io_bytes)
                events.append(
                    _DemandEvent(
                        time_us=write_time_us,
                        op_type="write",
                        payload_bytes=stream.payload_bytes,
                        logical_block_ops=stream.physical_blocks,
                        block_ops_lower_bound=stream.lower_bound_blocks,
                        block_ops_upper_bound=stream.upper_bound_blocks,
                        chunked_ios=(
                            stream.physical_blocks * chunked_per_block
                            if chunked_per_block is not None
                            else None
                        ),
                        io_size_bytes=stream.block_bytes,
                        component=stream.component,
                        access_pattern=stream.access_pattern,
                    )
                )
                if include_trace:
                    indexed_ids = _write_block_ids(
                        request,
                        stream.trace_block_indices,
                        stream.logical_blocks,
                        block_size_tokens,
                        write_prefix_tokens // block_size_tokens,
                    )
                    for block_id in indexed_ids:
                        ops.append(
                            DemandOp(
                                time_us=write_time_us,
                                request_id=request.request_id,
                                model_id=request.model_id,
                                op_type="write",
                                block_id=f"{block_id}:{stream.component}",
                                payload_bytes=stream.block_bytes,
                                component=stream.component,
                                operation_weight=1.0,
                                access_pattern=stream.access_pattern,
                                chunked_ios=chunked_per_block,
                                cache_shard=request.cache_shard,
                            )
                        )

        if decode_state_writes and request.generation_len > 0:
            decode_mapper = getattr(model.attention, "map_decode_to_writes", None)
            if callable(decode_mapper):
                decode_plan = decode_mapper(
                    model=model,
                    decode_tokens=request.generation_len,
                    block_size_tokens=block_size_tokens,
                    prefix_tokens=request.seq_len,
                    policy=prefix_cache_policy,
                )
            else:
                decode_plan = model.attention.map_miss_to_writes(
                    model=model,
                    miss_tokens=request.generation_len,
                    block_size_tokens=block_size_tokens,
                    prefix_tokens=request.seq_len,
                    policy=prefix_cache_policy,
                )
            decode_path = canonical_paths[request.request_id]
            decode_partial_rewrite_count += len(
                decode_path.rewritten_partial_objects
            )
            decode_written_blocks += decode_plan.logical_blocks
            decode_written_bytes += int(decode_plan.payload_bytes)
            written_block_count += decode_plan.logical_blocks
            written_bytes += int(decode_plan.payload_bytes)
            burst_time_us = (
                request.arrival_time_us
                + actual_prefill_time_us
                + request.generation_len * model.decode_us_per_token
            )
            for stream in decode_plan.streams:
                payload = int(stream.payload_bytes)
                key = f"{model.name}:{stream.component}"
                decode_written_bytes_by_component[key] = (
                    decode_written_bytes_by_component.get(key, 0) + payload
                )
                written_bytes_by_component[key] = (
                    written_bytes_by_component.get(key, 0) + payload
                )
                chunked_per_block = _chunk_count(stream.block_bytes, max_io_bytes)
                event_time_us = burst_time_us
                events.append(
                    _DemandEvent(
                        time_us=event_time_us,
                        op_type="write",
                        payload_bytes=stream.payload_bytes,
                        logical_block_ops=stream.physical_blocks,
                        block_ops_lower_bound=stream.lower_bound_blocks,
                        block_ops_upper_bound=stream.upper_bound_blocks,
                        chunked_ios=(
                            stream.physical_blocks * chunked_per_block
                            if chunked_per_block is not None
                            else None
                        ),
                        io_size_bytes=stream.block_bytes,
                        component=stream.component,
                        access_pattern=stream.access_pattern,
                    )
                )
                if include_trace:
                    indexed_ids = _decode_write_block_ids(
                        request,
                        decode_path,
                        stream.trace_block_indices,
                        stream.logical_blocks,
                        block_size_tokens,
                    )
                    for ordinal, block_id in enumerate(indexed_ids):
                        operation_time_us = event_time_us
                        if decode_write_step_us is not None:
                            completed_tokens = min(
                                request.generation_len,
                                max(
                                    1,
                                    (ordinal + 1)
                                    * math.ceil(
                                        request.generation_len
                                        / max(len(indexed_ids), 1)
                                    ),
                                ),
                            )
                            operation_time_us = (
                                request.arrival_time_us
                                + actual_prefill_time_us
                                + completed_tokens * decode_write_step_us
                            )
                        ops.append(
                            DemandOp(
                                time_us=operation_time_us,
                                request_id=request.request_id,
                                model_id=request.model_id,
                                op_type="write",
                                block_id=f"{block_id}:{stream.component}",
                                payload_bytes=stream.block_bytes,
                                component=stream.component,
                                operation_weight=1.0,
                                access_pattern=stream.access_pattern,
                                chunked_ios=chunked_per_block,
                                cache_shard=request.cache_shard,
                            )
                        )

    reusable_block_count = 0
    reusable_bytes_by_component: dict[str, int] = {}
    for (model_id, _), prefix_tokens in reusable_groups.items():
        model = models[model_id]
        capacity_plan = model.attention.map_prefix_to_capacity(
            model,
            prefix_tokens,
            block_size_tokens,
            prefix_cache_policy,
        )
        reusable_block_count += capacity_plan.logical_blocks
        for component_name, component_bytes in capacity_plan.bytes_by_component.items():
            key = f"{model.name}:{component_name}"
            reusable_bytes_by_component[key] = (
                reusable_bytes_by_component.get(key, 0) + component_bytes
            )
    reusable_bytes = sum(reusable_bytes_by_component.values())
    retain_by_component = dict(reusable_bytes_by_component)
    for key, value in written_bytes_by_component.items():
        retain_by_component[key] = retain_by_component.get(key, 0) + value
    capacity = CapacitySummary(
        reusable_prefix_bytes=reusable_bytes,
        cumulative_unique_write_bytes=written_bytes,
        retain_all_upper_bound_bytes=reusable_bytes + written_bytes,
        reusable_prefix_blocks=reusable_block_count,
        cumulative_unique_write_blocks=written_block_count,
        reusable_prefix_bytes_by_component=reusable_bytes_by_component,
        cumulative_unique_write_bytes_by_component=written_bytes_by_component,
        retain_all_upper_bound_bytes_by_component=retain_by_component,
    )
    ordered_ops = sorted(
        ops,
        key=lambda op: (op.time_us, op.request_id, op.op_type, op.block_id),
    )
    nominal_duration_us = len(requests) * 1_000_000.0 / qps
    report = build_demand_report(
        events=events,
        capacity=capacity,
        latency_budgets_us=latency_budgets_us,
        models=models,
        block_size_tokens=block_size_tokens,
        nominal_duration_us=nominal_duration_us,
        peak_windows_us=peak_windows_us,
        max_io_bytes=max_io_bytes,
        request_count=len(requests),
        ttft_jobs=ttft_jobs,
        prefix_cache_policy=prefix_cache_policy,
        max_seq_len_by_model=max_seq_len_by_model,
        decode_write_summary={
            "enabled": decode_state_writes,
            "schedule": (
                "uniform_step" if decode_write_step_us is not None else "burst_at_completion"
            ),
            "step_us_per_token": decode_write_step_us,
            "timing_validation": (
                "uncalibrated_uniform_step_scenario"
                if decode_write_step_us is not None
                else "uncalibrated_burst_scenario"
            ),
            "generated_token_count": sum(
                request.generation_len for request in requests
            ),
            "write_events_enabled": decode_state_writes,
            "logical_blocks": decode_written_blocks,
            "serialized_bytes": decode_written_bytes,
            "serialized_bytes_by_component": decode_written_bytes_by_component,
            "partial_tail_rewrite_count": decode_partial_rewrite_count,
        },
    )
    if compute_warnings:
        report["warnings"] = sorted(compute_warnings)
    if any(job.execution_plan is not None for job in ttft_jobs):
        report["io_accounting"] = {
            "generated_global_entries": {side: sum(job.generated_global_entries.get(side, 0) for job in ttft_jobs)
                for side in ("encoder", "decoder")},
            "generated_global_payload_bytes": sum(job.generated_global_payload_bytes for job in ttft_jobs),
            "logical_read_payload_bytes": sum(event.payload_bytes for event in events if event.op_type == "read"),
            "logical_write_payload_bytes": sum(event.payload_bytes for event in events if event.op_type == "write"),
            "semantics": "effective new entries, serialized object IO and physical SSD IO are separate",
        }
    if cache_replay_summary is not None:
        report["analysis_mode"] = "cache_replay"
        report["cache_replay"] = cache_replay_summary.to_dict()
        report["semantics"]["cache_hit_rate"] = (
            "object-shareable and resident prefixes derived by capacity-aware radix leaf-LRU replay"
        )
        report["semantics"]["capacity"] = (
            "retain_all remains a counterfactual upper bound; actual residency is under cache_replay.capacity"
        )
    else:
        report["analysis_mode"] = "assumed_prefix"
    return DemandAnalysis(
        ops=ordered_ops,
        capacity=capacity,
        latency_budgets_us=latency_budgets_us,
        ttft_jobs=ttft_jobs,
        report=report,
        cache_replay=cache_replay_summary,
    )


def _apply_actual_checkpoint_restore(
    read_plan,
    *,
    request: Request,
    model: ModelProfile,
    block_size_tokens: int,
    policy: PrefixCachePolicy,
):
    """Finalize state recovery before selecting the token streams to restore.

    Full-forward replay regenerates every token-state component after the
    common recovery boundary.  Reading that resident tail would be redundant,
    so the IO plan is rebuilt at the boundary and the tail is reported
    separately as stranded resident capacity.
    """

    if read_plan.restore_execution_plan is not None and read_plan.restore_execution_plan.execution_scope == "phased":
        return read_plan
    groups = _checkpoint_groups_for_model(model, policy)
    if not groups:
        reusable = (read_plan.reusable_prefix_tokens if read_plan.reusable_prefix_tokens is not None
                    else request.matched_prefix_len)
        recovery = min(request.matched_prefix_len, reusable,
                       request.matched_prefix_len - read_plan.restore_recompute_tokens)
        if request.recovery_boundary_tokens is not None:
            recovery = min(recovery, request.recovery_boundary_tokens)
        replay = request.matched_prefix_len - recovery
        restore = read_plan.restore_execution_plan
        if replay and restore is None:
            restore = RestoreExecutionPlan(
                matched_prefix_tokens=request.matched_prefix_len,
                recovery_boundary_tokens=recovery,
                replay_end_tokens=request.matched_prefix_len,
                causes=("partial_prefix_not_restored",),
            )
        return replace(
            read_plan,
            streams=tuple(
                _with_stream_phase(stream, "replay" if replay else "suffix", policy)
                for stream in read_plan.streams
            ),
            restore_recompute_tokens=max(read_plan.restore_recompute_tokens, replay),
            restore_execution_plan=restore,
        )
    ideal_restore = read_plan.restore_execution_plan
    recovery = (
        request.recovery_boundary_tokens
        if request.recovery_boundary_tokens is not None
        else (
            ideal_restore.recovery_boundary_tokens
            if ideal_restore is not None
            else request.matched_prefix_len
        )
    )
    actual_replay = request.recovery_boundary_tokens is not None
    restored_names = (
        {key.rsplit("@", 1)[0] for key in request.restored_checkpoint_keys}
        if actual_replay
        else set(ideal_restore.restored_objects if ideal_restore is not None else ())
    )
    checkpoint_names = {group.name for group in groups}
    boundary_plan = model.attention.map_hit_to_reads(
        model=model,
        hit_tokens=recovery,
        block_size_tokens=block_size_tokens,
        seq_len=request.seq_len,
        decode_len=request.generation_len,
        policy=policy,
        request_key=request.request_id,
    )
    phase = "replay" if recovery < request.matched_prefix_len else "suffix"
    streams = []
    for stream in boundary_plan.streams:
        if stream.component in checkpoint_names and (
            stream.component not in restored_names or recovery <= 0
        ):
            continue
        streams.append(_with_stream_phase(stream, phase, policy))
    blocking = sum(
        int(stream.payload_bytes)
        for stream in streams
        if stream.component in checkpoint_names
    )
    causes = (
        ()
        if recovery == request.matched_prefix_len
        else ("checkpoint_not_resident_or_tail_replay",)
    )
    restore = RestoreExecutionPlan(
        matched_prefix_tokens=request.matched_prefix_len,
        recovery_boundary_tokens=recovery,
        replay_end_tokens=request.matched_prefix_len,
        causes=causes,
        restored_objects=tuple(sorted(restored_names)),
        blocking_checkpoint_bytes=blocking,
    )
    return replace(
        boundary_plan,
        streams=tuple(streams),
        materialization_us=boundary_plan.materialization_us,
        reusable_prefix_tokens=recovery,
        restore_recompute_tokens=restore.restore_recompute_tokens,
        restore_execution_plan=restore,
    )


def _with_stream_phase(stream, phase: str, policy: PrefixCachePolicy):
    optional_draft = stream.component.startswith("draft_") and not policy.include_draft_execution
    return replace(
        stream,
        consumer_phase="optional" if optional_draft else phase,
        criticality="non_blocking" if optional_draft else "blocking",
    )


def _selected_time_us(estimate) -> float:
    return estimate.time_us if estimate.time_us is not None else estimate.normalized_reference_us


def _exact_flop_count(value: int | float) -> str | None:
    """Keep integer counts lossless in JSON without asserting lost float bits.

    Small integral legacy/custom-formula values remain representable exactly;
    a large float has already crossed the precision boundary and is not an
    exact source, even when its displayed value ends in .0.
    """
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value) and abs(value) <= 2**53 and value == int(value):
        return str(int(value))
    return None


def _component_read_deadlines(
    phases: tuple[PhaseTiming, ...], no_cache_compute_us: float, overhead_us: float
) -> dict[str, float]:
    """Each restored component must arrive before its first consumer starts.

    The target is the no-cache first-token boundary; the common first-token
    surcharge cancels, leaving no-cache compute less the remaining phase work.
    Keep non-positive deadlines visible instead of clamping away a failed budget.
    """
    remaining = sum(phase.compute_us for phase in phases)
    deadlines: dict[str, float] = {}
    for phase in phases:
        for component in phase.read_components:
            deadlines.setdefault(component, no_cache_compute_us - overhead_us - remaining)
        remaining -= phase.compute_us
    return deadlines


def _chunk_count(payload_bytes: int, max_io_bytes: int | None) -> int | None:
    if max_io_bytes is None:
        return None
    return math.ceil(payload_bytes / max_io_bytes)


def _read_block_ids(
    request: Request,
    token_count: int,
    block_size_tokens: int,
) -> list[str]:
    source_ids = request.restore_block_ids or request.block_ids
    if not source_ids:
        return build_block_ids(
            request.model_id,
            request.prefix_group_id,
            token_count,
            block_size_tokens,
        )
    block_count = math.ceil(token_count / block_size_tokens) if token_count > 0 else 0
    if block_count > len(source_ids):
        raise ValueError(
            f"request {request.request_id} has too few block_ids for a {token_count}-token read"
        )
    return list(source_ids[:block_count])


def _write_block_ids(
    request: Request,
    trace_block_indices: tuple[int, ...] | None,
    logical_blocks: int,
    block_size_tokens: int,
    default_start_index: int | None = None,
) -> list[str]:
    if request.block_ids:
        indices = (
            trace_block_indices
            if trace_block_indices is not None
            else tuple(
                range(
                    (
                        default_start_index
                        if default_start_index is not None
                        else request.matched_prefix_len // block_size_tokens
                    ),
                    (
                        default_start_index
                        if default_start_index is not None
                        else request.matched_prefix_len // block_size_tokens
                    )
                    + logical_blocks,
                )
            )
        )
        missing = [index for index in indices if not 0 <= index < len(request.block_ids)]
        if missing:
            raise ValueError(
                f"request {request.request_id} has no block_ids for write indices {missing}"
            )
        return [request.block_ids[index] for index in indices]

    token_count = request.miss_len
    if trace_block_indices:
        token_count = (max(trace_block_indices) + 1) * block_size_tokens
    write_ids = build_block_ids(
        request.model_id,
        f"{request.prefix_group_id}-miss-{request.request_id}",
        token_count,
        block_size_tokens,
    )
    if trace_block_indices is None:
        return write_ids
    return [write_ids[index] for index in trace_block_indices if 0 <= index < len(write_ids)]


def _decode_write_block_ids(
    request: Request,
    path: CanonicalRequestPath,
    trace_block_indices: tuple[int, ...] | None,
    logical_blocks: int,
    block_size_tokens: int,
) -> list[str]:
    start_index = request.seq_len // block_size_tokens
    indices = (
        trace_block_indices
        if trace_block_indices is not None
        else tuple(range(start_index, start_index + logical_blocks))
    )
    missing = [index for index in indices if not 0 <= index < len(path.output_block_ids)]
    if missing:
        raise ValueError(
            f"request {request.request_id} has no generated block ids for indices {missing}"
        )
    return [path.output_block_ids[index] for index in indices]
