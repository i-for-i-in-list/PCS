"""Aggregate device-independent demand reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ssu_sim.models.llm_model import ModelProfile
from ssu_sim.models.state import (
    PhaseTiming, RestoreReadGroup, ordered_restore_groups, phase_chain_time_us,
    shared_read_ready_times, phase_bandwidth_threshold, phase_read_deadlines,
)

if TYPE_CHECKING:
    from ssu_sim.demand.analyzer import (
        CapacitySummary,
        TTFTJob,
        _DemandEvent,
    )


def _unmodeled_full_hits(jobs) -> list:
    return [j for j in jobs if getattr(j, "full_hit_finalization_status", "not_applicable") == "not_evaluated"]


def _null_distribution() -> dict[str, None]:
    return {key: None for key in ("p50", "p95", "p99", "max")}


def _finalization_evaluation(jobs) -> dict[str, Any]:
    affected = _unmodeled_full_hits(jobs)
    return {
        "scope": "full_hit_finalization; other counting omissions and timing assumptions still apply",
        "status": "not_evaluated" if affected else "evaluated",
        "reason": "full_hit_finalization_unmodeled" if affected else None,
        "affected_request_count": len(affected),
        "affected_request_ids": [j.request_id for j in affected],
        "total_request_count": len(jobs),
        "full_hit_finalization_status_counts": dict(sorted(Counter(
            getattr(j, "full_hit_finalization_status", "not_applicable") for j in jobs).items())),
    }


def build_demand_report(
    *,
    events: list[_DemandEvent],
    capacity: CapacitySummary,
    latency_budgets_us: list[float],
    models: dict[str, ModelProfile],
    block_size_tokens: int,
    nominal_duration_us: float,
    peak_windows_us: tuple[int, ...],
    max_io_bytes: int | None,
    request_count: int,
    ttft_jobs: list[TTFTJob] | None = None,
    prefix_cache_policy: Any = None,
    max_seq_len_by_model: dict[str, int] | None = None,
    decode_write_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_seq_len_by_model = max_seq_len_by_model or {}
    model_cache = {}
    ttft_jobs = ttft_jobs or []
    for name, model in sorted(models.items()):
        seq_len = max_seq_len_by_model.get(name, 0)
        serialized_prefix_bytes = sum(
            value
            for key, value in capacity.reusable_prefix_bytes_by_component.items()
            if key.startswith(f"{model.name}:")
        )
        model_cache[name] = _model_cache_report(
            model,
            seq_len=seq_len,
            block_size_tokens=block_size_tokens,
            serialized_prefix_bytes=serialized_prefix_bytes,
            prefix_cache_policy=prefix_cache_policy,
        )
    report: dict[str, Any] = {
        "demand_report_schema_version": 3,
        "evaluation": _finalization_evaluation(ttft_jobs),
        "semantics": {
            "rates": "offered demand, not achieved device throughput",
            "latency": "maximum storage restore budget, not predicted device latency",
            "logical_block_ops": "one operation per accessed cache-component block",
            "expected_sparse_ops": (
                "fractional weights are uniform top-k expectations; bounds are separate"
            ),
            "chunked_ios": (
                f"logical blocks split at max_io_bytes={max_io_bytes}"
                if max_io_bytes is not None
                else "not derived; set max_io_bytes to estimate chunked SSD IOPS"
            ),
            "recovery": "configured, shareable, resident, and recoverable prefixes are distinct",
            "ttft": (
                "first-token boundary estimate, not end-to-end serving or measured TTFT"
            ),
            "serialized_payload": (
                "excludes allocator, page alignment, and filesystem overhead"
            ),
        },
        "request_count": request_count,
        "nominal_duration_us": nominal_duration_us,
        "block_size_tokens": block_size_tokens,
        "max_io_bytes": max_io_bytes,
        "swa_policy": asdict(prefix_cache_policy) if prefix_cache_policy is not None else None,
        "models": model_cache,
        "capacity": asdict(capacity),
        "read": _operation_report(
            [event for event in events if event.op_type == "read"],
            nominal_duration_us,
            peak_windows_us,
            max_io_bytes is not None,
        ),
        "write": _operation_report(
            [event for event in events if event.op_type == "write"],
            nominal_duration_us,
            peak_windows_us,
            max_io_bytes is not None,
        ),
        "latency_budget_us": _latency_budget_report(latency_budgets_us),
        "ttft": _ttft_report(ttft_jobs, max_io_bytes is not None),
        "recovery": _recovery_report(ttft_jobs, max_io_bytes is not None),
        "sparse_access_scenarios": _sparse_access_report(
            models, prefix_cache_policy
        ),
        "decode_write_summary": decode_write_summary
        or {
            "enabled": False,
            "schedule": "burst_at_completion",
            "step_us_per_token": None,
            "timing_validation": "uncalibrated_burst_scenario",
            "generated_token_count": 0,
            "write_events_enabled": False,
        },
        "timing_validation": _timing_validation_report(models, ttft_jobs),
    }
    if _unmodeled_full_hits(ttft_jobs):
        report["latency_budget_us"] = {
            **_null_distribution(), "status": "not_evaluated",
            "hit_request_count": sum(j.matched_prefix_tokens > 0 for j in ttft_jobs),
            "negative_budget_count": None, "negative_budget_ratio": None,
        }
    v4_components = {
        component
        for model in models.values()
        if model.attention.name == "deepseek_v4"
        for component in ("csa_main", "csa_indexer", "hca_main", "swa")
    }
    for operation in ("read", "write"):
        for component in sorted(v4_components):
            report[operation]["by_component"].setdefault(
                component,
                _operation_report(
                    [],
                    nominal_duration_us,
                    peak_windows_us,
                    max_io_bytes is not None,
                    include_component_breakdown=False,
                ),
            )
    return report


def _model_cache_report(
    model: ModelProfile,
    *,
    seq_len: int,
    block_size_tokens: int,
    serialized_prefix_bytes: int,
    prefix_cache_policy: Any,
) -> dict[str, Any]:
    identity = model.identity
    assert identity is not None
    footprint_method = getattr(model.attention, "cache_footprint", None)
    footprint: dict[str, Any] | None = None
    if callable(footprint_method):
        kwargs = {
            "seq_len": seq_len,
            "block_size_tokens": block_size_tokens,
            "sequence_count": 1,
        }
        try:
            value = footprint_method(
                model,
                policy=prefix_cache_policy,
                **kwargs,
            )
        except TypeError:
            value = footprint_method(model, **kwargs)
        footprint = asdict(value)

    resident_state = _resident_state_bytes(model, seq_len)
    if footprint is None:
        nominal = int(seq_len * model.cache_bytes_per_token())
        footprint = {
            "calculator_reference": {
                "total_bytes": nominal,
                "formula_id": model.attention.name,
                "settings": {"seq_len": seq_len, "sequence_count": 1},
                "snapshot_sha256": model.calculator_snapshot_sha256 or "unverified",
                "semantic_class": "token_payload",
                "by_component": {},
            },
            "active_execution": {
                "nominal_payload_bytes": nominal,
                "serialized_bytes": int(resident_state),
                "reserved_bytes": int(resident_state),
                "by_component": {},
            },
            "prefix_store": {
                "nominal_payload_bytes": nominal,
                "serialized_bytes": serialized_prefix_bytes,
                "object_count": (
                    (seq_len + block_size_tokens - 1) // block_size_tokens
                    if seq_len
                    else 0
                ),
                "checkpoint_count": 0,
                "by_component": {},
            },
        }

    return {
        "model_identity": asdict(identity),
        "catalog": asdict(model.catalog),
        "evidence": model.evidence.to_dict(),
        "config_digests": {
            "model_config_sha256": model.model_config_sha256,
            "runtime_config_sha256": model.runtime_config_sha256,
            "calculator_snapshot_sha256": model.calculator_snapshot_sha256,
        },
        "formula_implementation_id": model.formula_implementation_id,
        "draft_support": model.draft_support,
        "attention": model.attention.name,
        "compute": asdict(model.compute),
        "max_context_tokens": model.max_context_tokens,
        "cache_bytes_per_token": model.cache_bytes_per_token(),
        "cache_bytes_per_block": model.cache_bytes_per_block(block_size_tokens),
        "cache_components": _cache_components_report(model, block_size_tokens),
        "cache_layout": _cache_layout(model, block_size_tokens),
        "resident_state_bytes": resident_state,
        "resident_state_tokens": seq_len,
        "serialized_prefix_bytes": serialized_prefix_bytes,
        "calculator_reference": footprint["calculator_reference"],
        "active_execution_footprint": footprint["active_execution"],
        "prefix_store_footprint": footprint["prefix_store"],
    }


def _operation_report(
    events: list[_DemandEvent],
    nominal_duration_us: float,
    peak_windows_us: tuple[int, ...],
    chunking_enabled: bool,
    *,
    include_component_breakdown: bool = True,
) -> dict[str, Any]:
    total_bytes = sum(event.payload_bytes for event in events)
    logical_ops = sum(event.logical_block_ops for event in events)
    lower_ops = sum(event.block_ops_lower_bound for event in events)
    upper_ops = sum(event.block_ops_upper_bound for event in events)
    chunked_ios = sum(event.chunked_ios or 0 for event in events) if chunking_enabled else None
    duration_us = max(nominal_duration_us, 1.0)
    pattern_ops = Counter()
    pattern_bytes = Counter()
    for event in events:
        pattern_ops[event.access_pattern] += event.logical_block_ops
        pattern_bytes[event.access_pattern] += event.payload_bytes

    result: dict[str, Any] = {
        "total_bytes": total_bytes,
        "logical_block_ops": logical_ops,
        "expected_block_ops": logical_ops,
        "block_ops_lower_bound": lower_ops,
        "block_ops_upper_bound": upper_ops,
        "chunked_ios": chunked_ios,
        "offered_avg_GBps": _bytes_per_window_to_gbps(total_bytes, duration_us),
        "offered_avg_logical_block_ops_per_s": logical_ops * 1_000_000.0 / duration_us,
        "offered_avg_chunked_iops": (
            chunked_ios * 1_000_000.0 / duration_us
            if chunked_ios is not None
            else None
        ),
        "io_size_bytes": _weighted_percentiles(
            [(event.io_size_bytes, event.logical_block_ops) for event in events]
        ),
        "access_pattern_by_ops": dict(sorted(pattern_ops.items())),
        "access_pattern_by_bytes": dict(sorted(pattern_bytes.items())),
        "peak_windows": {
            _window_label(window): _peak_window(events, window, chunking_enabled)
            for window in sorted(set(peak_windows_us))
        },
    }
    if include_component_breakdown:
        result["by_component"] = {
            component: _operation_report(
                [event for event in events if event.component == component],
                nominal_duration_us,
                peak_windows_us,
                chunking_enabled,
                include_component_breakdown=False,
            )
            for component in sorted({event.component for event in events})
        }
    return result


def _peak_window(
    events: list[_DemandEvent],
    window_us: int,
    chunking_enabled: bool,
) -> dict[str, float | int | None]:
    if not events:
        return {
            "bytes": 0,
            "logical_block_ops": 0,
            "expected_block_ops": 0,
            "block_ops_lower_bound": 0,
            "block_ops_upper_bound": 0,
            "chunked_ios": 0 if chunking_enabled else None,
            "GBps": 0.0,
            "logical_block_ops_per_s": 0.0,
            "chunked_iops": 0.0 if chunking_enabled else None,
        }

    ordered = sorted(events, key=lambda event: event.time_us)
    left = 0
    current_bytes = 0.0
    current_logical_ops = 0.0
    current_lower_ops = 0.0
    current_upper_ops = 0.0
    current_chunked_ios = 0.0
    max_bytes = 0.0
    max_logical_ops = 0.0
    max_lower_ops = 0.0
    max_upper_ops = 0.0
    max_chunked_ios = 0.0

    for right, event in enumerate(ordered):
        current_bytes += event.payload_bytes
        current_logical_ops += event.logical_block_ops
        current_lower_ops += event.block_ops_lower_bound
        current_upper_ops += event.block_ops_upper_bound
        current_chunked_ios += event.chunked_ios or 0
        while ordered[right].time_us - ordered[left].time_us >= window_us:
            old = ordered[left]
            current_bytes -= old.payload_bytes
            current_logical_ops -= old.logical_block_ops
            current_lower_ops -= old.block_ops_lower_bound
            current_upper_ops -= old.block_ops_upper_bound
            current_chunked_ios -= old.chunked_ios or 0
            left += 1
        max_bytes = max(max_bytes, current_bytes)
        max_logical_ops = max(max_logical_ops, current_logical_ops)
        max_lower_ops = max(max_lower_ops, current_lower_ops)
        max_upper_ops = max(max_upper_ops, current_upper_ops)
        max_chunked_ios = max(max_chunked_ios, current_chunked_ios)

    return {
        "bytes": max_bytes,
        "logical_block_ops": max_logical_ops,
        "expected_block_ops": max_logical_ops,
        "block_ops_lower_bound": max_lower_ops,
        "block_ops_upper_bound": max_upper_ops,
        "chunked_ios": max_chunked_ios if chunking_enabled else None,
        "GBps": _bytes_per_window_to_gbps(max_bytes, window_us),
        "logical_block_ops_per_s": max_logical_ops * 1_000_000.0 / window_us,
        "chunked_iops": (
            max_chunked_ios * 1_000_000.0 / window_us
            if chunking_enabled
            else None
        ),
    }


def _latency_budget_report(values: list[float]) -> dict[str, float | int]:
    negative = sum(value < 0 for value in values)
    result: dict[str, float | int] = {
        "hit_request_count": len(values),
        "negative_budget_count": negative,
        "negative_budget_ratio": negative / len(values) if values else 0.0,
    }
    result.update(_percentiles(values))
    return result


def phase_timing_samples(jobs: list[TTFTJob]) -> list[dict[str, Any]]:
    """Compact repeated request shapes for rated-bandwidth evaluation.

    Keep per-request correlations without exporting block traces or adding
    independently aggregated phase percentiles.
    """
    if _unmodeled_full_hits(jobs):
        return []  # Never feed an incomplete timeline into rated-bandwidth timing.
    counts: Counter[tuple] = Counter()
    for job in jobs:
        if not job.phase_timings:
            continue
        reads = [0.0] * len(job.phase_timings)
        for index, group in ordered_restore_groups(job.phase_timings, job.restore_read_groups):
            reads[index] += group.payload_bytes
        key = (tuple(p.compute_us for p in job.phase_timings), tuple(reads),
               job.no_cache_ttft_us, job.cache_path_overhead_us, job.first_token_decode_us)
        counts[key] += 1
    return [{"count": count, "compute_us": list(key[0]), "read_bytes": list(key[1]),
             "no_cache_us": key[2], "overhead_us": key[3], "first_token_us": key[4]}
            for key, count in counts.items()]


def rated_phase_timing_report(samples: list[dict[str, Any]], bandwidth_GBps: float) -> dict[str, Any]:
    times, savings = [], []
    for sample in samples:
        reads = sample["read_bytes"]
        phases = tuple(PhaseTiming(str(i), float(cost), (str(i),) if reads[i] > 0 else ())
                       for i, cost in enumerate(sample["compute_us"]))
        groups = tuple(RestoreReadGroup(str(i), float(size)) for i, size in enumerate(reads) if size > 0)
        ready = shared_read_ready_times(phases, groups, bandwidth_GBps)
        elapsed = phase_chain_time_us(phases, ready, overhead_us=sample["overhead_us"],
                                     first_token_us=sample["first_token_us"])
        count = int(sample["count"])
        times.append((elapsed, count))
        savings.append((sample["no_cache_us"] - elapsed, count))
    return {"ttft_us": _weighted_percentiles(times), "savings_us": _weighted_percentiles(savings),
            "semantics": "isolated request, shared read bandwidth, first-consumer order; no queueing"}


def _execution_phase_report(jobs: list[TTFTJob]) -> list[dict[str, Any]]:
    # Internal replay/suffix timing adapters are not CED architecture phases.
    jobs = [job for job in jobs if job.execution_plan is not None]
    names = list(dict.fromkeys(p.name for j in jobs for p in j.phase_timings))
    result = []
    for name in names:
        costs = [p for j in jobs for p in j.phase_timings if p.name == name]
        shapes = [p for j in jobs if j.execution_plan for p in j.execution_plan.phases if p.name == name]
        result.append({"name": name,
            "layer_ranges": sorted({(p.layer_start, p.layer_end) for p in shapes}),
            "tokens": _distribution([float(p.tokens) for p in shapes]),
            "token_start": _distribution([float(p.token_start) for p in shapes]),
            "token_end": _distribution([float(p.token_end) for p in shapes]),
            "flops": _distribution([p.flops for p in costs]),
            "compute_us": _distribution([p.compute_us for p in costs]),
            "read_components": sorted({c for p in shapes for c in p.read_components}),
            "write_components": sorted({c for p in shapes for c in p.write_components}),
            "approximate": any(p.approximate for p in shapes)})
    return result


def _ttft_report(jobs: list[TTFTJob], chunking_enabled: bool) -> dict[str, Any]:
    optimistic = _bandwidth_threshold_report(jobs, "optimistic")
    dependency = _bandwidth_threshold_report(jobs, "dependency")
    dependency_at_infinite = [
        phase_chain_time_us(job.phase_timings, overhead_us=job.cache_path_overhead_us,
                            first_token_us=job.first_token_decode_us) if job.phase_timings else
        job.cache_path_overhead_us
        + job.replay_compute_time_us
        + job.miss_suffix_compute_time_us
        + job.first_token_decode_us
        for job in jobs
    ]
    optimistic_at_infinite = [
        max(
            sum(p.compute_us for p in job.phase_timings) if job.phase_timings else
            job.replay_compute_time_us + job.miss_suffix_compute_time_us,
            job.cache_path_overhead_us,
        )
        + job.first_token_decode_us
        for job in jobs
    ]
    calibrated = [
        job.calibrated_ttft_us
        for job in jobs
        if job.calibrated_ttft_us is not None
    ]
    timing_status = Counter(job.timing_validation for job in jobs)
    missing_classes = Counter(
        value for job in jobs for value in job.missing_operation_classes
    )
    result = {
        "model": "first_token_boundary",
        "evaluation": _finalization_evaluation(jobs),
        **({"execution_phases": _execution_phase_report(jobs)}
           if any(j.execution_plan is not None for j in jobs) else {}),
        **({"phase_timing_samples": phase_timing_samples(jobs)}
           if any(j.phase_timings for j in jobs) else {}),
        "semantics": (
            "arrival to first-token boundary with prefill, restore hiding, "
            "cache-path overhead, and optional first-token decode"
        ),
        "request_count": len(jobs),
        "restore_request_count": sum(job.restore_payload_bytes > 0 for job in jobs),
        "no_cache_ttft_us": _distribution([job.no_cache_ttft_us for job in jobs]),
        "optimistic_overlap_estimate_us": _distribution(
            optimistic_at_infinite
        ),
        "dependency_aware_estimate_us": _distribution(
            dependency_at_infinite
        ),
        "dependency_aware_estimate_semantics": (
            "zero-restore-latency lower bound; use rated recommendation timing "
            "or SSD simulation for a realizable estimate"
        ),
        "no_compute_io_overlap_estimate_us": None,
        "no_compute_io_overlap_status": "requires_storage_simulation",
        "dependency_ttft_savings_us": _distribution(
            [
                job.no_cache_ttft_us - value
                for job, value in zip(jobs, dependency_at_infinite)
            ]
        ),
        "dependency_ttft_savings_semantics": (
            "upper bound at zero restore latency"
        ),
        "no_cache_prefill_flops": _distribution(
            [job.no_cache_prefill_flops for job in jobs]
        ),
        "cache_path_flops": _distribution([job.cache_path_flops for job in jobs]),
        "cache_path_flops_semantics": "modeled operator subtotal; not a complete request cost when evaluation.status is not_evaluated",
        "net_saved_flops": _distribution([job.net_saved_flops for job in jobs]),
        "miss_suffix_flops": _distribution([job.miss_suffix_flops for job in jobs]),
        "suffix_body_flops": _distribution([job.suffix_body_flops for job in jobs]),
        "mandatory_conversion_flops": _distribution([job.mandatory_conversion_flops for job in jobs]),
        "restore_conversion_flops": _distribution([job.restore_conversion_flops for job in jobs]),
        "finalization_flops": _distribution([job.finalization_flops for job in jobs]),
        "output_head_flops": _distribution([job.output_head_flops for job in jobs]),
        "special_operation_counts": {
            name: _distribution([float(job.special_operation_counts.get(name, 0)) for job in jobs])
            for name in sorted({name for job in jobs for name in job.special_operation_counts})
        },
        "counting_omissions": sorted({item for job in jobs for item in job.counting_omissions}),
        "compute_algorithms": sorted({job.compute_algorithm_id for job in jobs if job.compute_algorithm_id}),
        "audit_metadata": {
            "exact_flops_by_request": [
                {"request_id": job.request_id, "model_id": job.model_id,
                 "full_hit_finalization_status": job.full_hit_finalization_status, **job.flops_exact}
                for job in jobs
            ],
            "exact_flops_semantics": "decimal integer strings; null means the formula did not supply a lossless integer count",
        },
        "flop_timing_semantics": "FLOP-equivalent timing excludes separately reported special operations and counting omissions",
        "suffix_compute_semantics": (
            "work required by new suffix tokens, including mandatory conversion and algorithm finalization; "
            "excludes state restoration and output head; zero for an empty suffix"
        ),
        "restore_recompute_tokens": _distribution(
            [float(job.restore_recompute_tokens) for job in jobs]
        ),
        "restore_recompute_flops": _distribution(
            [job.restore_recompute_flops for job in jobs]
        ),
        "compression_tail_flops": _distribution(
            [job.compression_tail_flops for job in jobs]
        ),
        "swa_restore_flops": _distribution([job.swa_restore_flops for job in jobs]),
        "flops_by_component": _flops_component_report(jobs),
        "flops_by_operation_class": _flops_operation_class_report(jobs),
        "analytical_flop_equivalent_ttft_us": _distribution(
            [job.analytical_flop_equivalent_ttft_us for job in jobs]
        ),
        "calibrated_ttft_us": _distribution(calibrated) if calibrated else None,
        "calibrated_request_count": len(calibrated),
        "timing_validation": dict(sorted(timing_status.items())),
        "missing_operation_classes": dict(sorted(missing_classes.items())),
        "timing_omissions": sorted(
            {value for job in jobs for value in job.timing_omissions}
        ),
        "saved_recompute_time_us": _distribution(
            [job.saved_recompute_time_us for job in jobs]
        ),
        "cache_path_overhead_us": _distribution(
            [job.cache_path_overhead_us for job in jobs]
        ),
        "replay_compute_time_us": _distribution(
            [job.replay_compute_time_us for job in jobs]
        ),
        "suffix_compute_time_us": _distribution(
            [job.miss_suffix_compute_time_us for job in jobs]
        ),
        "mandatory_conversion_time_us": _distribution([job.mandatory_conversion_time_us for job in jobs]),
        "finalization_time_us": _distribution([job.finalization_time_us for job in jobs]),
        "output_head_time_us": _distribution([job.output_head_time_us for job in jobs]),
        "first_token_decode_us": _distribution(
            [job.first_token_decode_us for job in jobs]
        ),
        "optimistic_overlap_GBps": optimistic,
        "dependency_aware_break_even_GBps": dependency,
        "optimistic_overlap_chunked_iops": _optimistic_iops_report(jobs)
        if chunking_enabled
        else None,
    }
    if _unmodeled_full_hits(jobs):
        # Mark the entire aggregate incomplete; do not compute percentiles from
        # only the evaluable subset or interpret a missing finalization as zero.
        for key in ("optimistic_overlap_estimate_us", "dependency_aware_estimate_us",
                    "dependency_ttft_savings_us", "net_saved_flops",
                    "analytical_flop_equivalent_ttft_us", "saved_recompute_time_us"):
            result[key] = _null_distribution()
        result["calibrated_ttft_us"] = None
        result["calibrated_request_count"] = 0
        result["no_compute_io_overlap_status"] = "not_evaluated"
    return result


def _recovery_report(jobs: list[TTFTJob], chunking_enabled: bool) -> dict[str, Any]:
    causes = Counter(
        value
        for job in jobs
        for value in (*job.restore_causes, *job.recovery_loss_causes)
    )
    restored = Counter(value for job in jobs for value in job.restored_objects)
    configured = [float(job.configured_prefix_tokens) for job in jobs]
    shareable = [float(job.shareable_prefix_tokens) for job in jobs]
    resident = [float(job.resident_prefix_tokens) for job in jobs]
    recovery = [float(job.recovery_boundary_tokens) for job in jobs]
    seq_lens = [max(job.seq_len, 1) for job in jobs]
    result = {
        "execution_scope": "phased" if any(j.execution_plan is not None for j in jobs) else "full_forward",
        **({"execution_phases": _execution_phase_report(jobs),
            "state_sources": {name: dict(Counter(j.execution_plan.state_sources.get(name, "unknown")
                for j in jobs if j.execution_plan))
                for name in ("encoder_state", "encoder_outputs", "encoder_tail_policy")},
            "generated_global_payload_bytes": sum(j.generated_global_payload_bytes for j in jobs)}
           if any(j.execution_plan is not None for j in jobs) else {}),
        "request_count": len(jobs),
        "funnel": {
            "configured_prefix_tokens": _distribution(configured),
            "shareable_prefix_tokens": _distribution(shareable),
            "resident_prefix_tokens": _distribution(resident),
            "recovery_boundary_tokens": _distribution(recovery),
            "replay_tokens": _distribution(
                [float(job.restore_recompute_tokens) for job in jobs]
            ),
            "suffix_tokens": _distribution(
                [float(job.suffix_tokens) for job in jobs]
            ),
        },
        "ratios": {
            "configured_prefix": _distribution(
                [value / length for value, length in zip(configured, seq_lens)]
            ),
            "shareable_prefix": _distribution(
                [value / length for value, length in zip(shareable, seq_lens)]
            ),
            "resident_prefix": _distribution(
                [value / length for value, length in zip(resident, seq_lens)]
            ),
            "recovery_boundary": _distribution(
                [value / length for value, length in zip(recovery, seq_lens)]
            ),
        },
        "bytes": {
            "resident_cache": _distribution([job.resident_cache_bytes for job in jobs]),
            "planned_restore": _distribution(
                [
                    (
                        job.planned_restore_payload_bytes
                        if job.planned_restore_payload_bytes > 0
                        else job.restore_payload_bytes
                        + job.optional_restore_payload_bytes
                    )
                    for job in jobs
                ]
            ),
            "replay_gate": _distribution([job.replay_gate_payload_bytes for job in jobs]),
            "suffix_gate": _distribution([job.suffix_gate_payload_bytes for job in jobs]),
            "optional_restore": _distribution([job.optional_restore_payload_bytes for job in jobs]),
            "blocking_checkpoint": _distribution([float(job.blocking_checkpoint_bytes) for job in jobs]),
            "stranded_resident": _distribution([job.stranded_resident_bytes for job in jobs]),
        },
        "compute": {
            "replay_tokens": _distribution(
            [float(job.restore_recompute_tokens) for job in jobs]
            ),
            "replay_flops": _distribution([job.restore_recompute_flops for job in jobs]),
            "suffix_flops": _distribution([job.miss_suffix_flops for job in jobs]),
            "net_saved_flops": _distribution([job.net_saved_flops for job in jobs]),
            "net_saved_flops_ratio": _distribution(
                [
                    job.net_saved_flops / job.no_cache_prefill_flops
                    if job.no_cache_prefill_flops > 0
                    else 0.0
                    for job in jobs
                ]
            ),
        },
        "critical_path": {
            "optimistic_overlap_GBps": _bandwidth_threshold_report(jobs, "optimistic"),
            "dependency_aware_break_even_GBps": _bandwidth_threshold_report(jobs, "dependency"),
            "aggregate_dependency": _aggregate_dependency_pressure(jobs, chunking_enabled),
        },
        "shareability_sources": dict(sorted(Counter(job.shareability_source for job in jobs).items())),
        "causes": dict(sorted(causes.items())),
        "restored_objects": dict(sorted(restored.items())),
    }
    if _unmodeled_full_hits(jobs):
        result["compute"]["net_saved_flops"] = _null_distribution()
        result["compute"]["net_saved_flops_ratio"] = _null_distribution()
    return result


def _bandwidth_threshold_report(jobs: list[TTFTJob], mode: str) -> dict[str, Any]:
    if _unmodeled_full_hits(jobs):
        return {**_null_distribution(), "status": "not_evaluated",
                "reason": "full_hit_finalization_unmodeled",
                "status_counts": {"not_evaluated": len(jobs)},
                "finite_count": 0, "non_finite_count": 0, "not_applicable_count": 0,
                "not_evaluated_count": len(jobs),
                "affected_request_count": len(_unmodeled_full_hits(jobs))}
    finite: list[float] = []
    statuses: Counter[str] = Counter()
    for job in jobs:
        if job.phase_timings:
            value, status = phase_bandwidth_threshold(job.phase_timings, job.restore_read_groups,
                job.no_cache_ttft_us, mode=mode, overhead_us=job.cache_path_overhead_us,
                first_token_us=job.first_token_decode_us)
            statuses[status] += 1
            if value is not None:
                finite.append(value)
            continue
        dependency_infinite = (
            job.cache_path_overhead_us
            + job.replay_compute_time_us
            + job.miss_suffix_compute_time_us
            + job.first_token_decode_us
        )
        if mode == "optimistic":
            if job.restore_payload_bytes <= 0:
                status = "no_restore_required"
                value = 0.0
            else:
                window = (
                    job.replay_compute_time_us
                    + job.miss_suffix_compute_time_us
                    - job.cache_path_overhead_us
                )
                if window <= 0:
                    status = "no_overlap_window"
                    value = None
                else:
                    status = "finite"
                    value = _bytes_per_window_to_gbps(job.restore_payload_bytes, window)
        else:
            if dependency_infinite >= job.no_cache_ttft_us - 1e-9:
                status = "no_compute_saving"
                value = None
            elif job.restore_payload_bytes <= 0:
                status = "no_restore_required"
                value = 0.0
            else:
                suffix_budget = (
                    job.no_cache_ttft_us
                    - job.miss_suffix_compute_time_us
                    - job.first_token_decode_us
                )
                all_window = suffix_budget - job.cache_path_overhead_us
                gate_window = (
                    suffix_budget
                    - job.cache_path_overhead_us
                    - job.replay_compute_time_us
                )
                if all_window <= 0:
                    status = "no_compute_saving"
                    value = None
                elif job.replay_gate_payload_bytes > 0 and gate_window <= 0:
                    status = "no_replay_slack"
                    value = None
                else:
                    all_required = _bytes_per_window_to_gbps(
                        job.restore_payload_bytes, all_window
                    )
                    gate_required = (
                        _bytes_per_window_to_gbps(
                            job.replay_gate_payload_bytes, gate_window
                        )
                        if job.replay_gate_payload_bytes > 0
                        else 0.0
                    )
                    status = "finite"
                    value = max(all_required, gate_required)
        statuses[status] += 1
        if value is not None:
            finite.append(value)
    result: dict[str, Any] = (
        _distribution(finite)
        if finite
        else {"p50": None, "p95": None, "p99": None, "max": None}
    )
    result.update(
        {
            "finite_count": len(finite),
            "non_finite_count": len(jobs) - len(finite),
            "not_applicable_count": statuses.get("no_restore_required", 0),
            "status_counts": dict(sorted(statuses.items())),
            "status": _summary_status(statuses),
        }
    )
    return result


def _optimistic_iops_report(jobs: list[TTFTJob]) -> dict[str, Any]:
    if _unmodeled_full_hits(jobs):
        return {**_null_distribution(), "status": "not_evaluated"}
    values = []
    for job in jobs:
        window = (
            (sum(phase.compute_us for phase in job.phase_timings) if job.phase_timings else
             job.replay_compute_time_us + job.miss_suffix_compute_time_us)
            - job.cache_path_overhead_us
        )
        if job.restore_chunked_ios is not None and window > 0:
            values.append(job.restore_chunked_ios * 1_000_000.0 / window)
    return _distribution(values)


def _aggregate_dependency_pressure(
    jobs: list[TTFTJob], chunking_enabled: bool
) -> dict[str, Any]:
    if _unmodeled_full_hits(jobs):
        return {"model": "phase_gated_uniform_rate_over_dependency_windows",
                "status": "not_evaluated", "reason": "full_hit_finalization_unmodeled",
                "invalid_window_count": None, "peak_required_GBps": None,
                "peak_required_logical_block_ops_per_s": None,
                "peak_required_chunked_iops": None}
    deltas: dict[float, tuple[float, float, float]] = {}
    invalid = 0

    def add_interval(
        job: TTFTJob,
        *,
        start_offset_us: float,
        end_offset_us: float,
        payload_bytes: float,
        logical_ops: float,
        chunked_ios: float,
    ) -> bool:
        if payload_bytes <= 0:
            return True
        window_us = end_offset_us - start_offset_us
        if window_us <= 0:
            return False
        gbps = _bytes_per_window_to_gbps(payload_bytes, window_us)
        logical = logical_ops * 1_000_000.0 / window_us
        iops = (
            chunked_ios * 1_000_000.0 / window_us
            if chunking_enabled
            else 0.0
        )
        start = job.arrival_time_us + start_offset_us
        end = job.arrival_time_us + end_offset_us
        _add_rate_delta(deltas, start, gbps, logical, iops)
        _add_rate_delta(deltas, end, -gbps, -logical, -iops)
        return True

    for job in jobs:
        if job.phase_timings:
            previous_deadline = 0.0
            for group in phase_read_deadlines(job.phase_timings, job.restore_read_groups,
                    job.no_cache_ttft_us, overhead_us=job.cache_path_overhead_us,
                    first_token_us=job.first_token_decode_us):
                valid = add_interval(job, start_offset_us=previous_deadline,
                    end_offset_us=group["deadline_us"], payload_bytes=group["payload_bytes"],
                    logical_ops=group["logical_ops"], chunked_ios=group["chunked_ios"] or 0.0)
                if not valid:
                    invalid += 1
                    break
                previous_deadline = group["deadline_us"]
            continue
        all_window = (
            job.no_cache_ttft_us
            - job.miss_suffix_compute_time_us
            - job.first_token_decode_us
            - job.cache_path_overhead_us
        )
        if job.restore_payload_bytes <= 0:
            continue
        if all_window <= 0:
            invalid += 1
            continue
        gate_bytes = min(
            max(job.replay_gate_payload_bytes, 0.0),
            job.restore_payload_bytes,
        )
        if gate_bytes <= 0:
            if not add_interval(
                job,
                start_offset_us=0.0,
                end_offset_us=all_window,
                payload_bytes=job.restore_payload_bytes,
                logical_ops=job.restore_logical_block_ops,
                chunked_ios=job.restore_chunked_ios or 0.0,
            ):
                invalid += 1
            continue

        gate_window = all_window - job.replay_compute_time_us
        if gate_window <= 0:
            invalid += 1
            continue
        gate_logical = min(
            max(job.replay_gate_logical_block_ops, 0.0),
            job.restore_logical_block_ops,
        )
        gate_iops = min(
            max(job.replay_gate_chunked_ios or 0.0, 0.0),
            job.restore_chunked_ios or 0.0,
        )
        valid = add_interval(
            job,
            start_offset_us=0.0,
            end_offset_us=gate_window,
            payload_bytes=gate_bytes,
            logical_ops=gate_logical,
            chunked_ios=gate_iops,
        )
        valid = add_interval(
            job,
            start_offset_us=gate_window,
            end_offset_us=all_window,
            payload_bytes=job.restore_payload_bytes - gate_bytes,
            logical_ops=job.restore_logical_block_ops - gate_logical,
            chunked_ios=(job.restore_chunked_ios or 0.0) - gate_iops,
        ) and valid
        if not valid:
            invalid += 1
    current = [0.0, 0.0, 0.0]
    peak = [0.0, 0.0, 0.0]
    for timestamp in sorted(deltas):
        for index, delta in enumerate(deltas[timestamp]):
            current[index] += delta
            peak[index] = max(peak[index], current[index])
    return {
        "model": "phase_gated_uniform_rate_over_dependency_windows",
        "invalid_window_count": invalid,
        "peak_required_GBps": peak[0],
        "peak_required_logical_block_ops_per_s": peak[1],
        "peak_required_chunked_iops": peak[2] if chunking_enabled else None,
    }


def _summary_status(statuses: Counter[str]) -> str:
    if not statuses:
        return "not_applicable"
    if len(statuses) == 1:
        return next(iter(statuses))
    if statuses.get("finite", 0) + statuses.get("no_restore_required", 0) == sum(statuses.values()):
        return "finite"
    return "mixed"


def _sparse_access_report(
    models: dict[str, ModelProfile], prefix_cache_policy: Any
) -> dict[str, Any]:
    selected = getattr(prefix_cache_policy, "sparse_access_policy", None)
    result: dict[str, Any] = {
        "selected_runtime_policy": asdict(selected) if selected is not None else None,
        "models": {},
        "semantics": (
            "capacity is independent of sparse SSD restore; sparse-on-demand values "
            "are explicit locality/layout/reuse scenarios"
        ),
    }
    for model_id, model in sorted(models.items()):
        profile_policy = getattr(model.attention, "sparse_access_policy", None)
        if profile_policy is not None or selected is not None:
            result["models"][model_id] = asdict(selected or profile_policy)
    return result


def _timing_validation_report(
    models: dict[str, ModelProfile], jobs: list[TTFTJob]
) -> dict[str, Any]:
    by_model: dict[str, list[TTFTJob]] = {}
    for job in jobs:
        by_model.setdefault(job.model_id, []).append(job)
    result: dict[str, Any] = {}
    for name, model in sorted(models.items()):
        model_jobs = by_model.get(name, [])
        statuses = sorted({job.timing_validation for job in model_jobs})
        missing = sorted(
            {
                value
                for job in model_jobs
                for value in job.missing_operation_classes
            }
        )
        if model.compute.timing_mode == "normalized_reference":
            status = "uncalibrated"
        elif model_jobs and all(job.calibrated_ttft_us is not None for job in model_jobs):
            status = "component_calibrated"
        else:
            status = "component_calibration_incomplete"
        result[name] = {
            "mode": model.compute.timing_mode,
            "status": status,
            "observed_evidence": statuses,
            "missing_operation_classes": missing,
            "omissions": list(model.compute.timing_omissions),
        }
    return result


def _flops_component_report(jobs: list[TTFTJob]) -> dict[str, Any]:
    components = sorted(
        {
            component
            for job in jobs
            for component in (
                set(job.cache_path_flops_by_component)
                | set(job.no_cache_flops_by_component)
            )
        }
    )
    return {
        component: {
            "no_cache_flops": _distribution(
                [job.no_cache_flops_by_component.get(component, 0.0) for job in jobs]
            ),
            "cache_path_flops": _distribution(
                [job.cache_path_flops_by_component.get(component, 0.0) for job in jobs]
            ),
        }
        for component in components
    }


def _flops_operation_class_report(jobs: list[TTFTJob]) -> dict[str, Any]:
    operation_classes = sorted(
        {
            operation_class
            for job in jobs
            for operation_class in (
                set(job.cache_path_flops_by_operation_class)
                | set(job.no_cache_flops_by_operation_class)
            )
        }
    )
    return {
        operation_class: {
            "no_cache_flops": _distribution(
                [
                    job.no_cache_flops_by_operation_class.get(operation_class, 0.0)
                    for job in jobs
                ]
            ),
            "cache_path_flops": _distribution(
                [
                    job.cache_path_flops_by_operation_class.get(operation_class, 0.0)
                    for job in jobs
                ]
            ),
        }
        for operation_class in operation_classes
    }


def _cache_layout(model: ModelProfile, block_size_tokens: int) -> dict[str, Any]:
    method = getattr(model.attention, "cache_layout", None)
    if method is not None:
        return method(model, block_size_tokens)
    return {
        "type": model.attention.name,
        "components": {
            component.name: component.report_dict(model, block_size_tokens)
            for component in model.cache_components()
        },
    }


def _cache_components_report(
    model: ModelProfile, block_size_tokens: int
) -> dict[str, Any]:
    components = {
        component.name: component.report_dict(model, block_size_tokens)
        for component in model.cache_components()
    }
    if model.attention.name == "deepseek_v4":
        layout = _cache_layout(model, block_size_tokens)
        components["swa"] = dict(layout["components"]["swa"])
    return components


def _resident_state_bytes(model: ModelProfile, seq_len: int) -> int | float:
    method = getattr(model.attention, "resident_state_bytes", None)
    if method is not None:
        return method(model, seq_len)
    return seq_len * model.cache_bytes_per_token()


def _add_rate_delta(
    deltas: dict[float, tuple[float, float, float]],
    time_us: float,
    gbps: float,
    logical_ops: float,
    chunked_iops: float,
) -> None:
    old_gbps, old_logical_ops, old_chunked_iops = deltas.get(time_us, (0.0, 0.0, 0.0))
    deltas[time_us] = (
        old_gbps + gbps,
        old_logical_ops + logical_ops,
        old_chunked_iops + chunked_iops,
    )


def _distribution(values: list[float]) -> dict[str, float]:
    result = _percentiles(values)
    result["max"] = float(max(values)) if values else 0.0
    return result


def _percentiles(values: list[float | int]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _weighted_percentiles(values: list[tuple[int, float]]) -> dict[str, float]:
    if not values:
        return _percentiles([])
    ordered = sorted((value, weight) for value, weight in values if weight > 0)
    if not ordered:
        return _percentiles([])
    total_weight = sum(weight for _, weight in ordered)

    def select(q: float) -> float:
        target = (q / 100.0) * total_weight
        seen = 0.0
        for value, weight in ordered:
            seen += weight
            if seen >= target:
                return float(value)
        return float(ordered[-1][0])

    return {"p50": select(50), "p95": select(95), "p99": select(99)}


def _percentile(values: list[float | int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((q / 100.0) * (len(ordered) - 1))
    return float(ordered[index])


def _bytes_per_window_to_gbps(payload_bytes: float, window_us: float) -> float:
    return payload_bytes * 1_000_000.0 / window_us / 1_000_000_000.0


def _window_label(window_us: int) -> str:
    if window_us % 1_000_000 == 0:
        return f"{window_us // 1_000_000}s"
    if window_us % 1_000 == 0:
        return f"{window_us // 1_000}ms"
    return f"{window_us}us"
