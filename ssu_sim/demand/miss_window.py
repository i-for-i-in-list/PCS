"""Legacy miss-window helper retained only for the original dashboard."""

from __future__ import annotations

from typing import Any, Iterable

from ssu_sim.models.llm_model import ModelProfile, PrefillComputeContext
from ssu_sim.models.execution import validate_execution_combination
from ssu_sim.workload.request import Request


INFINITY_LABEL = "Infinity"
MISS_COMPUTE_WINDOW_METRIC = "miss_compute_window_required_GBps_per_request"


def miss_compute_window_metrics(
    *, requests: Iterable[Request], model: ModelProfile, block_size_tokens: int
) -> dict[str, Any]:
    """Return the pre-v3 metric used by ``dashboard/`` compatibility code."""

    validate_execution_combination(model)
    deltas: dict[float, float] = {}
    aggregate_peak_gbps = 0.0
    first_payload_bytes = 0.0
    first_required_gbps: float | str | None = 0.0
    first_matched_prefix_tokens = 0
    first_miss_tokens = 0
    first_actual_prefill_flops = 0.0
    first_actual_prefill_time_us: float | None = 0.0
    is_full_hit_infinite = False
    not_evaluated_count = 0
    for index, request in enumerate(requests):
        context = PrefillComputeContext(
            seq_len=request.seq_len,
            matched_prefix_len=request.matched_prefix_len,
            generation_len=request.generation_len,
        )
        read_plan = model.attention.map_hit_to_reads(
            model=model,
            hit_tokens=request.matched_prefix_len,
            block_size_tokens=block_size_tokens,
            seq_len=request.seq_len,
            decode_len=request.generation_len,
        )
        payload_bytes = read_plan.payload_bytes
        breakdown = model.compute.cache_compute_breakdown(model, context)
        actual_flops = breakdown.cache_path_flops
        actual_us = (None if breakdown.full_hit_finalization_status == "not_evaluated"
                     else model.compute.actual_prefill_time_us(model, context))
        if index == 0:
            first_payload_bytes = payload_bytes
            first_matched_prefix_tokens = request.matched_prefix_len
            first_miss_tokens = request.miss_len
            first_actual_prefill_flops = actual_flops
            first_actual_prefill_time_us = actual_us
        if actual_us is None:
            not_evaluated_count += 1
            if index == 0:
                first_required_gbps = None
            continue
        if actual_us <= 0:
            if payload_bytes > 0:
                is_full_hit_infinite = True
                if index == 0:
                    first_required_gbps = INFINITY_LABEL
            continue
        required = payload_bytes * 1_000_000.0 / actual_us / 1_000_000_000.0
        if index == 0:
            first_required_gbps = required
        deltas[request.arrival_time_us] = deltas.get(request.arrival_time_us, 0.0) + required
        end = request.arrival_time_us + actual_us
        deltas[end] = deltas.get(end, 0.0) - required
    current = 0.0
    for timestamp in sorted(deltas):
        current += deltas[timestamp]
        aggregate_peak_gbps = max(aggregate_peak_gbps, current)
    return {
        "hit_cache_payload_bytes": first_payload_bytes,
        "matched_prefix_tokens": first_matched_prefix_tokens,
        "miss_tokens": first_miss_tokens,
        "actual_prefill_flops": first_actual_prefill_flops,
        "actual_prefill_time_us": first_actual_prefill_time_us,
        "required_GBps_per_request": first_required_gbps,
        "aggregate_peak_GBps": (None if not_evaluated_count else
                                INFINITY_LABEL if is_full_hit_infinite else aggregate_peak_gbps),
        "is_full_hit_infinite": is_full_hit_infinite,
        "evaluation_status": "not_evaluated" if not_evaluated_count else "evaluated",
        "not_evaluated_count": not_evaluated_count,
    }
