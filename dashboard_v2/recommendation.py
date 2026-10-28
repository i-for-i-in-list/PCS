"""Candidate-grid recommendation engine for Dashboard V3."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import statistics
from typing import Any, Callable

from ssu_sim.models.state import ENCODER_SCENARIO_VALUES, normalize_encoder_scenarios
from ssu_sim.demand.report import rated_phase_timing_report


MAX_RECOMMENDATION_COMPUTE_CASES = 1_200
OBJECTIVES = {
    "balanced_headroom": "Maximize worst-case normalized resource headroom",
    "max_coverage": "Maximize workload-envelope coverage",
    "min_storage_pressure": "Minimize average storage pressure",
    "max_dependency_ttft_margin": "Maximize worst-case dependency-aware TTFT margin",
}

MAX_CONSTRAINTS = {
    "max_read_avg_GBps": (
        "read_avg_GBps",
        "sequential_read_GBps",
        "Average read bandwidth",
    ),
    "max_write_avg_GBps": (
        "write_avg_GBps",
        "sequential_write_GBps",
        "Average write bandwidth",
    ),
    "max_aggregate_dependency_required_GBps": (
        "aggregate_dependency_required_GBps",
        "sequential_read_GBps",
        "Aggregate dependency bandwidth",
    ),
    "max_aggregate_dependency_chunked_iops": (
        "aggregate_dependency_chunked_iops",
        "random_read_iops",
        "Aggregate dependency read IOPS",
    ),
    "max_retain_all_capacity_GiB": (
        "retain_all_capacity_GiB",
        "logical_capacity_GiB",
        "Retain-all capacity",
    ),
    "max_dependency_aware_break_even_GBps": (
        "dependency_aware_break_even_GBps_max",
        "sequential_read_GBps",
        "Dependency-aware break-even bandwidth",
    ),
    "max_optimistic_overlap_GBps": (
        "optimistic_overlap_GBps_max",
        "sequential_read_GBps",
        "Optimistic fully-hidden bandwidth",
    ),
}
MIN_CONSTRAINTS = {
    "min_dependency_ttft_savings_us_p50": (
        "dependency_ttft_savings_at_rated_bandwidth_us_p50",
        0.0,
        "Dependency-aware TTFT savings p50",
    ),
}

MODE_CONSTRAINTS = {
    "bandwidth_envelope": {
        "max_dependency_aware_break_even_GBps",
        "max_optimistic_overlap_GBps",
        "min_dependency_ttft_savings_us_p50",
    },
    "cache_replay": {
        "max_read_avg_GBps",
        "max_write_avg_GBps",
        "max_aggregate_dependency_required_GBps",
        "max_aggregate_dependency_chunked_iops",
        "max_retain_all_capacity_GiB",
        "min_dependency_ttft_savings_us_p50",
    },
}


class RecommendationCanceled(Exception):
    """Cooperative cancellation boundary for recommendation jobs."""


def recommendation_definitions() -> dict[str, Any]:
    return {
        "objectives": [
            {"value": value, "label": label}
            for value, label in OBJECTIVES.items()
        ],
        "constraints": [
            {
                "key": key,
                "metric": metric,
                "direction": "max",
                "storage_threshold": storage_threshold,
                "label": label,
            }
            for key, (metric, storage_threshold, label) in MAX_CONSTRAINTS.items()
        ]
        + [
            {
                "key": key,
                "metric": metric,
                "direction": "min",
                "default": default,
                "label": label,
            }
            for key, (metric, default, label) in MIN_CONSTRAINTS.items()
        ],
    }


def normalize_recommendation_payload(
    payload: dict[str, Any],
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("recommendation request must be an object")
    design = payload.get("design_space")
    workload = payload.get("workload_envelope")
    if not isinstance(design, dict) or not isinstance(workload, dict):
        raise ValueError("design_space and workload_envelope are required")
    analysis_mode = str(payload.get("analysis_mode") or "bandwidth_envelope")
    if analysis_mode not in {"bandwidth_envelope", "cache_replay"}:
        raise ValueError("invalid recommendation analysis_mode")

    # The picker exposes only active catalog entries, while saved V1/V2
    # recommendation experiments may still reference hidden sample profiles.
    allowed_models = {
        item["path"]
        for key in ("models", "legacy_models")
        for item in config_payload.get(key, [])
    }
    allowed_runtimes = {item["path"] for item in config_payload.get("runtimes", [])}
    allowed_storages = {item["path"] for item in config_payload.get("storages", [])}
    allowed_workloads = {item["path"] for item in config_payload.get("workloads", [])}

    raw_models = design.get("model_configs")
    if not isinstance(raw_models, list) or not raw_models:
        raw_models = list(
            (config_payload.get("defaults") or {}).get(
                "recommendation_model_configs",
                (config_payload.get("defaults") or {}).get("model_configs", ()),
            )
        )
    model_configs = sorted(
        _paths({**design, "model_configs": raw_models}, "model_configs", allowed_models)
    )
    runtime_configs = sorted(_paths(design, "runtime_configs", allowed_runtimes))
    storage_configs = sorted(_paths(design, "storage_configs", allowed_storages))
    block_sizes = sorted(
        _numbers(design, "block_size_tokens", integer=True, positive=True)
    )
    max_io_values = sorted(
        _optional_integers(design.get("max_io_bytes")),
        key=lambda item: -1 if item is None else item,
    )

    workload_config = str(workload.get("workload_config") or "")
    if workload_config not in allowed_workloads:
        raise ValueError("unknown workload_config")
    seq_lens = sorted(
        _numbers(workload, "seq_len", integer=True, positive=True)
    )
    hit_rates = sorted(_numbers(workload, "prefix_ratio"))
    if any(value < 0 or value > 1 for value in hit_rates):
        raise ValueError("prefix_ratio must stay within [0, 1]")
    qps_values = sorted(_numbers(workload, "qps", positive=True))
    arrivals = sorted(_strings(workload, "arrival_model"))
    if any(value not in {"fixed", "poisson"} for value in arrivals):
        raise ValueError("arrival_model must be fixed or poisson")

    objective = str(payload.get("objective") or "balanced_headroom")
    if objective not in OBJECTIVES:
        raise ValueError("invalid recommendation objective")
    raw_constraints = payload.get("constraints") or {}
    if not isinstance(raw_constraints, dict):
        raise ValueError("constraints must be an object")
    constraints: dict[str, Any] = {
        "use_storage_limits": bool(raw_constraints.get("use_storage_limits", True))
    }
    for key in MAX_CONSTRAINTS:
        value = _optional_number(raw_constraints.get(key), key)
        if value is not None and value < 0:
            raise ValueError(f"{key} must be non-negative")
        constraints[key] = value
    for key, (_metric, default, _label) in MIN_CONSTRAINTS.items():
        value = raw_constraints.get(key, default)
        constraints[key] = _optional_number(value, key)

    overrides = dict(payload.get("workload_overrides") or {})
    overrides.pop("qps", None)
    modeling_options = payload.get("modeling_options")
    if modeling_options is None:
        modeling_options = (config_payload.get("defaults") or {}).get(
            "modeling_options",
            {},
        )
    if not isinstance(modeling_options, dict):
        raise ValueError("modeling_options must be an object")
    config_digests = payload.get("config_digests") or {}
    if not isinstance(config_digests, dict):
        raise ValueError("config_digests must be an object")
    cache_replay_config = payload.get("cache_replay_config")
    if analysis_mode == "cache_replay" and not isinstance(cache_replay_config, dict):
        raise ValueError("cache_replay recommendation requires cache_replay_config")
    if analysis_mode == "bandwidth_envelope":
        cache_replay_config = None
    return {
        "analysis_mode": analysis_mode,
        "design_space": {
            "model_configs": model_configs,
            "runtime_configs": runtime_configs,
            "storage_configs": storage_configs,
            "block_size_tokens": block_sizes,
            "max_io_bytes": max_io_values,
        },
        "workload_envelope": {
            "workload_config": workload_config,
            "seq_len": seq_lens,
            "prefix_ratio": hit_rates,
            "arrival_model": arrivals,
            "qps": qps_values,
            **(normalize_encoder_scenarios(workload) if any(
                item["path"] in model_configs and item.get("recovery_capability", {}).get("encoder_scenarios_supported")
                for item in config_payload.get("models", [])) else {}),
        },
        "workload_overrides": overrides,
        "write_miss_blocks": bool(payload.get("write_miss_blocks", True)),
        "modeling_options": copy.deepcopy(modeling_options),
        "config_digests": copy.deepcopy(config_digests),
        "constraints": constraints,
        "objective": objective,
        "cache_replay_config": copy.deepcopy(cache_replay_config),
    }


def recommendation_case_count(
    params: dict[str, Any],
    config_payload: dict[str, Any],
) -> int:
    designs = _compatible_compute_designs(params, config_payload)
    return sum(len(_scenarios(params, design)) for design in designs)


def validate_recommendation_params(
    params: dict[str, Any],
    config_payload: dict[str, Any],
) -> list[str]:
    """Validate constraints that must be known before any Demand Case runs."""

    storage_thresholds = {
        item["path"]: item.get("thresholds") or {}
        for item in config_payload.get("storages", [])
    }
    iops_active = False
    for storage_config in params["design_space"]["storage_configs"]:
        effective = _effective_constraints(
            params["constraints"],
            storage_thresholds.get(storage_config, {}),
            params["analysis_mode"],
        )
        iops_active = iops_active or "max_aggregate_dependency_chunked_iops" in effective

    max_io_values = params["design_space"]["max_io_bytes"]
    if iops_active and all(value is None for value in max_io_values):
        raise ValueError(
            "max IO bytes is required when a read IOPS constraint is active"
        )
    if iops_active and any(value is None for value in max_io_values):
        return [
            "Candidates without max IO bytes cannot evaluate read IOPS and "
            "will be marked infeasible."
        ]
    return []


def run_recommendation(
    payload: dict[str, Any],
    config_payload: dict[str, Any],
    *,
    case_runner: Callable[[dict[str, Any]], dict[str, Any]],
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    params = normalize_recommendation_payload(payload, config_payload)
    preflight_warnings = validate_recommendation_params(params, config_payload)
    compute_designs = _compatible_compute_designs(params, config_payload)
    scenarios = _scenarios(params)
    total_compute_cases = sum(len(_scenarios(params, design)) for design in compute_designs)
    if not compute_designs:
        raise ValueError("no compatible design candidates remain")
    if total_compute_cases > MAX_RECOMMENDATION_COMPUTE_CASES:
        raise ValueError(
            "recommendation has "
            f"{total_compute_cases} compute cases; reduce the grid to at most "
            f"{MAX_RECOMMENDATION_COMPUTE_CASES}"
        )

    computed: dict[tuple[str, str], dict[str, Any]] = {}
    completed = 0
    default_storage = params["design_space"]["storage_configs"][0]
    for design in compute_designs:
        for scenario in _scenarios(params, design):
            if cancel_requested is not None and cancel_requested():
                raise RecommendationCanceled()
            case = {
                "id": f"recommendation-case-{completed + 1:04d}",
                "analysis_mode": params["analysis_mode"],
                "model_config": design["model_config"],
                "runtime_config": design["runtime_config"],
                "workload_config": params["workload_envelope"]["workload_config"],
                "storage_config": default_storage,
                "seq_len": scenario["seq_len"],
                "prefix_ratio": scenario["prefix_ratio"],
                "block_size_tokens": design["block_size_tokens"],
                "arrival_model": scenario["arrival_model"],
                "max_io_bytes": design["max_io_bytes"],
                "workload_overrides": {
                    **params["workload_overrides"],
                    "qps": scenario["qps"],
                },
                "write_miss_blocks": params["write_miss_blocks"],
                "modeling_options": params["modeling_options"],
                "config_digests": params["config_digests"],
            }
            case.update({key: scenario[key] for key in ENCODER_SCENARIO_VALUES if key in scenario})
            if params["analysis_mode"] == "cache_replay":
                case["cache_replay_config"] = copy.deepcopy(
                    params["cache_replay_config"]
                )
            case["case_identity"] = _case_identity(case)
            row = case_runner(case)
            computed[(_stable_key(design), _stable_key(scenario))] = row
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_compute_cases)
            if cancel_requested is not None and cancel_requested():
                raise RecommendationCanceled()

    storage_thresholds = {
        item["path"]: item.get("thresholds") or {}
        for item in config_payload.get("storages", [])
    }
    candidates = []
    all_rows = []
    for compute_design in compute_designs:
        for storage_config in params["design_space"]["storage_configs"]:
            design = {
                **compute_design,
                "storage_config": storage_config,
                "storage_config_sha256": (
                    params.get("config_digests", {})
                    .get("storages", {})
                    .get(storage_config)
                ),
            }
            candidate_id = _candidate_id(design)
            rows = []
            design_scenarios = _scenarios(params, compute_design)
            for scenario_index, scenario in enumerate(design_scenarios, start=1):
                source = computed[
                    (_stable_key(compute_design), _stable_key(scenario))
                ]
                row = copy.deepcopy(source)
                row["id"] = f"{candidate_id}-case-{scenario_index:04d}"
                row["candidate_id"] = candidate_id
                row["case"]["id"] = row["id"]
                row["case"]["storage_config"] = storage_config
                row["case"]["case_identity"] = _case_identity(row["case"])
                rows.append(row)
                all_rows.append(row)
            candidates.append(
                _evaluate_candidate(
                    candidate_id=candidate_id,
                    design=design,
                    rows=rows,
                    constraints=params["constraints"],
                    storage_thresholds=storage_thresholds.get(storage_config, {}),
                    analysis_mode=params["analysis_mode"],
                    required_combinations={(x.get("encoder_state"), x.get("encoder_tail_policy")) for x in design_scenarios},
                )
            )

    ranked, recommendation = _rank_evaluated_candidates(candidates, params["objective"])
    fully_feasible = sum(item["fully_feasible"] for item in ranked)
    warnings = list(preflight_warnings)
    warnings.extend(sorted({message for row in all_rows for message in row.get("compute_warnings", [])}
                           - set(warnings)))
    if recommendation is None and ranked:
        warnings.append("No design has a fully evaluated timeline; incomplete cases are excluded from performance ranking.")
    elif not fully_feasible:
        warnings.append(
            "No design covers the full workload envelope; showing the best "
            "partial coverage."
        )
    result = {
        "sweep_result_schema_version": 3,
        "analysis_mode": params["analysis_mode"],
        "kind": "recommendation",
        "objective": params["objective"],
        "case_count": total_compute_cases,
        "evaluated_case_count": len(all_rows),
        "scenario_count": len(scenarios),
        "candidate_count": len(ranked),
        "fully_feasible_candidate_count": fully_feasible,
        "recommended_candidate_id": recommendation["candidate_id"] if recommendation else None,
        "candidates": ranked,
        "rows": all_rows,
        "summary": {
            "case_count": total_compute_cases,
            "evaluated_case_count": len(all_rows),
            "scenario_count": len(scenarios),
            "candidate_count": len(ranked),
            "fully_feasible_candidate_count": fully_feasible,
            "recommended_candidate_id": recommendation["candidate_id"] if recommendation else None,
            "recommended_coverage_ratio": recommendation["coverage_ratio"] if recommendation else None,
            "recommended_worst_headroom": recommendation["worst_headroom"] if recommendation else None,
            "recommended_binding_constraint": recommendation["binding_constraint"] if recommendation else None,
            **_risk_summary(all_rows),
        },
        "request": params,
        "warnings": warnings,
    }
    return result


def _rank_evaluated_candidates(candidates, objective):
    evaluated = sorted((c for c in candidates if c.get("evaluation_status") != "not_evaluated"),
                       key=lambda c: _ranking_key(c, objective))
    unevaluated = [c for c in candidates if c.get("evaluation_status") == "not_evaluated"]
    for rank, candidate in enumerate(evaluated, start=1):
        candidate["rank"] = rank
    for candidate in unevaluated:
        candidate["rank"] = None
    return evaluated + unevaluated, evaluated[0] if evaluated else None


def rerank_recommendation(result: dict[str, Any], objective: str) -> dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ValueError("invalid recommendation objective")
    reranked = copy.deepcopy(result)
    candidates, recommendation = _rank_evaluated_candidates(reranked.get("candidates") or [], objective)
    reranked["objective"] = objective
    reranked["candidates"] = candidates
    reranked["recommended_candidate_id"] = (
        recommendation["candidate_id"] if recommendation else None
    )
    return reranked


def _compatible_compute_designs(
    params: dict[str, Any],
    config_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    runtimes = {
        item["path"]: item for item in config_payload.get("runtimes", [])
    }
    max_seq_len = max(params["workload_envelope"]["seq_len"])
    designs = []
    for runtime_path in params["design_space"]["runtime_configs"]:
        compatibility = runtimes[runtime_path].get("model_compatibility") or {}
        for model_path in params["design_space"]["model_configs"]:
            model = compatibility.get(model_path) or {}
            if not model.get("compatible"):
                continue
            constraints = model.get("constraints") or {}
            if constraints.get("max_context_tokens") and max_seq_len > int(
                constraints["max_context_tokens"]
            ):
                continue
            multiple = int(constraints.get("block_size_multiple") or 1)
            minimum = int(constraints.get("minimum_block_size_tokens") or 1)
            for block_size in params["design_space"]["block_size_tokens"]:
                if block_size < minimum or block_size % multiple != 0:
                    continue
                for max_io_bytes in params["design_space"]["max_io_bytes"]:
                    designs.append(
                        {
                            "model_config": model_path,
                            "model_name": model.get("model_name") or model_path,
                            "attention": model.get("attention") or "unknown",
                            **({"encoder_scenarios_supported": True} if model.get("recovery_capability", {}).get("encoder_scenarios_supported") else {}),
                            "runtime_config": runtime_path,
                            "block_size_tokens": block_size,
                            "max_io_bytes": max_io_bytes,
                            "modeling_options": params["modeling_options"],
                            "config_digests": {
                                "model": (
                                    params.get("config_digests", {})
                                    .get("models", {})
                                    .get(model_path)
                                ),
                                "runtime": (
                                    params.get("config_digests", {})
                                    .get("runtimes", {})
                                    .get(runtime_path)
                                ),
                                "workload": params.get("config_digests", {}).get(
                                    "workload"
                                ),
                            },
                        }
                    )
    return designs


def _scenarios(params: dict[str, Any], design: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    workload = params["workload_envelope"]
    variants = [{}]
    if design and design.get("encoder_scenarios_supported"):
        values = normalize_encoder_scenarios(workload)
        variants = [{"encoder_state": state, "encoder_tail_policy": tail}
                    for state in values["encoder_state"] for tail in values["encoder_tail_policy"]]
    return [{"seq_len": seq_len, "prefix_ratio": hit_rate, "arrival_model": arrival, "qps": qps, **variant}
            for seq_len in workload["seq_len"] for hit_rate in workload["prefix_ratio"]
            for arrival in workload["arrival_model"] for qps in workload["qps"] for variant in variants]


def _evaluate_candidate(
    *,
    candidate_id: str,
    design: dict[str, Any],
    rows: list[dict[str, Any]],
    constraints: dict[str, Any],
    storage_thresholds: dict[str, Any],
    analysis_mode: str,
    required_combinations: set[tuple] | None = None,
) -> dict[str, Any]:
    effective = _effective_constraints(
        constraints, storage_thresholds, analysis_mode
    )
    scenario_results = []
    resource_headrooms = []
    scenario_pressures = []
    ttft_margins = []
    resource_binding: tuple[float, str, dict[str, Any]] | None = None
    resource_binding_evidence: dict[str, Any] | None = None
    worst_ttft: tuple[float, dict[str, Any]] | None = None
    worst_ttft_evidence: dict[str, Any] | None = None
    failure_counts: Counter[str] = Counter()
    missing_failure_counts: Counter[str] = Counter()
    row_ids = []

    for row in rows:
        row_ids.append(row["id"])
        evaluation = row.get("evaluation") or {}
        if evaluation.get("status") in {"not_evaluated", "unsupported"}:
            failure = {"constraint": "evaluation", "metric": None,
                       "label": "Case not evaluated", "value": None, "limit": None,
                       "direction": None, "reason": evaluation.get("reason", "not_evaluated")}
            failure_counts["evaluation"] += 1
            scenario_results.append({**_scenario_from_row(row), "row_id": row["id"],
                                     "feasible": False, "failures": [failure]})
            row["recommendation_evaluation"] = {"candidate_id": candidate_id,
                "feasible": False, "status": "not_evaluated", "failures": [failure]}
            continue
        _add_rated_storage_ttft(row, storage_thresholds)
        failures = []
        scenario_headrooms = []
        scenario = _scenario_from_row(row)
        for key, spec in effective.items():
            metric = spec["metric"]
            value = _numeric_value((row.get("metrics") or {}).get(metric))
            if spec["direction"] == "max":
                valid = value is not None and math.isfinite(value) and value >= 0
                passed = valid and value <= spec["limit"]
                if valid and spec["limit"] > 0:
                    utilization = value / spec["limit"]
                elif valid and spec["limit"] == 0 and value == 0:
                    utilization = 0.0
                else:
                    utilization = math.inf
                headroom = 1 - utilization
                scenario_headrooms.append(headroom)
                resource_headrooms.append(headroom)
                pressure = utilization
                if (
                    resource_binding is None
                    or pressure > resource_binding[0]
                ):
                    resource_binding = (pressure, key, scenario)
                    resource_binding_evidence = {
                        "constraint": key,
                        "metric": metric,
                        "value": _json_number(value),
                        "limit": spec["limit"],
                        "utilization": _json_number(utilization),
                        "headroom": _json_number(headroom),
                        "scenario": copy.deepcopy(scenario),
                    }
            else:
                passed = value is not None and value >= spec["limit"]
                margin = value - spec["limit"] if value is not None else -math.inf
                ttft_margins.append(margin)
                if worst_ttft is None or margin < worst_ttft[0]:
                    worst_ttft = (margin, scenario)
                    worst_ttft_evidence = {
                        "value": _json_number(value),
                        "limit": spec["limit"],
                        "margin": _json_number(margin),
                        "scenario": copy.deepcopy(scenario),
                    }
            if not passed:
                failure_counts[key] += 1
                reason = _failure_reason(value, spec)
                if reason == "missing_metric":
                    missing_failure_counts[key] += 1
                failure = {
                    "constraint": key,
                    "metric": metric,
                    "label": spec["label"],
                    "value": _json_number(value),
                    "limit": spec["limit"],
                    "direction": spec["direction"],
                    "reason": reason,
                }
                if reason == "missing_metric" and metric == "aggregate_dependency_chunked_iops":
                    failure["hint"] = "Set max IO bytes to evaluate read IOPS."
                failures.append(failure)
        scenario_results.append(
            {
                **scenario,
                "row_id": row["id"],
                "feasible": not failures,
                "failures": failures,
            }
        )
        row["recommendation_evaluation"] = {
            "candidate_id": candidate_id,
            "feasible": not failures,
            "failures": copy.deepcopy(failures),
        }
        if scenario_headrooms:
            scenario_pressures.append(1 - min(scenario_headrooms))

    feasible_count = sum(item["feasible"] for item in scenario_results)
    scenario_count = len(scenario_results)
    coverage = feasible_count / scenario_count if scenario_count else 0
    evaluated = failure_counts["evaluation"] == 0
    hit_boundaries = _sampled_hit_boundaries(scenario_results, required_combinations=required_combinations)
    return {
        "candidate_id": candidate_id,
        "evaluation_status": "evaluated" if evaluated else "not_evaluated",
        "design": design,
        "effective_state_policies": copy.deepcopy(
            rows[0].get("effective_state_policies", {}) if rows else {}
        ),
        "heterogeneous_state_policy": bool(
            (design.get("modeling_options") or {}).get("model_state_policies")
            or (design.get("modeling_options") or {}).get("state_policies")
        ),
        "fully_feasible": evaluated and feasible_count == scenario_count,
        "feasible_scenario_count": feasible_count,
        "scenario_count": scenario_count,
        "coverage_ratio": coverage if evaluated else None,
        "worst_headroom": (min(resource_headrooms) if resource_headrooms else math.inf) if evaluated else None,
        "mean_storage_pressure": (
            (statistics.fmean(scenario_pressures) if scenario_pressures else 0) if evaluated else None
        ),
        "min_ttft_margin_us": (min(ttft_margins) if ttft_margins else math.inf) if evaluated else None,
        "binding_resource_constraint": (
            resource_binding[1] if resource_binding else None
        ),
        "worst_resource_scenario": (
            resource_binding[2] if resource_binding else None
        ),
        "worst_ttft_scenario": worst_ttft[1] if worst_ttft else None,
        "binding_resource_evidence": resource_binding_evidence,
        "worst_ttft_evidence": worst_ttft_evidence,
        # Compatibility aliases now intentionally describe resource pressure only.
        "binding_constraint": resource_binding[1] if resource_binding else None,
        "worst_scenario": resource_binding[2] if resource_binding else None,
        "constraint_failures": [
            {
                "constraint": key,
                "count": count,
                "missing_count": missing_failure_counts[key],
            }
            for key, count in sorted(
                failure_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "sampled_hit_rate_boundaries": hit_boundaries,
        "row_ids": row_ids,
        "rank": 0,
    }


def _add_rated_storage_ttft(
    row: dict[str, Any], storage_thresholds: dict[str, Any]
) -> None:
    metrics = row.get("metrics") or {}
    if (row.get("evaluation") or {}).get("status") in {"not_evaluated", "unsupported"}:
        metrics["dependency_ttft_at_rated_bandwidth_us_p50"] = None
        metrics["dependency_ttft_savings_at_rated_bandwidth_us_p50"] = None
        return
    if metrics.get("dependency_ttft_savings_at_rated_bandwidth_us_p50") is not None:
        return
    bandwidth = _numeric_value(storage_thresholds.get("sequential_read_GBps"))
    if bandwidth is None or bandwidth <= 0:
        metrics["dependency_ttft_at_rated_bandwidth_us_p50"] = None
        metrics["dependency_ttft_savings_at_rated_bandwidth_us_p50"] = None
        return
    if row.get("phase_timing_samples"):
        result = rated_phase_timing_report(row["phase_timing_samples"], bandwidth)
        metrics["dependency_ttft_at_rated_bandwidth_us_p50"] = result["ttft_us"]["p50"]
        metrics["dependency_ttft_savings_at_rated_bandwidth_us_p50"] = result["savings_us"]["p50"]
        row["rated_timing_semantics"] = result["semantics"]
        return
    gate_bytes = _numeric_value(metrics.get("replay_gate_bytes_p50")) or 0.0
    all_bytes = _numeric_value(metrics.get("suffix_gate_bytes_p50")) or 0.0
    replay = _numeric_value(metrics.get("replay_compute_time_us_p50")) or 0.0
    suffix = _numeric_value(metrics.get("suffix_compute_time_us_p50")) or 0.0
    overhead = _numeric_value(metrics.get("cache_path_overhead_us_p50")) or 0.0
    decode = _numeric_value(metrics.get("first_token_decode_us_p50")) or 0.0
    no_cache = _numeric_value(metrics.get("no_cache_ttft_us_p50"))
    gate_us = gate_bytes / (bandwidth * 1_000.0)
    all_us = all_bytes / (bandwidth * 1_000.0)
    dependency = max(gate_us + overhead + replay, all_us + overhead) + suffix + decode
    metrics["dependency_ttft_at_rated_bandwidth_us_p50"] = dependency
    metrics["dependency_ttft_savings_at_rated_bandwidth_us_p50"] = (
        no_cache - dependency if no_cache is not None else None
    )


def _effective_constraints(
    constraints: dict[str, Any],
    storage_thresholds: dict[str, Any],
    analysis_mode: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    use_storage = constraints.get("use_storage_limits", True)
    for key, (metric, storage_key, label) in MAX_CONSTRAINTS.items():
        if key not in MODE_CONSTRAINTS[analysis_mode]:
            continue
        limit = constraints.get(key)
        if limit is None and use_storage:
            limit = storage_thresholds.get(storage_key)
        if limit is None:
            continue
        result[key] = {
            "metric": metric,
            "limit": float(limit),
            "direction": "max",
            "label": label,
        }
    for key, (metric, default, label) in MIN_CONSTRAINTS.items():
        if key not in MODE_CONSTRAINTS[analysis_mode]:
            continue
        limit = constraints.get(key, default)
        if limit is None:
            continue
        result[key] = {
            "metric": metric,
            "limit": float(limit),
            "direction": "min",
            "label": label,
        }
    if not result:
        raise ValueError("enable at least one recommendation constraint")
    return result


def _failure_reason(
    value: float | None,
    spec: dict[str, Any],
) -> str:
    if value is None:
        return "missing_metric"
    if spec["direction"] == "max":
        if not math.isfinite(value) or value < 0:
            return "invalid_metric"
        return "over_limit"
    if math.isnan(value):
        return "invalid_metric"
    return "below_limit"


def _sampled_hit_boundaries(
    scenario_results: list[dict[str, Any]],
    *, required_combinations: set[tuple] | None = None,
) -> list[dict[str, Any]]:
    required = required_combinations or {(x.get("encoder_state"), x.get("encoder_tail_policy")) for x in scenario_results}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for scenario in scenario_results:
        key = (
            scenario["seq_len"],
            scenario["qps"],
            scenario["arrival_model"],
        )
        groups.setdefault(key, []).append(scenario)
    boundaries = []
    for (seq_len, qps, arrival), scenarios in sorted(groups.items()):
        points: dict[float, dict[tuple, bool]] = {}
        for scenario in scenarios:
            point = points.setdefault(scenario["prefix_ratio"], {})
            key = (scenario.get("encoder_state"), scenario.get("encoder_tail_policy"))
            point[key] = point.get(key, True) and bool(scenario["feasible"])
        ordered = [{"prefix_ratio": ratio, "feasible": set(point) == required and all(point.values())}
                   for ratio, point in sorted(points.items())]
        intervals = []
        current = None
        for scenario in ordered:
            if scenario["feasible"]:
                if current is None:
                    current = {
                        "start": scenario["prefix_ratio"],
                        "end": scenario["prefix_ratio"],
                        "sample_count": 1,
                    }
                else:
                    current["end"] = scenario["prefix_ratio"]
                    current["sample_count"] += 1
            elif current is not None:
                intervals.append(current)
                current = None
        if current is not None:
            intervals.append(current)
        boundaries.append(
            {
                "seq_len": seq_len,
                "qps": qps,
                "arrival_model": arrival,
                "intervals": intervals,
                "non_monotonic": len(intervals) > 1,
            }
        )
    return boundaries


def _ranking_key(candidate: dict[str, Any], objective: str) -> tuple[Any, ...]:
    fully_feasible_rank = 0 if candidate["fully_feasible"] else 1
    coverage = _sortable(candidate.get("coverage_ratio"), negative=True)
    headroom = _sortable(candidate.get("worst_headroom"), negative=True)
    pressure = _sortable(candidate.get("mean_storage_pressure"))
    ttft = _sortable(candidate.get("min_ttft_margin_us"), negative=True)
    stable = candidate["candidate_id"]
    if objective == "max_coverage":
        return (coverage, fully_feasible_rank, headroom, pressure, stable)
    if objective == "min_storage_pressure":
        return (fully_feasible_rank, coverage, pressure, ttft, stable)
    if objective == "max_dependency_ttft_margin":
        return (fully_feasible_rank, coverage, ttft, headroom, stable)
    return (fully_feasible_rank, coverage, headroom, ttft, stable)


def _sortable(value: Any, *, negative: bool = False) -> float:
    number = _numeric_value(value)
    if number is None:
        return math.inf
    return -number if negative else number


def _scenario_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "seq_len": row.get("seq_len"),
        "prefix_ratio": row.get("prefix_ratio"),
        "arrival_model": row.get("arrival_model"),
        "qps": row.get("qps"),
        **{key: (row.get("case") or {}).get(key, row.get(key))
           for key in ENCODER_SCENARIO_VALUES if key in row or key in (row.get("case") or {})},
    }


def _paths(
    source: dict[str, Any],
    key: str,
    allowed: set[str],
) -> list[str]:
    values = _strings(source, key)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unknown {key}: {unknown[0]}")
    return values


def _strings(source: dict[str, Any], key: str) -> list[str]:
    value = source.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty array")
    result = [str(item) for item in value if str(item)]
    if len(result) != len(value):
        raise ValueError(f"{key} contains an empty value")
    return list(dict.fromkeys(result))


def _numbers(
    source: dict[str, Any],
    key: str,
    *,
    integer: bool = False,
    positive: bool = False,
) -> list[int | float]:
    value = source.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty array")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{key} must contain numbers")
        try:
            parsed = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must contain numbers") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{key} contains an invalid value")
        if integer:
            if not parsed.is_integer():
                raise ValueError(f"{key} must contain integers")
            number: int | float = int(parsed)
        else:
            number = parsed
        if positive and number <= 0:
            raise ValueError(f"{key} contains an invalid value")
        result.append(number)
    return list(dict.fromkeys(result))


def _optional_integers(value: Any) -> list[int | None]:
    if not isinstance(value, list) or not value:
        raise ValueError("max_io_bytes must be a non-empty array")
    result: list[int | None] = []
    for item in value:
        if item is None or item == "":
            number = None
        else:
            if isinstance(item, bool):
                raise ValueError("max_io_bytes must contain integers or null")
            try:
                parsed = float(item)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("max_io_bytes must contain integers or null") from exc
            if not math.isfinite(parsed) or not parsed.is_integer():
                raise ValueError("max_io_bytes must contain integers or null")
            number = int(parsed)
            if number <= 0:
                raise ValueError("max_io_bytes must be positive")
        if number not in result:
            result.append(number)
    return result


def _optional_number(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric or null")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric or null") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value in {"Infinity", "+Infinity"}:
            return math.inf
        if value == "-Infinity":
            return -math.inf
        if value in {"", "NaN"}:
            return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if math.isnan(number) else number


def _json_number(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _risk_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    resource_metrics = (
        "read_avg_GBps",
        "write_avg_GBps",
        "aggregate_dependency_required_GBps",
        "aggregate_dependency_chunked_iops",
        "retain_all_capacity_GiB",
    )
    negative_ttft = 0
    zero_ttft = 0
    infinite = 0
    missing = 0
    for row in rows:
        metrics = row.get("metrics") or {}
        ttft = _numeric_value(
            metrics.get("dependency_ttft_savings_at_rated_bandwidth_us_p50")
        )
        negative_ttft += int(ttft is not None and ttft < 0)
        zero_ttft += int(ttft == 0)
        values = [_numeric_value(metrics.get(key)) for key in resource_metrics]
        infinite += int(
            metrics.get("dependency_aware_break_even_status")
            in {"no_compute_saving", "no_replay_slack", "mixed"}
        )
        missing += int(any(value is None for value in values))
    return {
        "negative_ttft_case_count": negative_ttft,
        "zero_ttft_case_count": zero_ttft,
        "infinite_resource_case_count": infinite,
        "missing_resource_case_count": missing,
    }


def _candidate_id(design: dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_key(design).encode("utf-8")).hexdigest()[:12]
    return f"candidate-{digest}"


def _stable_key(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _case_identity(case: dict[str, Any]) -> str:
    identity = {
        key: value
        for key, value in case.items()
        if key not in {"id", "case_identity"}
    }
    return hashlib.sha256(_stable_key(identity).encode("utf-8")).hexdigest()[:16]
