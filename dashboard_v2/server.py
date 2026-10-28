#!/usr/bin/env python3
"""Local dashboard server for the prefix-cache demand workbench."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from itertools import product
from dataclasses import asdict, is_dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import uuid


DASHBOARD_ROOT = Path(__file__).resolve().parent
ROOT = DASHBOARD_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssu_sim.cache import CacheReplayConfig, cache_replay_config_from_dict
from ssu_sim.demand import (
    DemandAnalysis,
    analyze_bandwidth_envelope,
    analyze_workload,
)
from ssu_sim.models.llm_model import ModelProfile
from ssu_sim.models.attention import PrefixCachePolicy
from ssu_sim.models.catalog import (
    active_catalog,
    discover_model_catalog,
    legacy_catalog,
)
from ssu_sim.models.model_registry import model_from_dict, ATTENTION_PROFILE_FACTORIES
from ssu_sim.models.state import (
    ENCODER_SCENARIO_VALUES, normalize_encoder_scenarios,
    SparseAccessPolicy,
    StatePersistencePolicy,
    checkpoint_count,
)
from ssu_sim.storage import simulate_ssd, ssd_profile_from_dict
from ssu_sim.utils.config import load_config
from ssu_sim.workload.synthetic import SyntheticWorkloadConfig, generate_synthetic_workload

from dashboard_v2.experiment_store import (
    ExperimentNotFound,
    ExperimentStore,
    ExperimentValidationError,
    fingerprint_payload,
)
from dashboard_v2.parallel_sweep import (
    initialize_worker as _initialize_parallel_sweep_worker,
    run_case as _run_parallel_sweep_worker_case,
)
from dashboard_v2.recommendation import (
    MAX_RECOMMENDATION_COMPUTE_CASES,
    RecommendationCanceled,
    normalize_recommendation_payload,
    recommendation_case_count,
    recommendation_definitions,
    run_recommendation,
    validate_recommendation_params,
)


DEFAULT_RUNTIME_CONFIG = "configs/runtime/bf16_reference.yaml"
DEFAULT_STORAGE_CONFIG = "configs/storage/single_nvme_ftl.yaml"
DEFAULT_CACHE_CONFIG = "configs/cache/radix_leaf_lru.yaml"
MAX_SWEEP_CASES = 1_200
SYNC_SWEEP_CASES = 100
SYNC_PARALLEL_CACHE_BLOCK_VISITS = 2_000_000
MAX_REQUEST_BODY_BYTES = 2_000_000
MAX_INTERACTIVE_STORAGE_LOGICAL_OPS = 200_000
MAX_INTERACTIVE_STORAGE_PHYSICAL_IOS = 2_000_000
MAX_TERMINAL_SWEEP_JOBS = 8
SWEEP_JOB_TTL_SECONDS = 60 * 60
MAX_PARALLEL_SWEEP_WORKERS = 8
PARALLEL_SWEEP_INFLIGHT_MULTIPLIER = 8
PARALLEL_SWEEP_POLL_SECONDS = 0.05

FOOTPRINT_VIEWS = frozenset(
    {
        "calculator-reference",
        "active-execution",
        "prefix-store",
        "replay-resident",
    }
)
DRAFT_EXECUTION_MODES = frozenset({"disabled", "prefill"})
SPARSE_RESTORE_MODES = frozenset({"full_restore", "sparse_on_demand"})
SPARSE_LOCALITY_SCENARIOS = frozenset(
    {"clustered_lower", "window_global", "seeded_uniform", "dispersed_upper"}
)
SPARSE_REUSE_SCOPES = frozenset({"request_union", "per_query_no_reuse"})
SPARSE_STORAGE_LAYOUTS = frozenset(
    {"all_kv_heads_together", "per_gqa_group"}
)
ANALYSIS_MODES = frozenset({"bandwidth_envelope", "cache_replay"})
RECOVERY_ASSUMPTIONS = frozenset(
    {"model_default", "assume_exact_at_resident", "periodic", "none"}
)

DEFAULT_MODELING_OPTIONS: dict[str, Any] = {
    "footprint_view": "prefix-store",
    "state_policies": {},
    "model_state_policies": {},
    "sparse_access_policy": {
        "restore_mode": "full_restore",
        "locality_scenario": "window_global",
        "reuse_scope": "request_union",
        "storage_layout": "all_kv_heads_together",
        "query_granularity_tokens": 1,
        "seed": 0,
    },
    "include_draft_cache": False,
    "draft_execution": "disabled",
    "decode_state_writes": False,
    "decode_write_step_us": None,
    "recovery_assumption": "model_default",
    "checkpoint_interval_tokens": 2048,
    "retain_prompt_end": False,
}

TERMINAL_SWEEP_JOB_STATES = frozenset({"completed", "failed", "canceled"})


class ApiError(Exception):
    """User-facing API error with an HTTP status code."""

    def __init__(
        self,
        status: HTTPStatus,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details or {}


class SweepCanceled(Exception):
    """Internal cooperative-cancellation signal for a background sweep."""


class SweepJob:
    """In-memory state for one background compute task."""

    def __init__(
        self,
        *,
        id: str,
        payload: dict[str, Any],
        total_cases: int,
        kind: str = "sweep",
    ) -> None:
        self.id = id
        self.payload = payload
        self.request_fingerprint = fingerprint_payload(payload)
        self.total_cases = total_cases
        self.kind = kind
        self.status = "queued"
        self.completed_cases = 0
        self.cancel_requested = False
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.cancel_event = threading.Event()

    def public_payload(self, *, reused: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.id,
            "job_kind": self.kind,
            "status": self.status,
            "completed_cases": self.completed_cases,
            "total_cases": self.total_cases,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "request": self.payload,
            "request_fingerprint": self.request_fingerprint,
        }
        if reused:
            payload["reused"] = True
        if self.finished_at is not None:
            payload["finished_at"] = self.finished_at
        if self.error:
            payload["error"] = self.error
        if self.status == "completed" and self.result is not None:
            payload["result"] = self.result
        return payload


_SWEEP_JOBS: dict[str, SweepJob] = {}
_SWEEP_JOBS_LOCK = threading.Lock()
_ACTIVE_SWEEP_JOB_ID: str | None = None
_EXPERIMENT_STORE: ExperimentStore | None = None
_EXPERIMENT_STORE_LOCK = threading.Lock()


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve dashboard assets and JSON API endpoints."""

    server_version = "PCSDashboardV3/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/configs":
                self._send_json(build_config_payload())
                return
            job_id = _sweep_job_id_from_path(path)
            if job_id is not None:
                self._send_json(get_sweep_job(job_id))
                return
            job_id = _recommendation_job_id_from_path(path)
            if job_id is not None:
                self._send_json(get_recommendation_job(job_id))
                return
            if path == "/api/experiments":
                self._send_json(list_experiments(parse_qs(parsed.query)))
                return
            experiment_id = _experiment_id_from_path(path)
            if experiment_id is not None:
                self._send_json(get_experiment(experiment_id))
                return
            self._serve_static(path)
        except ApiError as exc:
            self._send_error_json(exc.status, exc.message, exc.details)
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback
        try:
            path = urlparse(self.path).path
            if path == "/api/sweep":
                self._send_json(run_synchronous_sweep(self._read_json_body()))
                return
            if path == "/api/sweep/jobs":
                self._send_json(
                    create_sweep_job(self._read_json_body()),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if path == "/api/recommendations/jobs":
                self._send_json(
                    create_recommendation_job(self._read_json_body()),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if path == "/api/experiments/compare":
                body = self._read_json_body()
                self._send_json(
                    compare_experiments(
                        str(body.get("baseline_id") or ""),
                        str(body.get("candidate_id") or ""),
                    )
                )
                return
            if path == "/api/case":
                self._send_json(run_case_endpoint(self._read_json_body()))
                return
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown endpoint: {path}")
        except ApiError as exc:
            self._send_error_json(exc.status, exc.message, exc.details)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib callback
        try:
            path = urlparse(self.path).path
            experiment_id = _experiment_id_from_path(path)
            if experiment_id is None:
                raise ApiError(HTTPStatus.NOT_FOUND, f"unknown endpoint: {path}")
            self._send_json(update_experiment(experiment_id, self._read_json_body()))
        except ApiError as exc:
            self._send_error_json(exc.status, exc.message, exc.details)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib callback
        try:
            path = urlparse(self.path).path
            job_id = _sweep_job_id_from_path(path)
            if job_id is not None:
                self._send_json(cancel_sweep_job(job_id), status=HTTPStatus.ACCEPTED)
                return
            job_id = _recommendation_job_id_from_path(path)
            if job_id is not None:
                self._send_json(
                    cancel_recommendation_job(job_id),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            experiment_id = _experiment_id_from_path(path)
            if experiment_id is not None:
                delete_experiment(experiment_id)
                self._send_json({"deleted": True, "id": experiment_id})
                return
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown endpoint: {path}")
        except ApiError as exc:
            self._send_error_json(exc.status, exc.message, exc.details)
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the local server quiet unless there is a real error."""

        if args and isinstance(args[1] if len(args) > 1 else None, str):
            status = args[1]
            if status.startswith("4") or status.startswith("5"):
                super().log_message(format, *args)

    def _read_json_body(self) -> dict[str, Any]:
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if length > MAX_REQUEST_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        return payload

    def _serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            path = "/index.html"
        relative = unquote(path).lstrip("/")
        if not relative:
            relative = "index.html"
        file_path = (DASHBOARD_ROOT / relative).resolve()
        if not file_path.is_relative_to(DASHBOARD_ROOT) or not file_path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "file not found")
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _content_type(file_path))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_json(
        self,
        status: HTTPStatus,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._send_json({"error": message, **(details or {})}, status=status)


def build_config_payload() -> dict[str, Any]:
    raw_runtime_options = _config_options("runtime")
    model_config_options = _config_options("models")
    legacy_model_config_options = _config_options("models", include_legacy=True)
    compatible_model_options = model_config_options + legacy_model_config_options
    default_runtime = _first_existing(raw_runtime_options, DEFAULT_RUNTIME_CONFIG)
    runtime_data = load_config(_resolve_config_path(default_runtime))

    def load_model_options(
        options: list[dict[str, Any]],
        *,
        include_kv_calculator: bool = False,
    ) -> list[dict[str, Any]]:
        loaded = []
        for option in options:
            try:
                model_data = load_config(_resolve_config_path(option["path"]))
                model = model_from_dict(model_data, runtime_data)
                option = {
                    **option,
                    "model_name": model.display_name,
                    "model_id": model.model_id,
                    "attention": model.attention.name,
                    "cache_bytes_per_token": model.cache_bytes_per_token(),
                    "constraints": _model_constraints(model),
                    "recovery_capability": _model_recovery_capability(model),
                }
                if include_kv_calculator:
                    reference = model_data.get("calculator_reference")
                    if not isinstance(reference, dict) or not isinstance(
                        reference.get("fields"), dict
                    ):
                        raise ValueError(
                            f"model {model.model_id} lacks calculator_reference fields"
                        )
                    option["kv_calculator"] = {
                        "formula_id": str(reference["formula_id"]),
                        "semantic_class": str(reference["semantic_class"]),
                        "default_tokens": int(reference["default_tokens"]),
                        "max_context_tokens": int(model.max_context_tokens),
                        "draft_support": str(model_data.get("draft_support", "none")),
                        "fields": reference["fields"],
                    }
            except Exception as exc:
                option = {**option, "error": str(exc)}
            loaded.append(option)
        return loaded

    model_options = load_model_options(
        model_config_options,
        include_kv_calculator=True,
    )
    legacy_model_options = load_model_options(legacy_model_config_options)

    runtime_options = []
    for runtime_option in raw_runtime_options:
        compatibility: dict[str, dict[str, Any]] = {}
        try:
            selected_runtime = load_config(_resolve_config_path(runtime_option["path"]))
            # Hidden legacy profiles remain valid API inputs for saved-experiment
            # replay even though the V3 chooser only renders the active catalog.
            for model_option in compatible_model_options:
                try:
                    model = model_from_dict(
                        load_config(_resolve_config_path(model_option["path"])),
                        selected_runtime,
                    )
                    compatibility[model_option["path"]] = {
                        "compatible": True,
                        "model_name": model.display_name,
                        "model_id": model.model_id,
                        "attention": model.attention.name,
                        "cache_bytes_per_token": model.cache_bytes_per_token(),
                        "constraints": _model_constraints(model),
                        "recovery_capability": _model_recovery_capability(model),
                    }
                except Exception as exc:
                    compatibility[model_option["path"]] = {
                        "compatible": False,
                        "error": str(exc),
                    }
            runtime_options.append(
                {**runtime_option, "model_compatibility": compatibility,
                 "compute_config": _runtime_compute_config(selected_runtime)}
            )
        except Exception as exc:
            runtime_options.append(
                {
                    **runtime_option,
                    "error": str(exc),
                    "model_compatibility": compatibility,
                }
            )

    workload_options = []
    for option in _config_options("workloads"):
        data = load_config(_resolve_config_path(option["path"]))
        workload_options.append(
            {
                **option,
                "request_count": data.get("request_count"),
                "qps": data.get("qps"),
                "seq_len": data.get("seq_len"),
                "prefix_ratio": data.get("prefix_ratio"),
                "workload_overrides": _default_workload_overrides(option["path"]),
                "preset_defaults": _workload_preset_defaults(data),
            }
        )

    storage_options = []
    for option in _config_options("storage"):
        try:
            profile = ssd_profile_from_dict(
                load_config(_resolve_config_path(option["path"]))
            )
            option = {
                **option,
                "thresholds": {
                    "sequential_read_GBps": profile.sequential_read_bandwidth_GBps,
                    "sequential_write_GBps": profile.sequential_write_bandwidth_GBps,
                    "random_read_iops": profile.random_read_iops,
                    "random_write_iops": profile.random_write_iops,
                    "logical_capacity_GiB": _bytes_to_gib(
                        profile.logical_capacity_bytes
                    ),
                },
            }
        except Exception as exc:
            option = {**option, "error": str(exc)}
        storage_options.append(option)

    cache_options = []
    for option in _config_options("cache"):
        try:
            config = cache_replay_config_from_dict(
                load_config(_resolve_config_path(option["path"]))
            )
            option = {
                **option,
                "name": config.name,
                "policy": config.policy,
                "total_capacity_bytes": config.total_capacity_bytes,
                "shard_count": config.shard_count,
                "placement_seed": config.placement_seed,
                "warmup_fraction": config.warmup_fraction,
            }
        except Exception as exc:
            option = {**option, "error": str(exc)}
        cache_options.append(option)
    default_workload = _first_existing(workload_options, "configs/workloads/synthetic_chat.yaml")
    default_storage = _first_existing(storage_options, DEFAULT_STORAGE_CONFIG)
    default_cache = _first_existing(cache_options, DEFAULT_CACHE_CONFIG)
    default_cache_data = cache_replay_config_from_dict(
        load_config(_resolve_config_path(default_cache))
    )
    default_models = _curated_model_paths(
        [option for option in model_options if "error" not in option]
    )
    runtime_analysis = _runtime_analysis_defaults(runtime_data)
    modeling_defaults = _runtime_modeling_defaults(runtime_data)
    modeling_defaults["recovery_assumption"] = "model_default"
    modeling_defaults.setdefault("checkpoint_interval_tokens", 2048)
    modeling_defaults.setdefault("retain_prompt_end", False)

    payload = {
        "models": model_options,
        "legacy_models": legacy_model_options,
        "runtimes": runtime_options,
        "workloads": workload_options,
        "storages": storage_options,
        "caches": cache_options,
        "defaults": {
            "analysis_mode": "bandwidth_envelope",
            "model_configs": default_models,
            "recommendation_model_configs": default_models,
            "recommendation_runtime_configs": [default_runtime],
            "runtime_config": default_runtime,
            "compute_config": _runtime_compute_config(runtime_data),
            "workload_config": default_workload,
            "storage_config": default_storage,
            "cache_replay_enabled": False,
            "cache_config": default_cache,
            "cache_total_capacity_bytes": [default_cache_data.total_capacity_bytes],
            "cache_shard_count": default_cache_data.shard_count,
            "cache_placement_seed": default_cache_data.placement_seed,
            "cache_warmup_fraction": default_cache_data.warmup_fraction,
            "workload_shape": "shared_prefix_groups",
            "seq_len": [65536, 131072, 262144],
            "prefix_ratio": [
                0.5,
                0.5729580301,
                0.6352703119,
                0.688490231,
                0.7339445092,
                0.7727662782,
                0.8059233276,
                0.834242231,
                0.8584289516,
                0.8790864412,
                0.8967296714,
                0.9117984708,
                0.9246684905,
                0.9356605676,
                0.9450487241,
                0.9530669977,
                0.9599152765,
                0.9657642814,
                0.9707598226,
                0.9750264341,
                0.9786704784,
                0.9817827982,
                0.9844409805,
                0.9867112913,
                0.9886503273,
                0.9903064269,
                0.9917208749,
                0.9929289322,
                0.9939607145,
                0.9948419433,
                0.9955945866,
                0.9962374072,
                0.9967864299,
                0.9972553414,
                0.9976558312,
                0.997997883,
                0.9982900241,
                0.998539537,
                0.998752642,
                0.9989346516,
                0.999090103,
                0.9992228716,
                0.9993362671,
                0.9994331164,
                0.9995158338,
                0.9995864814,
                0.9996468204,
                0.999698355,
                0.9997423699,
                0.9997799622,
                0.9998120693,
                0.9998394914,
                0.9998629122,
                0.9998829155,
                0.9999,
            ],
            "block_size_tokens": [
                int(runtime_analysis.get("block_size_tokens", 64) or 64)
            ],
            "arrival_model": ["fixed"],
            "max_io_bytes": [
                runtime_analysis.get("max_io_bytes")
                if runtime_analysis.get("max_io_bytes") is not None
                else 131072
            ],
            "write_miss_blocks": bool(runtime_analysis.get("write_miss_blocks", True)),
            "modeling_options": modeling_defaults,
            "workload_overrides": _default_workload_overrides(default_workload),
        },
        "limits": {
            "max_sweep_cases": MAX_SWEEP_CASES,
            "sync_sweep_cases": SYNC_SWEEP_CASES,
            "max_recommendation_compute_cases": (
                MAX_RECOMMENDATION_COMPUTE_CASES
            ),
        },
        "recommendation": recommendation_definitions(),
    }

    full_tail = {
        "label": "Full Tail",
        "axes": {
            key: list(payload["defaults"][key])
            for key in (
                "seq_len",
                "prefix_ratio",
                "block_size_tokens",
                "arrival_model",
                "max_io_bytes",
            )
        },
    }
    # The original tail used the runtime's legacy 64-token block. V2 compares all
    # models, including DeepSeek V4, whose exposed constraint requires a 128-token
    # multiple. Keep the original tail/seq axes while using the native 256 block.
    full_tail["axes"]["block_size_tokens"] = [256]
    max_full_tail_context = max(full_tail["axes"]["seq_len"])
    long_context_models = [
        option
        for option in model_options
        if "error" not in option
        and (option.get("constraints", {}).get("max_context_tokens") or 0)
        >= max_full_tail_context
    ]
    full_tail["model_configs"] = _curated_model_paths(long_context_models)
    if not full_tail["model_configs"]:
        full_tail["model_configs"] = list(default_models)
    quick_compare = {
        "label": "Quick Compare",
        "model_configs": list(default_models),
        "axes": {
            "seq_len": [32768],
            "prefix_ratio": [0.7, 0.95, 0.99, 0.999],
            "block_size_tokens": [256],
            "arrival_model": ["fixed"],
            "max_io_bytes": [131072],
        },
    }
    payload["presets"] = {
        "quick_compare": quick_compare,
        "full_tail": full_tail,
    }
    payload["defaults"].update(quick_compare["axes"])
    payload["defaults"]["preset"] = "quick_compare"
    return payload


def run_sweep(payload: dict[str, Any], *, progress_callback=None, cancel_event=None) -> dict[str, Any]:
    params = _sweep_params(payload)
    descriptors = _parallel_sweep_case_descriptors(params)
    runtime_config = _load_effective_runtime(params)
    models = {path: _load_model(path, runtime_config) for path in params["model_configs"]}
    rows = []
    for descriptor in descriptors:
        if cancel_event is not None and cancel_event.is_set():
            raise SweepCanceled()
        case = descriptor["case"]
        model = models[descriptor["model_config"]]
        analysis, workload, miss_window = _run_demand_case(
            model=model, runtime_config=runtime_config, workload_config=params["workload_config"],
            case=case, include_trace=False)
        rows.append(_row_from_analysis(case, model, workload, analysis, miss_window))
        if progress_callback is not None:
            progress_callback(len(rows), len(descriptors))
        if cancel_event is not None and cancel_event.is_set():
            raise SweepCanceled()
    return {"sweep_result_schema_version": 3, "analysis_mode": params["analysis_mode"],
            "case_count": len(rows), "rows": rows, "summary": _summary_from_rows(rows),
            "warnings": _compute_warnings_from_rows(rows), "request": params}


def _should_run_parallel_sweep(job: SweepJob) -> bool:
    """Use processes only for large, background, cache-enabled sweeps."""

    return (
        job.total_cases > SYNC_SWEEP_CASES
        and isinstance(job.payload.get("cache_replay_config"), dict)
    )


def _estimated_cache_block_visits(params: dict[str, Any]) -> int | None:
    """Estimate exact-path work for choosing a synchronous execution backend.

    This estimate is deliberately best-effort.  It must never introduce a new
    validation or failure path: when the workload does not expose a reliable
    integer request count, the caller simply retains the serial implementation.
    """

    try:
        workload_data = load_config(
            _resolve_config_path(str(params["workload_config"]))
        )
        workload_data.update(params.get("workload_overrides") or {})
        request_count = workload_data.get("request_count")
        if isinstance(request_count, bool) or not isinstance(request_count, int):
            return None
        if request_count < 0:
            return None
        return sum(
            request_count
            * (
                (
                    int(descriptor["case"]["seq_len"])
                    + int(descriptor["case"]["block_size_tokens"])
                    - 1
                )
                // int(descriptor["case"]["block_size_tokens"])
            )
            for descriptor in _parallel_sweep_case_descriptors(params)
        )
    except Exception:
        return None


def _should_run_parallel_synchronous_sweep(
    params: dict[str, Any], total_cases: int
) -> bool:
    """Use the existing ordered process pool only when spawn is safely amortized."""

    if not (
        isinstance(params.get("cache_replay_config"), dict)
        and 2 <= total_cases <= SYNC_SWEEP_CASES
    ):
        return False
    estimated_visits = _estimated_cache_block_visits(params)
    return (
        estimated_visits is not None
        and estimated_visits >= SYNC_PARALLEL_CACHE_BLOCK_VISITS
    )


def _parallel_sweep_worker_count(total_cases: int) -> int:
    available_after_parent = max(1, (os.cpu_count() or 1) - 1)
    return min(MAX_PARALLEL_SWEEP_WORKERS, available_after_parent, total_cases)


def _model_encoder_scenarios(model_config: str, axes: dict[str, Any]) -> list[dict[str, str]]:
    raw = load_config(_resolve_config_path(model_config))
    factory = ATTENTION_PROFILE_FACTORIES.get(raw.get("attention", {}).get("type"))
    if not callable(getattr(factory, "recovery_plan", None)):
        return [{}]
    values = normalize_encoder_scenarios(axes)
    return [{"encoder_state": state, "encoder_tail_policy": tail}
            for state in values["encoder_state"] for tail in values["encoder_tail_policy"]]


def _parallel_sweep_case_descriptors(params: dict[str, Any]) -> list[dict[str, Any]]:
    """One expansion used by serial execution, processes and work estimates."""
    descriptors = []
    axes = params["axes"]
    names = ("seq_len", "prefix_ratio", "block_size_tokens", "arrival_model", "max_io_bytes")
    for model_config in params["model_configs"]:
        scenarios = _model_encoder_scenarios(model_config, axes)
        for values in product(*(axes[name] for name in names), axes.get("cache_total_capacity_bytes", [None])):
            for scenario in scenarios:
                case = {"id": f"case-{len(descriptors) + 1:04d}",
                    "analysis_mode": params["analysis_mode"], "model_config": model_config,
                    **{name: params[name] for name in ("runtime_config", "workload_config", "storage_config",
                        "workload_overrides", "write_miss_blocks", "modeling_options", "config_digests")},
                    **dict(zip(names, values[:5])), **scenario}
                if values[5] is not None:
                    case["cache_replay_config"] = {**params["cache_replay_config"], "total_capacity_bytes": values[5]}
                if "compute_config" in params:
                    case["compute_config"] = params["compute_config"]
                case["case_identity"] = _case_identity(case)
                descriptors.append({"ordinal": len(descriptors), "model_config": model_config, "case": case})
    return descriptors


def _create_parallel_sweep_executor(
    *,
    worker_count: int,
    runtime_config: dict[str, Any],
    models: dict[str, ModelProfile],
    worker_cancel_event: Any,
    mp_context: Any,
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_initialize_parallel_sweep_worker,
        initargs=(runtime_config, models, worker_cancel_event),
    )


def _serial_sweep_suffix(
    *,
    params: dict[str, Any],
    descriptors: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    first_ordinal: int,
    runtime_config: dict[str, Any],
    models: dict[str, ModelProfile],
    progress_callback: Any | None,
    cancel_event: threading.Event | None,
) -> dict[str, Any]:
    """Finish an uncommitted suffix after process infrastructure failure."""

    total_cases = len(descriptors)
    for descriptor in descriptors[first_ordinal:]:
        if cancel_event is not None and cancel_event.is_set():
            raise SweepCanceled()
        case = descriptor["case"]
        model = models[descriptor["model_config"]]
        analysis, workload, miss_window = _run_demand_case(
            model=model,
            runtime_config=runtime_config,
            workload_config=params["workload_config"],
            case=case,
            include_trace=False,
        )
        rows.append(
            _row_from_analysis(
                case,
                model,
                workload,
                analysis,
                miss_window,
            )
        )
        completed = descriptor["ordinal"] + 1
        if progress_callback is not None:
            progress_callback(completed, total_cases)
        if cancel_event is not None and cancel_event.is_set():
            raise SweepCanceled()

    return {
        "sweep_result_schema_version": 3,
        "analysis_mode": params["analysis_mode"],
        "case_count": len(rows),
        "rows": rows,
        "summary": _summary_from_rows(rows),
        "warnings": _compute_warnings_from_rows(rows),
        "request": params,
    }


def _run_parallel_sweep(
    payload: dict[str, Any],
    *,
    progress_callback: Any | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run independent cache cases in spawn workers and commit them in order."""

    params = _sweep_params(payload)
    descriptors = _parallel_sweep_case_descriptors(params)
    total_cases = len(descriptors)
    if cancel_event is not None and cancel_event.is_set():
        raise SweepCanceled()

    # Setup happens before any observable progress. A setup failure can therefore
    # fall back to the untouched serial oracle without resetting anything.
    try:
        runtime_config = _load_effective_runtime(params)
        models = {
            model_config: _load_model(model_config, runtime_config)
            for model_config in params["model_configs"]
        }
        worker_count = _parallel_sweep_worker_count(total_cases)
        mp_context = multiprocessing.get_context("spawn")
        worker_cancel_event = mp_context.Event()
        executor = _create_parallel_sweep_executor(
            worker_count=worker_count,
            runtime_config=runtime_config,
            models=models,
            worker_cancel_event=worker_cancel_event,
            mp_context=mp_context,
        )
    except SweepCanceled:
        raise
    except Exception:
        return run_sweep(
            payload,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    rows: list[dict[str, Any]] = []
    futures: dict[Any, int] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    next_submit = 0
    next_commit = 0
    max_uncommitted = max(
        1, worker_count * PARALLEL_SWEEP_INFLIGHT_MULTIPLIER
    )
    infrastructure_failed = False
    canceled = False
    worker_error: Exception | None = None
    parent_error: Exception | None = None

    def submit_window() -> None:
        nonlocal next_submit
        while (
            next_submit < total_cases
            and next_submit - next_commit < max_uncommitted
        ):
            future = executor.submit(
                _run_parallel_sweep_worker_case,
                descriptors[next_submit],
            )
            futures[future] = next_submit
            next_submit += 1

    try:
        try:
            submit_window()
        except Exception:
            infrastructure_failed = True

        while (
            next_commit < total_cases
            and not infrastructure_failed
            and worker_error is None
            and parent_error is None
            and not canceled
        ):
            if cancel_event is not None and cancel_event.is_set():
                canceled = True
                break
            if not futures:
                infrastructure_failed = True
                break

            try:
                done, _ = wait(
                    tuple(futures),
                    timeout=PARALLEL_SWEEP_POLL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
            except Exception:
                infrastructure_failed = True
                break
            if not done:
                continue
            for future in done:
                ordinal = futures.pop(future)
                try:
                    outcome = future.result()
                except BrokenProcessPool:
                    infrastructure_failed = True
                    break
                except Exception:
                    # The worker entry point converts ordinary Case failures into
                    # data. An exception escaping the Future therefore means the
                    # worker result could not be transported safely.
                    infrastructure_failed = True
                    break
                if not isinstance(outcome, dict) or "kind" not in outcome:
                    infrastructure_failed = True
                    break
                outcomes[ordinal] = outcome
            if infrastructure_failed:
                break

            while next_commit in outcomes:
                outcome = outcomes.pop(next_commit)
                kind = outcome["kind"]
                if kind == "row":
                    rows.append(outcome["row"])
                    next_commit += 1
                    if progress_callback is not None:
                        try:
                            progress_callback(next_commit, total_cases)
                        except Exception as exc:
                            parent_error = exc
                            break
                    if cancel_event is not None and cancel_event.is_set():
                        canceled = True
                        break
                    continue
                if kind == "api_error":
                    worker_error = ApiError(
                        HTTPStatus(outcome["status"]),
                        outcome["message"],
                        outcome["details"],
                    )
                elif kind == "exception" and isinstance(
                    outcome.get("exception"), Exception
                ):
                    worker_error = outcome["exception"]
                elif kind == "canceled":
                    canceled = True
                else:
                    infrastructure_failed = True
                break

            if (
                not infrastructure_failed
                and worker_error is None
                and parent_error is None
                and not canceled
            ):
                try:
                    submit_window()
                except Exception:
                    infrastructure_failed = True
    finally:
        if (
            infrastructure_failed
            or worker_error is not None
            or parent_error is not None
            or canceled
        ):
            try:
                worker_cancel_event.set()
            except Exception:
                pass
            for future in futures:
                try:
                    future.cancel()
                except Exception:
                    pass
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            infrastructure_failed = True

    if canceled:
        raise SweepCanceled()
    if parent_error is not None:
        raise parent_error
    if worker_error is not None:
        raise worker_error
    if infrastructure_failed:
        if next_commit == 0:
            return run_sweep(
                payload,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        return _serial_sweep_suffix(
            params=params,
            descriptors=descriptors,
            rows=rows,
            first_ordinal=next_commit,
            runtime_config=runtime_config,
            models=models,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    return {
        "sweep_result_schema_version": 3,
        "analysis_mode": params["analysis_mode"],
        "case_count": len(rows),
        "rows": rows,
        "summary": _summary_from_rows(rows),
        "warnings": _compute_warnings_from_rows(rows),
        "request": params,
    }


def run_synchronous_sweep(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a small sweep while participating in the shared compute lock."""

    global _ACTIVE_SWEEP_JOB_ID
    params = _sweep_params(payload)
    total_cases = _case_count(params)
    if total_cases > SYNC_SWEEP_CASES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            (
                f"synchronous sweep has {total_cases} cases; use /api/sweep/jobs "
                f"for sweeps above {SYNC_SWEEP_CASES} cases"
            ),
        )
    job = SweepJob(
        id=str(uuid.uuid4()),
        payload=payload,
        total_cases=total_cases,
        kind="sweep",
    )
    job.status = "running"
    with _SWEEP_JOBS_LOCK:
        _cleanup_sweep_jobs_locked()
        if _ACTIVE_SWEEP_JOB_ID:
            active = _SWEEP_JOBS.get(_ACTIVE_SWEEP_JOB_ID)
            if active and active.status not in TERMINAL_SWEEP_JOB_STATES:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "another compute job is already running",
                    {
                        "active_job_id": active.id,
                        "active_job_kind": active.kind,
                    },
                )
            _ACTIVE_SWEEP_JOB_ID = None
        _SWEEP_JOBS[job.id] = job
        _ACTIVE_SWEEP_JOB_ID = job.id
    try:
        sweep_runner = (
            _run_parallel_sweep
            if _should_run_parallel_synchronous_sweep(params, total_cases)
            else run_sweep
        )
        result = _save_successful_result(sweep_runner(payload), "sweep")
        with _SWEEP_JOBS_LOCK:
            job.status = "completed"
            job.completed_cases = job.total_cases
            job.result = result
            job.finished_at = time.time()
        return result
    except Exception:
        with _SWEEP_JOBS_LOCK:
            job.status = "failed"
            job.finished_at = time.time()
        raise
    finally:
        with _SWEEP_JOBS_LOCK:
            if _ACTIVE_SWEEP_JOB_ID == job.id:
                _ACTIVE_SWEEP_JOB_ID = None
            _cleanup_sweep_jobs_locked()


def create_sweep_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and start one background sweep."""

    global _ACTIVE_SWEEP_JOB_ID
    params = _sweep_params(payload)
    total_cases = _case_count(params)
    request_fingerprint = fingerprint_payload(params)
    with _SWEEP_JOBS_LOCK:
        _cleanup_sweep_jobs_locked()
        if _ACTIVE_SWEEP_JOB_ID:
            active = _SWEEP_JOBS.get(_ACTIVE_SWEEP_JOB_ID)
            if active and active.status not in TERMINAL_SWEEP_JOB_STATES:
                if (
                    active.kind == "sweep"
                    and active.request_fingerprint == request_fingerprint
                ):
                    return active.public_payload(reused=True)
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "another compute job is already running",
                    {
                        "active_job_id": active.id,
                        "active_job_kind": active.kind,
                        "active_request_fingerprint": (
                            active.request_fingerprint
                        ),
                    },
                )
            _ACTIVE_SWEEP_JOB_ID = None
        job = SweepJob(
            id=str(uuid.uuid4()),
            payload=params,
            total_cases=total_cases,
            kind="sweep",
        )
        _SWEEP_JOBS[job.id] = job
        _ACTIVE_SWEEP_JOB_ID = job.id
    threading.Thread(
        target=_run_sweep_job,
        args=(job.id,),
        name=f"dashboard-v2-sweep-{job.id[:8]}",
        daemon=True,
    ).start()
    return job.public_payload()


def get_sweep_job(job_id: str) -> dict[str, Any]:
    with _SWEEP_JOBS_LOCK:
        _cleanup_sweep_jobs_locked()
        job = _SWEEP_JOBS.get(job_id)
        if job is None or job.kind != "sweep":
            raise ApiError(HTTPStatus.NOT_FOUND, "sweep job not found")
        return job.public_payload()


def cancel_sweep_job(job_id: str) -> dict[str, Any]:
    with _SWEEP_JOBS_LOCK:
        _cleanup_sweep_jobs_locked()
        job = _SWEEP_JOBS.get(job_id)
        if job is None or job.kind != "sweep":
            raise ApiError(HTTPStatus.NOT_FOUND, "sweep job not found")
        if job.status in TERMINAL_SWEEP_JOB_STATES:
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"cannot cancel a {job.status} sweep job",
            )
        job.cancel_requested = True
        job.cancel_event.set()
        return job.public_payload()


def _run_sweep_job(job_id: str) -> None:
    global _ACTIVE_SWEEP_JOB_ID
    with _SWEEP_JOBS_LOCK:
        job = _SWEEP_JOBS.get(job_id)
        if job is None:
            return
        if job.cancel_event.is_set():
            job.status = "canceled"
            job.finished_at = time.time()
            if _ACTIVE_SWEEP_JOB_ID == job_id:
                _ACTIVE_SWEEP_JOB_ID = None
            return
        job.status = "running"

    def progress(completed: int, total: int) -> None:
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.completed_cases = completed
                current.total_cases = total

    try:
        sweep_runner = (
            _run_parallel_sweep
            if _should_run_parallel_sweep(job)
            else run_sweep
        )
        result = sweep_runner(
            job.payload,
            progress_callback=progress,
            cancel_event=job.cancel_event,
        )
        result = _save_successful_result(result, "sweep")
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.status = "completed"
                current.completed_cases = current.total_cases
                current.result = result
                current.finished_at = time.time()
    except SweepCanceled:
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.status = "canceled"
                current.finished_at = time.time()
    except Exception as exc:  # pragma: no cover - defensive thread boundary
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.status = "failed"
                current.error = str(exc)
                current.finished_at = time.time()
    finally:
        with _SWEEP_JOBS_LOCK:
            if _ACTIVE_SWEEP_JOB_ID == job_id:
                _ACTIVE_SWEEP_JOB_ID = None
            _cleanup_sweep_jobs_locked()


def create_recommendation_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and start a background candidate-grid recommendation."""

    global _ACTIVE_SWEEP_JOB_ID
    config_payload = build_config_payload()
    try:
        params = normalize_recommendation_payload(payload, config_payload)
        params["modeling_options"] = _normalize_modeling_options(
            params,
            defaults=(config_payload.get("defaults") or {}).get(
                "modeling_options",
                DEFAULT_MODELING_OPTIONS,
            ),
        )
        for runtime_path in params["design_space"]["runtime_configs"]:
            _validate_modeling_state_policies(
                params["design_space"]["model_configs"],
                load_config(_resolve_config_path(runtime_path)),
                params["modeling_options"],
            )
        params["config_digests"] = {
            "models": {
                path: _file_sha256(_resolve_config_path(path))
                for path in params["design_space"]["model_configs"]
            },
            "runtimes": {
                path: _file_sha256(_resolve_config_path(path))
                for path in params["design_space"]["runtime_configs"]
            },
            "storages": {
                path: _file_sha256(_resolve_config_path(path))
                for path in params["design_space"]["storage_configs"]
            },
            "workload": {
                "path": params["workload_envelope"]["workload_config"],
                "sha256": _file_sha256(
                    _resolve_config_path(
                        params["workload_envelope"]["workload_config"]
                    )
                ),
            },
        }
        validate_recommendation_params(params, config_payload)
        total_cases = recommendation_case_count(params, config_payload)
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    cleaned_overrides = _clean_workload_overrides(params["workload_overrides"])
    params["workload_overrides"] = cleaned_overrides
    for qps in params["workload_envelope"]["qps"]:
        _validate_workload_inputs(
            params["workload_envelope"]["workload_config"],
            {**cleaned_overrides, "qps": qps},
        )
    if total_cases <= 0:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "no compatible design candidates remain",
        )
    if total_cases > MAX_RECOMMENDATION_COMPUTE_CASES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            (
                f"recommendation has {total_cases} compute cases; reduce the grid "
                f"to at most {MAX_RECOMMENDATION_COMPUTE_CASES}"
            ),
        )
    request_fingerprint = fingerprint_payload(params)
    with _SWEEP_JOBS_LOCK:
        _cleanup_sweep_jobs_locked()
        if _ACTIVE_SWEEP_JOB_ID:
            active = _SWEEP_JOBS.get(_ACTIVE_SWEEP_JOB_ID)
            if active and active.status not in TERMINAL_SWEEP_JOB_STATES:
                if (
                    active.kind == "recommendation"
                    and active.request_fingerprint == request_fingerprint
                ):
                    return active.public_payload(reused=True)
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "another compute job is already running",
                    {
                        "active_job_id": active.id,
                        "active_job_kind": active.kind,
                        "active_request_fingerprint": (
                            active.request_fingerprint
                        ),
                    },
                )
            _ACTIVE_SWEEP_JOB_ID = None
        job = SweepJob(
            id=str(uuid.uuid4()),
            payload=params,
            total_cases=total_cases,
            kind="recommendation",
        )
        _SWEEP_JOBS[job.id] = job
        _ACTIVE_SWEEP_JOB_ID = job.id
    threading.Thread(
        target=_run_recommendation_job,
        args=(job.id, config_payload),
        name=f"dashboard-v3-recommendation-{job.id[:8]}",
        daemon=True,
    ).start()
    return job.public_payload()


def get_recommendation_job(job_id: str) -> dict[str, Any]:
    with _SWEEP_JOBS_LOCK:
        _cleanup_sweep_jobs_locked()
        job = _SWEEP_JOBS.get(job_id)
        if job is None or job.kind != "recommendation":
            raise ApiError(HTTPStatus.NOT_FOUND, "recommendation job not found")
        return job.public_payload()


def cancel_recommendation_job(job_id: str) -> dict[str, Any]:
    with _SWEEP_JOBS_LOCK:
        _cleanup_sweep_jobs_locked()
        job = _SWEEP_JOBS.get(job_id)
        if job is None or job.kind != "recommendation":
            raise ApiError(HTTPStatus.NOT_FOUND, "recommendation job not found")
        if job.status in TERMINAL_SWEEP_JOB_STATES:
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"cannot cancel a {job.status} recommendation job",
            )
        job.cancel_requested = True
        job.cancel_event.set()
        return job.public_payload()


def _run_recommendation_job(
    job_id: str,
    config_payload: dict[str, Any] | None = None,
) -> None:
    global _ACTIVE_SWEEP_JOB_ID
    with _SWEEP_JOBS_LOCK:
        job = _SWEEP_JOBS.get(job_id)
        if job is None:
            return
        if job.cancel_event.is_set():
            job.status = "canceled"
            job.finished_at = time.time()
            if _ACTIVE_SWEEP_JOB_ID == job_id:
                _ACTIVE_SWEEP_JOB_ID = None
            return
        job.status = "running"

    def progress(completed: int, total: int) -> None:
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.completed_cases = completed
                current.total_cases = total

    try:
        result = run_recommendation(
            job.payload,
            config_payload if config_payload is not None else build_config_payload(),
            case_runner=_recommendation_case_runner(),
            progress_callback=progress,
            cancel_requested=job.cancel_event.is_set,
        )
        result = _save_successful_result(result, "recommendation")
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.status = "completed"
                current.completed_cases = current.total_cases
                current.result = result
                current.finished_at = time.time()
    except RecommendationCanceled:
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.status = "canceled"
                current.finished_at = time.time()
    except Exception as exc:  # pragma: no cover - defensive thread boundary
        with _SWEEP_JOBS_LOCK:
            current = _SWEEP_JOBS.get(job_id)
            if current is not None:
                current.status = "failed"
                current.error = str(exc)
                current.finished_at = time.time()
    finally:
        with _SWEEP_JOBS_LOCK:
            if _ACTIVE_SWEEP_JOB_ID == job_id:
                _ACTIVE_SWEEP_JOB_ID = None
            _cleanup_sweep_jobs_locked()


def _recommendation_case_runner() -> Any:
    runtime_cache: dict[str, dict[str, Any]] = {}
    model_cache: dict[tuple[str, str], ModelProfile] = {}

    def run(case: dict[str, Any]) -> dict[str, Any]:
        runtime_path = str(case["runtime_config"])
        runtime_config = runtime_cache.get(runtime_path)
        if runtime_config is None:
            runtime_config = load_config(_resolve_config_path(runtime_path))
            runtime_cache[runtime_path] = runtime_config
        model_path = str(case["model_config"])
        model_key = (runtime_path, model_path)
        model = model_cache.get(model_key)
        if model is None:
            model = _load_model(model_path, runtime_config)
            model_cache[model_key] = model
        analysis, workload, miss_window = _run_demand_case(
            model=model,
            runtime_config=runtime_config,
            workload_config=str(case["workload_config"]),
            case=case,
            include_trace=False,
        )
        return _row_from_analysis(case, model, workload, analysis, miss_window)

    return run


def _cleanup_sweep_jobs_locked() -> None:
    now = time.time()
    terminal = [
        job
        for job in _SWEEP_JOBS.values()
        if job.status in TERMINAL_SWEEP_JOB_STATES
    ]
    for job in terminal:
        if job.finished_at is not None and now - job.finished_at > SWEEP_JOB_TTL_SECONDS:
            _SWEEP_JOBS.pop(job.id, None)
    terminal = sorted(
        (
            job
            for job in _SWEEP_JOBS.values()
            if job.status in TERMINAL_SWEEP_JOB_STATES
        ),
        key=lambda item: item.finished_at or item.created_at,
        reverse=True,
    )
    for job in terminal[MAX_TERMINAL_SWEEP_JOBS:]:
        _SWEEP_JOBS.pop(job.id, None)


def _sweep_job_id_from_path(path: str) -> str | None:
    prefix = "/api/sweep/jobs/"
    if not path.startswith(prefix):
        return None
    job_id = unquote(path[len(prefix):]).strip("/")
    if not job_id or "/" in job_id:
        raise ApiError(HTTPStatus.NOT_FOUND, "sweep job not found")
    return job_id


def _recommendation_job_id_from_path(path: str) -> str | None:
    prefix = "/api/recommendations/jobs/"
    if not path.startswith(prefix):
        return None
    job_id = unquote(path[len(prefix):]).strip("/")
    if not job_id or "/" in job_id:
        raise ApiError(HTTPStatus.NOT_FOUND, "recommendation job not found")
    return job_id


def _experiment_id_from_path(path: str) -> str | None:
    prefix = "/api/experiments/"
    if not path.startswith(prefix):
        return None
    experiment_id = unquote(path[len(prefix):]).strip("/")
    if not experiment_id or "/" in experiment_id:
        raise ApiError(HTTPStatus.NOT_FOUND, "experiment not found")
    return experiment_id


def configure_experiment_store(path: str | Path | None = None) -> None:
    """Reset the lazy store; intended for isolated tests and embedding."""

    global _EXPERIMENT_STORE
    with _EXPERIMENT_STORE_LOCK:
        _EXPERIMENT_STORE = ExperimentStore(path)


def _get_experiment_store() -> ExperimentStore:
    global _EXPERIMENT_STORE
    with _EXPERIMENT_STORE_LOCK:
        if _EXPERIMENT_STORE is None:
            _EXPERIMENT_STORE = ExperimentStore()
        return _EXPERIMENT_STORE


def _save_successful_result(
    result: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    response = dict(result)
    try:
        metadata = _get_experiment_store().save(
            kind=kind,
            request=dict(response.get("request") or {}),
            result=response,
        )
        response["experiment_id"] = metadata["id"]
    except Exception as exc:  # Persistence must never discard a computed result.
        response["history_error"] = f"Result was not saved to history: {exc}"
    return response


def list_experiments(query: dict[str, list[str]]) -> dict[str, Any]:
    kind = (query.get("kind") or [None])[0] or None
    search = (query.get("q") or query.get("search") or [None])[0] or None
    pinned_text = (query.get("pinned") or [None])[0]
    pinned: bool | None = None
    if pinned_text is not None:
        normalized = pinned_text.strip().lower()
        if normalized not in {"true", "false", "1", "0"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "pinned must be true or false")
        pinned = normalized in {"true", "1"}
    try:
        experiments = _get_experiment_store().list(
            kind=kind,
            pinned=pinned,
            search=search,
        )
    except ExperimentValidationError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    return {"experiments": experiments}


def get_experiment(experiment_id: str) -> dict[str, Any]:
    try:
        return _get_experiment_store().get(experiment_id)
    except ExperimentNotFound as exc:
        raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc


def update_experiment(
    experiment_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(payload) - {"name", "pinned"})
    if unknown:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"unsupported experiment field: {unknown[0]}",
        )
    try:
        return _get_experiment_store().update(
            experiment_id,
            name=payload.get("name") if "name" in payload else None,
            pinned=payload.get("pinned") if "pinned" in payload else None,
        )
    except ExperimentNotFound as exc:
        raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
    except ExperimentValidationError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc


def delete_experiment(experiment_id: str) -> None:
    try:
        _get_experiment_store().delete(experiment_id)
    except ExperimentNotFound as exc:
        raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc


def compare_experiments(
    baseline_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    if not baseline_id or not candidate_id:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "baseline_id and candidate_id are required",
        )
    try:
        return _get_experiment_store().compare(baseline_id, candidate_id)
    except ExperimentNotFound as exc:
        raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc


def run_case_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    case = payload.get("case")
    case = (
        _validated_case(case)
        if isinstance(case, dict)
        else _first_case_from_payload(payload)
    )
    include_storage = bool(payload.get("include_storage", False))
    if include_storage and case.get("analysis_mode") != "cache_replay":
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "storage queue simulation is available only for cache_replay cases",
        )
    cache_config = case.get("cache_replay_config")
    if (
        include_storage
        and isinstance(cache_config, dict)
        and int(cache_config.get("shard_count", 1)) != 1
    ):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            (
                "Interactive SSD simulation requires cache shard_count=1; "
                "a multi-shard cache cannot be collapsed into one SSD trace."
            ),
        )
    runtime_config = _load_effective_runtime(case)
    model = _load_model(str(case["model_config"]), runtime_config)
    analysis, workload, miss_window = _run_demand_case(
        model=model,
        runtime_config=runtime_config,
        workload_config=str(case["workload_config"]),
        case=case,
        include_trace=False,
    )
    demand_report = dict(analysis.report)
    demand_report["runtime"] = runtime_config
    demand_report["workload"] = {
        key: value
        for key, value in asdict(workload).items()
        if key != "token_hit_rate"
    }
    storage_estimate = _storage_work_estimate(analysis.report)

    response: dict[str, Any] = {
        "case": case,
        "row": _row_from_analysis(case, model, workload, analysis, miss_window),
        "demand_report": demand_report,
        "storage_estimate": storage_estimate,
    }
    if analysis.report.get("evaluation", {}).get("status") in {"not_evaluated", "unsupported"}:
        response["storage_estimate"] = None
        if include_storage:
            response["storage_report"] = {"evaluation": analysis.report["evaluation"]}
        return response
    if include_storage:
        _validate_interactive_storage_work(storage_estimate)
        trace_analysis, _, _ = _run_demand_case(
            model=model,
            runtime_config=runtime_config,
            workload_config=str(case["workload_config"]),
            case=case,
            include_trace=True,
        )
        storage_path = str(case.get("storage_config") or DEFAULT_STORAGE_CONFIG)
        storage_profile = ssd_profile_from_dict(load_config(_resolve_config_path(storage_path)))
        result = simulate_ssd(
            trace_analysis.ops,
            storage_profile,
            max_io_bytes=_optional_int(case.get("max_io_bytes")),
            ttft_jobs=trace_analysis.ttft_jobs,
        )
        response["storage_report"] = result.report
    return response


def _sweep_params(payload: dict[str, Any]) -> dict[str, Any]:
    analysis_mode = str(
        payload.get("analysis_mode")
        or (
            "cache_replay"
            if isinstance(payload.get("cache_replay_config"), dict)
            else "bandwidth_envelope"
        )
    )
    if analysis_mode not in ANALYSIS_MODES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "analysis_mode must be bandwidth_envelope or cache_replay",
        )
    runtime_config = str(payload.get("runtime_config") or DEFAULT_RUNTIME_CONFIG)
    workload_config = str(payload.get("workload_config") or "configs/workloads/synthetic_chat.yaml")
    storage_config = str(payload.get("storage_config") or DEFAULT_STORAGE_CONFIG)
    _resolve_config_path(runtime_config)
    _resolve_config_path(workload_config)
    _resolve_config_path(storage_config)

    model_configs = payload.get("model_configs")
    if not isinstance(model_configs, list) or not model_configs:
        model_configs = _curated_model_paths(_config_options("models"))
    model_configs = sorted({str(path) for path in model_configs})
    for model_config in model_configs:
        _resolve_config_path(model_config)

    runtime_data = load_config(_resolve_config_path(runtime_config))
    compute_config = None
    if "compute_config" in payload:
        compute_config = _parse_compute_config(payload["compute_config"], runtime_data)
        runtime_data = _runtime_with_compute(runtime_data, compute_config)
    runtime_analysis = _runtime_analysis_defaults(runtime_data)
    modeling_options = _normalize_modeling_options(
        payload,
        defaults=_runtime_modeling_defaults(runtime_data),
    )
    axes_payload = payload.get("axes") if isinstance(payload.get("axes"), dict) else {}
    axes = {
        "seq_len": _int_axis(axes_payload, "seq_len", [8192]),
        "prefix_ratio": _float_axis(axes_payload, "prefix_ratio", [0.8]),
        "block_size_tokens": _int_axis(
            axes_payload,
            "block_size_tokens",
            [int(runtime_analysis.get("block_size_tokens", 64) or 64)],
        ),
        "arrival_model": _arrival_axis(axes_payload),
        "max_io_bytes": _optional_int_axis(
            axes_payload,
            "max_io_bytes",
            [runtime_analysis.get("max_io_bytes")],
        ),
    }
    if any(_model_encoder_scenarios(path, {}) != [{}] for path in model_configs):
        try:
            axes.update(normalize_encoder_scenarios(axes_payload))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    cache_replay_config = _cache_replay_params(payload, axes_payload)
    if analysis_mode == "cache_replay" and cache_replay_config is None:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "cache_replay analysis requires cache_replay_config",
        )
    if analysis_mode == "bandwidth_envelope" and cache_replay_config is not None:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "bandwidth_envelope does not accept cache_replay_config",
        )
    if cache_replay_config is not None:
        axes["cache_total_capacity_bytes"] = cache_replay_config.pop(
            "capacity_axis"
        )
    params_for_count = {"model_configs": model_configs, "axes": axes}
    case_count = _case_count(params_for_count)
    if case_count <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "sweep must contain at least one case")
    if case_count > MAX_SWEEP_CASES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"sweep has {case_count} cases; reduce axes to at most {MAX_SWEEP_CASES}",
        )
    _validate_sweep_models(model_configs, runtime_data, axes)
    _validate_modeling_state_policies(
        model_configs,
        runtime_data,
        modeling_options,
    )

    overrides = payload.get("workload_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    write_miss_default = bool(runtime_analysis.get("write_miss_blocks", True))
    cleaned_overrides = _clean_workload_overrides(overrides)
    _validate_workload_inputs(workload_config, cleaned_overrides)
    config_digests = _request_config_digests(
        model_configs=model_configs,
        runtime_config=runtime_config,
        workload_config=workload_config,
        storage_config=storage_config,
        cache_config=(
            str(cache_replay_config["source_path"])
            if cache_replay_config is not None
            else None
        ),
    )
    params = {
        "analysis_mode": analysis_mode,
        "model_configs": model_configs,
        "runtime_config": runtime_config,
        "workload_config": workload_config,
        "storage_config": storage_config,
        "axes": axes,
        "workload_overrides": cleaned_overrides,
        "write_miss_blocks": bool(payload.get("write_miss_blocks", write_miss_default)),
        "modeling_options": modeling_options,
        "config_digests": config_digests,
    }
    if compute_config is not None:
        params["compute_config"] = compute_config
    if cache_replay_config is not None:
        params["cache_replay_config"] = cache_replay_config
    return params


def _normalize_modeling_options(
    payload: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one authoritative set of non-axis modeling assumptions.

    These values deliberately do not multiply the sweep case count. A sweep
    represents one explicitly selected checkpoint/sparse/decode scenario; a
    caller that wants a sensitivity grid submits separate sweeps. Keeping the
    assumptions in each normalized request still makes worker descriptors,
    experiment fingerprints, exports, and replay identities complete.
    """

    values: dict[str, Any] = {
        **DEFAULT_MODELING_OPTIONS,
        **(defaults or {}),
    }
    raw = payload.get("modeling_options")
    if raw is not None:
        if not isinstance(raw, dict):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "modeling_options must be a mapping",
            )
        values.update(raw)
    # Accept the documented CLI-style names at the API boundary for one
    # compatibility cycle. The normalized response always nests them.
    for key in DEFAULT_MODELING_OPTIONS:
        if key in payload:
            values[key] = payload[key]

    footprint_view = str(values.get("footprint_view") or "prefix-store")
    footprint_view = footprint_view.replace("_", "-")
    if footprint_view not in FOOTPRINT_VIEWS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "footprint_view must be calculator-reference, active-execution, "
            "prefix-store, or replay-resident",
        )

    raw_state_policies = values.get("state_policies") or {}
    if not isinstance(raw_state_policies, dict):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "state_policies must map component names to persistence policies",
        )
    state_policies: dict[str, dict[str, Any]] = {}
    for component, policy in sorted(raw_state_policies.items()):
        name = str(component).strip()
        if not name:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "state_policies contains an empty component name",
            )
        try:
            state_policies[name] = asdict(_state_policy_from_value(policy))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"invalid state policy for {name}: {exc}",
            ) from exc

    raw_model_state_policies = values.get("model_state_policies") or {}
    if not isinstance(raw_model_state_policies, dict):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "model_state_policies must map model paths to state policies",
        )
    model_state_policies: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_model_path, raw_policies in sorted(raw_model_state_policies.items()):
        model_path = str(raw_model_path).strip()
        if not model_path or not isinstance(raw_policies, dict):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "model_state_policies entries must map a model path to policies",
            )
        normalized_policies: dict[str, dict[str, Any]] = {}
        for component, policy in sorted(raw_policies.items()):
            name = str(component).strip()
            if not name:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"model_state_policies[{model_path}] contains an empty target",
                )
            try:
                normalized_policies[name] = asdict(_state_policy_from_value(policy))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"invalid state policy for {model_path}:{name}: {exc}",
                ) from exc
        model_state_policies[model_path] = normalized_policies

    raw_sparse = values.get("sparse_access_policy") or {}
    if not isinstance(raw_sparse, dict):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "sparse_access_policy must be a mapping",
        )
    locality = str(raw_sparse.get("locality_scenario", "window_global"))
    locality = {
        "clustered": "clustered_lower",
        "dispersed": "dispersed_upper",
    }.get(locality, locality)
    try:
        sparse_values = {
            "restore_mode": str(raw_sparse.get("restore_mode", "full_restore")),
            "locality_scenario": locality,
            "reuse_scope": str(raw_sparse.get("reuse_scope", "request_union")),
            "storage_layout": str(
                raw_sparse.get("storage_layout", "all_kv_heads_together")
            ),
            "query_granularity_tokens": _strict_positive_int(
                raw_sparse.get("query_granularity_tokens", 1),
                "sparse_access_policy.query_granularity_tokens",
            ),
            "seed": _strict_int(
                raw_sparse.get("seed", 0),
                "sparse_access_policy.seed",
            ),
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    if sparse_values["restore_mode"] not in SPARSE_RESTORE_MODES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "sparse restore_mode must be full_restore or sparse_on_demand",
        )
    if sparse_values["locality_scenario"] not in SPARSE_LOCALITY_SCENARIOS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "sparse locality_scenario must be clustered_lower, window_global, "
            "seeded_uniform, or dispersed_upper",
        )
    if sparse_values["reuse_scope"] not in SPARSE_REUSE_SCOPES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "sparse reuse_scope must be request_union or per_query_no_reuse",
        )
    if sparse_values["storage_layout"] not in SPARSE_STORAGE_LAYOUTS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "sparse storage_layout must be all_kv_heads_together or per_gqa_group",
        )

    draft_execution = str(values.get("draft_execution") or "disabled")
    if draft_execution not in DRAFT_EXECUTION_MODES:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "draft_execution must be disabled or prefill",
        )
    try:
        decode_write_step_us = _optional_positive_number(
            values.get("decode_write_step_us"),
            "decode_write_step_us",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    decode_state_writes = bool(values.get("decode_state_writes", False))
    if decode_write_step_us is not None:
        decode_state_writes = True
    recovery_assumption = str(
        values.get("recovery_assumption") or "model_default"
    )
    if recovery_assumption not in RECOVERY_ASSUMPTIONS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "recovery_assumption must be model_default, "
            "assume_exact_at_resident, periodic, or none",
        )
    checkpoint_interval_tokens = _strict_positive_int(
        values.get("checkpoint_interval_tokens", 2048),
        "checkpoint_interval_tokens",
    )

    return {
        "footprint_view": footprint_view,
        "state_policies": state_policies,
        "model_state_policies": model_state_policies,
        "sparse_access_policy": sparse_values,
        "include_draft_cache": bool(values.get("include_draft_cache", False)),
        "draft_execution": draft_execution,
        "decode_state_writes": decode_state_writes,
        "decode_write_step_us": decode_write_step_us,
        "recovery_assumption": recovery_assumption,
        "checkpoint_interval_tokens": checkpoint_interval_tokens,
        "retain_prompt_end": bool(values.get("retain_prompt_end", False)),
    }


def _state_policy_from_value(value: Any) -> StatePersistencePolicy:
    if isinstance(value, StatePersistencePolicy):
        return value
    if isinstance(value, dict):
        return StatePersistencePolicy(
            storage_form=str(value.get("storage_form", "snapshot")),
            interval_tokens=(
                _strict_positive_int(
                    value["interval_tokens"],
                    "interval_tokens",
                )
                if value.get("interval_tokens") is not None
                else None
            ),
            retain_prompt_end=bool(value.get("retain_prompt_end", True)),
        )
    text = str(value).strip().lower()
    if text in {"prompt_end", "terminal", "terminal_only"}:
        return StatePersistencePolicy.terminal()
    if text == "token_history":
        return StatePersistencePolicy.token_history()
    if text in {"none", "zero"}:
        return StatePersistencePolicy.none()
    if text.startswith("periodic+prompt_end:"):
        return StatePersistencePolicy.periodic(
            _strict_positive_int(text.split(":", 1)[1], "interval_tokens"),
            retain_prompt_end=True,
        )
    if text.startswith("periodic:"):
        return StatePersistencePolicy.periodic(
            _strict_positive_int(text.split(":", 1)[1], "interval_tokens"),
            retain_prompt_end=False,
        )
    raise ValueError(
        "expected prompt_end, periodic:N, periodic+prompt_end:N, "
        "token_history, none, or a structured policy"
    )


def _case_count(params: dict[str, Any]) -> int:
    base = math.prod(len(values) for name, values in params["axes"].items()
                     if name not in ENCODER_SCENARIO_VALUES)
    return base * sum(len(_model_encoder_scenarios(path, params["axes"])) for path in params["model_configs"])


def _case_identity(case: dict[str, Any]) -> str:
    identity = {
        key: value
        for key, value in case.items()
        if key not in {"id", "case_identity"}
    }
    return fingerprint_payload(identity)[:16]


def _cache_replay_params(
    payload: dict[str, Any],
    axes_payload: dict[str, Any],
) -> dict[str, Any] | None:
    raw = payload.get("cache_replay_config")
    if raw is None:
        if "cache_total_capacity_bytes" in axes_payload:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "cache_total_capacity_bytes requires cache_replay_config",
            )
        return None
    if not isinstance(raw, dict):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "cache_replay_config must be a mapping",
        )

    allowed = {
        "source_path",
        "name",
        "policy",
        "total_capacity_bytes",
        "shard_count",
        "placement_seed",
        "warmup_fraction",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"unsupported cache replay field: {unknown[0]}",
        )

    source_path = str(raw.get("source_path") or DEFAULT_CACHE_CONFIG)
    source = _resolve_config_path(source_path)
    cache_root = (ROOT / "configs" / "cache").resolve()
    if not source.is_relative_to(cache_root):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "cache replay source_path must stay under configs/cache/",
        )
    base = load_config(source)
    values = {
        **base,
        **{key: value for key, value in raw.items() if key != "source_path"},
    }
    capacity_default = values.get("total_capacity_bytes", 0)
    capacity_axis = _nonnegative_int_axis(
        axes_payload,
        "cache_total_capacity_bytes",
        [capacity_default],
    )
    try:
        validated = cache_replay_config_from_dict(
            {**values, "total_capacity_bytes": capacity_axis[0]}
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    return {
        "source_path": source_path,
        "name": validated.name,
        "policy": validated.policy,
        "shard_count": validated.shard_count,
        "placement_seed": validated.placement_seed,
        "warmup_fraction": validated.warmup_fraction,
        "capacity_axis": capacity_axis,
    }


def _validate_sweep_models(
    model_configs: list[str],
    runtime_data: dict[str, Any],
    axes: dict[str, list[Any]],
) -> None:
    runtime_name = str(runtime_data.get("name") or "selected runtime")
    for model_config in model_configs:
        try:
            model = _load_model(model_config, runtime_data)
        except Exception as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model_config} is incompatible with runtime {runtime_name}: {exc}",
            ) from exc
        constraints = _model_constraints(model)
        max_context = constraints.get("max_context_tokens")
        if max_context and any(value > max_context for value in axes["seq_len"]):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model.name}: seq_len exceeds max_context_tokens={max_context}",
            )
        block_minimum = int(constraints.get("minimum_block_size_tokens") or 1)
        if any(value < block_minimum for value in axes["block_size_tokens"]):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model.name}: block size must be at least {block_minimum}",
            )
        block_multiple = int(constraints.get("block_size_multiple") or 1)
        if block_multiple > 1 and any(
            value % block_multiple != 0 for value in axes["block_size_tokens"]
        ):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model.name}: block size must be a multiple of {block_multiple}",
            )


def _first_case_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params = _sweep_params(payload)
    case = {
        "id": "case-0001",
        "analysis_mode": params["analysis_mode"],
        "model_config": params["model_configs"][0],
        "runtime_config": params["runtime_config"],
        "workload_config": params["workload_config"],
        "storage_config": params["storage_config"],
        "seq_len": params["axes"]["seq_len"][0],
        "prefix_ratio": params["axes"]["prefix_ratio"][0],
        "block_size_tokens": params["axes"]["block_size_tokens"][0],
        "arrival_model": params["axes"]["arrival_model"][0],
        "max_io_bytes": params["axes"]["max_io_bytes"][0],
        "workload_overrides": params["workload_overrides"],
        "write_miss_blocks": params["write_miss_blocks"],
        "modeling_options": params["modeling_options"],
        "config_digests": params["config_digests"],
    }
    if "compute_config" in params:
        case["compute_config"] = params["compute_config"]
    if "cache_replay_config" in params:
        case["cache_replay_config"] = {
            **params["cache_replay_config"],
            "total_capacity_bytes": params["axes"][
                "cache_total_capacity_bytes"
            ][0],
        }
    case.update(_model_encoder_scenarios(case["model_config"], params["axes"])[0])
    case["case_identity"] = _case_identity(case)
    return case


def _validated_case(case: dict[str, Any]) -> dict[str, Any]:
    axes: dict[str, list[Any]] = {}
    for key in (
        "seq_len",
        "prefix_ratio",
        "block_size_tokens",
        "arrival_model",
        "max_io_bytes",
        "encoder_state",
        "encoder_tail_policy",
    ):
        if key in case:
            axes[key] = [case[key]]
    payload: dict[str, Any] = {
        "analysis_mode": case.get("analysis_mode", "bandwidth_envelope"),
        "model_configs": [case.get("model_config")],
        "runtime_config": case.get("runtime_config"),
        "workload_config": case.get("workload_config"),
        "storage_config": case.get("storage_config"),
        "axes": axes,
        "workload_overrides": case.get("workload_overrides", {}),
        "modeling_options": case.get("modeling_options", {}),
    }
    if "write_miss_blocks" in case:
        payload["write_miss_blocks"] = case["write_miss_blocks"]
    if "compute_config" in case:
        payload["compute_config"] = case["compute_config"]
    raw_cache = case.get("cache_replay_config")
    if raw_cache is not None:
        if not isinstance(raw_cache, dict):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "cache_replay_config must be a mapping",
            )
        payload["cache_replay_config"] = {
            key: value
            for key, value in raw_cache.items()
            if key != "total_capacity_bytes"
        }
        axes["cache_total_capacity_bytes"] = [
            raw_cache.get("total_capacity_bytes", 0)
        ]
    params = _sweep_params(payload)
    validated = {
        "id": str(case.get("id") or "case-0001"),
        "analysis_mode": params["analysis_mode"],
        "model_config": params["model_configs"][0],
        "runtime_config": params["runtime_config"],
        "workload_config": params["workload_config"],
        "storage_config": params["storage_config"],
        "seq_len": params["axes"]["seq_len"][0],
        "prefix_ratio": params["axes"]["prefix_ratio"][0],
        "block_size_tokens": params["axes"]["block_size_tokens"][0],
        "arrival_model": params["axes"]["arrival_model"][0],
        "max_io_bytes": params["axes"]["max_io_bytes"][0],
        "workload_overrides": params["workload_overrides"],
        "write_miss_blocks": params["write_miss_blocks"],
        "modeling_options": params["modeling_options"],
        "config_digests": params["config_digests"],
    }
    if "compute_config" in params:
        validated["compute_config"] = params["compute_config"]
    if "cache_replay_config" in params:
        validated["cache_replay_config"] = {
            **params["cache_replay_config"],
            "total_capacity_bytes": params["axes"][
                "cache_total_capacity_bytes"
            ][0],
        }
    validated.update(_model_encoder_scenarios(validated["model_config"], params["axes"])[0])
    validated["case_identity"] = _case_identity(validated)
    return validated


def _run_demand_case(
    *,
    model: ModelProfile,
    runtime_config: dict[str, Any],
    workload_config: str,
    case: dict[str, Any],
    include_trace: bool,
) -> tuple[DemandAnalysis, SyntheticWorkloadConfig, dict[str, Any]]:
    runtime_analysis = _runtime_analysis_defaults(runtime_config)
    modeling_options = _normalize_modeling_options(
        {"modeling_options": case.get("modeling_options", {})},
        defaults=_runtime_modeling_defaults(runtime_config),
    )
    model = _model_with_modeling_options(model, modeling_options)
    prefix_cache_policy = _prefix_cache_policy(
        modeling_options,
        model,
        str(case.get("model_config") or ""),
    )
    if callable(getattr(model.attention, "recovery_plan", None)):
        prefix_cache_policy = replace(prefix_cache_policy,
            encoder_state=case.get("encoder_state", "unavailable"),
            encoder_tail_policy=case.get("encoder_tail_policy", "recompute"))
    block_size_tokens = int(case.get("block_size_tokens") or runtime_analysis.get("block_size_tokens", 64))
    max_io_bytes = _optional_int(case.get("max_io_bytes"))
    peak_windows_us = _runtime_peak_windows(runtime_analysis)
    write_miss_blocks = bool(
        case.get("write_miss_blocks", runtime_analysis.get("write_miss_blocks", True))
    )

    workload_data = load_config(_resolve_config_path(workload_config))
    overrides = case.get("workload_overrides")
    if isinstance(overrides, dict):
        workload_data.update(_clean_workload_overrides(overrides))
    workload_data.update(
        {
            "model_id": model.name,
            "seq_len": int(case["seq_len"]),
            "prefix_ratio": float(case["prefix_ratio"]),
            "arrival_model": str(case["arrival_model"]),
        }
    )
    workload = SyntheticWorkloadConfig(**workload_data)
    analysis_mode = str(case.get("analysis_mode") or "bandwidth_envelope")
    raw_cache_config = case.get("cache_replay_config")
    cache_replay_config = None
    if raw_cache_config is not None:
        if not isinstance(raw_cache_config, dict):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "cache_replay_config must be a mapping",
            )
        try:
            cache_replay_config = cache_replay_config_from_dict(raw_cache_config)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    from ssu_sim.models.execution import CaseEvaluationError
    from ssu_sim.demand.analyzer import unevaluated_analysis
    try:
        if analysis_mode == "bandwidth_envelope":
            analysis = analyze_bandwidth_envelope(
                model,
                seq_len=workload.seq_len,
                prefix_ratio=workload.prefix_ratio,
                block_size_tokens=block_size_tokens,
                max_io_bytes=max_io_bytes,
                prefix_cache_policy=prefix_cache_policy,
            )
        else:
            requests = generate_synthetic_workload(workload)
            analysis = analyze_workload(
                requests,
                {model.name: model},
                block_size_tokens=block_size_tokens,
                qps=workload.qps,
                max_io_bytes=max_io_bytes,
                peak_windows_us=peak_windows_us,
                write_miss_blocks=write_miss_blocks,
                include_trace=include_trace,
                prefix_cache_policy=prefix_cache_policy,
                cache_replay_config=cache_replay_config,
                decode_state_writes=bool(modeling_options["decode_state_writes"]),
                decode_write_step_us=modeling_options["decode_write_step_us"],
            )
    except CaseEvaluationError as exc:
        # Expected limitations stop only this case, including in workers.
        # Discard partial replay state; unrelated exceptions still propagate.
        analysis = unevaluated_analysis(exc)
    analysis.report["selected_footprint_view"] = modeling_options["footprint_view"]
    analysis.report["modeling_options"] = modeling_options
    analysis.report["effective_state_policies"] = _effective_state_policy_report(
        model,
        str(case.get("model_config") or ""),
        modeling_options,
        seq_len=workload.seq_len,
        block_size_tokens=block_size_tokens,
    )
    analysis.report["analysis_mode"] = analysis_mode
    return analysis, workload, {}


def _prefix_cache_policy(
    modeling_options: dict[str, Any],
    model: ModelProfile | None = None,
    model_config: str | None = None,
) -> PrefixCachePolicy:
    explicit_state_policies = {
        name: _state_policy_from_value(value)
        for name, value in modeling_options.get("state_policies", {}).items()
    }
    sparse = SparseAccessPolicy(
        **dict(modeling_options.get("sparse_access_policy") or {})
    )
    state_policies: dict[str, StatePersistencePolicy] = {}
    assumption = modeling_options.get("recovery_assumption", "model_default")
    if model is not None and assumption != "model_default":
        factory = getattr(model.attention, "effective_checkpoint_groups", None)
        if not callable(factory):
            factory = getattr(model.attention, "checkpoint_groups", None)
        groups = tuple(factory(model)) if callable(factory) else ()
        for group in groups:
            if assumption == "assume_exact_at_resident":
                state_policies[group.name] = StatePersistencePolicy.terminal()
            elif assumption == "periodic":
                state_policies[group.name] = StatePersistencePolicy.periodic(
                    int(modeling_options.get("checkpoint_interval_tokens", 2048)),
                    retain_prompt_end=bool(
                        modeling_options.get("retain_prompt_end", False)
                    ),
                )
            elif assumption == "none":
                state_policies[group.name] = StatePersistencePolicy.none()
    state_policies.update(explicit_state_policies)
    if model_config is not None:
        state_policies.update(
            {
                name: _state_policy_from_value(value)
                for name, value in (
                    modeling_options.get("model_state_policies", {})
                    .get(model_config, {})
                    .items()
                )
            }
        )
    return PrefixCachePolicy(
        state_policies=state_policies,
        sparse_access_policy=sparse,
        include_draft_execution=(
            modeling_options.get("draft_execution") == "prefill"
        ),
        assume_exact_at_resident=(
            assumption == "assume_exact_at_resident"
        ),
    )


def _effective_state_policy_report(
    model: ModelProfile,
    model_config: str,
    modeling_options: dict[str, Any],
    *,
    seq_len: int,
    block_size_tokens: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    group_overrides = modeling_options.get("state_policies", {})
    model_overrides = modeling_options.get("model_state_policies", {}).get(
        model_config, {}
    )
    assumption = modeling_options.get("recovery_assumption", "model_default")
    for target in _model_state_policy_targets(model):
        name = target["name"]
        accepted_names = (name, *target.get("aliases", ()))
        source = "model_default"
        value = next(
            (model_overrides[key] for key in accepted_names if key in model_overrides),
            None,
        )
        if value is not None:
            source = "model_override"
        else:
            value = next(
                (group_overrides[key] for key in accepted_names if key in group_overrides),
                None,
            )
            if value is not None:
                source = "group_override"
        if value is not None:
            policy = _state_policy_from_value(value)
        elif target.get("global_assumption_applicable") and assumption != "model_default":
            source = "global"
            if assumption == "assume_exact_at_resident":
                policy = StatePersistencePolicy.terminal()
            elif assumption == "periodic":
                policy = StatePersistencePolicy.periodic(
                    int(modeling_options.get("checkpoint_interval_tokens", 2048)),
                    retain_prompt_end=bool(
                        modeling_options.get("retain_prompt_end", False)
                    ),
                )
            else:
                policy = StatePersistencePolicy.none()
        else:
            policy = _state_policy_from_value(target["default_policy"])

        count = 0
        state_bytes = 0
        if policy.storage_form == "snapshot":
            count = checkpoint_count(
                policy,
                prefix_tokens=seq_len,
                block_size_tokens=block_size_tokens,
            )
            state_bytes = count * int(target["checkpoint_bytes"])
        elif policy.storage_form == "token_history":
            state_bytes = seq_len * int(
                target.get("token_history_bytes_per_token") or 0
            )
        result[name] = {
            "policy": asdict(policy),
            "source": source,
            "policy_origin": target.get("policy_origin"),
            "checkpoint_bytes": target["checkpoint_bytes"],
            "checkpoint_count": count,
            "state_bytes": state_bytes,
            "storage_form_conversion": (
                target["default_policy"].get("storage_form")
                != policy.storage_form
            ),
        }
    return result


def _model_with_modeling_options(
    model: ModelProfile,
    modeling_options: dict[str, Any],
) -> ModelProfile:
    """Apply opt-in draft controls while policies remain request-scoped.

    State and sparse choices are intentionally passed via PrefixCachePolicy so
    cached ModelProfile instances remain immutable and safe to share between
    worker cases. Draft capacity/execution are model-shape choices, so only
    profiles that explicitly expose a dataclass field may be replaced.
    """

    attention = model.attention
    include_draft_cache = bool(modeling_options.get("include_draft_cache", False))
    if include_draft_cache:
        if model.draft_support in {"none", "unsupported"}:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model.display_name}: draft cache is {model.draft_support}",
            )
        if not is_dataclass(attention) or not hasattr(attention, "include_draft_cache"):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model.display_name}: attention profile does not expose draft cache",
            )
        attention = replace(attention, include_draft_cache=True)

    compute = model.compute
    draft_execution = str(modeling_options.get("draft_execution", "disabled"))
    if draft_execution == "prefill":
        if model.draft_support in {"none", "unsupported"}:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model.display_name}: draft execution is {model.draft_support}",
            )
        formula = compute.prefill_formula
        if hasattr(formula, "draft_execution_mode"):
            formula = replace(formula, draft_execution_mode="prefill")
        elif hasattr(formula, "include_draft_execution"):
            formula = replace(formula, include_draft_execution=True)
        else:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{model.display_name}: prefill formula does not expose draft execution",
            )
        compute = replace(compute, prefill_formula=formula)

    if attention is model.attention and compute is model.compute:
        return model
    return replace(model, attention=attention, compute=compute)


def _peak_pressure_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Expose reported windows without recomputing demand or filling missing data."""
    metrics: dict[str, Any] = {}
    for direction in ("read", "write"):
        operation = report.get(direction) or {}
        average = _number_or_none(operation.get("offered_avg_GBps"))
        for window in ("1ms", "10ms", "100ms", "1s"):
            peak = (operation.get("peak_windows") or {}).get(window) or {}
            bandwidth = _number_or_none(peak.get("GBps"))
            metrics[f"{direction}_peak_{window}_GBps"] = peak.get("GBps")
            metrics[f"{direction}_peak_{window}_chunked_iops"] = peak.get("chunked_iops")
            metrics[f"{direction}_peak_{window}_to_avg"] = (
                bandwidth / average
                if average is not None and math.isfinite(average) and average > 0
                and bandwidth is not None and math.isfinite(bandwidth)
                else None
            )
    return metrics


def _row_from_analysis(
    case: dict[str, Any],
    model: ModelProfile,
    workload: SyntheticWorkloadConfig,
    analysis: DemandAnalysis,
    miss_window: dict[str, Any],
) -> dict[str, Any]:
    report = analysis.report
    read = report.get("read", {})
    write = report.get("write", {})
    ttft = report.get("ttft", {})
    recovery = report.get("recovery", {})
    capacity = report.get("capacity", {})
    funnel = recovery.get("funnel", {})
    ratios = recovery.get("ratios", {})
    recovery_bytes = recovery.get("bytes", {})
    recovery_compute = recovery.get("compute", {})
    critical = recovery.get("critical_path", {})
    optimistic = critical.get("optimistic_overlap_GBps", {})
    dependency = critical.get("dependency_aware_break_even_GBps", {})
    aggregate = critical.get("aggregate_dependency", {})
    metrics = {
        "read_avg_GBps": _path(read, "offered_avg_GBps"),
        "write_avg_GBps": _path(write, "offered_avg_GBps"),
        "read_logical_block_ops": _path(read, "logical_block_ops"),
        "write_logical_block_ops": _path(write, "logical_block_ops"),
        "read_chunked_ios": _path(read, "chunked_ios"),
        "write_chunked_ios": _path(write, "chunked_ios"),
        "read_peak_1ms_GBps": _path(read, "peak_windows.1ms.GBps"),
        "write_peak_1ms_GBps": _path(write, "peak_windows.1ms.GBps"),
        "read_avg_chunked_iops": _path(read, "offered_avg_chunked_iops"),
        "configured_prefix_ratio": _path(ratios, "configured_prefix.p50"),
        "shareable_prefix_ratio": _path(ratios, "shareable_prefix.p50"),
        "resident_prefix_ratio": _path(ratios, "resident_prefix.p50"),
        "recovery_boundary_ratio": _path(ratios, "recovery_boundary.p50"),
        "configured_prefix_tokens_p50": _path(funnel, "configured_prefix_tokens.p50"),
        "shareable_prefix_tokens_p50": _path(funnel, "shareable_prefix_tokens.p50"),
        "resident_prefix_tokens_p50": _path(funnel, "resident_prefix_tokens.p50"),
        "recovery_boundary_tokens_p50": _path(funnel, "recovery_boundary_tokens.p50"),
        "replay_tokens_p50": _path(funnel, "replay_tokens.p50"),
        "suffix_tokens_p50": _path(funnel, "suffix_tokens.p50"),
        "resident_cache_bytes_p50": _path(recovery_bytes, "resident_cache.p50"),
        "planned_restore_bytes_p50": _path(recovery_bytes, "planned_restore.p50"),
        "replay_gate_bytes_p50": _path(recovery_bytes, "replay_gate.p50"),
        "suffix_gate_bytes_p50": _path(recovery_bytes, "suffix_gate.p50"),
        "blocking_checkpoint_bytes_p50": _path(recovery_bytes, "blocking_checkpoint.p50"),
        "stranded_resident_bytes_p50": _path(recovery_bytes, "stranded_resident.p50"),
        "net_saved_flops_p50": _path(recovery_compute, "net_saved_flops.p50"),
        "net_saved_flops_ratio_p50": _path(recovery_compute, "net_saved_flops_ratio.p50"),
        "optimistic_overlap_GBps_p50": optimistic.get("p50"),
        "optimistic_overlap_GBps_p99": optimistic.get("p99"),
        "optimistic_overlap_GBps_max": optimistic.get("max"),
        "optimistic_overlap_status": optimistic.get("status"),
        "dependency_aware_break_even_GBps_p50": dependency.get("p50"),
        "dependency_aware_break_even_GBps_p99": dependency.get("p99"),
        "dependency_aware_break_even_GBps_max": dependency.get("max"),
        "dependency_aware_break_even_status": dependency.get("status"),
        "aggregate_dependency_required_GBps": aggregate.get("peak_required_GBps"),
        "aggregate_dependency_chunked_iops": aggregate.get("peak_required_chunked_iops"),
        "no_cache_ttft_us_p50": _path(ttft, "no_cache_ttft_us.p50"),
        "optimistic_ttft_us_p50": _path(ttft, "optimistic_overlap_estimate_us.p50"),
        "dependency_aware_ttft_us_p50": _path(ttft, "dependency_aware_estimate_us.p50"),
        "no_overlap_ttft_us_p50": _path(ttft, "no_compute_io_overlap_estimate_us.p50"),
        "no_overlap_ttft_status": ttft.get(
            "no_compute_io_overlap_status"
        ),
        "dependency_ttft_savings_us_p50": _path(ttft, "dependency_ttft_savings_us.p50"),
        "replay_compute_time_us_p50": _path(ttft, "replay_compute_time_us.p50"),
        "suffix_compute_time_us_p50": _path(ttft, "suffix_compute_time_us.p50"),
        "cache_path_overhead_us_p50": _path(ttft, "cache_path_overhead_us.p50"),
        "first_token_decode_us_p50": _path(ttft, "first_token_decode_us.p50"),
        "retain_all_capacity_GiB": _bytes_to_gib(
            _path(capacity, "retain_all_upper_bound_bytes")
        ),
        "reusable_capacity_GiB": _bytes_to_gib(_path(capacity, "reusable_prefix_bytes")),
    }
    metrics.update(_peak_pressure_metrics(report))
    cache_replay = report.get("cache_replay")
    if isinstance(cache_replay, dict):
        cache_capacity = cache_replay.get("capacity", {})
        cache_hits = cache_replay.get("hit_rate", {})
        admission = cache_replay.get("admission", {})
        eviction = cache_replay.get("eviction", {})
        tree = cache_replay.get("tree", {})
        total_capacity_bytes = _path(
            cache_capacity, "total_capacity_bytes", 0
        )
        metrics.update(
            {
                "configured_prefix_ratio": _path(cache_hits, "configured_prefix"),
                "shareable_prefix_ratio": _path(cache_hits, "shareable_prefix"),
                "resident_prefix_ratio": _path(cache_hits, "resident_prefix"),
                "cache_candidate_block_hit_rate": _path(
                    cache_hits, "candidate_block"
                ),
                "cache_actual_block_hit_rate": _path(
                    cache_hits, "actual_block"
                ),
                "cache_candidate_byte_hit_rate": _path(
                    cache_hits, "candidate_byte"
                ),
                "cache_actual_byte_hit_rate": _path(
                    cache_hits, "actual_byte"
                ),
                "cache_total_capacity_bytes": total_capacity_bytes,
                "cache_total_capacity_GiB": _bytes_to_gib(total_capacity_bytes),
                "cache_inserted_blocks": _path(admission, "inserted_blocks", 0),
                "cache_inserted_bytes": _path(admission, "inserted_bytes", 0),
                "cache_deduplicated_insert_blocks": _path(
                    admission, "deduplicated_insert_blocks", 0
                ),
                "cache_deduplicated_insert_bytes": _path(
                    admission, "deduplicated_insert_bytes", 0
                ),
                "cache_rejected_blocks": _path(admission, "rejected_blocks", 0),
                "cache_rejected_bytes": _path(admission, "rejected_bytes", 0),
                "cache_eviction_blocks": _path(eviction, "blocks", 0),
                "cache_eviction_bytes": _path(eviction, "bytes", 0),
                "cache_eviction_GiB": _bytes_to_gib(_path(eviction, "bytes", 0)),
                "cache_self_eviction_blocks": _path(
                    eviction, "self_eviction_blocks", 0
                ),
                "cache_self_eviction_bytes": _path(
                    eviction, "self_eviction_bytes", 0
                ),
                "cache_initial_used_bytes": _path(
                    cache_capacity, "initial_used_bytes", 0
                ),
                "cache_peak_used_bytes": _path(
                    cache_capacity, "peak_used_bytes", 0
                ),
                "cache_peak_evictable_bytes": _path(
                    cache_capacity, "peak_evictable_bytes", 0
                ),
                "cache_peak_protected_bytes": _path(
                    cache_capacity, "peak_protected_bytes", 0
                ),
                "cache_final_used_bytes": _path(
                    cache_capacity, "final_used_bytes", 0
                ),
                "cache_final_evictable_bytes": _path(
                    cache_capacity, "final_evictable_bytes", 0
                ),
                "cache_final_protected_bytes": _path(
                    cache_capacity, "final_protected_bytes", 0
                ),
                "cache_radix_node_count": _path(tree, "node_count", 0),
                "cache_radix_leaf_count": _path(tree, "leaf_count", 0),
                "cache_radix_max_depth": _path(tree, "max_depth", 0),
                "cache_shard_count": len(cache_capacity.get("shards", [])),
                "cache_bootstrap_requests": _path(cache_replay, "requests.bootstrap", 0),
                "cache_warmup_requests": _path(cache_replay, "requests.warmup", 0),
                "cache_measured_requests": _path(cache_replay, "requests.measured", 0),
                "capacity_bottleneck_status": (
                    "capacity_thrashing"
                    if _path(eviction, "bytes", 0) or _path(admission, "rejected_bytes", 0)
                    else "none"
                ),
            }
        )
    row_case = {
        "id": str(case.get("id", "case")),
        "analysis_mode": str(case.get("analysis_mode", "bandwidth_envelope")),
        "model_config": str(case["model_config"]),
        "runtime_config": str(case["runtime_config"]),
        "workload_config": str(case["workload_config"]),
        "storage_config": str(case.get("storage_config") or DEFAULT_STORAGE_CONFIG),
        "seq_len": int(case["seq_len"]),
        "prefix_ratio": float(case["prefix_ratio"]),
        "block_size_tokens": int(case["block_size_tokens"]),
        "arrival_model": str(case["arrival_model"]),
        "max_io_bytes": _optional_int(case.get("max_io_bytes")),
        "workload_overrides": case.get("workload_overrides", {}),
        "write_miss_blocks": bool(case.get("write_miss_blocks", True)),
        "modeling_options": dict(case.get("modeling_options") or {}),
        "config_digests": dict(case.get("config_digests") or {}),
        "case_identity": str(case.get("case_identity") or _case_identity(case)),
    }
    if "compute_config" in case:
        row_case["compute_config"] = case["compute_config"]
    for key in ENCODER_SCENARIO_VALUES:
        if key in case:
            row_case[key] = case[key]
    if isinstance(case.get("cache_replay_config"), dict):
        row_case["cache_replay_config"] = dict(case["cache_replay_config"])
    return {
        "id": str(case.get("id", "case")),
        "case": row_case,
        "model": model.model_id,
        "model_display_name": model.display_name,
        "model_identity": asdict(model.identity),
        "attention": model.attention.name,
        "attention_display": _model_recovery_capability(model)["display_attention"],
        "formula_id": _model_recovery_capability(model)["formula_id"],
        "analysis_mode": str(case.get("analysis_mode", "bandwidth_envelope")),
        "seq_len": workload.seq_len,
        "prefix_ratio": workload.prefix_ratio,
        "workload_shape": workload.workload_shape,
        "block_size_tokens": int(case["block_size_tokens"]),
        "arrival_model": workload.arrival_model,
        "max_io_bytes": _optional_int(case.get("max_io_bytes")),
        "request_count": workload.request_count,
        "qps": workload.qps,
        "metrics": metrics,
        "evaluation": report.get("evaluation", {}),
        "compute_algorithms": ttft.get("compute_algorithms", []),
        **{key: value for key, value in report.get("io_accounting", {}).items() if key != "semantics"},
        **{key: case[key] for key in ENCODER_SCENARIO_VALUES if key in case},
        **({"phase_timing_samples": report["ttft"]["phase_timing_samples"]}
           if "phase_timing_samples" in report.get("ttft", {}) else {}),
        "result_semantics": _result_semantics(
            case=case,
            analysis=analysis,
            report=report,
        ),
        "effective_state_policies": report.get("effective_state_policies", {}),
        "compute_warnings": list(report.get("warnings", [])),
    }


def _compute_warnings_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({message for row in rows for message in row.get("compute_warnings", [])})


def _result_semantics(
    *,
    case: dict[str, Any],
    analysis: DemandAnalysis,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return additive, display-oriented semantics without changing raw metrics.

    The Dashboard historically flattened aggregate percentiles into each sweep
    row.  Percentiles alone cannot distinguish a meaningful zero from a cache
    object that is resident but cannot be used as a recovery boundary.  Keep
    every raw value intact and expose the per-request state counts alongside it.
    """

    recovery_outcomes: Counter[str] = Counter()
    ttft_outcomes: Counter[str] = Counter()
    for job in analysis.ttft_jobs:
        if job.resident_prefix_tokens <= 0 or job.resident_cache_bytes <= 0:
            recovery_outcomes["no_resident_prefix"] += 1
        elif job.execution_plan is not None:
            recovery_outcomes["bounded_replay" if job.restore_recompute_tokens else "encoder_ready"] += 1
        elif job.recovery_boundary_tokens <= 0:
            recovery_outcomes["resident_not_recoverable"] += 1
        elif job.restore_recompute_tokens > 0:
            recovery_outcomes["partial_replay"] += 1
        else:
            recovery_outcomes["direct_resume"] += 1

        if job.full_hit_finalization_status == "not_evaluated":
            ttft_outcomes["not_evaluated"] += 1
            continue
        dependency_floor = (
            job.cache_path_overhead_us
            + job.replay_compute_time_us
            + job.miss_suffix_compute_time_us
            + job.first_token_decode_us
        )
        if job.phase_timings:
            dependency_floor = job.cache_path_overhead_us + sum(p.compute_us for p in job.phase_timings) + job.first_token_decode_us
        saving = job.no_cache_ttft_us - dependency_floor
        tolerance = max(1e-9, abs(job.no_cache_ttft_us) * 1e-12)
        if not math.isfinite(saving):
            ttft_outcomes["infinite"] += 1
        elif abs(saving) <= tolerance:
            ttft_outcomes["zero"] += 1
        elif saving < 0:
            ttft_outcomes["negative"] += 1
        else:
            ttft_outcomes["positive"] += 1

    recovery = report.get("recovery", {})
    critical = recovery.get("critical_path", {})
    optimistic = critical.get("optimistic_overlap_GBps", {})
    dependency = critical.get("dependency_aware_break_even_GBps", {})
    effective_policies = report.get("effective_state_policies", {})
    recovery_assumption = str(
        (case.get("modeling_options") or {}).get(
            "recovery_assumption", "model_default"
        )
    )

    return {
        "evaluation": report.get("evaluation", {}),
        "recovery": {
            "dominant_status": _dominant_semantic_status(recovery_outcomes),
            "status_counts": dict(sorted(recovery_outcomes.items())),
            "reason_counts": dict(sorted((recovery.get("causes") or {}).items())),
            "total_request_count": sum(recovery_outcomes.values()),
            "assumption": recovery_assumption,
            "policy_summary": _effective_policy_summary(effective_policies),
        },
        "ttft_savings": {
            "dominant_status": ("not_evaluated" if ttft_outcomes.get("not_evaluated")
                                else _dominant_semantic_status(ttft_outcomes)),
            "status_counts": dict(sorted(ttft_outcomes.items())),
            "total_request_count": sum(ttft_outcomes.values()),
        },
        "optimistic_overlap": _threshold_result_semantics(optimistic),
        "dependency_break_even": _threshold_result_semantics(dependency),
    }


def _threshold_result_semantics(report: dict[str, Any]) -> dict[str, Any]:
    status_counts = {
        str(key): int(value)
        for key, value in (report.get("status_counts") or {}).items()
    }
    total = sum(status_counts.values())
    return {
        "status": str(report.get("status") or "not_applicable"),
        "status_counts": dict(sorted(status_counts.items())),
        # ``finite_count`` in the historical report also includes numeric
        # no-restore zeros.  This new count means finite bandwidth was actually
        # required, which is the denominator users need for a mixed result.
        "finite_required_count": status_counts.get("finite", 0),
        "total_request_count": total,
    }


def _dominant_semantic_status(counts: Counter[str]) -> str:
    if not counts:
        return "missing"
    maximum = max(counts.values())
    leaders = sorted(key for key, value in counts.items() if value == maximum)
    return leaders[0] if len(leaders) == 1 else "mixed"


def _effective_policy_summary(effective_policies: dict[str, Any]) -> str:
    parts: list[str] = []
    for name, value in sorted(effective_policies.items()):
        policy = value.get("policy") or {}
        storage = str(policy.get("storage_form") or "unknown")
        if storage == "snapshot":
            interval = policy.get("interval_tokens")
            cadence = "prompt_end" if interval is None else f"periodic:{int(interval)}"
            if interval is not None and policy.get("retain_prompt_end"):
                cadence += "+prompt_end"
        else:
            cadence = storage
        source = str(value.get("source") or "model_default")
        parts.append(f"{name}={cadence}({source})")
    return " | ".join(parts) if parts else "token_state"


def _storage_work_estimate(report: dict[str, Any]) -> dict[str, Any]:
    read = report.get("read", {})
    write = report.get("write", {})
    read_logical_ops = float(_path(read, "logical_block_ops", 0) or 0)
    write_logical_ops = float(_path(write, "logical_block_ops", 0) or 0)
    read_chunked_ios = _path(read, "chunked_ios")
    write_chunked_ios = _path(write, "chunked_ios")
    chunking_known = read_chunked_ios is not None and write_chunked_ios is not None
    estimated_physical_ios = (
        float(read_chunked_ios or 0) + float(write_chunked_ios or 0)
        if chunking_known
        else None
    )
    return {
        "logical_block_ops": read_logical_ops + write_logical_ops,
        "read_logical_block_ops": read_logical_ops,
        "write_logical_block_ops": write_logical_ops,
        "physical_ios": estimated_physical_ios,
        "read_physical_ios": float(read_chunked_ios or 0) if chunking_known else None,
        "write_physical_ios": float(write_chunked_ios or 0) if chunking_known else None,
        "limits": {
            "logical_block_ops": MAX_INTERACTIVE_STORAGE_LOGICAL_OPS,
            "physical_ios": MAX_INTERACTIVE_STORAGE_PHYSICAL_IOS,
        },
    }


def _validate_interactive_storage_work(estimate: dict[str, Any]) -> None:
    logical_ops = float(estimate.get("logical_block_ops") or 0)
    physical_ios = estimate.get("physical_ios")
    if logical_ops > MAX_INTERACTIVE_STORAGE_LOGICAL_OPS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            (
                "Selected case is too large for interactive SSD simulation: "
                f"estimated {logical_ops:,.0f} logical block ops, limit "
                f"{MAX_INTERACTIVE_STORAGE_LOGICAL_OPS:,.0f}. Reduce request_count, "
                "seq_len, or hit-rate, then try again."
            ),
        )
    if physical_ios is not None and float(physical_ios) > MAX_INTERACTIVE_STORAGE_PHYSICAL_IOS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            (
                "Selected case is too large for interactive SSD simulation: "
                f"estimated {float(physical_ios):,.0f} physical IOs, limit "
                f"{MAX_INTERACTIVE_STORAGE_PHYSICAL_IOS:,.0f}. Reduce request_count, "
                "seq_len, or max_io_bytes pressure, then try again."
            ),
        )


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "case_count": len(rows),
        "evaluation_status_counts": dict(Counter(
            row.get("evaluation", {}).get("status", "legacy_unspecified") for row in rows)),
        "max_read_avg_GBps": _max_metric(rows, "read_avg_GBps"),
        "max_write_avg_GBps": _max_metric(rows, "write_avg_GBps"),
        "max_optimistic_overlap_GBps": _max_metric(rows, "optimistic_overlap_GBps_max"),
        "max_dependency_aware_break_even_GBps": _max_metric(
            rows, "dependency_aware_break_even_GBps_max"
        ),
        "max_aggregate_dependency_required_GBps": _max_metric(
            rows, "aggregate_dependency_required_GBps"
        ),
        "max_aggregate_dependency_chunked_iops": _max_metric(
            rows, "aggregate_dependency_chunked_iops"
        ),
        "max_retain_all_capacity_GiB": _max_metric(
            rows,
            "retain_all_capacity_GiB",
        ),
        "min_dependency_ttft_savings_us_p50": _min_metric(
            rows, "dependency_ttft_savings_us_p50"
        ),
    }
    summary.update(_risk_summary(rows))
    incomplete = [row for row in rows if row.get("evaluation", {}).get("status") in {"not_evaluated", "unsupported"}]
    summary["evaluation_status"] = "not_evaluated" if incomplete else "evaluated"
    summary["not_evaluated_case_count"] = len(incomplete)
    if incomplete:
        for key in ("max_optimistic_overlap_GBps", "max_dependency_aware_break_even_GBps",
                    "max_aggregate_dependency_required_GBps", "max_aggregate_dependency_chunked_iops",
                    "min_dependency_ttft_savings_us_p50"):
            summary[key] = None
        if any(row.get("evaluation", {}).get("aborted") for row in incomplete):
            for key in ("max_read_avg_GBps", "max_write_avg_GBps", "max_retain_all_capacity_GiB"):
                summary[key] = None
    return summary


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
        ttft = _number_or_none(metrics.get("dependency_ttft_savings_us_p50"))
        negative_ttft += int(ttft is not None and ttft < 0)
        zero_ttft += int(ttft == 0)
        numbers = [_number_or_none(metrics.get(key)) for key in resource_metrics]
        non_finite_status = metrics.get("dependency_aware_break_even_status") in {
            "no_compute_saving",
            "no_replay_slack",
            "no_overlap_window",
            "mixed",
        }
        infinite += int(non_finite_status)
        missing += int(any(value is None for value in numbers))
    return {
        "negative_ttft_case_count": negative_ttft,
        "zero_ttft_case_count": zero_ttft,
        "infinite_resource_case_count": infinite,
        "missing_resource_case_count": missing,
    }


def _config_options(
    kind: str,
    *,
    include_legacy: bool = False,
) -> list[dict[str, Any]]:
    directory = ROOT / "configs" / kind
    if not directory.exists():
        return []
    if kind == "models":
        entries = discover_model_catalog(directory)
        selected = legacy_catalog(entries) if include_legacy else active_catalog(entries)
        return [
            {
                "path": item.path.relative_to(ROOT).as_posix(),
                "config_sha256": _file_sha256(item.path),
                "label": item.identity.display_name,
                "model_id": item.identity.id,
                "family": item.identity.family,
                "aliases": list(item.identity.aliases),
                "formula_id": item.formula_id,
                "evidence_statuses": sorted(
                    {
                        str(value.get("status"))
                        for value in (
                            item.raw.get("evidence", {}).values()
                            if isinstance(item.raw.get("evidence"), dict)
                            else ()
                        )
                        if isinstance(value, dict) and value.get("status")
                    }
                ),
                "catalog": {
                    "visible": item.catalog.visible,
                    "legacy_visible": item.catalog.legacy_visible,
                    "deprecated_since": item.catalog.deprecated_since,
                    "replacement_id": item.catalog.replacement_id,
                },
            }
            for item in selected
        ]
    options = []
    for path in sorted(directory.glob("*.y*ml")):
        if path.name.startswith("."):
            continue
        options.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "label": path.stem,
                "config_sha256": _file_sha256(path),
            }
        )
    return options


def _curated_model_paths(options: list[dict[str, Any]]) -> list[str]:
    """Choose five formula-diverse defaults without silently selecting the entire catalog."""

    formula_order = (
        "standard_gqa",
        "mla",
        "dsa_mla",
        "kimi_kda_mla_hybrid",
        "deepseek_v4_hybrid",
    )
    selected: list[str] = []
    for formula_id in formula_order:
        match = next(
            (item for item in options if item.get("formula_id") == formula_id),
            None,
        )
        if match is not None:
            selected.append(str(match["path"]))
    # Keep established formula families ahead of newly added families when
    # filling a preset. Catalog expansion must not replace its control models.
    for item in sorted(options, key=lambda item: item.get("formula_id") not in formula_order):
        if len(selected) >= 5:
            break
        path = str(item["path"])
        if path not in selected:
            selected.append(path)
    return selected[:5]


def _model_constraints(model: ModelProfile) -> dict[str, Any]:
    """Expose only constraints the V2 form can validate before a sweep."""

    attention = model.attention
    provider = getattr(attention, "block_size_constraints", None)
    if callable(provider):
        provided = provider()
        constraints = asdict(provided) if is_dataclass(provided) else dict(provided)
    else:
        constraints = {
            "minimum": 1,
            "multiple": 1,
            "native": getattr(attention, "native_block_size_tokens", None),
        }
    return {
        "max_context_tokens": model.max_context_tokens,
        "minimum_block_size_tokens": int(constraints.get("minimum", 1) or 1),
        "native_block_size_tokens": constraints.get("native"),
        "block_size_multiple": int(constraints.get("multiple", 1) or 1),
    }


def _model_recovery_capability(model: ModelProfile) -> dict[str, Any]:
    attention = model.attention
    factory = getattr(attention, "effective_checkpoint_groups", None)
    if callable(factory):
        try:
            groups = tuple(factory(model, PrefixCachePolicy()))
        except TypeError:
            groups = tuple(factory(model))
    else:
        factory = getattr(attention, "checkpoint_groups", None)
        groups = tuple(factory(model)) if callable(factory) else ()
    token_components = []
    for component in model.cache_components():
        token_components.append(
            {
                "name": component.name,
                "category": component.category,
                "layer_count": component.layer_count or model.num_layers,
                "bytes_per_token": component.bytes_per_token(model),
            }
        )
    state_groups = []
    for group in groups:
        state_groups.append(
            {
                "name": group.name,
                "default_policy": asdict(group.policy),
                "supported_scenarios": [
                    "model_default",
                    "assume_exact_at_resident",
                    "periodic",
                    "none",
                ],
                "checkpoint_bytes": group.serialized_bytes_per_checkpoint,
                "layer_kinds": [kind.value for kind in group.layer_kinds],
                "components": [
                    {
                        "name": component.name,
                        "category": component.category,
                        "bytes_per_checkpoint": component.serialized_bytes_per_checkpoint(),
                    }
                    for component in group.components
                ],
            }
        )
    object_policy = getattr(attention, "prefix_granularity", None)
    formula_id = getattr(
        model.compute.prefill_formula,
        "type",
        getattr(attention, "calculator_formula_id", attention.name),
    )
    return {
        "attention_family": (
            "gdn_linear_attention_hybrid"
            if formula_id == "qwen_linear_full_hybrid"
            else attention.name
        ),
        "display_attention": (
            "GDN / Linear Attention Hybrid"
            if formula_id == "qwen_linear_full_hybrid"
            else attention.name
        ),
        "formula_id": formula_id,
        "prefix_object_policy": (
            asdict(object_policy) if object_policy is not None else {"partial_tail_storage": "sealed_padded"}
        ),
        "token_components": token_components,
        "state_groups": state_groups,
        "state_policy_targets": _model_state_policy_targets(model, groups),
        "has_blocking_state": bool(state_groups),
        "sparse_restore_supported": getattr(attention, "sparse_access_policy", None) is not None,
        "draft_cache_supported": getattr(model, "draft_support", "none") not in {"none", "unsupported"},
        "draft_execution_supported": getattr(model, "draft_support", "none") not in {"none", "unsupported", "cache_only"},
        "encoder_scenarios_supported": callable(getattr(attention, "recovery_plan", None)),
        "timing_basis": model.compute.timing_mode,
    }


def _model_state_policy_targets(
    model: ModelProfile,
    default_groups: tuple[Any, ...] | None = None,
) -> list[dict[str, Any]]:
    """Describe every state target that the Dashboard may safely configure."""

    attention = model.attention
    if default_groups is None:
        factory = getattr(attention, "effective_checkpoint_groups", None)
        if callable(factory):
            try:
                default_groups = tuple(factory(model, PrefixCachePolicy()))
            except TypeError:
                default_groups = tuple(factory(model))
        else:
            factory = getattr(attention, "checkpoint_groups", None)
            default_groups = tuple(factory(model)) if callable(factory) else ()

    groups_by_name = {group.name: group for group in default_groups}
    sliding_default = getattr(attention, "sliding_persistence", None)
    if isinstance(sliding_default, StatePersistencePolicy):
        method = getattr(attention, "_sliding_checkpoint_group", None)
        if not callable(method):
            method = getattr(attention, "_sliding_group", None)
        if callable(method) and "sliding_window_state" not in groups_by_name:
            generated = method(model, StatePersistencePolicy.terminal())
            if generated is not None:
                groups_by_name["sliding_window_state"] = generated

    targets: list[dict[str, Any]] = []
    for name, group in sorted(groups_by_name.items()):
        default_policy = (
            sliding_default
            if name == "sliding_window_state"
            and isinstance(sliding_default, StatePersistencePolicy)
            else group.policy
        )
        supports_token_history = name in {"swa", "sliding_window_state"}
        checkpoint_bytes = group.serialized_bytes_per_checkpoint
        window = int(getattr(attention, "sliding_window", 0) or 0)
        token_history_bytes_per_token = (
            checkpoint_bytes // window
            if supports_token_history and window and checkpoint_bytes % window == 0
            else None
        )
        origin = "model_config"
        if name == "swa":
            origin = str(
                getattr(attention, "swa_persistence_origin", "engine_fallback")
            )
        aliases = [component.name for component in group.components]
        if name == "swa":
            aliases.append("sliding_window_state")
        targets.append(
            {
                "name": name,
                "aliases": sorted(set(aliases)),
                "default_policy": asdict(default_policy),
                "policy_origin": origin,
                "supported_storage_forms": [
                    *(["token_history"] if supports_token_history else []),
                    "snapshot",
                    "none",
                ],
                "checkpoint_bytes": checkpoint_bytes,
                "token_history_bytes_per_token": token_history_bytes_per_token,
                "requires_explicit_conversion": (
                    default_policy.storage_form == "token_history"
                ),
                "global_assumption_applicable": (
                    default_policy.storage_form == "snapshot"
                ),
                "components": [
                    {
                        "name": component.name,
                        "category": component.category,
                        "bytes_per_checkpoint": component.serialized_bytes_per_checkpoint(),
                    }
                    for component in group.components
                ],
            }
        )
    return targets


def _validate_modeling_state_policies(
    model_configs: list[str],
    runtime_data: dict[str, Any],
    modeling_options: dict[str, Any],
) -> None:
    selected = set(model_configs)
    model_overrides = modeling_options.get("model_state_policies", {})
    unknown_models = sorted(set(model_overrides) - selected)
    if unknown_models:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"model_state_policies references an unselected model: {unknown_models[0]}",
        )

    targets_by_model: dict[str, list[dict[str, Any]]] = {}
    for path in model_configs:
        model = _load_model(path, runtime_data)
        targets_by_model[path] = _model_state_policy_targets(model)

    def names_for(path: str) -> set[str]:
        return {
            value
            for target in targets_by_model[path]
            for value in (target["name"], *target.get("aliases", ()))
        }

    for name in modeling_options.get("state_policies", {}):
        if not any(name in names_for(path) for path in model_configs):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"state policy target {name!r} does not match any selected model",
            )
    for path, policies in model_overrides.items():
        supported = names_for(path)
        for name in policies:
            if name not in supported:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"model state policy target {name!r} is not supported by {path}",
                )
    for path, targets in targets_by_model.items():
        for target in targets:
            names = (target["name"], *target.get("aliases", ()))
            for policies, label in (
                (modeling_options.get("state_policies", {}), "state_policies"),
                (model_overrides.get(path, {}), f"model_state_policies[{path}]"),
            ):
                values = [policies[name] for name in names if name in policies]
                if values and any(value != values[0] for value in values[1:]):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        f"{label} contains conflicting policies for atomic target {target['name']}",
                    )


def _first_existing(options: list[dict[str, Any]], preferred: str) -> str:
    paths = [option["path"] for option in options]
    return preferred if preferred in paths else (paths[0] if paths else preferred)


def _default_workload_overrides(workload_path: str) -> dict[str, Any]:
    data = load_config(_resolve_config_path(workload_path))
    keys = [
        "request_count",
        "qps",
        "generation_len",
        "prefix_group_count",
        "prefix_zipf_alpha",
        "tenant_count",
        "seed",
        "workload_shape",
    ]
    return {key: data[key] for key in keys if key in data}


def _workload_preset_defaults(data: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "request_count",
        "qps",
        "seq_len",
        "generation_len",
        "prefix_ratio",
        "prefix_group_count",
        "prefix_zipf_alpha",
        "arrival_model",
        "seed",
        "workload_shape",
    ]
    return {key: data[key] for key in keys if key in data}


def _clean_workload_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "request_count": int,
        "qps": float,
        "generation_len": int,
        "prefix_group_count": int,
        "prefix_zipf_alpha": float,
        "tenant_count": int,
        "seed": int,
    }
    cleaned: dict[str, Any] = {}
    for key, caster in allowed.items():
        if (
            key not in overrides
            or overrides[key] is None
            or overrides[key] == ""
        ):
            continue
        try:
            if isinstance(overrides[key], bool):
                raise ValueError(f"{key} must be numeric")
            if caster is int:
                cleaned[key] = _strict_int(overrides[key], key)
            else:
                cleaned[key] = caster(overrides[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid workload override: {key}") from exc
    if "workload_shape" in overrides:
        workload_shape = str(overrides["workload_shape"])
        if workload_shape not in {
            "shared_prefix_groups",
            "prompt_lineage_bootstrap",
        }:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "workload_shape must be shared_prefix_groups or prompt_lineage_bootstrap",
            )
        cleaned["workload_shape"] = workload_shape
    return cleaned


def _validate_workload_inputs(
    workload_config: str,
    overrides: dict[str, Any],
) -> None:
    data = load_config(_resolve_config_path(workload_config))
    data.update(overrides)
    if int(data.get("request_count", 0)) < 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "request_count must be non-negative")
    qps = float(data.get("qps", 0))
    if not math.isfinite(qps) or qps <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "qps must be positive and finite")
    if int(data.get("generation_len", 0)) < 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "generation_len must be non-negative")
    if int(data.get("prefix_group_count", 0)) <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "prefix_group_count must be positive")
    if int(data.get("tenant_count", 1)) <= 0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "tenant_count must be positive")
    alpha = float(data.get("prefix_zipf_alpha", 1))
    if not math.isfinite(alpha):
        raise ApiError(HTTPStatus.BAD_REQUEST, "prefix_zipf_alpha must be finite")


def _load_model(model_config: str, runtime_config: dict[str, Any]) -> ModelProfile:
    return model_from_dict(load_config(_resolve_config_path(model_config)), runtime_config)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_config_digests(
    *,
    model_configs: list[str],
    runtime_config: str,
    workload_config: str,
    storage_config: str,
    cache_config: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "models": {
            path: _file_sha256(_resolve_config_path(path))
            for path in sorted(model_configs)
        },
        "runtime": {
            "path": runtime_config,
            "sha256": _file_sha256(_resolve_config_path(runtime_config)),
        },
        "workload": {
            "path": workload_config,
            "sha256": _file_sha256(_resolve_config_path(workload_config)),
        },
        "storage": {
            "path": storage_config,
            "sha256": _file_sha256(_resolve_config_path(storage_config)),
        },
    }
    if cache_config is not None:
        result["cache"] = {
            "path": cache_config,
            "sha256": _file_sha256(_resolve_config_path(cache_config)),
        }
    return result


def _resolve_config_path(reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (ROOT / path).resolve()
    configs_root = (ROOT / "configs").resolve()
    if not resolved.is_relative_to(configs_root):
        raise ApiError(HTTPStatus.BAD_REQUEST, "config path must stay under configs/")
    if not resolved.is_file():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"config not found: {reference}")
    if resolved.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ApiError(HTTPStatus.BAD_REQUEST, "config must be YAML or JSON")
    return resolved


def _runtime_compute_config(runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Expose the existing timing settings without changing cache/analysis defaults."""
    result: dict[str, Any] = {
        "timing_mode": "normalized_reference",
        "effective_tflops": {"default": 2000},
    }
    for name in ("hardware", "compute"):
        section = runtime_config.get(name) or {}
        if "effective_prefill_tflops" in section:
            result["effective_tflops"]["default"] = section["effective_prefill_tflops"]
        rates = section.get("effective_tflops")
        if isinstance(rates, dict):
            result["effective_tflops"].update(rates)
        elif rates is not None:
            result["effective_tflops"]["default"] = rates
        if section.get("timing_mode") is not None:
            result["timing_mode"] = section["timing_mode"]
        if "timing_omissions" in section:
            result["timing_omissions"] = copy.deepcopy(section["timing_omissions"])
    return result


def _parse_compute_config(raw: Any, runtime_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "compute_config must be a mapping")
    unknown = set(raw) - {"timing_mode", "effective_tflops", "timing_omissions"}
    if unknown:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"unsupported compute field: {sorted(unknown)[0]}")
    rates = raw.get("effective_tflops")
    if not isinstance(rates, dict) or "default" not in rates:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Compute requires effective_tflops.default in TFLOPS")
    result = _runtime_compute_config(runtime_config)
    result["effective_tflops"].update(rates)
    for name, value in result["effective_tflops"].items():
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = float("nan")
        if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Compute {name} TFLOPS must be a finite positive number")
        result["effective_tflops"][name] = number
    result["timing_mode"] = raw.get("timing_mode", result["timing_mode"])
    if result["timing_mode"] not in ("normalized_reference", "component_calibrated"):
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported compute timing_mode")
    if "timing_omissions" in raw:
        omissions = raw["timing_omissions"]
        if not isinstance(omissions, list) or any(not isinstance(item, str) or not item.strip() for item in omissions):
            raise ApiError(HTTPStatus.BAD_REQUEST, "compute timing_omissions must be a list of non-empty strings")
        result["timing_omissions"] = list(omissions)
    return result


def _runtime_with_compute(runtime_config: dict[str, Any], compute_config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(runtime_config)
    # The registry accepts hardware and legacy compute sections; explicit input
    # must win in both, while cache precision and all modeling policies stay put.
    for name in ("hardware", "compute"):
        if name == "compute" and name not in result:
            continue
        section = result.setdefault(name, {})
        section.update(copy.deepcopy(compute_config))
        if "effective_prefill_tflops" in section:
            section["effective_prefill_tflops"] = compute_config["effective_tflops"]["default"]
    return result


def _load_effective_runtime(case_or_params: dict[str, Any]) -> dict[str, Any]:
    runtime = load_config(_resolve_config_path(str(case_or_params["runtime_config"])))
    if "compute_config" in case_or_params:
        runtime = _runtime_with_compute(runtime, case_or_params["compute_config"])
    return runtime


def _runtime_analysis_defaults(runtime_config: dict[str, Any]) -> dict[str, Any]:
    value = runtime_config.get("analysis", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "runtime_config.analysis must be a mapping")
    return value


def _runtime_modeling_defaults(runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Read optional v2 policies without changing legacy runtime semantics."""

    raw = runtime_config.get("modeling_options") or {}
    if not isinstance(raw, dict):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "runtime_config.modeling_options must be a mapping",
        )
    values = dict(raw)
    analysis = _runtime_analysis_defaults(runtime_config)
    for key in DEFAULT_MODELING_OPTIONS:
        if key in analysis and key not in values:
            values[key] = analysis[key]
    if "sparse_access_policy" in runtime_config and "sparse_access_policy" not in values:
        values["sparse_access_policy"] = runtime_config["sparse_access_policy"]

    schedule = runtime_config.get("decode_write_schedule")
    if schedule is None:
        schedule = analysis.get("decode_write_schedule")
    if schedule is not None:
        if not isinstance(schedule, dict):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "decode_write_schedule must be a mapping",
            )
        mode = str(schedule.get("mode") or "burst_at_completion")
        if mode not in {"burst_at_completion", "uniform_step"}:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "decode_write_schedule.mode must be burst_at_completion or uniform_step",
            )
        values.setdefault("decode_state_writes", True)
        if mode == "uniform_step":
            values.setdefault("decode_write_step_us", schedule.get("step_us_per_token"))
        else:
            values.setdefault("decode_write_step_us", None)
    return _normalize_modeling_options({"modeling_options": values})


def _runtime_peak_windows(runtime_analysis: dict[str, Any]) -> tuple[int, ...]:
    values = runtime_analysis.get("peak_windows_us", (1_000, 10_000, 100_000, 1_000_000))
    if not isinstance(values, (list, tuple)):
        raise ApiError(HTTPStatus.BAD_REQUEST, "peak_windows_us must be a list")
    return tuple(int(value) for value in values)


def _int_axis(axes: dict[str, Any], name: str, default: list[int]) -> list[int]:
    values = axes.get(name, default)
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        try:
            parsed = _strict_int(value, name)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must contain integers") from exc
        if parsed <= 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must contain positive values")
        result.append(parsed)
    return sorted(set(result))


def _nonnegative_int_axis(
    axes: dict[str, Any], name: str, default: list[Any]
) -> list[int]:
    values = axes.get(name, default)
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        try:
            parsed = _strict_int(value, name)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, f"{name} must contain integers"
            ) from exc
        if parsed < 0:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{name} must contain non-negative values",
            )
        result.append(parsed)
    if not result:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must not be empty")
    return sorted(set(result))


def _float_axis(axes: dict[str, Any], name: str, default: list[float]) -> list[float]:
    values = axes.get(name, default)
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must contain numbers")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must contain numbers") from exc
        if not math.isfinite(parsed):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must contain finite numbers")
        if name == "prefix_ratio" and not 0.0 <= parsed <= 1.0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "prefix_ratio must be in [0, 1]")
        result.append(parsed)
    return sorted(set(result))


def _optional_int_axis(axes: dict[str, Any], name: str, default: list[Any]) -> list[int | None]:
    values = axes.get(name, default)
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None and parsed <= 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must contain positive values")
        result.append(parsed)
    return sorted(set(result), key=lambda item: -1 if item is None else item)


def _arrival_axis(axes: dict[str, Any]) -> list[str]:
    values = axes.get("arrival_model", ["fixed"])
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        parsed = str(value)
        if parsed not in {"fixed", "poisson"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "arrival_model must be fixed or poisson")
        result.append(parsed)
    return sorted(set(result))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value in {"", "null", "none", "None"}:
        return None
    return _strict_int(value, "integer value")


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(parsed)


def _strict_positive_int(value: Any, label: str) -> int:
    parsed = _strict_int(value, label)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _optional_positive_number(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive number or null")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be a positive finite number or null")
    return parsed


def _path(payload: Any, dotted: str, default: Any = None) -> Any:
    current = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _bytes_to_gib(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return value / (1024**3)


def _max_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [_number_or_none(row["metrics"].get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _min_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [_number_or_none(row["metrics"].get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _content_type(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Serving Prefix Cache Demand Workbench at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
