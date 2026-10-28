"""Spawn-safe worker entry points for Dashboard cache sweep cases."""

from __future__ import annotations

from typing import Any


_WORKER_RUNTIME_CONFIG: dict[str, Any] | None = None
_WORKER_MODELS: dict[str, Any] | None = None
_WORKER_CANCEL_EVENT: Any | None = None


def initialize_worker(
    runtime_config: dict[str, Any],
    models: dict[str, Any],
    cancel_event: Any,
) -> None:
    """Install immutable sweep inputs once per spawned worker."""

    global _WORKER_RUNTIME_CONFIG, _WORKER_MODELS, _WORKER_CANCEL_EVENT
    _WORKER_RUNTIME_CONFIG = runtime_config
    _WORKER_MODELS = models
    _WORKER_CANCEL_EVENT = cancel_event


def run_case(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one original Dashboard case without changing its semantics."""

    if _WORKER_CANCEL_EVENT is not None and _WORKER_CANCEL_EVENT.is_set():
        return {"kind": "canceled"}
    if _WORKER_RUNTIME_CONFIG is None or _WORKER_MODELS is None:
        raise RuntimeError("parallel sweep worker was not initialized")

    # Keep this import lazy so the worker module remains spawn-safe when the
    # dashboard itself is launched as a script.
    from dashboard_v2 import server

    case = descriptor["case"]
    model = _WORKER_MODELS[descriptor["model_config"]]
    try:
        analysis, workload, miss_window = server._run_demand_case(
            model=model,
            runtime_config=_WORKER_RUNTIME_CONFIG,
            workload_config=case["workload_config"],
            case=case,
            include_trace=False,
        )
        row = server._row_from_analysis(
            case,
            model,
            workload,
            analysis,
            miss_window,
        )
    except server.ApiError as exc:
        # ApiError's constructor needs more than Exception.args. Transport its
        # complete public state and reconstruct the parent module's ApiError.
        return {
            "kind": "api_error",
            "status": int(exc.status),
            "message": exc.message,
            "details": exc.details,
        }
    except Exception as exc:
        # Built-in and model exceptions retain their exact type and args through
        # multiprocessing pickle. If an unusual exception cannot be transported,
        # the parent treats that as worker infrastructure failure and recomputes
        # the uncommitted suffix serially.
        return {"kind": "exception", "exception": exc}
    return {"kind": "row", "row": row}
