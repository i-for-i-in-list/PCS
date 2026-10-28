"""Persistent local experiment history for Dashboard V3."""

from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / ".data" / "dashboard.sqlite3"
SCHEMA_VERSION = 1
MAX_UNPINNED_EXPERIMENTS = 50
MAX_EXPERIMENT_NAME_LENGTH = 120


class ExperimentNotFound(LookupError):
    """Raised when an experiment identifier does not exist."""


class ExperimentValidationError(ValueError):
    """Raised when an experiment mutation is invalid."""


class ExperimentStore:
    """SQLite-backed experiment storage with bounded unpinned retention."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path if path is not None else os.environ.get("PCS_DASHBOARD_DB")
        self.path = Path(configured) if configured else DEFAULT_DATABASE_PATH
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save(
        self,
        *,
        kind: str,
        request: dict[str, Any],
        result: dict[str, Any],
        name: str | None = None,
    ) -> dict[str, Any]:
        if kind not in {"sweep", "recommendation"}:
            raise ExperimentValidationError("experiment kind must be sweep or recommendation")
        created_at = time.time()
        experiment_id = str(uuid.uuid4())
        display_name = _validated_name(name or _default_name(kind, created_at))
        safe_request = json_safe(request)
        safe_result = json_safe(result)
        summary = safe_result.get("summary") if isinstance(safe_result, dict) else {}
        if not isinstance(summary, dict):
            summary = {}
        rows = safe_result.get("rows") if isinstance(safe_result, dict) else []
        case_count = int(
            safe_result.get("case_count", len(rows) if isinstance(rows, list) else 0)
        )
        fingerprint = fingerprint_payload(safe_request)
        request_json = _json_dumps(safe_request)
        summary_json = _json_dumps(summary)
        result_gzip = gzip.compress(_json_dumps(safe_result).encode("utf-8"), mtime=0)

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    id, kind, name, created_at, pinned, fingerprint,
                    request_json, summary_json, result_gzip, case_count
                ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    kind,
                    display_name,
                    created_at,
                    fingerprint,
                    request_json,
                    summary_json,
                    result_gzip,
                    case_count,
                ),
            )
            self._prune_locked(connection)
        return self.get(experiment_id, include_result=False)

    def list(
        self,
        *,
        kind: str | None = None,
        pinned: bool | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if kind is not None and kind not in {"sweep", "recommendation"}:
            raise ExperimentValidationError("invalid experiment kind")
        limit = max(1, min(200, int(limit)))
        clauses: list[str] = []
        values: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        if pinned is not None:
            clauses.append("pinned = ?")
            values.append(1 if pinned else 0)
        if search:
            clauses.append("LOWER(name) LIKE ?")
            values.append(f"%{search.strip().lower()}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT id, kind, name, created_at, pinned, fingerprint,
                   summary_json, case_count
            FROM experiments
            {where}
            ORDER BY pinned DESC, created_at DESC
            LIMIT ?
        """
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [_metadata_from_row(row) for row in rows]

    def get(self, experiment_id: str, *, include_result: bool = True) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, name, created_at, pinned, fingerprint,
                       request_json, summary_json, result_gzip, case_count
                FROM experiments WHERE id = ?
                """,
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise ExperimentNotFound("experiment not found")
        payload = _metadata_from_row(row)
        payload["request"] = json.loads(row["request_json"])
        if include_result:
            payload["result"] = annotate_result_schema(
                json.loads(gzip.decompress(row["result_gzip"]))
            )
        return payload

    def update(
        self,
        experiment_id: str,
        *,
        name: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any]:
        if name is None and pinned is None:
            raise ExperimentValidationError("provide name or pinned")
        assignments: list[str] = []
        values: list[Any] = []
        if name is not None:
            assignments.append("name = ?")
            values.append(_validated_name(name))
        if pinned is not None:
            if not isinstance(pinned, bool):
                raise ExperimentValidationError("pinned must be a boolean")
            assignments.append("pinned = ?")
            values.append(1 if pinned else 0)
        values.append(experiment_id)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE experiments SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            if cursor.rowcount == 0:
                raise ExperimentNotFound("experiment not found")
            self._prune_locked(connection)
        return self.get(experiment_id, include_result=False)

    def delete(self, experiment_id: str) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM experiments WHERE id = ?",
                (experiment_id,),
            )
            if cursor.rowcount == 0:
                raise ExperimentNotFound("experiment not found")

    def compare(self, baseline_id: str, candidate_id: str) -> dict[str, Any]:
        baseline = self.get(baseline_id)
        candidate = self.get(candidate_id)
        return compare_experiments(baseline, candidate)

    def _initialize(self) -> None:
        with self._lock, self._connect(initializing=True) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    fingerprint TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    result_gzip BLOB NOT NULL,
                    case_count INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS experiments_created_at
                    ON experiments(created_at DESC);
                CREATE INDEX IF NOT EXISTS experiments_kind
                    ON experiments(kind, created_at DESC);
                """
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported experiment database schema: {row['value']}"
                )

    def _connect(self, *, initializing: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        if initializing:
            connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _prune_locked(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT id FROM experiments
            WHERE pinned = 0
            ORDER BY created_at DESC
            LIMIT -1 OFFSET ?
            """,
            (MAX_UNPINNED_EXPERIMENTS,),
        ).fetchall()
        if rows:
            connection.executemany(
                "DELETE FROM experiments WHERE id = ?",
                [(row["id"],) for row in rows],
            )


def compare_experiments(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_result = baseline.get("result") or {}
    candidate_result = candidate.get("result") or {}
    baseline_version = baseline_result.get("sweep_result_schema_version")
    candidate_version = candidate_result.get("sweep_result_schema_version")
    baseline_mode = baseline_result.get("analysis_mode")
    candidate_mode = candidate_result.get("analysis_mode")
    if (
        baseline_version != 3
        or candidate_version != 3
        or baseline_mode != candidate_mode
    ):
        return {
            "compatible": False,
            "baseline": _experiment_reference(baseline),
            "candidate": _experiment_reference(candidate),
            "request_diff": _object_diff(
                baseline.get("request") or {}, candidate.get("request") or {}
            ),
            "summary_delta": {},
            "matched_cases": 0,
            "added_cases": 0,
            "removed_cases": 0,
            "added_case_keys": [],
            "removed_case_keys": [],
            "row_deltas": [],
            "warning": (
                "Recovery metrics can be compared only between schema-v3 "
                "experiments with the same analysis_mode."
            ),
        }
    baseline_rows = _rows_by_case_key(baseline_result.get("rows") or [])
    candidate_rows = _rows_by_case_key(candidate_result.get("rows") or [])
    shared_keys = sorted(set(baseline_rows) & set(candidate_rows))
    added_keys = sorted(set(candidate_rows) - set(baseline_rows))
    removed_keys = sorted(set(baseline_rows) - set(candidate_rows))

    row_deltas = []
    for key in shared_keys:
        before = baseline_rows[key]
        after = candidate_rows[key]
        metric_names = sorted(
            set((before.get("metrics") or {})) | set((after.get("metrics") or {}))
        )
        row_deltas.append(
            {
                "case_key": key,
                "baseline_id": before.get("id"),
                "candidate_id": after.get("id"),
                "metrics": {
                    metric: metric_delta(
                        (before.get("metrics") or {}).get(metric),
                        (after.get("metrics") or {}).get(metric),
                    )
                    for metric in metric_names
                },
            }
        )

    baseline_summary = baseline.get("summary") or {}
    candidate_summary = candidate.get("summary") or {}
    summary_names = sorted(set(baseline_summary) | set(candidate_summary))
    return {
        "compatible": True,
        "baseline": _experiment_reference(baseline),
        "candidate": _experiment_reference(candidate),
        "request_diff": _object_diff(
            baseline.get("request") or {},
            candidate.get("request") or {},
        ),
        "summary_delta": {
            key: metric_delta(baseline_summary.get(key), candidate_summary.get(key))
            for key in summary_names
        },
        "matched_cases": len(shared_keys),
        "added_cases": len(added_keys),
        "removed_cases": len(removed_keys),
        "added_case_keys": added_keys,
        "removed_case_keys": removed_keys,
        "row_deltas": row_deltas,
    }


def annotate_result_schema(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("sweep_result_schema_version") == 3:
        annotated = dict(result)
        annotated.setdefault("legacy", False)
        annotated.setdefault("recovery_metrics_available", True)
        return annotated
    original = dict(result)
    return {
        **original,
        "legacy": True,
        "legacy_schema_version": original.get("sweep_result_schema_version", 1),
        "recovery_metrics_available": False,
        "legacy_metrics": original,
    }


def metric_delta(baseline: Any, candidate: Any) -> dict[str, Any]:
    before = _numeric_value(baseline)
    after = _numeric_value(candidate)
    absolute = None
    percent = None
    if before is not None and after is not None:
        if math.isfinite(before) and math.isfinite(after):
            absolute = after - before
            if before != 0:
                percent = absolute / abs(before) * 100
    return {
        "baseline": json_safe(baseline),
        "candidate": json_safe(candidate),
        "baseline_state": _value_state(baseline),
        "candidate_state": _value_state(candidate),
        "absolute": absolute,
        "percent": percent,
    }


def fingerprint_payload(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(json_safe(payload)).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _metadata_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "name": row["name"],
        "created_at": row["created_at"],
        "pinned": bool(row["pinned"]),
        "fingerprint": row["fingerprint"],
        "case_count": row["case_count"],
        "summary": json.loads(row["summary_json"]),
    }


def _rows_by_case_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        case = dict(row.get("case") or {})
        case.pop("id", None)
        case.pop("case_identity", None)
        case.pop("modeling_options", None)
        # encoder_state and encoder_tail_policy deliberately remain as scalar
        # Case fields. A multi-scenario run must not collapse them into one key.
        result[_json_dumps(json_safe(case))] = row
    return result


def _object_diff(baseline: Any, candidate: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        changes = []
        for key in sorted(set(baseline) | set(candidate)):
            next_path = f"{path}.{key}" if path else key
            changes.extend(
                _object_diff(baseline.get(key), candidate.get(key), next_path)
            )
        return changes
    if baseline == candidate:
        return []
    return [
        {
            "path": path,
            "baseline": json_safe(baseline),
            "candidate": json_safe(candidate),
        }
    ]


def _experiment_reference(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: experiment.get(key)
        for key in ("id", "kind", "name", "created_at", "pinned", "case_count", "summary")
    }


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


def _value_state(value: Any) -> str:
    number = _numeric_value(value)
    if number is None:
        return "missing"
    if math.isinf(number):
        return "positive_infinity" if number > 0 else "negative_infinity"
    if number == 0:
        return "zero"
    return "positive" if number > 0 else "negative"


def _validated_name(name: str) -> str:
    value = str(name).strip()
    if not value:
        raise ExperimentValidationError("experiment name must not be empty")
    if len(value) > MAX_EXPERIMENT_NAME_LENGTH:
        raise ExperimentValidationError(
            f"experiment name must be at most {MAX_EXPERIMENT_NAME_LENGTH} characters"
        )
    return value


def _default_name(kind: str, created_at: float) -> str:
    label = "Recommendation" if kind == "recommendation" else "Sweep"
    timestamp = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
    return f"{label} · {timestamp}"


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
