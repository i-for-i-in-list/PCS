"""Demand-report schema compatibility helpers.

CED phase timings, state sources and IO accounting are additive v3 fields.
Old reports remain readable without inventing phase measurements.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping


DEMAND_REPORT_SCHEMA_VERSION = 3


def adapt_demand_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Annotate legacy reports without inventing recovery-v3 measurements."""

    adapted = copy.deepcopy(dict(report))
    version = int(adapted.get("demand_report_schema_version", 1))
    if version > DEMAND_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported demand report schema version {version}; "
            f"maximum supported is {DEMAND_REPORT_SCHEMA_VERSION}"
        )
    if version == DEMAND_REPORT_SCHEMA_VERSION:
        adapted.setdefault("legacy", False)
        adapted.setdefault("recovery_metrics_available", True)
        return adapted

    original = copy.deepcopy(adapted)

    models = adapted.get("models", {})
    if isinstance(models, dict):
        for model_id, value in models.items():
            if not isinstance(value, dict):
                continue
            value.setdefault(
                "model_identity",
                {
                    "id": str(model_id),
                    "display_name": str(model_id),
                    "family": "legacy-report",
                    "aliases": [],
                },
            )
            value.setdefault(
                "active_execution_footprint",
                {
                    "nominal_payload_bytes": value.get("resident_state_bytes"),
                    "serialized_bytes": value.get("resident_state_bytes"),
                    "reserved_bytes": value.get("resident_state_bytes"),
                    "by_component": {},
                    "migration_status": "legacy_inferred",
                },
            )
            value.setdefault(
                "prefix_store_footprint",
                {
                    "nominal_payload_bytes": value.get("serialized_prefix_bytes"),
                    "serialized_bytes": value.get("serialized_prefix_bytes"),
                    "object_count": None,
                    "checkpoint_count": None,
                    "by_component": {},
                    "migration_status": "legacy_inferred",
                },
            )

    adapted.setdefault("timing_validation", "legacy_unspecified")
    adapted.setdefault("decode_write_summary", {"enabled": False, "legacy_report": True})
    adapted["legacy"] = True
    adapted["legacy_schema_version"] = version
    adapted["recovery_metrics_available"] = False
    adapted["legacy_metrics"] = original
    return adapted
