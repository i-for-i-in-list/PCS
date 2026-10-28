"""Stable model identity, catalog metadata, and per-metric evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping


EVIDENCE_STATUSES = frozenset(
    {
        "official",
        "derived_from_official_code",
        "calculator_reference",
        "analytical",
        "synthetic_scenario",
        "scenario_estimate",
        "assumed",
        "uncalibrated",
        "unsupported",
        "upper_bound",
        "lower_bound",
    }
)


@dataclass(frozen=True)
class ModelIdentity:
    id: str
    display_name: str
    family: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.display_name or not self.family:
            raise ValueError("model identity id, display_name, and family are required")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("model identity aliases must be unique")
        if self.id in self.aliases:
            raise ValueError("canonical model id must not also be an alias")


@dataclass(frozen=True)
class EvidenceRecord:
    status: str
    sources: tuple[str, ...] = ()
    assumptions: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    geometry_status: str | None = None
    operator_coverage: str | None = None
    math_oracle_status: str | None = None

    def __post_init__(self) -> None:
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(
                f"unsupported evidence status {self.status!r}; "
                f"expected one of {sorted(EVIDENCE_STATUSES)}"
            )
        if self.status == "assumed" and not self.assumptions:
            raise ValueError("assumed evidence must list explicit assumptions")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "sources": list(self.sources),
            "assumptions": [dict(item) for item in self.assumptions],
            "notes": list(self.notes),
        }
        for key in ("geometry_status", "operator_coverage", "math_oracle_status"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


def _unsupported() -> EvidenceRecord:
    return EvidenceRecord("unsupported")


@dataclass(frozen=True)
class ModelEvidence:
    calculator_capacity: EvidenceRecord = field(default_factory=_unsupported)
    active_state_layout: EvidenceRecord = field(default_factory=_unsupported)
    prefix_serialization: EvidenceRecord = field(default_factory=_unsupported)
    compute_formula: EvidenceRecord = field(default_factory=_unsupported)
    sparse_routing: EvidenceRecord = field(default_factory=_unsupported)
    timing_calibration: EvidenceRecord = field(default_factory=_unsupported)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name).to_dict()
            for name in (
                "calculator_capacity",
                "active_state_layout",
                "prefix_serialization",
                "compute_formula",
                "sparse_routing",
                "timing_calibration",
            )
        }


@dataclass(frozen=True)
class CatalogMetadata:
    visible: bool = True
    legacy_visible: bool = False
    deprecated_since: str | None = None
    replacement_id: str | None = None

    @property
    def active(self) -> bool:
        return self.visible and self.deprecated_since is None


def evidence_from_dict(data: Any) -> ModelEvidence:
    values = data if isinstance(data, dict) else {}

    def record(name: str) -> EvidenceRecord:
        raw = values.get(name, {})
        if not isinstance(raw, dict):
            raise ValueError(f"evidence.{name} must be a mapping")
        return EvidenceRecord(
            status=str(raw.get("status", "unsupported")),
            sources=tuple(str(value) for value in raw.get("sources", ())),
            assumptions=tuple(
                dict(value) for value in raw.get("assumptions", ())
            ),
            notes=tuple(str(value) for value in raw.get("notes", ())),
            geometry_status=raw.get("geometry_status"),
            operator_coverage=raw.get("operator_coverage"),
            math_oracle_status=raw.get("math_oracle_status"),
        )

    return ModelEvidence(
        calculator_capacity=record("calculator_capacity"),
        active_state_layout=record("active_state_layout"),
        prefix_serialization=record("prefix_serialization"),
        compute_formula=record("compute_formula"),
        sparse_routing=record("sparse_routing"),
        timing_calibration=record("timing_calibration"),
    )


def identity_from_dict(data: Mapping[str, Any]) -> ModelIdentity:
    raw = data.get("identity")
    if raw is None:
        legacy_name = str(data.get("name", ""))
        if not legacy_name:
            raise ValueError("model name or identity is required")
        return ModelIdentity(
            id=legacy_name,
            display_name=legacy_name,
            family="legacy",
        )
    if not isinstance(raw, dict):
        raise ValueError("model identity must be a mapping")
    return ModelIdentity(
        id=str(raw.get("id", "")),
        display_name=str(raw.get("display_name", "")),
        family=str(raw.get("family", "")),
        aliases=tuple(str(value) for value in raw.get("aliases", ())),
    )


def catalog_from_dict(data: Any) -> CatalogMetadata:
    values = data if isinstance(data, dict) else {}
    return CatalogMetadata(
        visible=bool(values.get("visible", True)),
        legacy_visible=bool(values.get("legacy_visible", False)),
        deprecated_since=(
            str(values["deprecated_since"])
            if values.get("deprecated_since") is not None
            else None
        ),
        replacement_id=(
            str(values["replacement_id"])
            if values.get("replacement_id") is not None
            else None
        ),
    )


def canonical_sha256(data: Mapping[str, Any]) -> str:
    """Digest normalized config content independently of YAML formatting."""

    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_to_dict(value: Any) -> dict[str, Any]:
    """JSON-safe helper for report adapters."""

    if hasattr(value, "to_dict"):
        return value.to_dict()
    return asdict(value)
