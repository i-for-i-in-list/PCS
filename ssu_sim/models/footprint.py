"""Cache-footprint result types shared by attention profiles.

The three top-level fields deliberately describe different semantic scopes.
They are not interchangeable byte counters: ``calculator_reference`` captures
the pinned external calculator convention, ``active_execution`` describes the
state required by a live sequence, and ``prefix_store`` describes persisted
objects used for prefix reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


CALCULATOR_SEMANTIC_CLASSES = frozenset(
    {
        "token_payload",
        "resident_window",
        "active_sequence_state",
        "retained_checkpoints",
        "hybrid",
    }
)


@dataclass(frozen=True)
class ComponentFootprint:
    """Nominal, serialized, and reserved bytes for one named component."""

    nominal_payload_bytes: int
    serialized_bytes: int | None
    reserved_bytes: int

    def __post_init__(self) -> None:
        _validate_non_negative(self.nominal_payload_bytes, "nominal_payload_bytes")
        _validate_optional_non_negative(self.serialized_bytes, "serialized_bytes")
        _validate_non_negative(self.reserved_bytes, "reserved_bytes")


@dataclass(frozen=True)
class CalculatorReferenceFootprint:
    """A reproducible value calculated with a pinned external convention."""

    total_bytes: int
    formula_id: str
    settings: Mapping[str, object] = field(default_factory=dict)
    snapshot_sha256: str = "unverified"
    semantic_class: str = "token_payload"
    by_component: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative(self.total_bytes, "total_bytes")
        _validate_non_empty(self.formula_id, "formula_id")
        _validate_non_empty(self.snapshot_sha256, "snapshot_sha256")
        _validate_non_empty(self.semantic_class, "semantic_class")
        if self.semantic_class not in CALCULATOR_SEMANTIC_CLASSES:
            raise ValueError(
                "semantic_class must describe the calculator's actual capacity scope"
            )
        _validate_component_bytes(self.by_component)


@dataclass(frozen=True)
class ActiveExecutionFootprint:
    """State used by active execution for the queried sequence population."""

    nominal_payload_bytes: int
    serialized_bytes: int | None
    reserved_bytes: int
    by_component: Mapping[str, ComponentFootprint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative(self.nominal_payload_bytes, "nominal_payload_bytes")
        _validate_optional_non_negative(self.serialized_bytes, "serialized_bytes")
        _validate_non_negative(self.reserved_bytes, "reserved_bytes")
        _validate_named_components(self.by_component, ComponentFootprint)


@dataclass(frozen=True)
class PrefixStoreFootprint:
    """Serialized prefix objects and checkpoint snapshots persisted for reuse."""

    nominal_payload_bytes: int
    serialized_bytes: int | None
    object_count: int
    checkpoint_count: int
    by_component: Mapping[str, ComponentFootprint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative(self.nominal_payload_bytes, "nominal_payload_bytes")
        _validate_optional_non_negative(self.serialized_bytes, "serialized_bytes")
        _validate_non_negative(self.object_count, "object_count")
        _validate_non_negative(self.checkpoint_count, "checkpoint_count")
        if self.checkpoint_count > self.object_count:
            raise ValueError("checkpoint_count must not exceed object_count")
        _validate_named_components(self.by_component, ComponentFootprint)


@dataclass(frozen=True)
class CacheFootprintReport:
    """The three non-interchangeable cache-footprint scopes."""

    calculator_reference: CalculatorReferenceFootprint
    active_execution: ActiveExecutionFootprint
    prefix_store: PrefixStoreFootprint


def _validate_non_negative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")


def _validate_optional_non_negative(value: int | None, name: str) -> None:
    if value is not None:
        _validate_non_negative(value, name)


def _validate_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_component_bytes(values: Mapping[str, int]) -> None:
    for name, value in values.items():
        _validate_non_empty(name, "component name")
        _validate_non_negative(value, f"component {name} bytes")


def _validate_named_components(values: Mapping[str, object], expected_type: type) -> None:
    for name, value in values.items():
        _validate_non_empty(name, "component name")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"component {name} must be {expected_type.__name__}, "
                f"found {type(value).__name__}"
            )
