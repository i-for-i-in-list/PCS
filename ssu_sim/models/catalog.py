"""Offline canonical model-catalog discovery and identity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ssu_sim.models.metadata import CatalogMetadata, ModelIdentity, catalog_from_dict, identity_from_dict
from ssu_sim.utils.config import load_config


EXPECTED_ACTIVE_FORMULA_COUNTS = {
    "standard_gqa": 22,
    "mla": 4,
    "dsa_mla": 4,
    "kimi_kda_mla_hybrid": 1,
    "inkling_hybrid": 2,
    "qwen_linear_full_hybrid": 10,
    "mixed_full_sliding_gqa": 9,
    "minimax_msa": 1,
    "deepseek_v4_hybrid": 2,
}


@dataclass(frozen=True)
class ModelCatalogEntry:
    path: Path
    identity: ModelIdentity
    catalog: CatalogMetadata
    formula_id: str
    config_schema_version: int
    raw: dict[str, Any]

    @property
    def active(self) -> bool:
        return self.catalog.active


def discover_model_catalog(config_dir: Path) -> tuple[ModelCatalogEntry, ...]:
    entries: list[ModelCatalogEntry] = []
    for path in sorted(config_dir.glob("*.y*ml")):
        if path.name.startswith("."):
            continue
        raw = load_config(path)
        identity = identity_from_dict(raw)
        catalog = catalog_from_dict(raw.get("catalog"))
        entries.append(
            ModelCatalogEntry(
                path=path,
                identity=identity,
                catalog=catalog,
                formula_id=_formula_id(raw),
                config_schema_version=int(raw.get("model_config_schema_version", 1)),
                raw=raw,
            )
        )
    _validate_unique_identities(entries)
    return tuple(entries)


def active_catalog(entries: Iterable[ModelCatalogEntry]) -> tuple[ModelCatalogEntry, ...]:
    return tuple(item for item in entries if item.active)


def legacy_catalog(entries: Iterable[ModelCatalogEntry]) -> tuple[ModelCatalogEntry, ...]:
    return tuple(item for item in entries if not item.active and item.catalog.legacy_visible)


def validate_canonical_catalog(
    entries: Iterable[ModelCatalogEntry],
    *,
    expected_active_count: int | None = None,
    expected_legacy_count: int | None = None,
) -> dict[str, int]:
    entries = tuple(entries)
    active = active_catalog(entries)
    legacy = legacy_catalog(entries)
    if expected_active_count is not None and len(active) != expected_active_count:
        raise ValueError(
            f"active model catalog must contain {expected_active_count}, found {len(active)}"
        )
    if expected_legacy_count is not None and len(legacy) != expected_legacy_count:
        raise ValueError(
            f"legacy model catalog must contain {expected_legacy_count}, found {len(legacy)}"
        )
    counts: dict[str, int] = {}
    _validate_unique_identities(entries)
    from ssu_sim.models.model_registry import ATTENTION_PROFILE_FACTORIES
    for item in active:
        if item.raw.get("attention", {}).get("type") not in ATTENTION_PROFILE_FACTORIES:
            raise ValueError(f"unregistered attention type for {item.identity.id}")
        if not item.formula_id:
            raise ValueError(f"missing formula for {item.identity.id}")
        counts[item.formula_id] = counts.get(item.formula_id, 0) + 1
    return counts


def catalog_alias_map(entries: Iterable[ModelCatalogEntry]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in entries:
        for alias in item.identity.aliases:
            existing = aliases.get(alias)
            if existing is not None and existing != item.identity.id:
                raise ValueError(f"model alias {alias!r} resolves to multiple canonical ids")
            aliases[alias] = item.identity.id
    return aliases


def _formula_id(raw: dict[str, Any]) -> str:
    sources = raw.get("sources")
    if isinstance(sources, dict):
        calculator = sources.get("calculator")
        if isinstance(calculator, dict) and calculator.get("formula_id"):
            return str(calculator["formula_id"])
    calculator = raw.get("calculator")
    if isinstance(calculator, dict) and calculator.get("formula_id"):
        return str(calculator["formula_id"])
    attention = raw.get("attention")
    kind = str(attention.get("type", "")) if isinstance(attention, dict) else ""
    return {
        "mha": "standard_gqa",
        "mqa": "standard_gqa",
        "gqa": "standard_gqa",
        "mla": "mla",
        "dsa": "dsa_mla",
        "kda_mla": "kimi_kda_mla_hybrid",
        "inkling": "inkling_hybrid",
        "qwen_linear_full": "qwen_linear_full_hybrid",
        "mixed_full_swa": "mixed_full_sliding_gqa",
        "minimax_msa": "minimax_msa",
        "deepseek_v4": "deepseek_v4_hybrid",
        "deepseek_v41": "deepseek_v41_csa2",
    }.get(kind, kind)


def _validate_unique_identities(entries: Iterable[ModelCatalogEntry]) -> None:
    ids: set[str] = set()
    aliases: set[str] = set()
    for item in entries:
        if item.identity.id in ids:
            raise ValueError(f"duplicate canonical model id: {item.identity.id}")
        if item.identity.id in aliases:
            raise ValueError(f"canonical model id conflicts with alias: {item.identity.id}")
        ids.add(item.identity.id)
        for alias in item.identity.aliases:
            if alias in ids or alias in aliases:
                raise ValueError(f"duplicate or conflicting model alias: {alias}")
            aliases.add(alias)
