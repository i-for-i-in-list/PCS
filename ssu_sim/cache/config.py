"""Configuration for capacity-aware prefix-cache replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CacheReplayConfig:
    """Byte-capacity configuration for block-radix leaf-LRU replay."""

    name: str = "radix-leaf-lru"
    policy: str = "radix_leaf_lru"
    total_capacity_bytes: int = 0
    shard_count: int = 1
    placement_seed: int = 0
    warmup_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cache replay name must not be empty")
        if self.policy != "radix_leaf_lru":
            raise ValueError("cache replay policy must be 'radix_leaf_lru'")
        if (
            isinstance(self.total_capacity_bytes, bool)
            or not isinstance(self.total_capacity_bytes, int)
            or self.total_capacity_bytes < 0
        ):
            raise ValueError("total_capacity_bytes must be a non-negative integer")
        if (
            isinstance(self.shard_count, bool)
            or not isinstance(self.shard_count, int)
            or self.shard_count <= 0
        ):
            raise ValueError("shard_count must be a positive integer")
        if isinstance(self.placement_seed, bool) or not isinstance(self.placement_seed, int):
            raise ValueError("placement_seed must be an integer")
        if isinstance(self.warmup_fraction, bool) or not isinstance(
            self.warmup_fraction, (int, float)
        ):
            raise ValueError("warmup_fraction must be a number")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")

    @property
    def capacity_bytes_by_shard(self) -> tuple[int, ...]:
        """Split total capacity exactly and deterministically across shards."""

        base, remainder = divmod(self.total_capacity_bytes, self.shard_count)
        return tuple(base + (1 if shard < remainder else 0) for shard in range(self.shard_count))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["capacity_bytes_by_shard"] = list(self.capacity_bytes_by_shard)
        return payload


def cache_replay_config_from_dict(data: dict[str, Any]) -> CacheReplayConfig:
    """Build :class:`CacheReplayConfig` from a direct or nested mapping."""

    if "cache_replay" in data and isinstance(data["cache_replay"], dict):
        values = dict(data["cache_replay"])
    elif "cache" in data and isinstance(data["cache"], dict):
        values = dict(data["cache"])
    else:
        values = dict(data)
    return CacheReplayConfig(
        name=str(values.get("name", "radix-leaf-lru")),
        policy=str(values.get("policy", "radix_leaf_lru")),
        total_capacity_bytes=_integer(
            values.get("total_capacity_bytes", 0), "total_capacity_bytes"
        ),
        shard_count=_integer(values.get("shard_count", 1), "shard_count"),
        placement_seed=_integer(values.get("placement_seed", 0), "placement_seed"),
        warmup_fraction=_number(values.get("warmup_fraction", 0.0), "warmup_fraction"),
    )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return converted


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
