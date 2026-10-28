"""SSD profile configuration for storage performance simulation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any


@dataclass(frozen=True)
class SSDProfile:
    """Parameterized single-SSD geometry and timing model."""

    name: str
    page_size_bytes: int
    pages_per_erase_block: int
    channel_count: int
    dies_per_channel: int
    planes_per_die: int
    queue_depth: int
    read_latency_us: float
    program_latency_us: float
    erase_latency_us: float
    sequential_read_bandwidth_GBps: float
    sequential_write_bandwidth_GBps: float
    random_read_iops: float
    random_write_iops: float
    overprovisioning_ratio: float
    gc_free_block_threshold: float
    write_amplification: float
    logical_capacity_bytes: int
    io_unit_bytes: int | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.page_size_bytes, "page_size_bytes")
        _require_positive_int(self.pages_per_erase_block, "pages_per_erase_block")
        _require_positive_int(self.channel_count, "channel_count")
        _require_positive_int(self.dies_per_channel, "dies_per_channel")
        _require_positive_int(self.planes_per_die, "planes_per_die")
        _require_positive_int(self.queue_depth, "queue_depth")
        _require_positive_float(self.read_latency_us, "read_latency_us")
        _require_positive_float(self.program_latency_us, "program_latency_us")
        _require_positive_float(self.erase_latency_us, "erase_latency_us")
        _require_positive_float(
            self.sequential_read_bandwidth_GBps, "sequential_read_bandwidth_GBps"
        )
        _require_positive_float(
            self.sequential_write_bandwidth_GBps, "sequential_write_bandwidth_GBps"
        )
        _require_positive_float(self.random_read_iops, "random_read_iops")
        _require_positive_float(self.random_write_iops, "random_write_iops")
        if not 0 <= self.overprovisioning_ratio < 1:
            raise ValueError("overprovisioning_ratio must be in [0, 1)")
        if not 0 <= self.gc_free_block_threshold < 1:
            raise ValueError("gc_free_block_threshold must be in [0, 1)")
        if self.write_amplification < 1:
            raise ValueError("write_amplification must be at least 1")
        _require_positive_int(self.logical_capacity_bytes, "logical_capacity_bytes")
        if self.io_unit_bytes is not None:
            _require_positive_int(self.io_unit_bytes, "io_unit_bytes")

    @property
    def parallel_units(self) -> int:
        return self.channel_count * self.dies_per_channel * self.planes_per_die

    @property
    def logical_page_count(self) -> int:
        return max(1, math.ceil(self.logical_capacity_bytes / self.page_size_bytes))

    @property
    def physical_page_count(self) -> int:
        usable_fraction = 1.0 - self.overprovisioning_ratio
        return max(
            self.logical_page_count,
            math.ceil(self.logical_page_count / usable_fraction),
        )

    @property
    def physical_block_count(self) -> int:
        return max(
            self.parallel_units,
            math.ceil(self.physical_page_count / self.pages_per_erase_block),
        )

    @property
    def physical_capacity_bytes(self) -> int:
        return (
            self.physical_block_count
            * self.pages_per_erase_block
            * self.page_size_bytes
        )

    def to_report(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "parallel_units": self.parallel_units,
                "logical_page_count": self.logical_page_count,
                "physical_block_count": self.physical_block_count,
                "physical_capacity_bytes": self.physical_capacity_bytes,
            }
        )
        return payload


def ssd_profile_from_dict(data: dict[str, Any]) -> SSDProfile:
    """Build an :class:`SSDProfile` from a YAML/JSON mapping."""

    if "storage" in data and isinstance(data["storage"], dict):
        data = data["storage"]
    return SSDProfile(
        name=str(data.get("name", "single-nvme-ftl")),
        page_size_bytes=_int_value(data, "page_size_bytes", 16 * 1024),
        pages_per_erase_block=_int_value(data, "pages_per_erase_block", 256),
        channel_count=_int_value(data, "channel_count", 8),
        dies_per_channel=_int_value(data, "dies_per_channel", 2),
        planes_per_die=_int_value(data, "planes_per_die", 1),
        queue_depth=_int_value(data, "queue_depth", 64),
        read_latency_us=_float_value(data, "read_latency_us", 85.0),
        program_latency_us=_float_value(data, "program_latency_us", 450.0),
        erase_latency_us=_float_value(data, "erase_latency_us", 3500.0),
        sequential_read_bandwidth_GBps=_float_value(
            data, "sequential_read_bandwidth_GBps", 7.0
        ),
        sequential_write_bandwidth_GBps=_float_value(
            data, "sequential_write_bandwidth_GBps", 5.0
        ),
        random_read_iops=_float_value(data, "random_read_iops", 1_000_000.0),
        random_write_iops=_float_value(data, "random_write_iops", 250_000.0),
        overprovisioning_ratio=_float_value(data, "overprovisioning_ratio", 0.20),
        gc_free_block_threshold=_float_value(data, "gc_free_block_threshold", 0.05),
        write_amplification=_float_value(data, "write_amplification", 1.0),
        logical_capacity_bytes=_int_value(data, "logical_capacity_bytes", 1 << 40),
        io_unit_bytes=(
            _int_value(data, "io_unit_bytes", 128 * 1024)
            if data.get("io_unit_bytes", 128 * 1024) is not None
            else None
        ),
    )


def _int_value(data: dict[str, Any], name: str, default: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _float_value(data: dict[str, Any], name: str, default: float) -> float:
    value = data.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_float(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
