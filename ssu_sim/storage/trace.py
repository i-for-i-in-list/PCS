"""Lower logical demand operations into physical SSD IOs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

from ssu_sim.demand import DemandOp
from ssu_sim.storage.profile import SSDProfile


@dataclass(frozen=True)
class PhysicalIO:
    """One physical IO submitted to the single-SSD simulator."""

    time_us: float
    request_id: str
    model_id: str
    op_type: str
    lba: int
    size_bytes: int
    component: str
    access_pattern: str
    source_block_id: str
    source_operation_weight: float = 1.0
    deadline_us: float | None = None
    allocation_table_sha256: str | None = None


@dataclass(frozen=True)
class ExtentAllocation:
    """A collision-free logical extent assigned to one serialized object."""

    lba: int
    page_count: int
    serialized_size_bytes: int


def build_extent_allocation_table(
    ops: Iterable[DemandOp],
    profile: SSDProfile,
    *,
    allocation_seed: int = 0,
) -> dict[str, ExtentAllocation]:
    """Assign stable, non-overlapping extents to all logical cache objects.

    Allocation is deliberately two-pass: the largest serialized extent for
    every object is validated first, then objects are hash-ordered and packed
    contiguously.  This avoids the collisions caused by independently taking
    each object's hash modulo the LBA space.
    """

    sizes: dict[str, int] = {}
    for op in ops:
        if op.payload_bytes <= 0:
            raise ValueError(f"{op.block_id} payload_bytes must be positive")
        previous = sizes.get(op.block_id)
        if previous is not None and previous != op.payload_bytes:
            raise ValueError(
                "serialized size changed for object "
                f"{op.block_id}: {previous} != {op.payload_bytes}; "
                "partial rewrites must use a new object version"
            )
        sizes[op.block_id] = op.payload_bytes

    def allocation_key(block_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(
            f"{allocation_seed}:{block_id}".encode("utf-8")
        ).digest()
        return digest, block_id

    cursor = 0
    result: dict[str, ExtentAllocation] = {}
    for block_id in sorted(sizes, key=allocation_key):
        size_bytes = sizes[block_id]
        page_count = math.ceil(size_bytes / profile.page_size_bytes)
        if cursor + page_count > profile.logical_page_count:
            required = cursor + page_count
            raise ValueError(
                "logical cache objects exceed SSD capacity: "
                f"required_pages={required}, available_pages={profile.logical_page_count}"
            )
        result[block_id] = ExtentAllocation(
            lba=cursor,
            page_count=page_count,
            serialized_size_bytes=size_bytes,
        )
        cursor += page_count
    return result


def allocation_table_sha256(table: dict[str, ExtentAllocation]) -> str:
    """Return a stable digest suitable for trace/report provenance."""

    digest = hashlib.sha256()
    for block_id, extent in sorted(table.items()):
        digest.update(
            (
                f"{block_id}|{extent.lba}|{extent.page_count}|"
                f"{extent.serialized_size_bytes}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def lower_demand_ops(
    ops: list[DemandOp],
    profile: SSDProfile,
    *,
    max_io_bytes: int | None = None,
    allocation_seed: int = 0,
) -> list[PhysicalIO]:
    """Split concrete logical operations into deterministic physical IOs.

    Fractional ``operation_weight`` values are analytical expectations, not
    physical requests.  Callers must first realize a concrete sparse scenario;
    silently shrinking an IO would create a block that does not exist in the
    serialized layout.
    """

    chunk_bytes = _chunk_bytes(profile, max_io_bytes)
    allocation_table = build_extent_allocation_table(
        ops,
        profile,
        allocation_seed=allocation_seed,
    )
    table_digest = allocation_table_sha256(allocation_table)
    physical: list[PhysicalIO] = []
    for op in ops:
        if op.op_type not in {"read", "write"}:
            raise ValueError(f"unsupported op_type for storage simulation: {op.op_type}")
        if op.operation_weight <= 0:
            continue
        if not float(op.operation_weight).is_integer():
            raise ValueError(
                "fractional analytical operation cannot be lowered to physical IO: "
                f"block_id={op.block_id}, operation_weight={op.operation_weight}; "
                "select a concrete sparse access scenario first"
            )
        repetitions = int(op.operation_weight)
        effective_bytes = op.payload_bytes
        extent = allocation_table[op.block_id]
        base_lba = extent.lba
        deadline_us = (
            op.time_us + op.latency_budget_us
            if op.latency_budget_us is not None
            else None
        )
        for repetition in range(repetitions):
            offset = 0
            chunk_index = 0
            while offset < effective_bytes:
                size_bytes = min(chunk_bytes, effective_bytes - offset)
                page_offset = offset // profile.page_size_bytes
                physical.append(
                    PhysicalIO(
                        time_us=op.time_us,
                        request_id=op.request_id,
                        model_id=op.model_id,
                        op_type=op.op_type,
                        lba=base_lba + page_offset,
                        size_bytes=size_bytes,
                        component=op.component,
                        access_pattern=op.access_pattern,
                        source_block_id=(
                            f"{op.block_id}#{repetition}:{chunk_index}"
                            if repetitions > 1
                            else f"{op.block_id}#{chunk_index}"
                        ),
                        source_operation_weight=1.0,
                        deadline_us=deadline_us,
                        allocation_table_sha256=table_digest,
                    )
                )
                offset += size_bytes
                chunk_index += 1
    return sorted(
        physical,
        key=lambda item: (item.time_us, item.request_id, item.op_type, item.lba),
    )


def _chunk_bytes(profile: SSDProfile, max_io_bytes: int | None) -> int:
    candidates = [
        value
        for value in (max_io_bytes, profile.io_unit_bytes)
        if value is not None
    ]
    chunk = min(int(value) for value in candidates) if candidates else profile.page_size_bytes
    if chunk <= 0:
        raise ValueError("storage IO chunk size must be positive")
    if chunk <= profile.page_size_bytes:
        return profile.page_size_bytes
    return (chunk // profile.page_size_bytes) * profile.page_size_bytes
