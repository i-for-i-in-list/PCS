"""Single-SSD storage simulation for logical prefix-cache demand traces."""

from ssu_sim.storage.profile import SSDProfile, ssd_profile_from_dict
from ssu_sim.storage.simulator import CompletedIO, SSDResult, simulate_ssd
from ssu_sim.storage.trace import (
    ExtentAllocation,
    PhysicalIO,
    allocation_table_sha256,
    build_extent_allocation_table,
    lower_demand_ops,
)

__all__ = [
    "CompletedIO",
    "ExtentAllocation",
    "PhysicalIO",
    "SSDProfile",
    "SSDResult",
    "allocation_table_sha256",
    "build_extent_allocation_table",
    "lower_demand_ops",
    "simulate_ssd",
    "ssd_profile_from_dict",
]
