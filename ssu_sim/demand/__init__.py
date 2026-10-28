"""SSD-independent prefix-cache demand analysis."""

from ssu_sim.demand.analyzer import (
    CapacitySummary,
    DemandAnalysis,
    DemandOp,
    TTFTJob,
    analyze_workload,
    analyze_bandwidth_envelope,
)
from ssu_sim.demand.schema import DEMAND_REPORT_SCHEMA_VERSION, adapt_demand_report
from ssu_sim.demand.miss_window import (
    INFINITY_LABEL,
    MISS_COMPUTE_WINDOW_METRIC,
    miss_compute_window_metrics,
)

__all__ = [
    "CapacitySummary",
    "DemandAnalysis",
    "DemandOp",
    "INFINITY_LABEL",
    "MISS_COMPUTE_WINDOW_METRIC",
    "TTFTJob",
    "analyze_workload",
    "analyze_bandwidth_envelope",
    "adapt_demand_report",
    "DEMAND_REPORT_SCHEMA_VERSION",
    "miss_compute_window_metrics",
]
