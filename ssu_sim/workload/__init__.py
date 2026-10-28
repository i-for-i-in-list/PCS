"""Workload models and generators."""

from ssu_sim.workload.request import Request
from ssu_sim.workload.synthetic import SyntheticWorkloadConfig, generate_synthetic_workload

__all__ = ["Request", "SyntheticWorkloadConfig", "generate_synthetic_workload"]

