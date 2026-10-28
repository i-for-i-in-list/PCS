"""Synthetic LLM workload generation."""

from __future__ import annotations

from dataclasses import dataclass
import random

from ssu_sim.utils.random import weighted_choice, zipf_weights
from ssu_sim.workload.request import Request


@dataclass(frozen=True)
class SyntheticWorkloadConfig:
    request_count: int = 1000
    qps: float = 100.0
    model_id: str = "qwen-gqa-sample"
    tenant_count: int = 1
    seq_len: int = 8192
    generation_len: int = 256
    prefix_ratio: float = 0.5
    prefix_group_count: int = 100
    prefix_zipf_alpha: float = 1.1
    workload_shape: str = "shared_prefix_groups"
    arrival_model: str = "fixed"
    seed: int = 1
    token_hit_rate: float | None = None

    def __post_init__(self) -> None:
        if self.token_hit_rate is not None:
            object.__setattr__(self, "prefix_ratio", float(self.token_hit_rate))
        else:
            object.__setattr__(self, "token_hit_rate", self.prefix_ratio)


def generate_synthetic_workload(config: SyntheticWorkloadConfig) -> list[Request]:
    if config.request_count < 0:
        raise ValueError("request_count must be non-negative")
    if config.qps <= 0:
        raise ValueError("qps must be positive")
    if config.tenant_count <= 0:
        raise ValueError("tenant_count must be positive")
    if config.seq_len < 0 or config.generation_len < 0:
        raise ValueError("seq_len and generation_len must be non-negative")
    if config.prefix_group_count <= 0:
        raise ValueError("prefix_group_count must be positive")
    if not 0.0 <= config.prefix_ratio <= 1.0:
        raise ValueError("prefix_ratio must be in [0, 1]")
    if config.arrival_model not in {"fixed", "poisson"}:
        raise ValueError("arrival_model must be 'fixed' or 'poisson'")
    if config.workload_shape not in {
        "shared_prefix_groups",
        "prompt_lineage_bootstrap",
    }:
        raise ValueError(
            "workload_shape must be shared_prefix_groups or prompt_lineage_bootstrap"
        )

    rng = random.Random(config.seed)
    arrival_rng = random.Random(config.seed + 1)
    prefix_groups = [f"prefix-{idx}" for idx in range(config.prefix_group_count)]
    weights = zipf_weights(config.prefix_group_count, config.prefix_zipf_alpha)
    interval_us = 1_000_000.0 / config.qps
    requests: list[Request] = []
    arrival_time_us = 0.0
    matched_prefix_len = int(config.seq_len * config.prefix_ratio)

    parent_by_group: dict[str, str] = {}
    if (
        config.workload_shape == "prompt_lineage_bootstrap"
        and matched_prefix_len > 0
    ):
        for group_index, prefix_group_id in enumerate(prefix_groups):
            parent_id = f"bootstrap-{group_index}"
            parent_by_group[prefix_group_id] = parent_id
            requests.append(
                Request(
                    request_id=parent_id,
                    arrival_time_us=float(group_index - len(prefix_groups)),
                    model_id=config.model_id,
                    tenant_id="bootstrap",
                    seq_len=matched_prefix_len,
                    generation_len=0,
                    prefix_group_id=prefix_group_id,
                    matched_prefix_len=0,
                    conversation_id=f"conversation:{prefix_group_id}",
                    is_bootstrap=True,
                )
            )

    for idx in range(config.request_count):
        seq_len = config.seq_len
        prefix_group_id = (
            weighted_choice(prefix_groups, weights, rng)
            if matched_prefix_len > 0
            else f"cold-{idx}"
        )
        tenant_id = f"tenant-{idx % config.tenant_count}"
        if idx > 0:
            if config.arrival_model == "fixed":
                arrival_time_us += interval_us
            else:
                arrival_time_us += arrival_rng.expovariate(config.qps) * 1_000_000.0
        requests.append(
            Request(
                request_id=f"req-{idx}",
                arrival_time_us=arrival_time_us,
                model_id=config.model_id,
                tenant_id=tenant_id,
                seq_len=seq_len,
                generation_len=config.generation_len,
                prefix_group_id=prefix_group_id,
                matched_prefix_len=matched_prefix_len,
                conversation_id=(
                    f"conversation:{prefix_group_id}"
                    if prefix_group_id in parent_by_group
                    else None
                ),
                parent_request_id=parent_by_group.get(prefix_group_id),
            )
        )
    return requests
