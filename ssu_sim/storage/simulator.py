"""Single-SSD FTL and queueing simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from ssu_sim.demand import DemandOp, TTFTJob
from ssu_sim.models.state import phase_chain_time_us
from ssu_sim.models.execution import require_evaluated_full_hit
from ssu_sim.storage.profile import SSDProfile
from ssu_sim.storage.trace import PhysicalIO, lower_demand_ops


@dataclass(frozen=True)
class CompletedIO:
    """A physical IO with simulated timing."""

    io: PhysicalIO
    start_time_us: float
    completion_time_us: float
    queue_wait_us: float
    service_time_us: float
    gc_penalty_us: float
    parallel_unit: int

    @property
    def latency_us(self) -> float:
        return self.completion_time_us - self.io.time_us


@dataclass(frozen=True)
class SSDResult:
    physical_ios: list[PhysicalIO]
    completed_ios: list[CompletedIO]
    report: dict[str, Any]


@dataclass
class _BlockState:
    block_id: int
    next_free_page: int = 0
    valid_pages: int = 0
    invalid_pages: int = 0
    pages: dict[int, int] = field(default_factory=dict)

    def reset_erased(self) -> None:
        self.next_free_page = 0
        self.valid_pages = 0
        self.invalid_pages = 0
        self.pages.clear()


@dataclass(frozen=True)
class _FTLOpResult:
    parallel_unit: int
    gc_penalty_us: float
    background_program_us: float = 0.0


class _FTLState:
    def __init__(self, profile: SSDProfile) -> None:
        self.profile = profile
        self.blocks = [
            _BlockState(block_id) for block_id in range(profile.physical_block_count)
        ]
        self.erased_blocks_by_unit: list[list[int]] = [
            [] for _ in range(profile.parallel_units)
        ]
        for block_id in reversed(range(profile.physical_block_count)):
            self.erased_blocks_by_unit[block_id % profile.parallel_units].append(block_id)
        self.open_blocks: list[int | None] = [None] * profile.parallel_units
        self.l2p: dict[int, int] = {}
        self.host_write_pages = 0
        self.nand_write_pages = 0
        self.preloaded_pages = 0
        self.gc_copied_pages = 0
        self.gc_count = 0
        self.erase_count = 0

    def read_pages(self, lba: int, page_count: int) -> _FTLOpResult:
        penalty = 0.0
        first_ppn: int | None = None
        for page_lba in self._page_range(lba, page_count):
            ppn = self.l2p.get(page_lba)
            if ppn is None:
                ppn, page_penalty = self._allocate_with_gc(page_lba)
                penalty += page_penalty
                self.l2p[page_lba] = ppn
                self.preloaded_pages += 1
            if first_ppn is None:
                first_ppn = ppn
        return _FTLOpResult(
            parallel_unit=self._parallel_unit(first_ppn or 0),
            gc_penalty_us=penalty,
        )

    def write_pages(self, lba: int, page_count: int) -> _FTLOpResult:
        penalty = 0.0
        first_ppn: int | None = None
        for page_lba in self._page_range(lba, page_count):
            old_ppn = self.l2p.get(page_lba)
            if old_ppn is not None:
                self._invalidate(old_ppn)
            ppn, page_penalty = self._allocate_with_gc(page_lba)
            penalty += page_penalty
            self.l2p[page_lba] = ppn
            self.host_write_pages += 1
            self.nand_write_pages += 1
            if first_ppn is None:
                first_ppn = ppn

        extra_pages = max(
            0,
            math.ceil(page_count * (self.profile.write_amplification - 1.0)),
        )
        self.nand_write_pages += extra_pages
        background_program_us = (
            extra_pages * self.profile.program_latency_us / self.profile.parallel_units
        )
        return _FTLOpResult(
            parallel_unit=self._parallel_unit(first_ppn or 0),
            gc_penalty_us=penalty,
            background_program_us=background_program_us,
        )

    @property
    def free_page_count(self) -> int:
        free_pages = sum(
            len(items) * self.profile.pages_per_erase_block
            for items in self.erased_blocks_by_unit
        )
        for block_id in self.open_blocks:
            if block_id is None:
                continue
            state = self.blocks[block_id]
            free_pages += self.profile.pages_per_erase_block - state.next_free_page
        return free_pages

    @property
    def free_page_ratio(self) -> float:
        return self.free_page_count / self.total_physical_pages

    @property
    def total_physical_pages(self) -> int:
        return self.profile.physical_block_count * self.profile.pages_per_erase_block

    def summary(self) -> dict[str, Any]:
        write_amplification = (
            self.nand_write_pages / self.host_write_pages
            if self.host_write_pages
            else 0.0
        )
        return {
            "mapped_logical_pages": len(self.l2p),
            "preloaded_pages": self.preloaded_pages,
            "host_write_pages": self.host_write_pages,
            "nand_write_pages": self.nand_write_pages,
            "gc_copied_pages": self.gc_copied_pages,
            "gc_count": self.gc_count,
            "erase_count": self.erase_count,
            "write_amplification": write_amplification,
            "free_page_ratio": self.free_page_ratio,
            "utilization": len(self.l2p) / self.total_physical_pages,
        }

    def _page_range(self, lba: int, page_count: int) -> range:
        start = lba % self.profile.logical_page_count
        end = min(start + page_count, self.profile.logical_page_count)
        return range(start, end)

    def _allocate_with_gc(self, lba: int) -> tuple[int, float]:
        penalty = 0.0
        if self.free_page_ratio <= self.profile.gc_free_block_threshold:
            penalty += self._garbage_collect()
        ppn = self._allocate_free_page(lba % self.profile.parallel_units, lba)
        if ppn is None:
            penalty += self._garbage_collect()
            ppn = self._allocate_free_page(lba % self.profile.parallel_units, lba)
        if ppn is None:
            raise RuntimeError("SSD model is out of free physical pages")
        return ppn, penalty

    def _allocate_free_page(self, preferred_unit: int, lba: int) -> int | None:
        for unit in self._unit_order(preferred_unit):
            ppn = self._allocate_from_unit(unit, lba)
            if ppn is not None:
                return ppn
        return None

    def _allocate_from_unit(self, unit: int, lba: int) -> int | None:
        block_id = self.open_blocks[unit]
        if block_id is not None:
            state = self.blocks[block_id]
            if state.next_free_page >= self.profile.pages_per_erase_block:
                self.open_blocks[unit] = None
            else:
                return self._claim_page(state, lba)

        if not self.erased_blocks_by_unit[unit]:
            return None
        block_id = self.erased_blocks_by_unit[unit].pop()
        self.open_blocks[unit] = block_id
        return self._claim_page(self.blocks[block_id], lba)

    def _claim_page(self, state: _BlockState, lba: int) -> int:
        offset = state.next_free_page
        state.next_free_page += 1
        state.valid_pages += 1
        state.pages[offset] = lba
        return state.block_id * self.profile.pages_per_erase_block + offset

    def _invalidate(self, ppn: int) -> None:
        block_id = ppn // self.profile.pages_per_erase_block
        offset = ppn % self.profile.pages_per_erase_block
        state = self.blocks[block_id]
        if offset in state.pages:
            del state.pages[offset]
            state.valid_pages -= 1
            state.invalid_pages += 1

    def _garbage_collect(self) -> float:
        victim = self._select_gc_victim()
        if victim is None:
            return 0.0

        state = self.blocks[victim]
        live_lbas = list(state.pages.values())
        moved = 0
        for lba in live_lbas:
            ppn = self._allocate_free_page(lba % self.profile.parallel_units, lba)
            if ppn is None:
                break
            self.l2p[lba] = ppn
            self.nand_write_pages += 1
            moved += 1

        state.reset_erased()
        self.erased_blocks_by_unit[victim % self.profile.parallel_units].append(victim)
        self.gc_count += 1
        self.erase_count += 1
        self.gc_copied_pages += moved
        return self.profile.erase_latency_us + moved * self.profile.program_latency_us

    def _select_gc_victim(self) -> int | None:
        open_block_ids = {block_id for block_id in self.open_blocks if block_id is not None}
        candidates = [
            state
            for state in self.blocks
            if state.block_id not in open_block_ids
            and (state.valid_pages + state.invalid_pages) > 0
        ]
        if not candidates:
            return None
        victim = max(
            candidates,
            key=lambda state: (state.invalid_pages, -state.valid_pages),
        )
        return victim.block_id

    def _parallel_unit(self, ppn: int) -> int:
        block_id = ppn // self.profile.pages_per_erase_block
        return block_id % self.profile.parallel_units

    def _unit_order(self, preferred_unit: int) -> list[int]:
        return [
            (preferred_unit + offset) % self.profile.parallel_units
            for offset in range(self.profile.parallel_units)
        ]


def simulate_ssd(
    ops: Sequence[DemandOp] | Sequence[PhysicalIO],
    profile: SSDProfile,
    *,
    max_io_bytes: int | None = None,
    ttft_jobs: Sequence[TTFTJob] | None = None,
) -> SSDResult:
    """Simulate a logical demand trace or a pre-lowered physical IO trace."""

    for job in ttft_jobs or ():
        require_evaluated_full_hit(job, model_id=job.model_id, request_id=job.request_id)
    if not ops:
        physical_ios: list[PhysicalIO] = []
    elif isinstance(ops[0], PhysicalIO):
        physical_ios = list(ops)  # type: ignore[list-item]
    else:
        physical_ios = lower_demand_ops(list(ops), profile, max_io_bytes=max_io_bytes)  # type: ignore[arg-type]

    ftl = _FTLState(profile)
    host_slots = [0.0] * profile.queue_depth
    unit_available = [0.0] * profile.parallel_units
    bandwidth_available = {"read": 0.0, "write": 0.0}
    random_iops_available = {"read": 0.0, "write": 0.0}
    completed: list[CompletedIO] = []
    max_observed_inflight = 0

    for io in sorted(physical_ios, key=lambda item: (item.time_us, item.request_id, item.lba)):
        page_count = math.ceil(io.size_bytes / profile.page_size_bytes)
        if io.op_type == "read":
            ftl_result = ftl.read_pages(io.lba, page_count)
        elif io.op_type == "write":
            ftl_result = ftl.write_pages(io.lba, page_count)
        else:
            raise ValueError(f"unsupported op_type for SSD simulation: {io.op_type}")

        max_observed_inflight = max(
            max_observed_inflight,
            sum(available > io.time_us for available in host_slots),
        )
        slot_index = min(range(profile.queue_depth), key=lambda index: host_slots[index])
        submit_time_us = max(io.time_us, host_slots[slot_index])
        if io.access_pattern == "random":
            submit_time_us = max(submit_time_us, random_iops_available[io.op_type])
            random_iops_available[io.op_type] = submit_time_us + _random_iops_gap_us(
                io, profile
            )
        unit = ftl_result.parallel_unit
        start_time_us = max(submit_time_us, unit_available[unit])
        latency_floor_us = (
            _latency_floor_us(io, profile)
            + ftl_result.gc_penalty_us
            + ftl_result.background_program_us
        )
        bandwidth_start_us = max(start_time_us, bandwidth_available[io.op_type])
        bandwidth_complete_us = bandwidth_start_us + _transfer_time_us(io, profile)
        completion_time_us = max(start_time_us + latency_floor_us, bandwidth_complete_us)
        bandwidth_available[io.op_type] = bandwidth_complete_us
        host_slots[slot_index] = completion_time_us
        unit_available[unit] = completion_time_us
        completed.append(
            CompletedIO(
                io=io,
                start_time_us=start_time_us,
                completion_time_us=completion_time_us,
                queue_wait_us=start_time_us - io.time_us,
                service_time_us=completion_time_us - start_time_us,
                gc_penalty_us=ftl_result.gc_penalty_us,
                parallel_unit=unit,
            )
        )

    report = _build_report(
        profile,
        physical_ios,
        completed,
        ftl,
        max_observed_inflight,
        ttft_jobs=ttft_jobs,
    )
    return SSDResult(physical_ios=physical_ios, completed_ios=completed, report=report)


def _latency_floor_us(io: PhysicalIO, profile: SSDProfile) -> float:
    if io.op_type == "read":
        base = profile.read_latency_us
        random_iops = profile.random_read_iops
    else:
        base = profile.program_latency_us
        random_iops = profile.random_write_iops

    iops_us = 1_000_000.0 / random_iops if io.access_pattern == "random" else 0.0
    return max(base, iops_us)


def _transfer_time_us(io: PhysicalIO, profile: SSDProfile) -> float:
    bandwidth = (
        profile.sequential_read_bandwidth_GBps
        if io.op_type == "read"
        else profile.sequential_write_bandwidth_GBps
    )
    return io.size_bytes * 1_000_000.0 / (bandwidth * 1_000_000_000.0)


def _random_iops_gap_us(io: PhysicalIO, profile: SSDProfile) -> float:
    iops = profile.random_read_iops if io.op_type == "read" else profile.random_write_iops
    return 1_000_000.0 / iops


def _build_report(
    profile: SSDProfile,
    physical_ios: list[PhysicalIO],
    completed: list[CompletedIO],
    ftl: _FTLState,
    max_observed_inflight: int,
    *,
    ttft_jobs: Sequence[TTFTJob] | None = None,
) -> dict[str, Any]:
    start_time = min((item.io.time_us for item in completed), default=0.0)
    finish_time = max((item.completion_time_us for item in completed), default=start_time)
    duration_us = max(finish_time - start_time, 1.0)
    total_bytes = sum(item.io.size_bytes for item in completed)
    allocation_digests = {
        io.allocation_table_sha256
        for io in physical_ios
        if io.allocation_table_sha256 is not None
    }
    return {
        "semantics": {
            "rates": "achieved by the simplified single-SSD simulator",
            "latency": "arrival-to-completion time including host queueing and FTL penalties",
            "ftl": "page-mapped research model, not vendor firmware",
            "ttft": (
                "first-token boundary estimate from demand model plus simulated "
                "read completion"
            ),
        },
        "device": profile.to_report(),
        "allocation": {
            "method": "deterministic_two_pass_non_overlapping_extents",
            "table_sha256": (
                next(iter(allocation_digests)) if len(allocation_digests) == 1 else None
            ),
            "digest_count": len(allocation_digests),
        },
        "io_count": len(physical_ios),
        "completed_io_count": len(completed),
        "io_accounting": {
            "physical_read_payload_bytes": sum(x.io.size_bytes for x in completed if x.io.op_type == "read"),
            "physical_write_payload_bytes": sum(x.io.size_bytes for x in completed if x.io.op_type == "write"),
            "physical_io_count": len(completed),
            "semantics": "transferred payload; allocated page capacity is a separate measure",
        },
        "throughput": {
            "total_bytes": total_bytes,
            "simulation_duration_us": duration_us,
            "achieved_GBps": total_bytes * 1_000_000.0 / duration_us / 1_000_000_000.0,
            "achieved_iops": len(completed) * 1_000_000.0 / duration_us,
            "read_GBps": _gbps_for(completed, "read", duration_us),
            "write_GBps": _gbps_for(completed, "write", duration_us),
            "read_iops": _iops_for(completed, "read", duration_us),
            "write_iops": _iops_for(completed, "write", duration_us),
        },
        "latency_us": {
            "overall": _distribution([item.latency_us for item in completed]),
            "read": _distribution(
                [item.latency_us for item in completed if item.io.op_type == "read"]
            ),
            "write": _distribution(
                [item.latency_us for item in completed if item.io.op_type == "write"]
            ),
            "by_component": {
                component: _distribution(
                    [
                        item.latency_us
                        for item in completed
                        if item.io.component == component
                    ]
                )
                for component in sorted({item.io.component for item in completed})
            },
        },
        "deadline": _deadline_report(completed),
        "ttft": _ttft_report(completed, list(ttft_jobs or [])),
        "queue": {
            "configured_queue_depth": profile.queue_depth,
            "max_observed_inflight": max_observed_inflight,
            "queue_wait_us": _distribution([item.queue_wait_us for item in completed]),
        },
        "ftl": ftl.summary(),
    }


def _ttft_report(completed: list[CompletedIO], jobs: list[TTFTJob]) -> dict[str, Any]:
    read_completion_by_request: dict[str, float] = {}
    read_completion_by_request_component: dict[tuple[str, str], float] = {}
    for item in completed:
        if item.io.op_type != "read":
            continue
        request_id = item.io.request_id
        read_completion_by_request[request_id] = max(
            read_completion_by_request.get(request_id, item.completion_time_us),
            item.completion_time_us,
        )
        component_key = (request_id, item.io.component)
        read_completion_by_request_component[component_key] = max(
            read_completion_by_request_component.get(
                component_key, item.completion_time_us
            ),
            item.completion_time_us,
        )

    dependency_ttft_savings_us: list[float] = []
    restore_elapsed_us: list[float] = []
    checkpoint_prerequisite_elapsed_us: list[float] = []
    optimistic_overlap_estimate_us: list[float] = []
    dependency_aware_estimate_us: list[float] = []
    no_compute_io_overlap_estimate_us: list[float] = []
    hidden_restore_missed = 0
    missing_restore_completion = 0
    restore_jobs = [job for job in jobs if job.restore_payload_bytes > 0]

    for job in jobs:
        if job.phase_timings:
            required = {c for p in job.phase_timings for c in p.read_components}
            if any((job.request_id, c) not in read_completion_by_request_component for c in required):
                missing_restore_completion += 1
                continue
            ready = {c: max(0.0, read_completion_by_request_component[(job.request_id, c)] - job.arrival_time_us)
                     for c in required}
            all_elapsed = max(ready.values(), default=0.0)
            replay_elapsed = max((ready[c] for c in job.phase_timings[0].read_components), default=0.0)
            compute = sum(p.compute_us for p in job.phase_timings)
            dependency = phase_chain_time_us(job.phase_timings, ready,
                overhead_us=job.cache_path_overhead_us, first_token_us=job.first_token_decode_us)
            restore_elapsed_us.append(all_elapsed)
            checkpoint_prerequisite_elapsed_us.append(replay_elapsed)
            optimistic_overlap_estimate_us.append(max(compute, all_elapsed + job.cache_path_overhead_us)
                                                  + job.first_token_decode_us)
            dependency_aware_estimate_us.append(dependency)
            no_compute_io_overlap_estimate_us.append(all_elapsed + job.cache_path_overhead_us + compute
                                                    + job.first_token_decode_us)
            dependency_ttft_savings_us.append(job.no_cache_ttft_us - dependency)
            hidden_restore_missed += int(job.restore_payload_bytes > 0 and all_elapsed > job.hidden_restore_budget_us)
            continue
        has_restore = job.restore_payload_bytes > 0
        completion_time_us = (
            max(
                (
                    read_completion_by_request_component.get(
                        (job.request_id, component), job.arrival_time_us
                    )
                    for component in job.suffix_gate_components
                ),
                default=job.arrival_time_us,
            )
            if job.suffix_gate_components
            else read_completion_by_request.get(job.request_id)
        )
        if has_restore and completion_time_us is None:
            missing_restore_completion += 1
        restore_elapsed = (
            max(0.0, completion_time_us - job.arrival_time_us)
            if has_restore and completion_time_us is not None
            else 0.0
        )
        restore_elapsed_us.append(restore_elapsed)
        checkpoint_completion = max(
            (
                read_completion_by_request_component.get(
                    (job.request_id, component), job.arrival_time_us
                )
                for component in (
                    job.replay_gate_components
                    or job.blocking_checkpoint_components
                )
            ),
            default=job.arrival_time_us,
        )
        checkpoint_elapsed = max(
            0.0, checkpoint_completion - job.arrival_time_us
        )
        checkpoint_prerequisite_elapsed_us.append(checkpoint_elapsed)
        if has_restore and restore_elapsed > job.hidden_restore_budget_us:
            hidden_restore_missed += 1
        simulated = (
            max(
                job.actual_prefill_time_us,
                restore_elapsed + job.cache_path_overhead_us,
            )
            + job.first_token_decode_us
        )
        optimistic_overlap_estimate_us.append(simulated)
        replay_complete = (
            checkpoint_elapsed
            + job.cache_path_overhead_us
            + job.replay_compute_time_us
        )
        dependency_aware = (
            max(
                replay_complete,
                restore_elapsed + job.cache_path_overhead_us,
            )
            + job.miss_suffix_compute_time_us
            + job.first_token_decode_us
        )
        fully_serialized = (
            restore_elapsed
            + job.cache_path_overhead_us
            + job.replay_compute_time_us
            + job.miss_suffix_compute_time_us
            + job.first_token_decode_us
        )
        dependency_aware_estimate_us.append(dependency_aware)
        no_compute_io_overlap_estimate_us.append(fully_serialized)
        dependency_ttft_savings_us.append(job.no_cache_ttft_us - dependency_aware)

    return {
        "model": "phase_gated_first_token_boundary_with_simulated_restore",
        "request_count": len(jobs),
        "restore_request_count": len(restore_jobs),
        "missing_restore_completion_count": missing_restore_completion,
        "hidden_restore_missed_count": hidden_restore_missed,
        "hidden_restore_miss_ratio": (
            hidden_restore_missed / len(restore_jobs) if restore_jobs else 0.0
        ),
        "restore_elapsed_us": _distribution(restore_elapsed_us),
        "replay_gate_elapsed_us": _distribution(
            checkpoint_prerequisite_elapsed_us
        ),
        "no_cache_ttft_us": _distribution([job.no_cache_ttft_us for job in jobs]),
        "optimistic_overlap_estimate_us": _distribution(
            optimistic_overlap_estimate_us
        ),
        "dependency_aware_estimate_us": _distribution(
            dependency_aware_estimate_us
        ),
        "no_compute_io_overlap_estimate_us": _distribution(
            no_compute_io_overlap_estimate_us
        ),
        "dependency_ttft_savings_us": _distribution(dependency_ttft_savings_us),
    }


def _gbps_for(completed: list[CompletedIO], op_type: str, duration_us: float) -> float:
    total = sum(item.io.size_bytes for item in completed if item.io.op_type == op_type)
    return total * 1_000_000.0 / duration_us / 1_000_000_000.0


def _iops_for(completed: list[CompletedIO], op_type: str, duration_us: float) -> float:
    count = sum(1 for item in completed if item.io.op_type == op_type)
    return count * 1_000_000.0 / duration_us


def _deadline_report(completed: list[CompletedIO]) -> dict[str, Any]:
    considered = [item for item in completed if item.io.deadline_us is not None]
    misses = [
        item for item in considered if item.completion_time_us > (item.io.deadline_us or 0.0)
    ]
    slacks = [
        (item.io.deadline_us or 0.0) - item.completion_time_us
        for item in considered
    ]
    return {
        "considered_io_count": len(considered),
        "met_count": len(considered) - len(misses),
        "missed_count": len(misses),
        "miss_ratio": len(misses) / len(considered) if considered else 0.0,
        "slack_us": _distribution(slacks),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": float(max(values)) if values else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((q / 100.0) * (len(ordered) - 1))
    return float(ordered[index])
