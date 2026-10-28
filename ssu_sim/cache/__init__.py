"""Capacity-aware block-radix prefix-cache replay."""

from ssu_sim.cache.config import CacheReplayConfig, cache_replay_config_from_dict
from ssu_sim.cache.radix import (
    CacheEvictionResult,
    CacheInsertResult,
    CacheMatch,
    CacheShardSnapshot,
    CheckpointAdmissionResult,
    CheckpointAttachment,
    RadixLeafLRUCache,
    stable_cache_shard,
)
from ssu_sim.cache.replay import (
    CacheReplayResult,
    CacheReplaySummary,
    PrefixRecoveryFunnel,
    replay_prefix_cache,
)

__all__ = [
    "CacheEvictionResult",
    "CacheInsertResult",
    "CacheMatch",
    "CacheReplayConfig",
    "CacheReplayResult",
    "CacheReplaySummary",
    "CacheShardSnapshot",
    "CheckpointAdmissionResult",
    "CheckpointAttachment",
    "PrefixRecoveryFunnel",
    "RadixLeafLRUCache",
    "cache_replay_config_from_dict",
    "replay_prefix_cache",
    "stable_cache_shard",
]
