"""Block-radix leaf-LRU state used by capacity-aware cache replay."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
from typing import Iterable, Sequence

from ssu_sim.cache.config import CacheReplayConfig


@dataclass(frozen=True)
class CheckpointAttachment:
    """Atomic serialized state attached to a precise prefix boundary."""

    key: str
    boundary_token: int
    size_bytes: int
    component_extents: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("checkpoint attachment key must not be empty")
        if self.boundary_token < 0:
            raise ValueError("checkpoint boundary_token must not be negative")
        if self.size_bytes <= 0:
            raise ValueError("checkpoint size_bytes must be positive")
        if self.component_extents and sum(
            size for _component, size in self.component_extents
        ) != self.size_bytes:
            raise ValueError("checkpoint component extents must sum to size_bytes")


@dataclass(eq=False)
class _RadixNode:
    block_id: str
    size_bytes: int
    shard: int
    depth: int
    serial: int
    path_key: str
    parent: _RadixNode | None = None
    children: dict[str, _RadixNode] = field(default_factory=dict)
    last_access: int = 0
    lock_ref: int = 0
    attachments: dict[str, CheckpointAttachment] = field(default_factory=dict)

    @property
    def resident(self) -> bool:
        return self.parent is not None

    @property
    def leaf(self) -> bool:
        return not self.children

    @property
    def attachment_bytes(self) -> int:
        return sum(item.size_bytes for item in self.attachments.values())

    @property
    def resident_size_bytes(self) -> int:
        return self.size_bytes + self.attachment_bytes


@dataclass(frozen=True)
class CacheMatch:
    shard: int
    matched_nodes: tuple[_RadixNode, ...]
    missing_was_admitted: bool

    @property
    def matched_blocks(self) -> int:
        return len(self.matched_nodes)


@dataclass(frozen=True)
class CacheInsertResult:
    inserted_count: int = 0
    inserted_bytes: int = 0
    deduplicated_count: int = 0
    deduplicated_bytes: int = 0
    rejected_count: int = 0
    rejected_bytes: int = 0
    new_nodes: tuple[_RadixNode, ...] = ()


@dataclass(frozen=True)
class CacheEvictionResult:
    eviction_count: int = 0
    eviction_bytes: int = 0
    self_eviction_count: int = 0
    self_eviction_bytes: int = 0


@dataclass(frozen=True)
class CheckpointAdmissionResult:
    admitted: bool = False
    deduplicated: bool = False
    rejected: bool = False
    attachment_bytes: int = 0
    eviction: CacheEvictionResult = CacheEvictionResult()


@dataclass(frozen=True)
class CacheShardSnapshot:
    shard: int
    capacity_bytes: int
    used_bytes: int
    evictable_bytes: int
    protected_bytes: int
    node_count: int
    leaf_count: int
    max_depth: int
    attachment_count: int = 0
    attachment_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "shard": self.shard,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "evictable_bytes": self.evictable_bytes,
            "protected_bytes": self.protected_bytes,
            "node_count": self.node_count,
            "leaf_count": self.leaf_count,
            "max_depth": self.max_depth,
            "attachment_count": self.attachment_count,
            "attachment_bytes": self.attachment_bytes,
        }


@dataclass
class _ShardState:
    capacity_bytes: int
    root: _RadixNode
    used_bytes: int = 0
    protected_bytes: int = 0
    attachment_bytes: int = 0
    eviction_heap: list[tuple[int, int, _RadixNode]] = field(default_factory=list)


class RadixLeafLRUCache:
    """A byte-weighted block trie that recursively evicts LRU leaves.

    Each shard owns an independent tree.  Paths, rather than individual blocks,
    are shard-affine so an ancestor and all descendants share one capacity and
    eviction domain.
    """

    def __init__(self, config: CacheReplayConfig) -> None:
        self.config = config
        self._clock = 0
        self._serial = 0
        self._ever_admitted: set[str] = set()
        self._shards: list[_ShardState] = []
        for shard, capacity in enumerate(config.capacity_bytes_by_shard):
            root = _RadixNode(
                block_id="",
                size_bytes=0,
                shard=shard,
                depth=0,
                serial=self._next_serial(),
                path_key=f"root:{shard}",
            )
            self._shards.append(_ShardState(capacity_bytes=capacity, root=root))

    def shard_for_path(self, block_ids: Sequence[str]) -> int:
        if not block_ids:
            raise ValueError("cannot place an empty cache path")
        return stable_cache_shard(
            block_ids[0], self.config.shard_count, self.config.placement_seed
        )

    def match_path(
        self,
        block_ids: Sequence[str],
        *,
        shard: int | None = None,
        lock: bool = True,
    ) -> CacheMatch:
        """Return and optionally protect the longest resident path prefix."""

        if not block_ids:
            target_shard = 0 if shard is None else shard
            self._validate_shard(target_shard)
            return CacheMatch(target_shard, (), False)
        target_shard = self.shard_for_path(block_ids) if shard is None else shard
        self._validate_shard(target_shard)
        state = self._shards[target_shard]
        node = state.root
        matched: list[_RadixNode] = []
        missing_seen = False
        for block_id in block_ids:
            child = node.children.get(block_id)
            if child is None:
                missing_seen = _path_key(node.path_key, block_id) in self._ever_admitted
                break
            self._touch(state, child)
            matched.append(child)
            node = child
        if lock:
            for item in matched:
                if item.lock_ref == 0:
                    state.protected_bytes += item.resident_size_bytes
                item.lock_ref += 1
        return CacheMatch(target_shard, tuple(matched), missing_seen)

    def insert_path(
        self,
        block_ids: Sequence[str],
        size_bytes: Sequence[int],
        *,
        shard: int | None = None,
        write_start_index: int = 0,
    ) -> CacheInsertResult:
        """Insert a complete path without enforcing capacity yet.

        ``write_start_index`` separates already-hit nodes from blocks actually
        recomputed by this request, so duplicate-write accounting only includes
        the latter.
        """

        if len(block_ids) != len(size_bytes):
            raise ValueError("block_ids and size_bytes must have equal length")
        if not block_ids:
            return CacheInsertResult()
        if not 0 <= write_start_index <= len(block_ids):
            raise ValueError("write_start_index must be within the cache path")
        target_shard = self.shard_for_path(block_ids) if shard is None else shard
        self._validate_shard(target_shard)
        state = self._shards[target_shard]
        node = state.root
        inserted_count = inserted_bytes = 0
        deduplicated_count = deduplicated_bytes = 0
        rejected_count = rejected_bytes = 0
        new_nodes: list[_RadixNode] = []

        try:
            for index, (block_id, block_bytes) in enumerate(zip(block_ids, size_bytes)):
                if block_bytes <= 0:
                    raise ValueError("cache block size must be positive")
                child = node.children.get(block_id)
                if child is not None:
                    if child.size_bytes != block_bytes:
                        raise ValueError(
                            f"cache block {block_id!r} changed size from "
                            f"{child.size_bytes} to {block_bytes}"
                        )
                    if index >= write_start_index:
                        self._touch(state, child)
                        deduplicated_count += 1
                        deduplicated_bytes += block_bytes
                    node = child
                    continue

                if block_bytes > state.capacity_bytes:
                    remaining = size_bytes[index:]
                    rejected_count = len(remaining)
                    rejected_bytes = sum(remaining)
                    break

                child = _RadixNode(
                    block_id=block_id,
                    size_bytes=block_bytes,
                    shard=target_shard,
                    depth=node.depth + 1,
                    serial=self._next_serial(),
                    path_key=_path_key(node.path_key, block_id),
                    parent=node,
                )
                node.children[block_id] = child
                state.used_bytes += block_bytes
                self._ever_admitted.add(child.path_key)
                # Preserve one clock tick and last_access assignment per block.
                # Intermediate nodes immediately become internal, so enqueueing
                # them here only creates stale heap entries.
                self._touch(state, child, enqueue_leaf=False)
                inserted_count += 1
                inserted_bytes += block_bytes
                new_nodes.append(child)
                node = child
        finally:
            # Restore the one valid leaf candidate on success, oversize
            # admission stop, and every partially-mutating exceptional exit.
            if new_nodes:
                tail = new_nodes[-1]
                if tail.resident and tail.leaf and tail.lock_ref == 0:
                    self._push_leaf(state, tail)

        return CacheInsertResult(
            inserted_count=inserted_count,
            inserted_bytes=inserted_bytes,
            deduplicated_count=deduplicated_count,
            deduplicated_bytes=deduplicated_bytes,
            rejected_count=rejected_count,
            rejected_bytes=rejected_bytes,
            new_nodes=tuple(new_nodes),
        )

    def release(self, nodes: Iterable[_RadixNode]) -> None:
        """Release one request reference from every node on a matched path."""

        for node in nodes:
            if node.lock_ref <= 0:
                raise ValueError("cache node lock_ref is already zero")
            node.lock_ref -= 1
            if node.lock_ref == 0:
                state = self._shards[node.shard]
                state.protected_bytes -= node.resident_size_bytes
                if node.resident and node.leaf:
                    self._push_leaf(state, node)

    def attach_checkpoint(
        self,
        node: _RadixNode,
        attachment: CheckpointAttachment,
    ) -> CheckpointAdmissionResult:
        """Atomically attach state without risking deletion of the token node.

        Attachments share shard capacity with token blocks.  They can be
        admitted, rejected, or deduplicated independently, but are not
        independently evicted; owner-node eviction removes them together.
        """

        if not node.resident:
            raise ValueError("checkpoint owner node is not resident")
        state = self._shards[node.shard]
        existing = node.attachments.get(attachment.key)
        if existing is not None:
            if existing != attachment:
                raise ValueError(
                    f"checkpoint attachment {attachment.key!r} changed serialization"
                )
            self._touch(state, node)
            return CheckpointAdmissionResult(
                deduplicated=True,
                attachment_bytes=attachment.size_bytes,
            )
        if attachment.size_bytes > state.capacity_bytes:
            return CheckpointAdmissionResult(
                rejected=True,
                attachment_bytes=attachment.size_bytes,
            )

        was_locked = node.lock_ref > 0
        required_reclaim = max(
            state.used_bytes + attachment.size_bytes - state.capacity_bytes,
            0,
        )
        owner_bytes_to_protect = 0 if was_locked else node.resident_size_bytes
        evictable_after_owner_lock = max(
            state.used_bytes
            - state.protected_bytes
            - owner_bytes_to_protect,
            0,
        )
        if required_reclaim > evictable_after_owner_lock:
            return CheckpointAdmissionResult(
                rejected=True,
                attachment_bytes=attachment.size_bytes,
            )

        if not was_locked:
            node.lock_ref += 1
            state.protected_bytes += node.resident_size_bytes
        node.attachments[attachment.key] = attachment
        state.used_bytes += attachment.size_bytes
        state.attachment_bytes += attachment.size_bytes
        state.protected_bytes += attachment.size_bytes
        try:
            eviction = self.enforce_capacity(node.shard)
        except RuntimeError:
            del node.attachments[attachment.key]
            state.used_bytes -= attachment.size_bytes
            state.attachment_bytes -= attachment.size_bytes
            state.protected_bytes -= attachment.size_bytes
            if not was_locked:
                state.protected_bytes -= node.resident_size_bytes
                node.lock_ref -= 1
                if node.leaf:
                    self._push_leaf(state, node)
            return CheckpointAdmissionResult(
                rejected=True,
                attachment_bytes=attachment.size_bytes,
            )
        if not was_locked:
            state.protected_bytes -= node.resident_size_bytes
            node.lock_ref -= 1
            if node.leaf:
                self._push_leaf(state, node)
        self._touch(state, node)
        return CheckpointAdmissionResult(
            admitted=True,
            attachment_bytes=attachment.size_bytes,
            eviction=eviction,
        )

    @staticmethod
    def latest_checkpoint(
        matched_nodes: Sequence[_RadixNode],
        *,
        key_prefix: str | None = None,
        coverage_tokens: int | None = None,
    ) -> CheckpointAttachment | None:
        """Find the newest complete attachment on a matched token path."""

        candidates = (
            attachment
            for node in matched_nodes
            for attachment in node.attachments.values()
            if (key_prefix is None or attachment.key.startswith(key_prefix))
            and (
                coverage_tokens is None
                or attachment.boundary_token <= coverage_tokens
            )
        )
        return max(candidates, key=lambda item: item.boundary_token, default=None)

    def enforce_capacity(
        self,
        shard: int,
        *,
        new_nodes: Iterable[_RadixNode] = (),
    ) -> CacheEvictionResult:
        """Recursively remove LRU leaves until the shard fits its capacity."""

        self._validate_shard(shard)
        state = self._shards[shard]
        inserted = set(new_nodes)
        eviction_count = eviction_bytes = 0
        self_eviction_count = self_eviction_bytes = 0
        while state.used_bytes > state.capacity_bytes:
            node = self._pop_evictable_leaf(state)
            if node is None:
                raise RuntimeError(
                    "cache capacity cannot be satisfied because every resident leaf is protected"
                )
            block_bytes = node.resident_size_bytes
            if node in inserted:
                self_eviction_count += 1
                self_eviction_bytes += block_bytes
            self._delete_leaf(state, node)
            eviction_count += 1
            eviction_bytes += block_bytes
        return CacheEvictionResult(
            eviction_count=eviction_count,
            eviction_bytes=eviction_bytes,
            self_eviction_count=self_eviction_count,
            self_eviction_bytes=self_eviction_bytes,
        )

    def snapshots(self) -> tuple[CacheShardSnapshot, ...]:
        return tuple(self._snapshot(shard) for shard in range(len(self._shards)))

    @property
    def used_bytes_by_shard(self) -> tuple[int, ...]:
        return tuple(state.used_bytes for state in self._shards)

    @property
    def protected_bytes_by_shard(self) -> tuple[int, ...]:
        return tuple(state.protected_bytes for state in self._shards)

    @property
    def attachment_bytes_by_shard(self) -> tuple[int, ...]:
        return tuple(state.attachment_bytes for state in self._shards)

    def assert_consistent(self) -> None:
        """Raise when tree reachability, byte accounting, or capacity is invalid."""

        for shard, state in enumerate(self._shards):
            visited: set[_RadixNode] = set()
            stack = list(state.root.children.values())
            total = 0
            protected = 0
            attachments = 0
            while stack:
                node = stack.pop()
                if node in visited:
                    raise AssertionError("cache tree contains a cycle or duplicate node")
                visited.add(node)
                if node.parent is None or node.parent.children.get(node.block_id) is not node:
                    raise AssertionError("cache tree contains an unreachable node")
                if node.shard != shard:
                    raise AssertionError("cache node belongs to the wrong shard")
                total += node.resident_size_bytes
                attachments += node.attachment_bytes
                if node.lock_ref > 0:
                    protected += node.resident_size_bytes
                stack.extend(node.children.values())
            if total != state.used_bytes:
                raise AssertionError(
                    f"cache byte accounting mismatch: tree={total}, state={state.used_bytes}"
                )
            if protected != state.protected_bytes:
                raise AssertionError(
                    "cache protected byte accounting mismatch: "
                    f"tree={protected}, state={state.protected_bytes}"
                )
            if attachments != state.attachment_bytes:
                raise AssertionError(
                    "cache checkpoint byte accounting mismatch: "
                    f"tree={attachments}, state={state.attachment_bytes}"
                )
            if state.used_bytes > state.capacity_bytes:
                raise AssertionError("cache shard exceeds configured capacity")

    def _snapshot(self, shard: int) -> CacheShardSnapshot:
        state = self._shards[shard]
        nodes: list[_RadixNode] = []
        stack = list(state.root.children.values())
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        protected = sum(node.resident_size_bytes for node in nodes if node.lock_ref > 0)
        return CacheShardSnapshot(
            shard=shard,
            capacity_bytes=state.capacity_bytes,
            used_bytes=state.used_bytes,
            evictable_bytes=state.used_bytes - protected,
            protected_bytes=protected,
            node_count=len(nodes),
            leaf_count=sum(1 for node in nodes if node.leaf),
            max_depth=max((node.depth for node in nodes), default=0),
            attachment_count=sum(len(node.attachments) for node in nodes),
            attachment_bytes=state.attachment_bytes,
        )

    def _touch(
        self,
        state: _ShardState,
        node: _RadixNode,
        *,
        enqueue_leaf: bool = True,
    ) -> None:
        self._clock += 1
        node.last_access = self._clock
        if enqueue_leaf and node.leaf and node.lock_ref == 0:
            self._push_leaf(state, node)

    @staticmethod
    def _push_leaf(state: _ShardState, node: _RadixNode) -> None:
        heapq.heappush(state.eviction_heap, (node.last_access, node.serial, node))

    @staticmethod
    def _pop_evictable_leaf(state: _ShardState) -> _RadixNode | None:
        while state.eviction_heap:
            access, _serial, node = heapq.heappop(state.eviction_heap)
            if (
                node.resident
                and node.leaf
                and node.lock_ref == 0
                and node.last_access == access
            ):
                return node
        return None

    def _delete_leaf(self, state: _ShardState, node: _RadixNode) -> None:
        if not node.resident or not node.leaf or node.lock_ref:
            raise ValueError("only unlocked resident leaves can be evicted")
        parent = node.parent
        assert parent is not None
        removed = parent.children.pop(node.block_id, None)
        if removed is not node:
            raise AssertionError("cache parent does not own the evicted leaf")
        state.used_bytes -= node.resident_size_bytes
        state.attachment_bytes -= node.attachment_bytes
        node.parent = None
        if parent.parent is not None and parent.leaf and parent.lock_ref == 0:
            self._push_leaf(state, parent)

    def _validate_shard(self, shard: int) -> None:
        if not 0 <= shard < len(self._shards):
            raise ValueError(f"cache shard must be in [0, {len(self._shards)})")

    def _next_serial(self) -> int:
        serial = self._serial
        self._serial += 1
        return serial


def stable_cache_shard(value: str, shard_count: int, seed: int = 0) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def _path_key(parent_key: str, block_id: str) -> str:
    return hashlib.sha256(f"{parent_key}|{block_id}".encode("utf-8")).hexdigest()
