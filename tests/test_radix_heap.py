"""Regression coverage for the leaf-LRU eviction index (issue #1).

Run with ``python -m unittest discover -s tests -v``.
"""

from dataclasses import fields
import random
import unittest

from ssu_sim.cache.config import CacheReplayConfig
from ssu_sim.cache.radix import CheckpointAttachment, RadixLeafLRUCache


class RecordingCache(RadixLeafLRUCache):
    def __init__(self, config):
        super().__init__(config)
        self.evicted_paths = []

    def _delete_leaf(self, state, node):
        self.evicted_paths.append((node.shard, node.path_key))
        super()._delete_leaf(state, node)


class ScanningCache(RecordingCache):
    """Choose each victim independently of the heap under test."""

    @staticmethod
    def _push_leaf(state, node):
        pass

    @staticmethod
    def _pop_evictable_leaf(state):
        candidates = []
        stack = list(state.root.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node.leaf and node.lock_ref == 0:
                candidates.append(node)
        return min(candidates, key=lambda node: (node.last_access, node.serial), default=None)


def resident_nodes(cache):
    nodes = []
    for state in cache._shards:
        stack = list(state.root.children.values())
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
    return sorted(nodes, key=lambda node: (node.shard, node.path_key))


def result_values(result):
    """Compare result records without comparing cache-specific node identities."""

    values = {}
    for item in fields(result):
        value = getattr(result, item.name)
        if item.name in {"new_nodes", "matched_nodes"}:
            value = tuple((node.shard, node.path_key) for node in value)
        values[item.name] = value
    return values


class RadixHeapTests(unittest.TestCase):
    def make_cache(self, capacity=20, shards=1):
        return RadixLeafLRUCache(
            CacheReplayConfig(total_capacity_bytes=capacity, shard_count=shards)
        )

    def assert_small_heap(self, cache):
        # Allow threshold tuning while detecting accumulation by access count.
        for state, snapshot in zip(cache._shards, cache.snapshots()):
            self.assertLessEqual(
                len(state.eviction_heap), max(1024, 16 * snapshot.node_count)
            )

    def test_hot_leaf_stays_bounded_and_preserves_lru(self):
        cache = self.make_cache()
        cache.insert_path(["A"], [10])
        cache.insert_path(["B"], [10])
        for _ in range(100_000):
            self.assertEqual(cache.match_path(["A"], lock=False).matched_blocks, 1)
        self.assert_small_heap(cache)
        self.assertEqual(cache.snapshots()[0].node_count, 2)
        self.assertEqual(cache.used_bytes_by_shard, (20,))

        inserted = cache.insert_path(["C"], [10])
        eviction = cache.enforce_capacity(0, new_nodes=inserted.new_nodes)
        self.assertEqual(sorted(cache._shards[0].root.children), ["A", "C"])
        self.assertEqual((eviction.eviction_count, eviction.eviction_bytes), (1, 10))
        self.assertEqual(eviction.self_eviction_count, 0)
        cache.assert_consistent()

    def test_lock_release_duplicates_and_nested_references(self):
        cache = self.make_cache()
        cache.insert_path(["A"], [10])
        cache.insert_path(["B"], [10])
        for _ in range(10_000):
            match = cache.match_path(["A"])
            timestamp = match.matched_nodes[0].last_access
            self.assertEqual(cache.protected_bytes_by_shard, (10,))
            cache.release(match.matched_nodes)
            self.assertEqual(match.matched_nodes[0].last_access, timestamp)
        self.assert_small_heap(cache)

        first = cache.match_path(["A"])
        second = cache.match_path(["A"])
        cache.release(first.matched_nodes)
        self.assertEqual(cache.protected_bytes_by_shard, (10,))
        cache.match_path(["B"], lock=False)
        cache.insert_path(["C"], [10])
        self.assertEqual(cache.enforce_capacity(0).eviction_bytes, 10)
        self.assertTrue(second.matched_nodes[0].resident)
        cache.release(second.matched_nodes)
        self.assertEqual(cache.protected_bytes_by_shard, (0,))
        cache.assert_consistent()

    def test_compaction_removes_invalid_entries_and_deduplicates_live_leaves(self):
        cache = self.make_cache(capacity=50)
        cache.insert_path(["victim"], [10])
        victim = cache.match_path(["victim"])
        cache.release(victim.matched_nodes)
        cache.insert_path(["A"], [10])
        child = cache.insert_path(["A", "child"], [10, 10]).new_nodes[0]
        cache.insert_path(["locked"], [10])
        locked = cache.match_path(["locked"])
        cache.insert_path(["released"], [10])
        released = cache.match_path(["released"])
        cache.release(released.matched_nodes)
        hot = cache.insert_path(["hot"], [10]).new_nodes[0]
        self.assertEqual(cache.enforce_capacity(0).eviction_count, 1)
        self.assertFalse(victim.matched_nodes[0].resident)

        state = cache._shards[0]
        for _ in range(10_000):
            previous_length = len(state.eviction_heap)
            cache.match_path(["hot"], lock=False)
            if len(state.eviction_heap) < previous_length:
                break
        else:
            self.fail("expected the heap to compact during the hot access burst")
        expected = {child, released.matched_nodes[0], hot}
        self.assertCountEqual(
            state.eviction_heap,
            [(node.last_access, node.serial, node) for node in expected],
        )
        cache.release(locked.matched_nodes)
        cache.assert_consistent()

    def test_leaf_becomes_internal_then_parent_is_evictable_again(self):
        cache = self.make_cache()
        parent = cache.insert_path(["A"], [10]).new_nodes[0]
        for _ in range(10_000):
            cache.match_path(["A"], lock=False)
        child = cache.insert_path(["A", "B"], [10, 10]).new_nodes[0]
        cache.insert_path(["C"], [10])
        self.assertEqual(cache.enforce_capacity(0).eviction_bytes, 10)
        self.assertFalse(child.resident)
        self.assertTrue(parent.resident)
        self.assertTrue(parent.leaf)

        cache.insert_path(["D"], [10])
        self.assertEqual(cache.enforce_capacity(0).eviction_bytes, 10)
        self.assertFalse(parent.resident)
        self.assertEqual(sorted(cache._shards[0].root.children), ["C", "D"])
        cache.assert_consistent()

    def test_checkpoints_preserve_owner_protection_and_byte_accounting(self):
        cache = self.make_cache(capacity=30)
        owner = cache.insert_path(["A"], [10]).new_nodes[0]
        cache.insert_path(["B"], [10])
        attachment = CheckpointAttachment("checkpoint", 64, 15, (("state", 15),))
        locked = cache.match_path(["B"])
        self.assertTrue(cache.attach_checkpoint(owner, attachment).rejected)
        cache.release(locked.matched_nodes)
        admitted = cache.attach_checkpoint(owner, attachment)
        self.assertTrue(admitted.admitted)
        self.assertEqual(admitted.eviction.eviction_bytes, 10)
        self.assertEqual(cache.attachment_bytes_by_shard, (15,))
        for _ in range(10_000):
            self.assertTrue(cache.attach_checkpoint(owner, attachment).deduplicated)
        self.assert_small_heap(cache)
        self.assertEqual(cache.used_bytes_by_shard, (25,))

        cache.insert_path(["C"], [10])
        self.assertEqual(cache.enforce_capacity(0).eviction_bytes, 25)
        self.assertFalse(owner.resident)
        self.assertEqual(cache.attachment_bytes_by_shard, (0,))
        cache.assert_consistent()

    def test_partial_insert_retains_a_candidate_after_error_or_oversize(self):
        for invalid_size in (0, 21):
            with self.subTest(invalid_size=invalid_size):
                cache = self.make_cache()
                cache.insert_path(["A"], [10])
                for _ in range(1000):
                    cache.match_path(["A"], lock=False)
                locked = cache.match_path(["A"])
                if invalid_size == 0:
                    with self.assertRaises(ValueError):
                        cache.insert_path(["B", "invalid"], [10, invalid_size])
                else:
                    result = cache.insert_path(["B", "invalid"], [10, invalid_size])
                    self.assertEqual((result.rejected_count, result.rejected_bytes), (1, 21))
                cache.insert_path(["C"], [10])
                self.assertEqual(cache.enforce_capacity(0).eviction_count, 1)
                self.assertEqual(sorted(cache._shards[0].root.children), ["A", "C"])
                cache.release(locked.matched_nodes)
                cache.assert_consistent()

    def test_eviction_compacts_heap_after_the_tree_shrinks(self):
        cache = self.make_cache(capacity=200)
        for index in range(100):
            cache.insert_path([f"cold{index}"], [1])
        hot = cache.insert_path(["hot"], [1]).new_nodes[0]
        for _ in range(250):
            cache.match_path(["hot"], lock=False)
        large = cache.insert_path(["large"], [199]).new_nodes[0]
        eviction = cache.enforce_capacity(0, new_nodes=[large])
        self.assertEqual((eviction.eviction_count, eviction.eviction_bytes), (100, 100))
        self.assertEqual(eviction.self_eviction_count, 0)
        self.assertEqual(sorted(cache._shards[0].root.children), ["hot", "large"])
        heap = cache._shards[0].eviction_heap
        self.assertEqual(len(heap), 2)
        self.assertCountEqual(
            heap, [(node.last_access, node.serial, node) for node in (hot, large)]
        )
        cache.assert_consistent()

    def test_compaction_is_independent_for_each_shard(self):
        cache = self.make_cache(capacity=60, shards=3)
        for shard in range(3):
            cache.insert_path(["A"], [10], shard=shard)
            cache.insert_path(["B"], [10], shard=shard)
        quiet_heaps = [tuple(state.eviction_heap) for state in cache._shards[1:]]
        for _ in range(10_000):
            cache.match_path(["A"], shard=0, lock=False)
        self.assert_small_heap(cache)
        self.assertEqual([tuple(state.eviction_heap) for state in cache._shards[1:]], quiet_heaps)
        for shard in range(3):
            cache.insert_path(["C"], [10], shard=shard)
            self.assertEqual(cache.enforce_capacity(shard).eviction_count, 1)
            expected = ["A", "C"] if shard == 0 else ["B", "C"]
            self.assertEqual(sorted(cache._shards[shard].root.children), expected)
        cache.assert_consistent()

    def assert_equivalent(self, actual, reference):
        self.assertEqual(actual.snapshots(), reference.snapshots())
        self.assertEqual(actual.evicted_paths, reference.evicted_paths)
        self.assertEqual(actual.used_bytes_by_shard, reference.used_bytes_by_shard)
        self.assertEqual(actual.protected_bytes_by_shard, reference.protected_bytes_by_shard)
        self.assertEqual(actual.attachment_bytes_by_shard, reference.attachment_bytes_by_shard)
        actual_nodes = resident_nodes(actual)
        reference_nodes = resident_nodes(reference)
        self.assertEqual(len(actual_nodes), len(reference_nodes))
        for left, right in zip(actual_nodes, reference_nodes):
            self.assertEqual(
                (left.shard, left.path_key, left.size_bytes),
                (right.shard, right.path_key, right.size_bytes),
            )
            self.assertEqual(
                (left.last_access, left.lock_ref, left.attachments),
                (right.last_access, right.lock_ref, right.attachments),
            )
        actual.assert_consistent()
        reference.assert_consistent()

    def test_random_operations_match_full_tree_lru_reference(self):
        for shards in (1, 3):
            for seed in (7, 91):
                with self.subTest(shards=shards, seed=seed):
                    self.run_random_trace(shards, seed)

    def run_random_trace(self, shards, seed):
        rng = random.Random(seed)
        config = CacheReplayConfig(total_capacity_bytes=90 * shards, shard_count=shards)
        actual = RecordingCache(config)
        reference = ScanningCache(config)
        held_paths = []
        paths = [
            [f"root{root}", f"branch{branch}", f"leaf{leaf}"][:depth]
            for root in range(5)
            for branch in range(3)
            for leaf in range(2)
            for depth in (1, 2, 3)
        ]

        for step in range(2500):
            action = rng.randrange(100)
            path = rng.choice(paths)
            if action < 35:
                sizes = [7, 9, 11][:len(path)]
                write_start = rng.randrange(len(path) + 1)
                left = actual.insert_path(path, sizes, write_start_index=write_start)
                right = reference.insert_path(path, sizes, write_start_index=write_start)
                self.assertEqual(result_values(left), result_values(right))
                shard = actual.shard_for_path(path)
                try:
                    expected = reference.enforce_capacity(shard, new_nodes=right.new_nodes)
                except RuntimeError:
                    with self.assertRaises(RuntimeError):
                        actual.enforce_capacity(shard, new_nodes=left.new_nodes)
                    self.assertEqual(actual.evicted_paths, reference.evicted_paths)
                    for actual_path, reference_path in held_paths:
                        actual.release(actual_path)
                        reference.release(reference_path)
                    held_paths.clear()
                    expected = reference.enforce_capacity(shard, new_nodes=right.new_nodes)
                observed = actual.enforce_capacity(shard, new_nodes=left.new_nodes)
                self.assertEqual(observed, expected)
            elif action < 75:
                lock = rng.random() < 0.4
                left = actual.match_path(path, lock=lock)
                right = reference.match_path(path, lock=lock)
                self.assertEqual(result_values(left), result_values(right))
                if lock:
                    held_paths.append((left.matched_nodes, right.matched_nodes))
                elif step % 5 == 0:
                    # Bursts exercise compaction while most other nodes are cold.
                    for _ in range(100):
                        actual.match_path(path, lock=False)
                        reference.match_path(path, lock=False)
            elif action < 85 and held_paths:
                left, right = held_paths.pop(rng.randrange(len(held_paths)))
                actual.release(left)
                reference.release(right)
            else:
                left_nodes = resident_nodes(actual)
                right_nodes = resident_nodes(reference)
                if left_nodes:
                    index = rng.randrange(len(left_nodes))
                    attachment = CheckpointAttachment(f"state{rng.randrange(3)}", 64, 3)
                    left = actual.attach_checkpoint(left_nodes[index], attachment)
                    right = reference.attach_checkpoint(right_nodes[index], attachment)
                    self.assertEqual(left, right)
            self.assert_equivalent(actual, reference)
            self.assert_small_heap(actual)

        for left, right in held_paths:
            actual.release(left)
            reference.release(right)
        self.assert_equivalent(actual, reference)


if __name__ == "__main__":
    unittest.main()
