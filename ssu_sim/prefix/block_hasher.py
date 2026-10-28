"""Stable block id generation for synthetic prefix groups."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ssu_sim.utils.units import ceil_div
from ssu_sim.workload.request import Request


def make_block_id(model_id: str, prefix_group_id: str, block_index: int, parent_block_id: str | None) -> str:
    payload = f"{model_id}|{prefix_group_id}|{block_index}|{parent_block_id or 'root'}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def build_block_ids(
    model_id: str,
    prefix_group_id: str,
    token_count: int,
    block_size_tokens: int,
) -> list[str]:
    block_count = ceil_div(token_count, block_size_tokens) if token_count > 0 else 0
    block_ids: list[str] = []
    parent: str | None = None
    for block_index in range(block_count):
        block_id = make_block_id(model_id, prefix_group_id, block_index, parent)
        block_ids.append(block_id)
        parent = block_id
    return block_ids


def build_request_block_ids(
    model_id: str,
    prefix_group_id: str,
    request_id: str,
    token_count: int,
    shareable_prefix_tokens: int,
    block_size_tokens: int,
) -> list[str]:
    """Build a full request path with shared-prefix and private-suffix IDs.

    Only complete shared blocks use the prefix-group namespace.  The remaining
    blocks, including a padded partial tail, are request-specific.  Every ID is
    chained through its parent so branches retain radix-path identity.
    """

    if token_count < 0:
        raise ValueError("token_count must not be negative")
    if not 0 <= shareable_prefix_tokens <= token_count:
        raise ValueError("shareable_prefix_tokens must be within token_count")
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    block_count = ceil_div(token_count, block_size_tokens) if token_count else 0
    shared_blocks = shareable_prefix_tokens // block_size_tokens
    block_ids: list[str] = []
    parent: str | None = None
    for block_index in range(block_count):
        namespace = (
            prefix_group_id
            if block_index < shared_blocks
            else f"{prefix_group_id}:request:{request_id}"
        )
        block_id = make_block_id(model_id, namespace, block_index, parent)
        block_ids.append(block_id)
        parent = block_id
    return block_ids


SharedPrefixBlockIdMemo = dict[tuple[str, str, int, int], tuple[str, ...]]


@dataclass(frozen=True)
class CanonicalRequestPath:
    """Stable object identities for one request's input and generated output."""

    request_id: str
    input_block_ids: tuple[str, ...]
    output_block_ids: tuple[str, ...]
    lookup_block_ids: tuple[str, ...] = ()
    configured_prefix_tokens: int = 0
    shareable_prefix_tokens: int = 0
    shareability_source: str = "complete_shared_blocks"
    inherited_parent_tokens: int = 0
    rewritten_partial_objects: tuple[tuple[str, str], ...] = ()

    @property
    def generated_block_ids(self) -> tuple[str, ...]:
        input_set = set(self.input_block_ids)
        return tuple(value for value in self.output_block_ids if value not in input_set)


def build_request_block_ids_with_shared_memo(
    model_id: str,
    prefix_group_id: str,
    request_id: str,
    token_count: int,
    shareable_prefix_tokens: int,
    block_size_tokens: int,
    shared_prefix_memo: SharedPrefixBlockIdMemo,
) -> tuple[str, ...]:
    """Build a request path while reusing only its shared block-ID chain.

    The caller owns ``shared_prefix_memo``.  Cache replay creates one memo per
    replay so IDs are never retained across cases or workers.  Private suffix
    IDs remain request-specific and are rebuilt for every request.
    """

    if token_count < 0:
        raise ValueError("token_count must not be negative")
    if not 0 <= shareable_prefix_tokens <= token_count:
        raise ValueError("shareable_prefix_tokens must be within token_count")
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")

    block_count = ceil_div(token_count, block_size_tokens) if token_count else 0
    shared_blocks = shareable_prefix_tokens // block_size_tokens
    memo_key = (
        model_id,
        prefix_group_id,
        shared_blocks,
        block_size_tokens,
    )
    shared_ids = shared_prefix_memo.get(memo_key)
    if shared_ids is None:
        generated_shared_ids: list[str] = []
        parent: str | None = None
        for block_index in range(shared_blocks):
            block_id = make_block_id(
                model_id,
                prefix_group_id,
                block_index,
                parent,
            )
            generated_shared_ids.append(block_id)
            parent = block_id
        shared_ids = tuple(generated_shared_ids)
        shared_prefix_memo[memo_key] = shared_ids

    block_ids = list(shared_ids)
    parent = shared_ids[-1] if shared_ids else None
    private_namespace = f"{prefix_group_id}:request:{request_id}"
    for block_index in range(shared_blocks, block_count):
        block_id = make_block_id(
            model_id,
            private_namespace,
            block_index,
            parent,
        )
        block_ids.append(block_id)
        parent = block_id
    return tuple(block_ids)


def canonicalize_request_paths(
    requests: Iterable[Request],
    block_size_tokens: int,
    *,
    version_partial_prefix_models: frozenset[str] = frozenset(),
) -> tuple[tuple[Request, ...], dict[str, CanonicalRequestPath]]:
    """Resolve canonical read/write/decode identities, including explicit lineage.

    A child may inherit its parent's complete generated prefix only by naming
    ``parent_request_id``.  Appending to a padded partial object creates a new
    immutable version; the old object remains valid for concurrent readers.
    """

    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    ordered = sorted(requests, key=lambda item: (item.arrival_time_us, item.request_id))
    request_by_id: dict[str, Request] = {}
    paths: dict[str, CanonicalRequestPath] = {}
    emitted: list[Request] = []
    shared_memo: SharedPrefixBlockIdMemo = {}

    for request in ordered:
        if request.request_id in request_by_id:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        expected_input_blocks = ceil_div(request.seq_len, block_size_tokens) if request.seq_len else 0
        inherited_tokens = 0
        shareability_source = "complete_shared_blocks"
        shareable_prefix_tokens = (
            request.matched_prefix_len // block_size_tokens * block_size_tokens
        )
        rewritten: list[tuple[str, str]] = []

        if request.parent_request_id is not None:
            parent = request_by_id.get(request.parent_request_id)
            parent_path = paths.get(request.parent_request_id)
            if parent is None or parent_path is None:
                raise ValueError(
                    f"request {request.request_id} references an unknown or later parent "
                    f"{request.parent_request_id}"
                )
            if parent.model_id != request.model_id:
                raise ValueError("parent and child requests must use the same model_id")
            if (
                parent.conversation_id is not None
                and request.conversation_id is not None
                and parent.conversation_id != request.conversation_id
            ):
                raise ValueError("parent and child conversation_id values do not match")
            inherited_tokens = parent.output_end_token
            shareability_source = "parent_lineage"
            shareable_prefix_tokens = min(
                request.matched_prefix_len,
                inherited_tokens,
            )
            if request.seq_len < inherited_tokens:
                raise ValueError(
                    f"request {request.request_id} seq_len={request.seq_len} does not "
                    f"contain parent output prefix of {inherited_tokens} tokens"
                )
            input_ids = list(parent_path.output_block_ids)
            lookup_ids = list(parent_path.output_block_ids)
            if request.seq_len > inherited_tokens:
                input_ids = _append_immutable_objects(
                    model_id=request.model_id,
                    namespace=f"conversation:{request.request_id}:input",
                    existing_ids=input_ids,
                    existing_tokens=inherited_tokens,
                    target_tokens=request.seq_len,
                    block_size_tokens=block_size_tokens,
                    rewritten=rewritten,
                )
        elif request.block_ids:
            shareability_source = "explicit_object_ids"
            shareable_prefix_tokens = request.matched_prefix_len
            if len(request.block_ids) != expected_input_blocks:
                raise ValueError(
                    f"request {request.request_id} expected {expected_input_blocks} block_ids, "
                    f"found {len(request.block_ids)}"
                )
            if any(not value for value in request.block_ids):
                raise ValueError(f"request {request.request_id} contains an empty block_id")
            input_ids = list(request.block_ids)
            lookup_ids = list(input_ids)
        else:
            input_ids = list(
                build_request_block_ids_with_shared_memo(
                    request.model_id,
                    request.prefix_group_id,
                    request.request_id,
                    request.seq_len,
                    request.matched_prefix_len,
                    block_size_tokens,
                    shared_memo,
                )
            )
            lookup_ids = list(input_ids)

        if (request.model_id in version_partial_prefix_models and not request.parent_request_id
                and not request.block_ids and 0 < request.matched_prefix_len < request.seq_len
                and request.matched_prefix_len % block_size_tokens):
            # An assumed envelope may restore a sealed partial prefix. Appending
            # its suffix must use the same immutable extension rule as lineage.
            lookup_ids = input_ids[:ceil_div(request.matched_prefix_len, block_size_tokens)]
            input_ids = _append_immutable_objects(
                model_id=request.model_id, namespace=f"input-extension:{request.request_id}",
                existing_ids=list(lookup_ids), existing_tokens=request.matched_prefix_len,
                target_tokens=request.seq_len, block_size_tokens=block_size_tokens, rewritten=rewritten)

        if len(input_ids) != expected_input_blocks:
            raise AssertionError("canonical input path has the wrong block count")
        output_ids = _append_immutable_objects(
            model_id=request.model_id,
            namespace=f"generated:{request.request_id}",
            existing_ids=list(input_ids),
            existing_tokens=request.seq_len,
            target_tokens=request.output_end_token,
            block_size_tokens=block_size_tokens,
            rewritten=rewritten,
        )
        canonical = CanonicalRequestPath(
            request_id=request.request_id,
            input_block_ids=tuple(input_ids),
            output_block_ids=tuple(output_ids),
            lookup_block_ids=tuple(lookup_ids),
            configured_prefix_tokens=request.matched_prefix_len,
            shareable_prefix_tokens=shareable_prefix_tokens,
            shareability_source=shareability_source,
            inherited_parent_tokens=inherited_tokens,
            rewritten_partial_objects=tuple(rewritten),
        )
        paths[request.request_id] = canonical
        request_by_id[request.request_id] = request
        emitted.append(
            Request(
                request_id=request.request_id,
                arrival_time_us=request.arrival_time_us,
                model_id=request.model_id,
                tenant_id=request.tenant_id,
                seq_len=request.seq_len,
                generation_len=request.generation_len,
                prefix_group_id=request.prefix_group_id,
                matched_prefix_len=request.matched_prefix_len,
                priority=request.priority,
                metadata=request.metadata,
                block_ids=canonical.input_block_ids,
                restore_block_ids=canonical.lookup_block_ids,
                cache_shard=request.cache_shard,
                conversation_id=request.conversation_id,
                parent_request_id=request.parent_request_id,
                is_bootstrap=request.is_bootstrap,
                recovery_boundary_tokens=request.recovery_boundary_tokens,
                restored_checkpoint_keys=request.restored_checkpoint_keys,
            )
        )
    return tuple(emitted), paths


def _append_immutable_objects(
    *,
    model_id: str,
    namespace: str,
    existing_ids: list[str],
    existing_tokens: int,
    target_tokens: int,
    block_size_tokens: int,
    rewritten: list[tuple[str, str]],
) -> list[str]:
    if target_tokens < existing_tokens:
        raise ValueError("target_tokens must not precede existing_tokens")
    result = list(existing_ids)
    if target_tokens == existing_tokens:
        return result

    first_changed_block = existing_tokens // block_size_tokens
    if existing_tokens % block_size_tokens:
        old_id = result.pop()
        parent = result[-1] if result else None
        new_id = make_block_id(model_id, f"{namespace}:rewrite:v1", first_changed_block, parent)
        result.append(new_id)
        rewritten.append((old_id, new_id))

    parent = result[-1] if result else None
    target_blocks = ceil_div(target_tokens, block_size_tokens) if target_tokens else 0
    for block_index in range(len(result), target_blocks):
        block_id = make_block_id(model_id, namespace, block_index, parent)
        result.append(block_id)
        parent = block_id
    return result
