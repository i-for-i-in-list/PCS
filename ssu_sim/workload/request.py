"""LLM request data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_time_us: float
    model_id: str
    tenant_id: str
    seq_len: int
    generation_len: int
    prefix_group_id: str
    matched_prefix_len: int
    priority: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    block_ids: tuple[str, ...] = ()
    restore_block_ids: tuple[str, ...] = ()
    cache_shard: int | None = None
    conversation_id: str | None = None
    parent_request_id: str | None = None
    is_bootstrap: bool = False
    recovery_boundary_tokens: int | None = None
    restored_checkpoint_keys: tuple[str, ...] = ()

    @property
    def miss_len(self) -> int:
        return max(self.seq_len - self.matched_prefix_len, 0)

    @property
    def output_end_token(self) -> int:
        """Exclusive token boundary after this request's generated output."""

        return self.seq_len + self.generation_len
