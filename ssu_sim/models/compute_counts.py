"""Integer operation counts, with MAC=2 and causal self-attention included.

These are algorithmic counts, not measurements of issued kernel instructions.
The linear-attention routines explicitly name the reference algorithm counted;
padding, triangular work and state updates must not be hidden in a coefficient.
"""

from __future__ import annotations

def token_count(value: int | float, name: str = "tokens") -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def causal_entry_sum(end: int, ratio: int = 1, cap: int | None = None) -> int:
    """Sum min(floor(i/ratio), cap), i=1..end; no continuous approximation."""
    end = token_count(end)
    if ratio <= 0 or int(ratio) != ratio:
        raise ValueError("ratio must be a positive integer")
    ratio = int(ratio)
    if cap is not None:
        cap = token_count(cap, "cap")
        if cap == 0:
            return 0
    limit = end if cap is None else min(end, ratio * cap - 1)
    quotient, remainder = divmod(limit, ratio)
    result = ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)
    return result + (end - limit) * (cap or 0)


def causal_pairs(start: int, count: int, *, ratio: int = 1, cap: int | None = None) -> int:
    start, count = token_count(start, "start_context"), token_count(count, "q_len")
    return causal_entry_sum(start + count, ratio, cap) - causal_entry_sum(start, ratio, cap)


def block_sparse_pairs(start: int, count: int, *, block_size: int, topk_blocks: int) -> int:
    """Top-k blocks including the current partial block (local boost >= 1).

    All other selected causal blocks are complete. Local blocks occupy top-k
    slots, rather than being appended to them. Token positions are unpadded.
    """
    start, count = token_count(start), token_count(count)
    block_size, topk_blocks = token_count(block_size), token_count(topk_blocks)
    if block_size <= 0 or topk_blocks <= 0:
        raise ValueError("block_size and topk_blocks must be positive")
    def prefix(n):
        dense = min(n, block_size * topk_blocks)
        groups, tail = divmod(n - dense, block_size)
        history = (topk_blocks - 1) * block_size
        return (dense * (dense + 1) // 2
                + groups * (block_size * history + block_size * (block_size + 1) // 2)
                + tail * history + tail * (tail + 1) // 2)
    return prefix(start + count) - prefix(start)


def delta_rule_work(tokens: int, key_dim: int, value_dim: int, *,
                    cached: bool, chunk_size: int = 64, channelwise: bool = False) -> dict[str, int]:
    """Per-head reference GDN/KDA arithmetic, excluding external projections.

    Multi-token GDN counts the published torch chunk algorithm with two unit
    triangular solves. KDA counts the FLA reference chunk algorithm (explicit
    triangular inverse). This is an analytical reference, not a claim that FLA's
    fused Triton kernel issues the same operations. Special functions each count
    separately in ``exp`` and ``rsqrt``; they are never converted to FLOPs.
    """
    t, k, v = token_count(tokens), key_dim, value_dim
    if t == 0:
        return {"contractions": 0, "elementwise": 0, "exp": 0, "rsqrt": 0}
    # Q/K L2 normalization (square, reduction, eps, rsqrt, multiply), Q scaling.
    elementwise = t * 7 * k
    rsqrt = 2 * t
    if t == 1 and cached:
        # decay S; read k^T S; delta; rank-one update; read q^T S.
        contractions = 4 * k * v
        elementwise += 3 * k * v + (k + v if channelwise else 2 * v)
        return {"contractions": contractions, "elementwise": elementwise,
                "exp": k if channelwise else 1, "rsqrt": rsqrt}
    c = chunk_size
    if c <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = (t + c - 1) // c
    if channelwise:
        # FLA naive_chunk_kda: Akk/Aqk, A@K/A@V, and four state/output matmuls.
        contractions = chunks * (6 * c * c * k + 4 * c * c * v + 6 * c * k * v)
        # Explicit inverse row updates: c multiplications plus c-1 reduction
        # additions and one addition to each of i row elements, for i=1..c-1.
        elementwise += chunks * (c * c * (c - 1) + 4 * c * c * k
                                + 4 * c * c + 4 * c * k + 2 * c * v
                                + 2 * k * v + (c - 1) * k)
        exp = chunks * (2 * c * c * k + 3 * c * k + k)
    else:
        contractions = chunks * (4 * c * c * k + c * (c - 1) * (k + v)
                                  + 6 * c * k * v + 2 * c * c * v)
        # Beta, cumsum, pair decays, elementwise attention weighting, chunk
        # decays, state scaling/addition, and residual output additions.
        elementwise += chunks * (c * (k + v) + (c - 1) + 3 * c * c
                                + 3 * c * k + c
                                + 2 * c * v + 2 * k * v)
        exp = chunks * (c * c + 3 * c + 1)
    return {"contractions": contractions, "elementwise": elementwise, "exp": exp, "rsqrt": rsqrt}


def rmsnorm_work(rows: int, width: int) -> tuple[int, dict[str, int]]:
    """square/reduce/mean/eps, rescale and learned weight; rsqrt separate."""
    return rows * (4 * width + 1), {"rsqrt": rows}


def softmax_work(rows: int, valid_entries: int) -> tuple[int, dict[str, int]]:
    """Stable softmax: subtract max, sum exponentials, divide; exp/max separate."""
    if valid_entries < rows:
        raise ValueError("softmax rows must each contain at least one valid entry")
    return 3 * valid_entries - rows, {"exp": valid_entries, "max_compare": valid_entries - rows}
