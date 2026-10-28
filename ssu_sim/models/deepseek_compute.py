"""Auditable DeepSeek V4/V4.1 logical operation ledgers.

Source pins: references/model_compute/deepseek_fla_source_manifest.json.
MACs count twice; reduction additions count once. exp/rsqrt/sigmoid/silu,
selection, quantization and copies have separate named counts. We count valid
causal entries, not a dense reference kernel's masked/padded instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ssu_sim.models.compute_counts import block_sparse_pairs, causal_pairs, rmsnorm_work, token_count


@dataclass
class Ledger:
    components: dict[str, int] = field(default_factory=dict)
    operations: dict[str, int] = field(default_factory=dict)
    special: dict[str, int] = field(default_factory=dict)

    def add(self, component: str, operation: str, count: int) -> None:
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid integer count for {component}: {count}")
        self.components[component] = self.components.get(component, 0) + count
        self.operations[operation] = self.operations.get(operation, 0) + count

    def event(self, name: str, count: int) -> None:
        if count:
            self.special[name] = self.special.get(name, 0) + count

    def norm(self, component: str, rows: int, width: int, *, affine: bool = True) -> None:
        work, special = rmsnorm_work(rows, width)
        if not affine:
            work -= rows * width
        self.add(component, "elementwise", work)
        for name, count in special.items():
            self.event(name, count)


def _sink_attention_scalars(ledger: Ledger, rows: int, entries: int) -> None:
    # Score scale; subtract maximum; sum exp; divide valid weights. The sink
    # contributes to the denominator, not to the value contraction.
    ledger.add("attention_elementwise", "elementwise", 4 * entries + rows)
    ledger.event("exp", entries + rows)
    ledger.event("max_compare", entries)


def _hyper_connections(ledger: Ledger, rows: int, width: int, copies: int, iterations: int) -> None:
    """Two sublayers, each with a learned mix, Sinkhorn, pre and post.

    Row/column normalizers are evaluated once per row/column, consistent with
    the tensor algorithm in the kernel comments (not repeated broadcast loads).
    """
    n, d, r = copies, width, rows
    mixes = n * (n + 2)
    ledger.add("mhc_projection", "dense_gemm", 4 * r * n * d * mixes)
    # Statistic over flattened residual; projection rescale; pre/post affine;
    # combination affine, stable softmax+eps, then alternating normalization.
    per_mix = (2 * n * d + 1) + mixes + 6 * n + 2 * n * n
    per_mix += (3 * n * n - n) + n * n + (2 * iterations - 1) * 2 * n * n
    mixing = 2 * (2 * n * n + 3 * n - 1) * d
    ledger.add("mhc_elementwise", "elementwise", r * (2 * per_mix + mixing))
    ledger.event("rsqrt", 2 * r)
    ledger.event("sigmoid", 4 * r * n)
    ledger.event("exp", 2 * r * n * n)
    ledger.event("max_compare", 2 * r * n * (n - 1))


def _experts(ledger: Ledger, rows: int, d: int, intermediate: int, experts: int,
             topk: int, shared: int, *, biased_rows: int, temperature: bool) -> None:
    active = topk + shared
    ledger.add("moe", "moe", rows * 6 * d * intermediate * active)
    # Hash routing chooses indices by token ID, but still computes these scores
    # for their weights: Gate.forward calls linear before its hash branch.
    ledger.add("routing", "routing", rows * 2 * d * experts)
    route = biased_rows * experts + rows * (3 * topk - 1)
    if temperature:
        route += rows * (experts + 1)  # divide scores; epsilon in selected sum
    ledger.add("routing_elementwise", "elementwise", route)
    ledger.add("expert_elementwise", "elementwise",
               rows * (intermediate * (active + topk) + d * active))
    ledger.event("silu", rows * intermediate * active)
    ledger.event("expert_clamp_compare", 3 * rows * intermediate * active)
    ledger.event("softplus", rows * experts)
    ledger.event("sqrt", rows * experts)
    ledger.event("topk_calls", biased_rows)
    ledger.event("topk_candidates", biased_rows * experts)


def _compressor(ledger: Ledger, *, start: int, end: int, batch: int, layers: int,
                d: int, width: int, ratio: int, rope: int, overlap: bool,
                indexer: bool = False, ape: bool = True, count_quantization: bool = True) -> None:
    """Input projections happen for *every* token, not once per pooled entry."""
    if not layers or start == end:
        return
    q = end - start
    copies = 2 if overlap else 1
    generated = end // ratio - start // ratio
    first = int(overlap and start < ratio <= end)
    terms = generated * ratio * copies - first * ratio
    scale = batch * layers
    ledger.add("compression", "compression", 4 * scale * q * d * width * copies)
    if ape:
        ledger.add("compression_elementwise", "elementwise", scale * q * width * copies)
    # Stable softmax + weighted sum: 3k-1 + 2k-1 per dimension.
    ledger.add("compression_elementwise", "elementwise", scale * width * (5 * terms - 2 * generated))
    ledger.event("exp", scale * width * terms)
    ledger.event("max_compare", scale * width * (terms - generated))
    ledger.norm("normalization", scale * generated, width)
    ledger.add("rope", "elementwise", 3 * scale * generated * rope)
    if indexer:
        ledger.add("hadamard", "elementwise", scale * generated * width * (width.bit_length()))
    if count_quantization:
        ledger.event("quantized_cache_elements", scale * generated * (width if indexer else width - rope))


def v4_segment(model, f, start: int, end: int, batch: int) -> Ledger:
    start, end, batch = token_count(start), token_count(end), token_count(batch)
    if end < start or batch <= 0:
        raise ValueError("invalid segment or batch")
    out = Ledger()
    q, d, h, dim, layers = end - start, model.d_model, model.num_q_heads, f.head_dim, model.num_layers
    if not q:
        return out
    ratios = model.attention.compress_ratios
    if len(ratios) != layers:
        raise ValueError("DeepSeek V4 requires its exact per-layer schedule")
    csa, hca, bq = ratios.count(4), ratios.count(128), batch * q
    out.add("projection", "dense_gemm", bq * layers * 2 * (
        d * f.q_lora_rank + f.q_lora_rank * h * dim + d * dim
        + h * dim * f.o_lora_rank + f.o_groups * f.o_lora_rank * d))
    swa = causal_pairs(start, q, cap=f.sliding_window) * layers
    selected = causal_pairs(start, q, ratio=4, cap=f.index_topk) * csa
    compressed = causal_pairs(start, q, ratio=128) * hca
    candidates = causal_pairs(start, q, ratio=4) * csa
    out.add("swa", "dense_attention", 4 * batch * h * dim * swa)
    out.add("csa_core", "sparse_attention", 4 * batch * h * dim * selected)
    out.add("hca_core", "dense_attention", 4 * batch * h * dim * compressed)
    _sink_attention_scalars(out, bq * layers * h, batch * h * (swa + selected + compressed))
    out.add("indexer", "indexer", batch * candidates * (f.index_n_heads * (2 * f.index_head_dim + 2) - 1))
    out.add("index_projection", "dense_gemm", 2 * bq * csa * (
        f.q_lora_rank * f.index_n_heads * f.index_head_dim + d * f.index_n_heads))
    out.add("index_elementwise", "elementwise", bq * csa * f.index_n_heads)
    out.event("relu", batch * candidates * f.index_n_heads)
    out.event("topk_calls", bq * csa)
    out.event("topk_candidates", batch * candidates)
    for ratio, count in ((4, csa), (128, hca)):
        _compressor(out, start=start, end=end, batch=batch, layers=count,
                    d=d, width=dim, ratio=ratio, rope=f.rope_head_dim, overlap=ratio == 4)
    _compressor(out, start=start, end=end, batch=batch, layers=csa,
                d=d, width=f.index_head_dim, ratio=4, rope=f.rope_head_dim, overlap=True, indexer=True)
    out.add("rope", "elementwise", 3 * bq * f.rope_head_dim * (
        layers * (2 * h + 1) + csa * f.index_n_heads))
    out.add("hadamard", "elementwise", bq * csa * f.index_n_heads * f.index_head_dim * f.index_head_dim.bit_length())
    out.norm("normalization", bq * layers, f.q_lora_rank)
    out.norm("normalization", bq * layers * h, dim, affine=False)
    out.norm("normalization", bq * layers, dim)
    out.norm("normalization", 2 * bq * layers, d)
    _experts(out, bq * layers, d, f.expert_intermediate_size, f.n_routed_experts,
             f.experts_per_token, f.n_shared_experts,
             biased_rows=bq * max(0, layers - f.hash_routing_layers), temperature=False)
    out.event("hash_router_lookup", bq * min(layers, f.hash_routing_layers) * f.experts_per_token)
    _hyper_connections(out, bq * layers, d, f.mhc_multiplier, f.hc_sinkhorn_iters)
    out.event("quantized_query_elements", bq * csa * f.index_n_heads * f.index_head_dim)
    out.event("quantized_cache_elements", bq * layers * (dim - f.rope_head_dim))
    return out


def v41_segment(model, f, start: int, end: int, *, encoder: bool, swa_start: int,
                 generate_global: bool, compression_tail: bool, batch: int) -> Ledger:
    start, end, batch = token_count(start), token_count(end), token_count(batch)
    if end < start or batch <= 0 or swa_start > start:
        raise ValueError("invalid CED segment")
    out = Ledger()
    q, d, h, dim, layers = end - start, model.d_model, model.num_q_heads, f.head_dim, 20
    if not q:
        return out
    bq = batch * q
    out.add("projection", "dense_gemm", bq * layers * 2 * (
        d * f.q_lora_rank + f.q_lora_rank * h * dim + d * dim
        + h * dim * f.o_lora_rank + f.o_groups * f.o_lora_rank * d))
    ratio, active_layers = (2, 18) if encoder else (1, 20)
    swa = causal_pairs(start - swa_start, q, cap=f.sliding_window) * layers
    selected = causal_pairs(start, q, ratio=ratio, cap=f.index_topk) * active_layers
    out.add("swa", "dense_attention", batch * 4 * h * dim * swa)
    out.add("csa2_core", "sparse_attention", batch * 4 * h * dim * selected)
    _sink_attention_scalars(out, bq * h * layers, batch * h * (swa + selected))
    full_pairs = causal_pairs(start, q, ratio=ratio)
    candidate_pairs = block_sparse_pairs(start, q, block_size=f.candidate_block_size,
                                        topk_blocks=f.candidate_positions // f.candidate_block_size)
    score_pairs = 3 * full_pairs if encoder else full_pairs + 4 * candidate_pairs
    out.add("indexer", "indexer", batch * score_pairs * (f.index_n_heads * (2 * f.index_head_dim + 2) - 1))
    # Logical production work skips query rows with no reachable global KV.
    # The dense reference may issue those projections; report them separately.
    index_queries = max(0, end-max(start,ratio-1)) * (3 if encoder else 5)
    reference_queries = q * (3 if encoder else 5) if end // ratio else 0
    out.event("excluded_empty_index_query_rows", batch * (reference_queries-index_queries))
    out.add("index_projection", "dense_gemm", batch * index_queries * 2 * (
        f.q_lora_rank * f.index_n_heads * f.index_head_dim + d * f.index_n_heads))
    out.add("index_elementwise", "elementwise", batch * index_queries * f.index_n_heads)
    out.event("relu", batch * score_pairs * f.index_n_heads)
    out.event("topk_calls", batch * index_queries)
    out.event("topk_candidates", batch * score_pairs)
    out.event("quantized_query_elements", batch * index_queries * f.index_n_heads * f.index_head_dim)
    if not encoder:
        # The full indexer first selects candidate blocks; its latest partial
        # block occupies one top-k slot. Reindexers see that union, not cap(L).
        blocks = causal_pairs(start+f.candidate_block_size-1, q, ratio=f.candidate_block_size)
        out.event("candidate_block_max_compare", batch * (full_pairs-blocks))
        out.event("candidate_block_topk_calls", batch * q)
        out.event("candidate_block_topk_candidates", batch * blocks)
    out.add("rope", "elementwise", 3 * bq * f.rope_head_dim * layers * (2 * h + 1)
            + 3 * batch * index_queries * f.index_n_heads * f.rope_head_dim)
    out.norm("normalization", bq * layers, f.q_lora_rank)
    out.norm("normalization", bq * layers, dim)
    out.norm("normalization", 2 * bq * layers, d)
    if encoder and generate_global:
        _compressor(out, start=start, end=end, batch=batch, layers=3, d=d,
                    width=dim, ratio=2, rope=f.rope_head_dim, overlap=False, ape=False,
                    count_quantization=False)
        entries = 3 * (end // 2 - start // 2)
        out.add("index_projection", "dense_gemm", 2 * batch * entries * dim * f.index_head_dim)
        out.norm("normalization", batch * entries, f.index_head_dim)
        out.add("rope", "elementwise", 3 * batch * entries * f.rope_head_dim)
        out.event("quantized_cache_elements", batch * entries * (dim + f.index_head_dim))
    if encoder and compression_tail and end % 2:
        out.add("compression_tail", "compression", batch * 3 * 4 * d * dim)
    _experts(out, bq * layers, d, f.expert_intermediate_size, f.n_routed_experts,
             f.experts_per_token, f.n_shared_experts, biased_rows=bq * layers, temperature=True)
    _hyper_connections(out, bq * layers, d, f.hc_mult, f.hc_sinkhorn_iters)
    if encoder:
        width = f.engram_ngram_orders * f.engram_n_heads * f.engram_head_dim
        out.add("engram", "dense_gemm", 2 * bq * 2 * width * d * (f.hc_mult + 1))
        # 2 modules: FP8 table rescale; two normalized vectors; weighted dot;
        # gate times value and residual add. q_weight*k_weight is one per call.
        n = f.hc_mult
        per_token = width + n * (9 * d + 4)
        out.add("engram_elementwise", "elementwise", 2 * (bq * per_token + n * d))
        out.event("rsqrt", 4 * bq * n)
        out.event("sqrt", 2 * bq * n)
        out.event("sigmoid", 2 * bq * n)
        out.event("abs_clamp_copysign", 2 * bq * n)
        out.event("ngram_hash_lookups", 2 * bq * f.engram_ngram_orders * f.engram_n_heads)
    out.event("quantized_cache_elements", bq * layers * dim)
    return out


def v4_head(model, f, batch: int) -> Ledger:
    """Generation's last-position HC head, separate from suffix backbone."""
    out = Ledger()
    d, n, b = model.d_model, f.mhc_multiplier, batch
    out.add("output_head_projection", "dense_gemm", b * (2*n*d*n + 2*d*f.vocab_size))
    arithmetic = 2*n*d+1 + n + 3*n + (2*n-1)*d + 4*d+1
    out.add("output_head_elementwise", "elementwise", b*arithmetic)
    out.event("rsqrt", 2*b)
    out.event("sigmoid", n*b)
    return out
