"""Reference calculator-capacity equations.

This module deliberately operates on the calculator fields retained verbatim
in each canonical model config.  It does not reuse runtime attention sizing,
so parity tests can detect drift between the external convention and the
simulator's active/prefix-store representations.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Any

from ssu_sim.models.footprint import CalculatorReferenceFootprint
from ssu_sim.models.llm_model import ModelProfile


def calculate_reference_footprint(
    model: ModelProfile,
    *,
    tokens: int,
    sequence_count: int = 1,
    precision_bytes: float = 2.0,
    indexer_precision_bytes: float | None = None,
    include_draft: bool = False,
    include_state: bool = False,
    checkpoint_interval_tokens: int | None = None,
    encoder_tail_policy: str = "recompute",
) -> CalculatorReferenceFootprint:
    """Evaluate a retained website equation or the official V4.1 layout.

    ``checkpoint_interval_tokens=None`` denotes a single prompt-end snapshot,
    matching the website's ``infinity`` choice.  The website UI domain starts
    at one token; the internal function also accepts zero for simulator edge
    tests.
    """

    if tokens < 0:
        raise ValueError("tokens must not be negative")
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if precision_bytes <= 0:
        raise ValueError("precision_bytes must be positive")
    if indexer_precision_bytes is None:
        indexer_precision_bytes = precision_bytes
    if indexer_precision_bytes <= 0:
        raise ValueError("indexer_precision_bytes must be positive")
    if checkpoint_interval_tokens is not None and checkpoint_interval_tokens <= 0:
        raise ValueError("checkpoint_interval_tokens must be positive")

    reference = dict(model.calculator_reference)
    formula = str(reference.get("formula_id", ""))
    fields = reference.get("fields")
    if not formula or not isinstance(fields, dict):
        raise ValueError(
            f"model {model.name} lacks a canonical calculator_reference"
        )
    semantic_class = str(reference.get("semantic_class", "token_payload"))
    by_component: dict[str, int] = {}

    if formula == "standard_gqa":
        draft_layers = 0
        if include_draft:
            draft_layers += int(fields.get("num_nextn_predict_layers", 0))
            draft_layers += int(fields.get("num_mtp_modules", 0)) * int(
                fields.get("mtp_transformer_layers", 0)
            )
        layers = int(fields["num_hidden_layers"]) + draft_layers
        elements = (
            tokens
            * sequence_count
            * layers
            * int(fields["num_key_value_heads"])
            * 2
            * int(fields["head_dim"])
        )
        by_component["kv"] = _integer_bytes(elements * precision_bytes)

    elif formula == "mla":
        layers = int(fields["num_hidden_layers"])
        if include_draft:
            layers += int(fields.get("num_nextn_predict_layers", 0))
        elements = (
            tokens
            * sequence_count
            * layers
            * (int(fields["kv_lora_rank"]) + int(fields["qk_rope_head_dim"]))
        )
        by_component["mla"] = _integer_bytes(elements * precision_bytes)

    elif formula == "dsa_mla":
        layers = int(fields["num_hidden_layers"])
        draft_layers = (
            int(fields.get("num_nextn_predict_layers", 0)) if include_draft else 0
        )
        kv_elements = (
            tokens
            * sequence_count
            * (layers + draft_layers)
            * (int(fields["kv_lora_rank"]) + int(fields["qk_rope_head_dim"]))
        )
        index_layers = int(fields.get("indexer_full_layers", layers))
        if include_draft:
            index_layers += int(fields.get("draft_indexer_layers", draft_layers))
        index_elements = (
            tokens
            * sequence_count
            * index_layers
            * int(fields["index_head_dim"])
        )
        by_component["mla"] = _integer_bytes(kv_elements * precision_bytes)
        by_component["indexer"] = _integer_bytes(
            index_elements * indexer_precision_bytes
        )

    elif formula == "kimi_kda_mla_hybrid":
        mla_elements = (
            tokens
            * sequence_count
            * int(fields["full_attention_layers"])
            * (int(fields["kv_lora_rank"]) + int(fields["qk_rope_head_dim"]))
        )
        by_component["mla"] = _integer_bytes(mla_elements * precision_bytes)
        if include_state:
            conv_elements = int(fields["kda_layers"]) * (
                int(fields["kda_conv_kernel_size"]) - 1
            ) * (
                int(fields["kda_num_heads"]) * int(fields["kda_head_dim"])
                + int(fields["kda_num_key_heads"])
                * int(fields["kda_key_head_dim"])
                + int(fields["kda_num_value_heads"])
                * int(fields["kda_value_head_dim"])
            )
            recurrent_elements = (
                int(fields["kda_layers"])
                * int(fields["kda_num_value_heads"])
                * int(fields["kda_value_head_dim"])
                * int(fields["kda_key_head_dim"])
            )
            checkpoint_bytes = (
                conv_elements * int(fields["kda_conv_state_bytes_per_element"])
                + recurrent_elements
                * int(fields["kda_recurrent_state_bytes_per_element"])
            )
            by_component["kda_state"] = (
                sequence_count
                * _checkpoint_count(tokens, checkpoint_interval_tokens)
                * checkpoint_bytes
            )

    elif formula == "inkling_hybrid":
        full_layers = int(fields["full_attention_layers"])
        sliding_layers = int(fields["sliding_attention_layers"])
        if include_draft:
            full_layers += int(fields.get("draft_full_attention_layers", 0))
            sliding_layers += int(fields.get("draft_sliding_attention_layers", 0))
        global_elements = (
            tokens
            * sequence_count
            * full_layers
            * int(fields["num_key_value_heads"])
            * 2
            * int(fields["head_dim"])
        )
        sliding_elements = (
            min(tokens, int(fields["sliding_window"]))
            * sequence_count
            * sliding_layers
            * int(fields["swa_num_key_value_heads"])
            * 2
            * int(fields["swa_head_dim"])
        )
        by_component["global_kv"] = _integer_bytes(global_elements * precision_bytes)
        by_component["sliding_kv"] = _integer_bytes(
            sliding_elements * precision_bytes
        )
        if include_state:
            history = int(fields["sconv_kernel_size"]) - 1
            global_per_layer = history * (
                2 * int(fields["num_key_value_heads"]) * int(fields["head_dim"])
                + 2 * int(fields["hidden_size"])
            )
            sliding_per_layer = history * (
                2
                * int(fields["swa_num_key_value_heads"])
                * int(fields["swa_head_dim"])
                + 2 * int(fields["hidden_size"])
            )
            checkpoint_bytes = (
                full_layers * global_per_layer + sliding_layers * sliding_per_layer
            ) * int(fields["sconv_state_bytes_per_element"])
            by_component["sconv_state"] = (
                sequence_count
                * _checkpoint_count(tokens, checkpoint_interval_tokens)
                * checkpoint_bytes
            )

    elif formula == "qwen_linear_full_hybrid":
        elements = (
            tokens
            * sequence_count
            * int(fields["full_attention_layers"])
            * 2
            * int(fields["num_key_value_heads"])
            * int(fields["head_dim"])
        )
        by_component["full_kv"] = _integer_bytes(elements * precision_bytes)
        if include_state:
            conv_elements = (
                int(fields["linear_attention_layers"])
                * int(fields["linear_conv_kernel_dim"])
                * (
                    2
                    * int(fields["linear_num_key_heads"])
                    * int(fields["linear_key_head_dim"])
                    + int(fields["linear_num_value_heads"])
                    * int(fields["linear_value_head_dim"])
                )
            )
            recurrent_elements = (
                int(fields["linear_attention_layers"])
                * int(fields["linear_num_value_heads"])
                * int(fields["linear_key_head_dim"])
                * int(fields["linear_value_head_dim"])
            )
            by_component["linear_state"] = sequence_count * (
                conv_elements * 2 + recurrent_elements * 4
            )

    elif formula == "mixed_full_sliding_gqa":
        kv_heads = int(fields["num_key_value_heads"])
        full_heads = int(fields.get("num_global_key_value_heads", kv_heads))
        full_key = int(fields.get("global_head_dim", fields["head_dim"]))
        full_value = int(
            fields.get("global_v_head_dim", fields.get("v_head_dim", full_key))
        )
        sliding_heads = int(
            fields.get(
                "swa_num_key_value_heads",
                fields.get("sliding_num_key_value_heads", kv_heads),
            )
        )
        sliding_key = int(
            fields.get(
                "swa_head_dim",
                fields.get("sliding_head_dim", fields["head_dim"]),
            )
        )
        sliding_value = int(
            fields.get(
                "swa_v_head_dim",
                fields.get(
                    "sliding_v_head_dim", fields.get("v_head_dim", sliding_key)
                ),
            )
        )
        full_elements = (
            tokens
            * sequence_count
            * int(fields["full_attention_layers"])
            * full_heads
            * (full_key + full_value)
        )
        sliding_elements = (
            min(tokens, int(fields["sliding_window"]))
            * sequence_count
            * int(fields["sliding_attention_layers"])
            * sliding_heads
            * (sliding_key + sliding_value)
        )
        by_component["full_kv"] = _integer_bytes(full_elements * precision_bytes)
        by_component["sliding_kv"] = _integer_bytes(
            sliding_elements * precision_bytes
        )

    elif formula == "minimax_msa":
        main_elements = (
            tokens
            * sequence_count
            * int(fields["num_hidden_layers"])
            * 2
            * int(fields["num_key_value_heads"])
            * int(fields["head_dim"])
        )
        index_elements = (
            tokens
            * sequence_count
            * int(fields["sparse_attention_layers"])
            * int(fields["index_head_dim"])
        )
        by_component["main_kv"] = _integer_bytes(main_elements * precision_bytes)
        by_component["indexer"] = _integer_bytes(
            index_elements * indexer_precision_bytes
        )

    elif formula == "deepseek_v4_hybrid":
        layers = int(fields["num_hidden_layers"])
        ratios = [int(value) for value in fields["compress_ratios"]]
        active = ratios if include_draft else ratios[:layers]
        window_elements = (
            len(active)
            * int(fields["sliding_window"])
            * int(fields["head_dim"])
            * sequence_count
        )
        compressed_elements = sequence_count * sum(
            floor(tokens / ratio) * int(fields["head_dim"])
            for ratio in active
            if ratio > 0
        )
        index_elements = sequence_count * sum(
            floor(tokens / 4) * int(fields["index_head_dim"])
            for ratio in active
            if ratio == 4
        )
        by_component["main_kv"] = _integer_bytes(
            (window_elements + compressed_elements) * precision_bytes
        )
        by_component["indexer"] = _integer_bytes(
            index_elements * indexer_precision_bytes
        )

    elif formula == "deepseek_v41_csa2":
        if (precision_bytes, indexer_precision_bytes) not in {(0.5, 0.5), (2.0, 2.0)}:
            raise ValueError("V4.1 calculator supports released FP4 and BF16 reference layouts")
        if encoder_tail_policy not in {"recompute", "retain"}:
            raise ValueError("encoder_tail_policy must be recompute or retain")
        released = precision_bytes == 0.5
        dim, index_dim = int(fields["head_dim"]), int(fields["index_head_dim"])
        main_entry = int(dim * precision_bytes) + (dim // 16 if released else 0)
        index_entry = int(index_dim * indexer_precision_bytes) + (index_dim // 32 if released else 0)
        swa_entry = dim + dim // 32 if released else dim * 2
        by_component.update(
            encoder_main_kv=sequence_count * 3 * (tokens // 2) * main_entry,
            encoder_indexer=sequence_count * 3 * (tokens // 2) * index_entry,
            decoder_main_kv=sequence_count * tokens * main_entry,
            decoder_indexer=sequence_count * tokens * index_entry,
        )
        window = min(tokens, int(fields["sliding_window"]))
        if include_state:
            by_component["encoder_swa"] = sequence_count * 20 * window * swa_entry
            by_component["decoder_swa"] = sequence_count * 20 * window * swa_entry
            by_component["compression_tail"] = sequence_count * 3 * 2 * dim * 4 * (tokens % 2)
            if encoder_tail_policy == "retain":
                hc, hidden = int(fields["hc_mult"]), int(fields["hidden_size"])
                by_component["encoder_outputs"] = sequence_count * window * (hc * hidden * 2 + hc * 4)
        if include_draft:
            by_component["draft_swa"] = sequence_count * 3 * window * swa_entry
    else:
        raise ValueError(f"unsupported calculator formula: {formula}")

    settings: dict[str, Any] = {
        "tokens": tokens,
        "sequence_count": sequence_count,
        "precision_bytes": precision_bytes,
        "indexer_precision_bytes": indexer_precision_bytes,
        "include_draft": include_draft,
        "include_state": include_state,
        "checkpoint_interval_tokens": checkpoint_interval_tokens,
    }
    if formula == "deepseek_v41_csa2":
        settings.update(encoder_tail_policy=encoder_tail_policy,
            capacity_scope="held_cache_state_not_recovery_peak",
            known_buffer_reservations={
                "compression_tail": sequence_count * 24576 if tokens and include_state else 0,
                "swa": sequence_count * 40 * 128 * swa_entry if tokens and include_state else 0,
                "draft_swa": sequence_count * 3 * 128 * swa_entry if tokens and include_draft else 0,
            })
    return CalculatorReferenceFootprint(
        total_bytes=sum(by_component.values()),
        formula_id=formula,
        settings=settings,
        snapshot_sha256=model.calculator_snapshot_sha256 or "unverified",
        semantic_class=semantic_class,
        by_component=by_component,
    )


def _checkpoint_count(tokens: int, interval_tokens: int | None) -> int:
    if tokens == 0:
        return 0
    return 1 if interval_tokens is None else ceil(tokens / interval_tokens)


def _integer_bytes(value: float | int) -> int:
    if isinstance(value, int):
        return value
    if not float(value).is_integer():
        raise ValueError(f"calculator result is a fractional byte count: {value}")
    return int(value)
