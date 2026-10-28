"""Model registry and strict config conversion."""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import Any

from ssu_sim.models.attention import (
    DeepSeekV4HybridAttentionProfile,
    DeepSeekV41AttentionProfile,
    DenseKVAttentionProfile,
    DSAAttentionProfile,
    InklingHybridAttentionProfile,
    KDAHybridAttentionProfile,
    MLAAttentionProfile,
    MiniMaxMSAAttentionProfile,
    MixedFullSlidingGQAAttentionProfile,
    QwenLinearFullHybridAttentionProfile,
)
from ssu_sim.models.llm_model import (
    AnalyticalFallbackPrefillFormula,
    ComputeProfile,
    DenseFFNProfile,
    DSAPrefillFormula,
    DeepSeekV4PrefillFormula,
    DeepSeekV41PrefillFormula,
    GQAPrefillFormula,
    InklingHybridPrefillFormula,
    KDAHybridPrefillFormula,
    LayeredFFNProfile,
    MLAPrefillFormula,
    MiniMaxMSAPrefillFormula,
    MixedFullSlidingPrefillFormula,
    MoEFFNProfile,
    MoEProfile,
    ParallelFFNProfile,
    ModelProfile,
    PrefillFormula,
    QwenLinearHybridPrefillFormula,
)
from ssu_sim.models.metadata import (
    canonical_sha256,
    catalog_from_dict,
    evidence_from_dict,
    identity_from_dict,
)
from ssu_sim.models.state import (
    PrefixGranularityPolicy,
    SparseAccessPolicy,
    StatePersistencePolicy,
)


_LEGACY_MODEL_FIELDS = {"head_dim", "dtype_bytes"}
_LEGACY_PREFILL_FIELDS = {"prefill_us_per_token"}
_LEGACY_ATTENTION_FIELDS = {
    "compression_ratio",
    "cache_ratio",
    "selected_block_ratio",
    "read_amplification",
    "index_bytes_per_block",
    "routing_us_per_request",
}
_LEGACY_COMPUTE_FIELDS = {
    "mode",
    "fixed_flops",
    "flops_per_token",
    "flops_per_token_squared",
}
_PRECISION_FORMAT_BITS = {
    "fp4": 4,
    "float4": 4,
    "int4": 4,
    "fp8": 8,
    "float8": 8,
    "int8": 8,
    "bf16": 16,
    "fp16": 16,
    "float16": 16,
    "int16": 16,
}
_QWEN_MOE = MoEProfile(
    n_experts=128,
    experts_per_token=8,
    expert_intermediate_size=1536,
    router_included=False,
)
_DEEPSEEK_MOE = MoEProfile(
    n_experts=256,
    experts_per_token=8,
    expert_intermediate_size=2048,
    router_included=False,
    first_dense_layers=3,
    shared_experts=1,
)
_SAMPLE_GQA_COMPUTE = ComputeProfile(
    effective_prefill_tflops=2000.0,
    prefill_formula=GQAPrefillFormula(moe=_QWEN_MOE),
    scheduler_overhead_us=0.0,
)
_SAMPLE_MLA_COMPUTE = ComputeProfile(
    effective_prefill_tflops=2000.0,
    prefill_formula=MLAPrefillFormula(
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        moe=_DEEPSEEK_MOE,
    ),
    scheduler_overhead_us=0.0,
)
_SAMPLE_DSA_COMPUTE = ComputeProfile(
    effective_prefill_tflops=2000.0,
    prefill_formula=DSAPrefillFormula(
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        window_size=2048,
        global_tokens=128,
        moe=_DEEPSEEK_MOE,
    ),
    scheduler_overhead_us=0.0,
)


ATTENTION_PROFILE_FACTORIES: dict[str, type] = {
    "mha": DenseKVAttentionProfile,
    "gqa": DenseKVAttentionProfile,
    "mqa": DenseKVAttentionProfile,
    "mla": MLAAttentionProfile,
    "dsa": DSAAttentionProfile,
    "deepseek_v4": DeepSeekV4HybridAttentionProfile,
    "deepseek_v41": DeepSeekV41AttentionProfile,
    "mixed_full_sliding_gqa": MixedFullSlidingGQAAttentionProfile,
    "mixed_full_swa": MixedFullSlidingGQAAttentionProfile,
    "minimax_msa": MiniMaxMSAAttentionProfile,
    "qwen_linear_full_hybrid": QwenLinearFullHybridAttentionProfile,
    "qwen_linear_full": QwenLinearFullHybridAttentionProfile,
    "kimi_kda_mla_hybrid": KDAHybridAttentionProfile,
    "kda_mla": KDAHybridAttentionProfile,
    "inkling_hybrid": InklingHybridAttentionProfile,
    "inkling": InklingHybridAttentionProfile,
}


def attention_from_dict(data: dict[str, Any]):
    kind = str(data.get("type", "")).lower()
    legacy = sorted(_LEGACY_ATTENTION_FIELDS.intersection(data))
    if legacy:
        raise ValueError(
            "legacy attention fields are no longer supported: "
            + ", ".join(legacy)
            + "; migrate to explicit dimensions, dtype_bits, index_topk, and read_mode"
        )
    profile_type = ATTENTION_PROFILE_FACTORIES.get(kind)
    if profile_type is None:
        raise ValueError(
            "attention.type must name a registered profile; got " + repr(kind)
        )
    raw = {key: copy.deepcopy(value) for key, value in data.items() if key != "type"}
    if profile_type is DenseKVAttentionProfile:
        raw["name"] = kind
    if "compress_ratios" in raw:
        values = raw["compress_ratios"]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("attention.compress_ratios must be a non-empty sequence")
        raw["compress_ratios"] = tuple(int(value) for value in values)
    for policy_name in (
        "state_persistence",
        "sliding_persistence",
        "sconv_persistence",
        "swa_persistence",
    ):
        if policy_name in raw:
            raw[policy_name] = state_persistence_from_value(raw[policy_name])
            if policy_name == "swa_persistence":
                raw["swa_persistence_origin"] = "model_config"
    if "prefix_granularity" in raw:
        raw["prefix_granularity"] = prefix_granularity_from_value(
            raw["prefix_granularity"]
        )
    if "sparse_access_policy" in raw:
        raw["sparse_access_policy"] = sparse_access_policy_from_value(
            raw["sparse_access_policy"]
        )
    return _strict_dataclass_construct(profile_type, raw, "attention")


def compute_from_dict(data: Any) -> ComputeProfile:
    if data is None:
        raise ValueError("model compute is required")
    if not isinstance(data, dict):
        raise ValueError("model compute must be a mapping")
    legacy = sorted(_LEGACY_COMPUTE_FIELDS.intersection(data))
    if legacy:
        raise ValueError(
            "legacy compute fields are no longer supported: "
            + ", ".join(legacy)
            + "; provide FLOPs fields directly"
        )
    if "effective_prefill_tflops" not in data:
        raise ValueError("missing required field: compute.effective_prefill_tflops")
    formula_data = data.get("prefill_formula")
    if not isinstance(formula_data, dict):
        raise ValueError("compute.prefill_formula must be a mapping")
    return ComputeProfile(
        effective_prefill_tflops=float(data["effective_prefill_tflops"]),
        prefill_formula=prefill_formula_from_dict(formula_data),
        scheduler_overhead_us=float(data.get("scheduler_overhead_us", 0.0)),
        timing_mode=str(data.get("timing_mode", "normalized_reference")),
        operation_class_tflops={
            str(key): float(value)
            for key, value in _mapping_or_empty(
                data.get("operation_class_tflops"),
                "compute.operation_class_tflops",
            ).items()
        },
        timing_omissions=tuple(str(value) for value in data.get("timing_omissions", ())),
        output_head_spec=(dict(data["output_head_spec"]) if data.get("output_head_spec") is not None else None),
    )


PREFILL_FORMULA_FACTORIES: dict[str, type] = {
    "mha": GQAPrefillFormula,
    "gqa": GQAPrefillFormula,
    "mqa": GQAPrefillFormula,
    "standard_gqa": GQAPrefillFormula,
    "mla": MLAPrefillFormula,
    "dsa": DSAPrefillFormula,
    "dsa_mla": DSAPrefillFormula,
    "deepseek_v4": DeepSeekV4PrefillFormula,
    "deepseek_v41": DeepSeekV41PrefillFormula,
    "mixed_full_sliding_gqa": MixedFullSlidingPrefillFormula,
    "mixed_full_swa": MixedFullSlidingPrefillFormula,
    "minimax_msa": MiniMaxMSAPrefillFormula,
    "qwen_linear_full_hybrid": QwenLinearHybridPrefillFormula,
    "qwen_linear_full": QwenLinearHybridPrefillFormula,
    "kimi_kda_mla_hybrid": KDAHybridPrefillFormula,
    "kda_mla": KDAHybridPrefillFormula,
    "inkling_hybrid": InklingHybridPrefillFormula,
    "inkling": InklingHybridPrefillFormula,
    "analytical_fallback": AnalyticalFallbackPrefillFormula,
}


def prefill_formula_from_dict(data: dict[str, Any]) -> PrefillFormula:
    kind = _required_str(data, "type").lower()
    formula_type = PREFILL_FORMULA_FACTORIES.get(kind)
    if formula_type is None:
        raise ValueError(
            "compute.prefill_formula.type must name a registered formula; got "
            + repr(kind)
        )
    raw = {key: copy.deepcopy(value) for key, value in data.items() if key != "type"}
    if "moe" in raw:
        raw["moe"] = moe_from_dict(raw["moe"])
    if "ffn" in raw:
        raw["ffn"] = ffn_from_dict(raw["ffn"])
    if formula_type is GQAPrefillFormula:
        raw["type"] = kind if kind in {"mha", "gqa", "mqa"} else "gqa"
    elif formula_type is AnalyticalFallbackPrefillFormula:
        raw["type"] = str(raw.pop("formula_name", kind))
    return _strict_dataclass_construct(
        formula_type,
        raw,
        "compute.prefill_formula",
    )


def ffn_from_dict(data: Any):
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("compute.prefill_formula.ffn must be a mapping")
    kind = str(data.get("type", "dense")).lower()
    raw = {key: copy.deepcopy(value) for key, value in data.items() if key != "type"}
    if kind == "dense":
        return _strict_dataclass_construct(DenseFFNProfile, raw, "ffn")
    if kind in {"moe", "expert"}:
        return _strict_dataclass_construct(MoEFFNProfile, raw, "ffn")
    if kind == "parallel":
        branches = raw.pop("branches", None)
        if not isinstance(branches, list) or not branches:
            raise ValueError("parallel ffn.branches must be a non-empty list")
        raw["branches"] = tuple(ffn_from_dict(value) for value in branches)
        return _strict_dataclass_construct(ParallelFFNProfile, raw, "ffn")
    if kind == "layered":
        segments = raw.pop("segments", None)
        if not isinstance(segments, list) or not segments:
            raise ValueError("layered ffn.segments must be a non-empty list")
        if raw:
            raise ValueError("unsupported layered ffn fields: " + ", ".join(sorted(raw)))
        return LayeredFFNProfile(tuple(ffn_from_dict(value) for value in segments))
    raise ValueError(f"unsupported ffn.type: {kind}")


def moe_from_dict(data: Any) -> MoEProfile | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("compute.prefill_formula.moe must be a mapping")
    return MoEProfile(
        n_experts=_required_int(data, "n_experts"),
        experts_per_token=_required_int(data, "experts_per_token"),
        expert_intermediate_size=_required_int(data, "expert_intermediate_size"),
        router_included=bool(data.get("router_included", True)),
        first_dense_layers=_non_negative_int(data, "first_dense_layers", default=0),
        shared_experts=_non_negative_int(data, "shared_experts", default=0),
        dense_intermediate_size=_optional_int(data, "dense_intermediate_size"),
        dense_gated=bool(data.get("dense_gated", True)),
    )


def model_from_dict(
    data: dict[str, Any], runtime_config: dict[str, Any] | None = None
) -> ModelProfile:
    model_config_sha256 = canonical_sha256(data)
    runtime_config_sha256 = (
        canonical_sha256(runtime_config) if runtime_config is not None else None
    )
    data = _apply_runtime_config(data, runtime_config)
    legacy_prefill = sorted(_LEGACY_PREFILL_FIELDS.intersection(data))
    if legacy_prefill:
        raise ValueError(
            "legacy prefill timing fields are no longer supported: "
            + ", ".join(legacy_prefill)
            + "; configure compute.effective_prefill_tflops and FLOPs coefficients"
        )
    legacy = sorted(_LEGACY_MODEL_FIELDS.intersection(data))
    if legacy:
        raise ValueError(
            "legacy model fields are no longer supported: "
            + ", ".join(legacy)
            + "; move cache dimensions and dtype_bits under attention"
        )
    identity = identity_from_dict(data)
    attention_data = data.get("attention")
    if not isinstance(attention_data, dict):
        raise ValueError("model attention must be a mapping")
    attention_data = _with_calculator_reference(attention_data, data)
    compute_data = data.get("compute")
    formula = (
        compute_data.get("prefill_formula", {})
        if isinstance(compute_data, dict)
        else {}
    )
    formula_id = (
        str(formula.get("type"))
        if isinstance(formula, dict) and formula.get("type")
        else None
    )
    model = ModelProfile(
        name=str(data.get("name", identity.id)),
        d_model=_required_int(data, "d_model"),
        num_layers=_required_int(data, "num_layers"),
        num_q_heads=_required_int(data, "num_q_heads"),
        num_kv_heads=_required_int(data, "num_kv_heads"),
        attention=attention_from_dict(attention_data),
        compute=compute_from_dict(data.get("compute")),
        decode_us_per_token=float(data.get("decode_us_per_token", 0.0)),
        max_context_tokens=(
            _required_int(data, "max_context_tokens")
            if "max_context_tokens" in data
            else None
        ),
        identity=identity,
        evidence=evidence_from_dict(data.get("evidence")),
        catalog=catalog_from_dict(data.get("catalog")),
        model_config_sha256=model_config_sha256,
        runtime_config_sha256=runtime_config_sha256,
        calculator_snapshot_sha256=_calculator_snapshot_sha256(data),
        formula_implementation_id=str(
            data.get("formula_implementation_id", formula_id or "unknown")
        ),
        draft_support=str(data.get("draft_support", "none")),
        calculator_reference=(
            dict(data["calculator_reference"])
            if isinstance(data.get("calculator_reference"), dict)
            else {}
        ),
    )
    if model.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    _validate_formula_matches_attention(model)
    model.cache_components()  # Validate dimensions, quantization, and head relationships.
    return model


def _apply_runtime_config(
    data: dict[str, Any], runtime_config: dict[str, Any] | None
) -> dict[str, Any]:
    if runtime_config is None:
        return data
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime_config must be a mapping")

    merged = copy.deepcopy(data)
    _apply_runtime_compute(merged, runtime_config)
    _apply_runtime_cache_precision(merged, runtime_config)
    return merged


def _apply_runtime_compute(data: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    hardware = _optional_mapping(runtime_config, "hardware")
    compute_runtime = _optional_mapping(runtime_config, "compute")
    effective_prefill_tflops = None
    timing_mode = None
    operation_rates: dict[str, float] = {}
    timing_omissions: list[str] | None = None
    if hardware is not None and "effective_prefill_tflops" in hardware:
        effective_prefill_tflops = hardware["effective_prefill_tflops"]
    if hardware is not None and "effective_tflops" in hardware:
        rates = hardware["effective_tflops"]
        if isinstance(rates, dict):
            if "default" in rates:
                effective_prefill_tflops = rates["default"]
            operation_rates.update(_normalize_operation_rates(rates))
        else:
            effective_prefill_tflops = rates
    if hardware is not None:
        timing_mode = hardware.get("timing_mode")
        if "timing_omissions" in hardware:
            timing_omissions = [str(value) for value in hardware["timing_omissions"]]
    if compute_runtime is not None and "effective_prefill_tflops" in compute_runtime:
        effective_prefill_tflops = compute_runtime["effective_prefill_tflops"]
    if compute_runtime is not None and "effective_tflops" in compute_runtime:
        rates = compute_runtime["effective_tflops"]
        if isinstance(rates, dict):
            if "default" in rates:
                effective_prefill_tflops = rates["default"]
            operation_rates.update(_normalize_operation_rates(rates))
        else:
            effective_prefill_tflops = rates
    if compute_runtime is not None:
        timing_mode = compute_runtime.get("timing_mode", timing_mode)
        if "timing_omissions" in compute_runtime:
            timing_omissions = [
                str(value) for value in compute_runtime["timing_omissions"]
            ]

    compute_data = data.setdefault("compute", {})
    if not isinstance(compute_data, dict):
        raise ValueError("model compute must be a mapping")
    if effective_prefill_tflops is not None:
        compute_data["effective_prefill_tflops"] = effective_prefill_tflops
    if timing_mode is not None:
        compute_data["timing_mode"] = str(timing_mode)
    if operation_rates:
        compute_data["operation_class_tflops"] = operation_rates
    if timing_omissions is not None:
        compute_data["timing_omissions"] = timing_omissions


def _apply_runtime_cache_precision(
    data: dict[str, Any], runtime_config: dict[str, Any]
) -> None:
    precision = _optional_mapping(runtime_config, "cache_precision")
    if precision is None:
        return

    attention = data.get("attention")
    if not isinstance(attention, dict):
        raise ValueError("model attention must be a mapping")
    kind = str(attention.get("type", "")).lower()
    handler = PRECISION_OVERRIDE_HANDLERS.get(kind)
    if handler is not None:
        for field_name, component_name in handler:
            _set_runtime_dtype(attention, precision, field_name, component_name)
        _validate_fixed_state_precision(attention, precision)
    elif kind == "deepseek_v41":
        bits = _runtime_dtype_value(precision, "main_dtype_bits", "dtype_bits")
        if bits is not None:
            if bits not in {4, 16}:
                raise ValueError("V4.1 cache precision must be released FP4 or BF16 reference")
            attention.update(main_dtype_bits=bits, indexer_dtype_bits=bits,
                             swa_dtype_bits=8 if bits == 4 else 16)
        expected_main = 288 if attention.get("main_dtype_bits", 4) == 4 else 1024
        expected_index = 68 if attention.get("indexer_dtype_bits", 4) == 4 else 256
        for name, expected in (("main_entry_bytes", expected_main), ("indexer_entry_bytes", expected_index)):
            if name in precision and int(precision[name]) != expected:
                raise ValueError(f"V4.1 {name} must be {expected} for this layout")
    elif kind == "deepseek_v4":
        explicit_entries = False
        for field in ("main_entry_bytes", "indexer_entry_bytes"):
            if field in precision:
                explicit_entries = True
                attention[field] = _positive_runtime_int(
                    precision[field], f"cache_precision.{field}"
                )
        if not explicit_entries and "format" in precision:
            bits = _dtype_bits_from_format(str(precision["format"]))
            if bits >= 16:
                attention["main_entry_bytes"] = (
                    int(attention.get("main_nope_elements", 448))
                    + int(attention.get("main_rope_elements", 64))
                ) * bits // 8
                attention["indexer_entry_bytes"] = (
                    int(attention.get("indexer_elements", 128)) * bits // 8
                )
                attention["main_scale_bytes"] = 0
                attention["main_padding_bytes"] = 0


PRECISION_OVERRIDE_HANDLERS: dict[str, tuple[tuple[str, str], ...]] = {
    kind: (("key_dtype_bits", "key"), ("value_dtype_bits", "value"))
    for kind in (
        "mha",
        "gqa",
        "mqa",
        "mixed_full_sliding_gqa",
        "mixed_full_swa",
        "qwen_linear_full_hybrid",
        "qwen_linear_full",
        "inkling_hybrid",
        "inkling",
    )
}
PRECISION_OVERRIDE_HANDLERS.update(
    {
        "mla": (("latent_dtype_bits", "latent"), ("rope_key_dtype_bits", "rope_key")),
        "dsa": (
            ("latent_dtype_bits", "latent"),
            ("rope_key_dtype_bits", "rope_key"),
            ("index_dtype_bits", "index"),
        ),
        "minimax_msa": (
            ("key_dtype_bits", "key"),
            ("value_dtype_bits", "value"),
            ("indexer_dtype_bits", "indexer"),
        ),
        "kimi_kda_mla_hybrid": (
            ("latent_dtype_bits", "latent"),
            ("rope_key_dtype_bits", "rope_key"),
        ),
        "kda_mla": (
            ("latent_dtype_bits", "latent"),
            ("rope_key_dtype_bits", "rope_key"),
        ),
    }
)


def _set_runtime_dtype(
    attention: dict[str, Any], precision: dict[str, Any], field: str, component: str
) -> None:
    value = _runtime_dtype_value(precision, field, f"{component}_dtype_bits")
    if value is not None:
        attention[field] = value


def _runtime_dtype_value(precision: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in precision:
            return _positive_runtime_int(precision[key], f"cache_precision.{key}")
    if "default_dtype_bits" in precision:
        return _positive_runtime_int(
            precision["default_dtype_bits"], "cache_precision.default_dtype_bits"
        )
    if "dtype_bits" in precision:
        return _positive_runtime_int(precision["dtype_bits"], "cache_precision.dtype_bits")
    if "format" in precision:
        return _dtype_bits_from_format(str(precision["format"]))
    return None


def _dtype_bits_from_format(value: str) -> int:
    normalized = value.lower().replace("-", "").replace("_", "")
    if normalized not in _PRECISION_FORMAT_BITS:
        raise ValueError(f"unsupported cache_precision.format: {value}")
    return _PRECISION_FORMAT_BITS[normalized]


def _optional_mapping(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = data.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"runtime_config.{name} must be a mapping")
    return value


def _positive_runtime_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def default_qwen_gqa() -> ModelProfile:
    return ModelProfile(
        name="qwen-gqa-sample",
        d_model=4096,
        num_layers=94,
        num_q_heads=64,
        num_kv_heads=4,
        attention=DenseKVAttentionProfile(128, 128, 4, 4, name="gqa"),
        compute=_SAMPLE_GQA_COMPUTE,
    )


def default_deepseek_mla() -> ModelProfile:
    return ModelProfile(
        name="deepseek-mla-sample",
        d_model=7168,
        num_layers=61,
        num_q_heads=128,
        num_kv_heads=128,
        attention=MLAAttentionProfile(512, 64, 4, 4, materialization_us_per_token=0.25),
        compute=_SAMPLE_MLA_COMPUTE,
    )


def default_deepseek_dsa(*, read_mode: str = "sparse_on_demand") -> ModelProfile:
    return ModelProfile(
        name="deepseek-dsa-sample",
        d_model=7168,
        num_layers=61,
        num_q_heads=128,
        num_kv_heads=128,
        attention=DSAAttentionProfile(
            latent_dim=512,
            rope_key_dim=64,
            index_key_dim=128,
            latent_dtype_bits=4,
            rope_key_dtype_bits=4,
            index_dtype_bits=4,
            index_topk=2048,
            read_mode=read_mode,
            block_mapping="window_global",
            window_size=2048,
            global_tokens=128,
            routing_us_per_query=10.0,
        ),
        compute=_SAMPLE_DSA_COMPUTE,
    )


def _require_attention_names(*names: str):
    def validate(model: ModelProfile) -> None:
        if model.attention.name not in names:
            raise ValueError(
                f"{model.compute.prefill_formula.type} prefill formula requires "
                f"attention profile in {names}; got {model.attention.name}"
            )

    return validate


def _validate_dsa_pair(model: ModelProfile) -> None:
    _require_attention_names("dsa")(model)
    formula = model.compute.prefill_formula
    if (
        getattr(model.attention, "window_size") != getattr(formula, "window_size")
        or getattr(model.attention, "global_tokens") != getattr(formula, "global_tokens")
    ):
        raise ValueError(
            "dsa attention window_size/global_tokens must match compute.prefill_formula"
        )


def _validate_v4_pair(model: ModelProfile) -> None:
    _require_attention_names("deepseek_v4")(model)
    formula = model.compute.prefill_formula
    if (
        getattr(model.attention, "index_topk") != getattr(formula, "index_topk")
        or getattr(model.attention, "sliding_window")
        != getattr(formula, "sliding_window")
    ):
        raise ValueError(
            "deepseek_v4 attention top-k/sliding_window must match prefill formula"
        )


def _validate_v41_pair(model: ModelProfile) -> None:
    _require_attention_names("deepseek_v41")(model)
    formula = model.compute.prefill_formula
    if model.num_layers != 40 or model.num_q_heads != 64 or model.d_model != 5120:
        raise ValueError("V4.1 requires 40 layers, 64 query heads and hidden size 5120")
    for name in ("head_dim", "index_head_dim", "sliding_window", "hc_mult"):
        if getattr(model.attention, name) != getattr(formula, name):
            raise ValueError(f"V4.1 attention and compute must agree on {name}")
    if formula.index_topk != 512 or formula.candidate_positions != 16384:
        raise ValueError("V4.1 uses top-512 selection and a 16384-position candidate bound")


PROFILE_FORMULA_VALIDATORS: dict[str, Any] = {
    "mha": _require_attention_names("mha"),
    "gqa": _require_attention_names("gqa"),
    "mqa": _require_attention_names("mqa"),
    "mla": _require_attention_names("mla"),
    "dsa": _validate_dsa_pair,
    "deepseek_v4": _validate_v4_pair,
    "deepseek_v41": _validate_v41_pair,
    "mixed_full_sliding_gqa": _require_attention_names("mixed_full_sliding_gqa"),
    "minimax_msa": _require_attention_names("minimax_msa"),
    "qwen_linear_full_hybrid": _require_attention_names("qwen_linear_full_hybrid"),
    "kimi_kda_mla_hybrid": _require_attention_names("kimi_kda_mla_hybrid"),
    "inkling_hybrid": _require_attention_names("inkling_hybrid"),
}


def _validate_formula_matches_attention(model: ModelProfile) -> None:
    formula_type = model.compute.prefill_formula.type
    validator = PROFILE_FORMULA_VALIDATORS.get(formula_type)
    if validator is None:
        if isinstance(model.compute.prefill_formula, AnalyticalFallbackPrefillFormula):
            return
        raise ValueError(f"no attention/formula validator registered for {formula_type}")
    validator(model)
    _validate_compute_geometry(model)


def _validate_compute_geometry(model: ModelProfile) -> None:
    """Validate execution geometry independently of cache storage ownership."""
    f, a = model.compute.prefill_formula, model.attention
    kind = f.type
    if kind == "mixed_full_sliding_gqa":
        if f.full_attention_layers + f.sliding_attention_layers != model.num_layers:
            raise ValueError("full/sliding compute layers must cover all execution layers")
        for group in ("full", "sliding"):
            execution = getattr(f, f"{group}_attention_layers")
            declared = getattr(f, f"{group}_kv_projection_layers")
            producers = execution if declared is None else declared
            stored = getattr(a, f"{group}_attention_layers")
            if producers != stored or not 0 < producers <= execution:
                raise ValueError(f"{group} KV producer layers must match stored layers and not exceed execution")
            for suffix in ("key_head_dim", "value_head_dim", "num_kv_heads"):
                x = getattr(f, f"{group}_{suffix}", None)
                y = getattr(a, f"{group}_{suffix}", None)
                if x is not None and y is not None and x != y:
                    raise ValueError(f"{group}_{suffix} differs between attention and compute")
        if f.sliding_window != a.sliding_window:
            raise ValueError("sliding window differs between attention and compute")
    elif kind == "qwen_linear_full_hybrid":
        if f.full_attention_layers + f.linear_attention_layers != model.num_layers:
            raise ValueError("Qwen full/linear compute layers must cover all execution layers")
        for key in ("full_attention_layers", "linear_attention_layers"):
            if getattr(f, key) != getattr(a, key):
                raise ValueError(f"Qwen {key} differs between attention and compute")
    elif kind == "kimi_kda_mla_hybrid":
        if f.mla_attention_layers + f.kda_attention_layers != model.num_layers:
            raise ValueError("Kimi MLA/KDA compute layers must cover all execution layers")
        if f.mla_attention_layers != a.mla_layers or f.kda_attention_layers != a.kda_layers:
            raise ValueError("Kimi attention layer partition differs from compute")
    elif kind == "inkling_hybrid":
        if f.full_attention_layers + f.sliding_attention_layers != model.num_layers:
            raise ValueError("Inkling compute layers must cover all execution layers")
        if f.full_attention_layers != a.global_attention_layers or f.sliding_attention_layers != a.sliding_attention_layers:
            raise ValueError("Inkling attention layer partition differs from compute")
    elif kind == "minimax_msa":
        if f.full_attention_layers + f.sparse_attention_layers != model.num_layers:
            raise ValueError("MSA compute layers must cover all execution layers")
        for key in ("full_attention_layers", "sparse_attention_layers"):
            if getattr(f, key) != getattr(a, key):
                raise ValueError(f"MSA {key} differs between attention and compute")
    elif kind == "dsa":
        producers = f.indexer_compute_layers or model.num_layers
        if producers != a.indexer_layer_count and a.indexer_layer_count is not None:
            raise ValueError("DSA indexer execution layers must match independent indexer producers")

    ffn = getattr(f, "ffn", None)
    if ffn is not None:
        # Layered/parallel profiles validate their own partitions, including branches.
        count = getattr(ffn, "layer_count", None)
        if count is not None and count != model.num_layers:
            raise ValueError("FFN layers must cover all execution layers")
        ffn.operation_breakdown(model=model, q_len=1, batch_size=1)


def state_persistence_from_value(value: Any) -> StatePersistencePolicy:
    if isinstance(value, StatePersistencePolicy):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"prompt_end", "terminal", "terminal_only"}:
            return StatePersistencePolicy.terminal()
        if normalized == "token_history":
            return StatePersistencePolicy.token_history()
        if normalized in {"none", "zero"}:
            return StatePersistencePolicy.none()
        if normalized.startswith("periodic:"):
            return StatePersistencePolicy.periodic(
                int(normalized.split(":", 1)[1]), retain_prompt_end=False
            )
        if normalized.startswith("periodic+prompt_end:"):
            return StatePersistencePolicy.periodic(
                int(normalized.split(":", 1)[1]), retain_prompt_end=True
            )
        raise ValueError(f"unsupported state persistence policy: {value}")
    if not isinstance(value, dict):
        raise ValueError("state persistence policy must be a mapping or string")
    return _strict_dataclass_construct(
        StatePersistencePolicy,
        dict(value),
        "state persistence policy",
    )


def prefix_granularity_from_value(value: Any) -> PrefixGranularityPolicy:
    if isinstance(value, PrefixGranularityPolicy):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"token", "sealed_padded"}:
            return PrefixGranularityPolicy.sealed_padded()
        if normalized in {"block", "complete_blocks_only"}:
            return PrefixGranularityPolicy.full_shared_blocks_only()
        if normalized == "checkpoint_boundary":
            return PrefixGranularityPolicy.sealed_padded()
        raise ValueError(f"unsupported prefix granularity policy: {value}")
    if not isinstance(value, dict):
        raise ValueError("prefix granularity policy must be a mapping or string")
    return _strict_dataclass_construct(
        PrefixGranularityPolicy,
        dict(value),
        "prefix granularity policy",
    )


def sparse_access_policy_from_value(value: Any) -> SparseAccessPolicy:
    if isinstance(value, SparseAccessPolicy):
        return value
    if not isinstance(value, dict):
        raise ValueError("sparse_access_policy must be a mapping")
    raw = dict(value)
    locality_aliases = {
        "clustered": "clustered_lower",
        "dispersed": "dispersed_upper",
    }
    if "locality_scenario" in raw:
        raw["locality_scenario"] = locality_aliases.get(
            str(raw["locality_scenario"]), raw["locality_scenario"]
        )
    return _strict_dataclass_construct(
        SparseAccessPolicy,
        raw,
        "sparse_access_policy",
    )


def _strict_dataclass_construct(profile_type: type, raw: dict[str, Any], scope: str):
    allowed = {item.name for item in fields(profile_type) if item.init}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unsupported {scope} fields: " + ", ".join(unknown))
    try:
        return profile_type(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid {scope} configuration: {exc}") from exc


def _mapping_or_empty(value: Any, scope: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{scope} must be a mapping")
    return value


def _normalize_operation_rates(values: dict[str, Any]) -> dict[str, float]:
    aliases = {
        "projection": "dense_gemm",
        "convolution": "convolution",
        "dense_attention": "dense_attention",
        "sparse_attention": "sparse_attention",
        "indexer": "indexer",
        "recurrent": "recurrent",
        "moe": "moe",
        "routing": "routing",
        "compression": "compression",
        "elementwise": "elementwise",
        "unclassified": "unclassified",
        "dense_gemm": "dense_gemm",
    }
    result: dict[str, float] = {}
    for name, value in values.items():
        if name == "default":
            continue
        operation_class = aliases.get(str(name))
        if operation_class is None:
            raise ValueError(f"unknown hardware effective_tflops class: {name}")
        result[operation_class] = float(value)
    return result


def _validate_fixed_state_precision(
    attention: dict[str, Any], precision: dict[str, Any]
) -> None:
    fixed_fields = (
        "conv_state_dtype_bits",
        "recurrent_state_dtype_bits",
        "kda_conv_state_dtype_bits",
        "kda_recurrent_state_dtype_bits",
        "sconv_state_dtype_bits",
    )
    for field_name in fixed_fields:
        if field_name not in precision:
            continue
        requested = _positive_runtime_int(
            precision[field_name], f"cache_precision.{field_name}"
        )
        configured = int(attention.get(field_name, requested))
        if requested != configured:
            raise ValueError(
                f"cache_precision.{field_name} cannot override fixed state "
                f"precision {configured} bits"
            )


def _calculator_snapshot_sha256(data: dict[str, Any]) -> str | None:
    direct = data.get("calculator_snapshot_sha256")
    if direct:
        return str(direct)
    sources = data.get("sources")
    if isinstance(sources, dict):
        calculator = sources.get("calculator")
        if isinstance(calculator, dict):
            for key in ("snapshot_sha256", "sha256"):
                if calculator.get(key):
                    return str(calculator[key])
    return None


def _with_calculator_reference(
    attention_data: dict[str, Any], model_data: dict[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(attention_data)
    kind = str(result.get("type", "")).lower()
    profile_type = ATTENTION_PROFILE_FACTORIES.get(kind)
    if profile_type is None:
        return result
    supported = {item.name for item in fields(profile_type) if item.init}
    reference = model_data.get("calculator_reference")
    if not isinstance(reference, dict):
        return result
    candidates = {
        "calculator_formula_id": reference.get("formula_id"),
        "calculator_semantic_class": reference.get("semantic_class"),
        "calculator_settings": reference.get("fields"),
        "calculator_snapshot_sha256": _calculator_snapshot_sha256(model_data),
    }
    for name, value in candidates.items():
        if name in supported and value is not None and name not in result:
            result[name] = value
    return result


def _required_int(data: dict[str, Any], name: str) -> int:
    if name not in data:
        raise ValueError(f"missing required field: {name}")
    value = int(data[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_int(data: dict[str, Any], name: str) -> int | None:
    return int(data[name]) if name in data else None


def _non_negative_int(data: dict[str, Any], name: str, *, default: int | None = None) -> int:
    if name not in data:
        if default is not None:
            return default
        raise ValueError(f"missing required field: {name}")
    value = int(data[name])
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _required_str(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required string field: {name}")
    return value
