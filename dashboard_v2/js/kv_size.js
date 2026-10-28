import { initializeKvCompare, configurationKey, aggregateKvGroups } from "./kv_compare.js";

const PRECISIONS = Object.freeze([
  { id: "bf16_fp16", label: "BF16 / FP16", halfBytes: 4 },
  { id: "fp8_int8", label: "FP8 / INT8", halfBytes: 2 },
  { id: "fp4_int4", label: "FP4 / INT4", halfBytes: 1 },
]);

const COMPONENT_LABELS = Object.freeze({
  encoder_main_kv: "Encoder global KV",
  decoder_main_kv: "Decoder global KV",
  encoder_indexer: "Encoder indexer",
  decoder_indexer: "Decoder indexer",
  encoder_swa: "Encoder SWA",
  decoder_swa: "Decoder SWA",
  compression_tail: "Compression tail (FP32)",
  encoder_outputs: "Encoder output snapshot",
  draft_swa: "Draft SWA",

  kv: "KV cache",
  mla: "MLA latent KV",
  indexer: "Indexer",
  kda_state: "KDA checkpoint state",
  global_kv: "Global KV",
  sliding_kv: "Sliding-window KV",
  sconv_state: "SConv checkpoint state",
  full_kv: "Full-attention KV",
  linear_state: "GDN checkpoint state",
  main_kv: "Main KV",
});

const EDITABLE_FIELDS = Object.freeze({
  standard_gqa: [
    "num_hidden_layers", "num_key_value_heads", "head_dim",
    "num_nextn_predict_layers", "num_mtp_modules", "mtp_transformer_layers",
  ],
  mla: [
    "num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim",
    "num_nextn_predict_layers",
  ],
  dsa_mla: [
    "num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim",
    "num_nextn_predict_layers", "indexer_full_layers",
    "draft_indexer_layers", "index_head_dim",
  ],
  kimi_kda_mla_hybrid: [
    "full_attention_layers", "kv_lora_rank", "qk_rope_head_dim",
    "kda_layers", "kda_conv_kernel_size", "kda_num_heads", "kda_head_dim",
    "kda_num_key_heads", "kda_key_head_dim", "kda_num_value_heads",
    "kda_value_head_dim", "kda_conv_state_bytes_per_element",
    "kda_recurrent_state_bytes_per_element",
  ],
  inkling_hybrid: [
    "full_attention_layers", "sliding_attention_layers",
    "num_key_value_heads", "head_dim", "swa_num_key_value_heads",
    "swa_head_dim", "sliding_window", "draft_full_attention_layers",
    "draft_sliding_attention_layers", "hidden_size", "sconv_kernel_size",
    "sconv_state_bytes_per_element",
  ],
  qwen_linear_full_hybrid: [
    "full_attention_layers", "num_key_value_heads", "head_dim",
    "linear_attention_layers", "linear_conv_kernel_dim",
    "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
    "linear_value_head_dim", "linear_conv_state_bytes_per_element",
    "linear_recurrent_state_bytes_per_element",
  ],
  mixed_full_sliding_gqa: [
    "full_attention_layers", "sliding_attention_layers", "num_key_value_heads",
    "num_global_key_value_heads", "head_dim", "global_head_dim",
    "v_head_dim", "global_v_head_dim", "swa_num_key_value_heads",
    "sliding_num_key_value_heads", "swa_head_dim", "sliding_head_dim",
    "swa_v_head_dim", "sliding_v_head_dim", "sliding_window",
  ],
  minimax_msa: [
    "num_hidden_layers", "num_key_value_heads", "head_dim",
    "sparse_attention_layers", "index_head_dim",
  ],
  deepseek_v4_hybrid: [
    "num_hidden_layers", "head_dim", "sliding_window", "index_head_dim",
    "compress_ratios",
  ],
});

const STATE_FORMULAS = new Set([
  "kimi_kda_mla_hybrid", "inkling_hybrid", "qwen_linear_full_hybrid", "deepseek_v41_csa2",
]);
const CHECKPOINT_FORMULAS = new Set([
  "kimi_kda_mla_hybrid", "inkling_hybrid", "qwen_linear_full_hybrid",
]);
const INDEXER_FORMULAS = new Set(["dsa_mla", "minimax_msa", "deepseek_v4_hybrid", "deepseek_v41_csa2"]);

const GDN_STATE_DEFAULTS = Object.freeze({
  linear_conv_state_bytes_per_element: 2,
  linear_recurrent_state_bytes_per_element: 4,
});

class KvInputError extends Error {}

function calculatorFields(calculator, overrideFields = null) {
  return {
    ...(calculator.formula_id === "qwen_linear_full_hybrid" ? GDN_STATE_DEFAULTS : {}),
    ...(overrideFields || calculator.fields || {}),
  };
}

export function initializeKvSize({ config }) {
  const models = (config.models || []).filter((model) => model.kv_calculator && !model.error);
  byId("kvCatalogBadge").textContent = `${models.length} models · ${new Set(models.map((model) => model.kv_calculator.formula_id)).size} formulas · local`;
  const state = {
    models,
    fields: {},
    model: null,
  };
  populatePrecisionSelect(byId("kvPrecision"));
  populatePrecisionSelect(byId("kvIndexerPrecision"));
  populateFamilies(state);
  wireControls(state);
  state.compare = initializeKvCompare({
    capture: () => captureCurrent(state),
    format: { bytes: formatBytes, ratio: formatBytesRatio, integer: formatInteger, field: formatFieldValue, fieldLabel: labelForField },
    componentLabels: COMPONENT_LABELS,
  });

  const preferred = models.find((model) => model.model_id === "deepseek-v4-pro") || models[0];
  if (preferred) {
    byId("kvFamily").value = preferred.family || "";
    populateModels(state, preferred.model_id);
    selectModel(state);
  } else {
    renderError("No active model provides KV calculator data.");
    state.compare.updateInput("No active model provides KV calculator data.");
  }
  return { recalculate: () => recalculate(state) };
}

function wireControls(state) {
  byId("kvFamily").addEventListener("change", () => {
    populateModels(state);
    selectModel(state);
  });
  byId("kvModel").addEventListener("change", () => selectModel(state));
  for (const id of [
    "kvTokens", "kvSequences", "kvPrecision", "kvIndexerPrecision",
    "kvIncludeDraft", "kvIncludeState", "kvCheckpointMode", "kvCheckpointInterval", "kvEncoderTailPolicy",
  ]) {
    byId(id).addEventListener("input", () => {
      if (state.model?.kv_calculator.formula_id === "deepseek_v41_csa2") byId("kvIndexerPrecision").value = byId("kvPrecision").value;
      syncDynamicControlState();
      recalculate(state);
    });
  }
  byId("kvResetDefaults").addEventListener("click", () => selectModel(state));
}

function populatePrecisionSelect(select) {
  select.replaceChildren();
  for (const precision of PRECISIONS) {
    const option = document.createElement("option");
    option.value = precision.id;
    option.textContent = `${precision.label} · ${formatPrecision(precision.halfBytes)}`;
    select.append(option);
  }
}

function populateFamilies(state) {
  const select = byId("kvFamily");
  select.replaceChildren();
  for (const family of [...new Set(state.models.map((model) => model.family).filter(Boolean))].sort()) {
    const option = document.createElement("option");
    option.value = family;
    option.textContent = family;
    select.append(option);
  }
}

function populateModels(state, selectedModelId = "") {
  const select = byId("kvModel");
  const family = byId("kvFamily").value;
  const models = state.models.filter((model) => model.family === family);
  select.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.model_name || model.label || model.model_id;
    select.append(option);
  }
  if (models.some((model) => model.model_id === selectedModelId)) select.value = selectedModelId;
}

function selectModel(state) {
  state.model = state.models.find((model) => model.model_id === byId("kvModel").value) || null;
  if (!state.model) {
    renderError("Choose a model with calculator data.");
    return;
  }
  state.fields = cloneValue(calculatorFields(state.model.kv_calculator));
  const calculator = state.model.kv_calculator;
  const formula = calculator.formula_id;
  populatePrecisionSelect(byId("kvPrecision"));
  populatePrecisionSelect(byId("kvIndexerPrecision"));
  if (formula === "deepseek_v41_csa2") {
    for (const id of ["kvPrecision", "kvIndexerPrecision"]) byId(id).querySelector('option[value="fp8_int8"]').remove();
    byId("kvPrecision").querySelector('option[value="fp4_int4"]').textContent = "Released · FP4 global + FP8 SWA + scales";
  }
  byId("kvEncoderTailPolicy").value = "recompute";
  byId("kvTokens").value = String(calculator.default_tokens || 1024);
  byId("kvSequences").value = "1";
  byId("kvPrecision").value = fieldPrecisionId(state.fields.default_precision_id)
    || (formula === "deepseek_v4_hybrid" ? "fp8_int8" : "bf16_fp16");
  byId("kvIndexerPrecision").value = fieldPrecisionId(state.fields.indexer_fixed_precision_id)
    || (formula === "deepseek_v4_hybrid" ? "fp4_int4" : byId("kvPrecision").value);
  byId("kvIncludeDraft").checked = false;
  byId("kvIncludeState").checked = defaultIncludeState(state.fields, formula);
  byId("kvCheckpointMode").value = "prompt_end";
  byId("kvCheckpointInterval").value = "2048";
  configureDynamicControls(state);
  renderAdvancedFields(state);
  recalculate(state);
}

function configureDynamicControls(state) {
  const formula = state.model.kv_calculator.formula_id;
  const hasIndexer = INDEXER_FORMULAS.has(formula);
  const hasDraft = supportsDraft(state.model, state.fields);
  const hasState = STATE_FORMULAS.has(formula);
  const hasCheckpoint = CHECKPOINT_FORMULAS.has(formula);
  setFieldVisibility("kvEncoderTailField", formula === "deepseek_v41_csa2");
  setFieldVisibility("kvIndexerPrecisionField", hasIndexer);
  setFieldVisibility("kvIncludeDraftField", hasDraft);
  setFieldVisibility("kvIncludeStateField", hasState);
  setFieldVisibility("kvCheckpointModeField", hasCheckpoint);
  setFieldVisibility("kvCheckpointIntervalField", hasCheckpoint);
  const fixedIndexer = fieldPrecisionId(state.fields.indexer_fixed_precision_id);
  byId("kvIndexerPrecision").disabled = Boolean(fixedIndexer) || formula === "deepseek_v41_csa2";
  if (fixedIndexer) byId("kvIndexerPrecision").value = fixedIndexer;
  syncDynamicControlState();
}

function syncDynamicControlState() {
  const stateEnabled = byId("kvIncludeState").checked;
  const fixedInterval = byId("kvCheckpointMode").value === "fixed";
  byId("kvCheckpointMode").disabled = !stateEnabled;
  byId("kvCheckpointInterval").disabled = !stateEnabled || !fixedInterval;
}

function setFieldVisibility(id, visible) {
  byId(id).hidden = !visible;
}

function renderAdvancedFields(state) {
  const container = byId("kvAdvancedFields");
  container.replaceChildren();
  const formula = state.model.kv_calculator.formula_id;
  const editable = (EDITABLE_FIELDS[formula] || []).filter((key) => key in state.fields);
  if (formula === "deepseek_v41_csa2") container.textContent = "V4.1 structure is fixed. Layer owners and sharing roles are shown below (zero-based layer numbers).";
  for (const key of editable) {
    const label = document.createElement("label");
    label.dataset.kvAdvancedField = key;
    const title = document.createElement("span");
    title.textContent = labelForField(key);
    const value = state.fields[key];
    const input = document.createElement(key === "compress_ratios" ? "textarea" : "input");
    input.dataset.kvField = key;
    input.spellcheck = false;
    if (key === "compress_ratios") {
      input.rows = 4;
      input.value = Array.isArray(value) ? value.join(", ") : String(value ?? "");
      label.classList.add("kv-wide-field");
    } else {
      input.type = "number";
      input.min = "0";
      input.step = "1";
      input.value = String(value);
    }
    input.addEventListener("input", () => {
      if (key === "compress_ratios") {
        state.fields[key] = input.value;
      } else {
        state.fields[key] = input.value;
      }
      recalculate(state);
    });
    label.append(title, input);
    container.append(label);
  }
}

function recalculate(state) {
  clearInputErrors();
  try {
    const snapshot = captureCurrent(state);
    renderResult(state.model, snapshot.settings, state.fields, snapshot.result, snapshot.explanations);
    state.compare.updateInput(null);
  } catch (error) {
    renderError(error instanceof Error ? error.message : String(error));
    state.compare.updateInput(error.message || String(error));
  }
}

function captureCurrent(state) {
  if (!state.model) throw new KvInputError("Choose a model with calculator data.");
  return createKvSnapshot(state.model, readSettings(state), state.fields);
}

function readSettings(state) {
  inputBigInt("kvTokens", "Tokens", 0n);
  inputBigInt("kvSequences", "Sequences", 1n);
  if (CHECKPOINT_FORMULAS.has(state.model.kv_calculator.formula_id)
    && byId("kvIncludeState").checked && byId("kvCheckpointMode").value === "fixed") {
    inputBigInt("kvCheckpointInterval", "Checkpoint interval", 1n);
  }
  return {
    tokens: byId("kvTokens").value,
    sequences: byId("kvSequences").value,
    precisionHalfBytes: precisionHalfBytes(byId("kvPrecision").value),
    indexerPrecisionHalfBytes: precisionHalfBytes(byId("kvIndexerPrecision").value),
    includeDraft: byId("kvIncludeDraft").checked,
    includeState: byId("kvIncludeState").checked,
    checkpointMode: byId("kvCheckpointMode").value,
    checkpointInterval: byId("kvCheckpointInterval").value,
    encoderTailPolicy: byId("kvEncoderTailPolicy").value,
  };
}

// Resolve only parameters used by the selected formula and enabled branches.
// Both calculation and identity consume this result; raw editor values stay in
// the snapshot so disabling a branch does not discard the user's edits.
export function resolveKvConfiguration(model, input, overrideFields = null) {
  const calculator = model.kv_calculator;
  const formula = calculator.formula_id;
  const fields = calculatorFields(calculator, overrideFields);
  const defaults = calculatorFields(calculator);
  const state = STATE_FORMULAS.has(formula);
  const draft = supportsDraft(model, fields) || supportsDraft(model, defaults);
  const settings = {
    tokens: requiredBigInt(input.tokens, "Tokens"),
    sequences: requiredBigInt(input.sequences, "Sequences", 1n),
    precisionHalfBytes: requiredSmallInt(input.precisionHalfBytes, "KV precision"),
    includeState: state && Boolean(input.includeState),
    includeDraft: draft && Boolean(input.includeDraft),
    checkpointInterval: null,
    encoderTailPolicy: "recompute",
  };
  const checkpoint = CHECKPOINT_FORMULAS.has(formula);
  if (checkpoint && settings.includeState && input.checkpointMode !== "prompt_end" && input.checkpointInterval != null) {
    settings.checkpointInterval = requiredBigInt(input.checkpointInterval, "Checkpoint interval", 1n);
  }
  if (formula === "deepseek_v41_csa2" && settings.includeState) settings.encoderTailPolicy = input.encoderTailPolicy ?? "recompute";

  const names = new Set((EDITABLE_FIELDS[formula] || []).filter(key => key in fields));
  const drop = predicate => { for (const key of names) if (predicate(key)) names.delete(key); };
  const used = {};
  const optional = (key, fallback = 0n) => {
    const value = optionalFieldInt(fields, key, fallback);
    names.add(key);
    used[key] = value;
    return value;
  };
  if (!settings.includeState) {
    drop(key => key.startsWith("kda_") || key.startsWith("linear_") || key.startsWith("sconv_") || key === "hidden_size");
  }
  const draftField = key => key.startsWith("draft_") || ["num_nextn_predict_layers", "num_mtp_modules", "mtp_transformer_layers"].includes(key);
  if (!settings.includeDraft) drop(draftField);
  else {
    for (const key of [...names].filter(draftField)) optional(key);
    if (formula === "standard_gqa" && optional("num_mtp_modules") === 0n) names.delete("mtp_transformer_layers");
  }
  let indexer = INDEXER_FORMULAS.has(formula);
  if (formula === "dsa_mla") {
    let count = optional("indexer_full_layers", fieldInt(fields, "num_hidden_layers", 1n));
    if (settings.includeDraft) count += optional("draft_indexer_layers", optionalFieldInt(fields, "num_nextn_predict_layers"));
    indexer = count > 0n;
  } else if (formula === "minimax_msa") {
    indexer = fieldInt(fields, "sparse_attention_layers") > 0n;
  } else if (formula === "deepseek_v4_hybrid") {
    const entries = Array.isArray(fields.compress_ratios) ? fields.compress_ratios : String(fields.compress_ratios ?? "").split(/[\s,]+/).filter(Boolean);
    used.compress_ratios = ratioFields(settings.includeDraft ? entries : entries.slice(0, layerLoopCount(fieldInt(fields, "num_hidden_layers", 1n))));
    indexer = used.compress_ratios.some(value => value === 4n);
  }
  if (!indexer) names.delete("index_head_dim");
  if (formula === "mixed_full_sliding_gqa") {
    const candidates = new Set();
    const choose = (...keys) => {
      const key = keys.find(name => fields[name] != null && fields[name] !== "");
      if (key) candidates.add(key);
    };
    choose("num_global_key_value_heads", "num_key_value_heads");
    choose("global_head_dim", "head_dim");
    choose("global_v_head_dim", "v_head_dim", "global_head_dim", "head_dim");
    choose("swa_num_key_value_heads", "sliding_num_key_value_heads", "num_key_value_heads");
    choose("swa_head_dim", "sliding_head_dim", "head_dim");
    choose("swa_v_head_dim", "sliding_v_head_dim", "v_head_dim", "swa_head_dim", "sliding_head_dim", "head_dim");
    drop(key => !["full_attention_layers", "sliding_attention_layers", "sliding_window"].includes(key) && !candidates.has(key));
  }
  const effectiveFields = {};
  for (const key of names) effectiveFields[key] = key in used ? used[key] : requiredBigInt(fields[key], labelForField(key));
  const calculationFields = { ...fields };
  for (const key of EDITABLE_FIELDS[formula] || []) delete calculationFields[key];
  Object.assign(calculationFields, effectiveFields);

  const fixedPrecision = fieldPrecisionId(fields.indexer_fixed_precision_id);
  settings.indexerPrecisionHalfBytes = indexer
    ? formula === "deepseek_v41_csa2" ? settings.precisionHalfBytes
      : fixedPrecision ? precisionHalfBytes(fixedPrecision)
        : requiredSmallInt(input.indexerPrecisionHalfBytes ?? settings.precisionHalfBytes, "Indexer precision")
    : null;
  return {
    settings, calculationFields,
    effective: { modelId: model.model_id, formula, settings, fields: effectiveFields },
    applicability: { state, draft, indexer, checkpoint, encoderTail: formula === "deepseek_v41_csa2" },
  };
}

function canonicalField(key, value) {
  if (key === "compress_ratios") {
    try { return ratioFields(value); } catch { return String(value).trim(); }
  }
  return value;
}

export function createKvSnapshot(model, input, overrideFields = null) {
  const fields = calculatorFields(model.kv_calculator, overrideFields);
  const resolved = resolveKvConfiguration(model, input, fields);
  const result = calculateKvSize(model, resolved.settings, resolved.calculationFields);
  const defaults = calculatorFields(model.kv_calculator);
  const custom = (EDITABLE_FIELDS[model.kv_calculator.formula_id] || []).some(key =>
    configurationKey(canonicalField(key, fields[key])) !== configurationKey(canonicalField(key, defaults[key])));
  return structuredClone({
    model, input, fields, custom,
    settings: resolved.settings, effective: resolved.effective, applicability: resolved.applicability,
    result, groups: aggregateKvGroups(result),
    explanations: describeKvResult(model, resolved.settings, result),
  });
}

export function calculateKvSize(model, settings, overrideFields = null) {
  const calculator = model.kv_calculator || model;
  const formula = calculator.formula_id;
  const fields = calculatorFields(calculator, overrideFields);
  const tokens = requiredBigInt(settings.tokens, "Tokens", 0n);
  const sequences = requiredBigInt(settings.sequences, "Sequences", 1n);
  const precisionHalfBytes = requiredSmallInt(settings.precisionHalfBytes, "KV precision");
  const indexerPrecisionHalfBytes = requiredSmallInt(
    settings.indexerPrecisionHalfBytes ?? precisionHalfBytes,
    "Indexer precision",
  );
  const includeDraft = Boolean(settings.includeDraft);
  const includeState = Boolean(settings.includeState);
  const checkpointInterval = settings.checkpointInterval === null
    || settings.checkpointInterval === undefined
    ? null
    : requiredBigInt(settings.checkpointInterval, "Checkpoint interval", 1n);
  const components = {};
  const layers = [];
  const equations = [];
  const reservedComponents = {};

  const addComponent = (name, bytes) => {
    components[name] = (components[name] || 0n) + bytes;
  };
  const addLayer = (index, type, tokenSpan, ratio, byComponent, draft = false) => {
    const totalBytes = Object.values(byComponent).reduce((sum, value) => sum + value, 0n);
    layers.push({ index, type, tokenSpan, ratio, byComponent, totalBytes, draft });
  };
  const equation = (component, expression, substitution, bytes) => {
    equations.push({ component, expression, substitution, bytes });
  };

  if (formula === "standard_gqa") {
    const baseLayers = fieldInt(fields, "num_hidden_layers", 1n);
    const nextn = includeDraft ? optionalFieldInt(fields, "num_nextn_predict_layers") : 0n;
    const mtp = includeDraft ? optionalFieldInt(fields, "num_mtp_modules")
      * optionalFieldInt(fields, "mtp_transformer_layers") : 0n;
    const draftLayers = includeDraft ? nextn + mtp : 0n;
    const layerCount = baseLayers + draftLayers;
    const heads = fieldInt(fields, "num_key_value_heads", 1n);
    const dim = fieldInt(fields, "head_dim", 1n);
    const perLayer = elementBytes(tokens * sequences * heads * 2n * dim, precisionHalfBytes);
    const count = layerLoopCount(layerCount);
    for (let index = 0; index < count; index += 1) {
      addComponent("kv", perLayer);
      addLayer(index + 1, "GQA / KV", tokens, null, { kv: perLayer }, BigInt(index) >= baseLayers);
    }
    equation(
      "kv", "B = T × S × L × Hkv × 2 × D × bkv",
      `${tokens} × ${sequences} × ${layerCount} × ${heads} × 2 × ${dim} × ${formatPrecision(precisionHalfBytes)}`,
      components.kv,
    );
  } else if (formula === "mla") {
    const baseLayers = fieldInt(fields, "num_hidden_layers", 1n);
    const draftLayers = includeDraft ? optionalFieldInt(fields, "num_nextn_predict_layers") : 0n;
    const layerCount = baseLayers + draftLayers;
    const latent = fieldInt(fields, "kv_lora_rank", 1n) + fieldInt(fields, "qk_rope_head_dim", 1n);
    const perLayer = elementBytes(tokens * sequences * latent, precisionHalfBytes);
    const count = layerLoopCount(layerCount);
    for (let index = 0; index < count; index += 1) {
      addComponent("mla", perLayer);
      addLayer(index + 1, "MLA latent", tokens, null, { mla: perLayer }, BigInt(index) >= baseLayers);
    }
    equation(
      "mla", "Bmla = T × S × L × (Rkv + Drope) × bkv",
      `${tokens} × ${sequences} × ${layerCount} × ${latent} × ${formatPrecision(precisionHalfBytes)}`,
      components.mla,
    );
  } else if (formula === "dsa_mla") {
    const baseLayers = fieldInt(fields, "num_hidden_layers", 1n);
    const draftLayers = includeDraft ? optionalFieldInt(fields, "num_nextn_predict_layers") : 0n;
    const kvLayers = baseLayers + draftLayers;
    const latent = fieldInt(fields, "kv_lora_rank", 1n) + fieldInt(fields, "qk_rope_head_dim", 1n);
    const kvPerLayer = elementBytes(tokens * sequences * latent, precisionHalfBytes);
    let indexLayers = optionalFieldInt(fields, "indexer_full_layers", baseLayers);
    if (includeDraft) indexLayers += optionalFieldInt(fields, "draft_indexer_layers", draftLayers);
    const indexDim = indexLayers > 0n ? fieldInt(fields, "index_head_dim", 1n) : 0n;
    const indexPerLayer = elementBytes(tokens * sequences * indexDim, indexerPrecisionHalfBytes);
    const count = layerLoopCount(kvLayers > indexLayers ? kvLayers : indexLayers);
    for (let index = 0; index < count; index += 1) {
      const row = {};
      if (BigInt(index) < kvLayers) {
        row.mla = kvPerLayer;
        addComponent("mla", kvPerLayer);
      }
      if (BigInt(index) < indexLayers) {
        row.indexer = indexPerLayer;
        addComponent("indexer", indexPerLayer);
      }
      addLayer(index + 1, "DSA + MLA", tokens, null, row, BigInt(index) >= baseLayers);
    }
    equation(
      "mla", "Bmla = T × S × Lmla × (Rkv + Drope) × bkv",
      `${tokens} × ${sequences} × ${kvLayers} × ${latent} × ${formatPrecision(precisionHalfBytes)}`,
      components.mla,
    );
    equation(
      "indexer", "Bidx = T × S × Lidx × Didx × bidx",
      `${tokens} × ${sequences} × ${indexLayers} × ${indexDim} × ${formatPrecision(indexerPrecisionHalfBytes)}`,
      components.indexer,
    );
  } else if (formula === "kimi_kda_mla_hybrid") {
    const fullLayers = fieldInt(fields, "full_attention_layers", 1n);
    const latent = fieldInt(fields, "kv_lora_rank", 1n) + fieldInt(fields, "qk_rope_head_dim", 1n);
    const mlaPerLayer = elementBytes(tokens * sequences * latent, precisionHalfBytes);
    for (let index = 0; index < layerLoopCount(fullLayers); index += 1) {
      addComponent("mla", mlaPerLayer);
      addLayer(index + 1, "MLA", tokens, null, { mla: mlaPerLayer });
    }
    equation(
      "mla", "Bmla = T × S × Lmla × (Rkv + Drope) × bkv",
      `${tokens} × ${sequences} × ${fullLayers} × ${latent} × ${formatPrecision(precisionHalfBytes)}`,
      components.mla,
    );
    if (includeState) {
      const kdaLayers = fieldInt(fields, "kda_layers", 1n);
      const history = fieldInt(fields, "kda_conv_kernel_size", 1n) - 1n;
      const convElements = history * (
        fieldInt(fields, "kda_num_heads", 1n) * fieldInt(fields, "kda_head_dim", 1n)
        + fieldInt(fields, "kda_num_key_heads", 1n) * fieldInt(fields, "kda_key_head_dim", 1n)
        + fieldInt(fields, "kda_num_value_heads", 1n) * fieldInt(fields, "kda_value_head_dim", 1n)
      );
      const recurrentElements = fieldInt(fields, "kda_num_value_heads", 1n)
        * fieldInt(fields, "kda_value_head_dim", 1n)
        * fieldInt(fields, "kda_key_head_dim", 1n);
      const statePerLayer = sequences * checkpointCount(tokens, checkpointInterval) * (
        convElements * fieldInt(fields, "kda_conv_state_bytes_per_element", 1n)
        + recurrentElements * fieldInt(fields, "kda_recurrent_state_bytes_per_element", 1n)
      );
      for (let index = 0; index < layerLoopCount(kdaLayers); index += 1) {
        addComponent("kda_state", statePerLayer);
        addLayer(Number(fullLayers) + index + 1, "KDA state", "checkpoint", null, { kda_state: statePerLayer });
      }
      equation(
        "kda_state", "Bkda = S × C(T) × Lkda × (Econv × bconv + Erecurrent × brecurrent)",
        `${sequences} × ${checkpointCount(tokens, checkpointInterval)} × ${kdaLayers} × (${convElements} × ${fields.kda_conv_state_bytes_per_element} + ${recurrentElements} × ${fields.kda_recurrent_state_bytes_per_element})`,
        components.kda_state,
      );
    }
  } else if (formula === "inkling_hybrid") {
    const baseFull = fieldInt(fields, "full_attention_layers", 0n);
    const baseSliding = fieldInt(fields, "sliding_attention_layers", 0n);
    const fullLayers = baseFull + (includeDraft ? optionalFieldInt(fields, "draft_full_attention_layers") : 0n);
    const slidingLayers = baseSliding + (includeDraft ? optionalFieldInt(fields, "draft_sliding_attention_layers") : 0n);
    const fullHeads = fieldInt(fields, "num_key_value_heads", 1n);
    const fullDim = fieldInt(fields, "head_dim", 1n);
    const slidingHeads = fieldInt(fields, "swa_num_key_value_heads", 1n);
    const slidingDim = fieldInt(fields, "swa_head_dim", 1n);
    const window = fieldInt(fields, "sliding_window", 1n);
    const slidingTokens = minBigInt(tokens, window);
    const fullPerLayer = elementBytes(tokens * sequences * fullHeads * 2n * fullDim, precisionHalfBytes);
    const slidingPerLayer = elementBytes(slidingTokens * sequences * slidingHeads * 2n * slidingDim, precisionHalfBytes);
    const stateBytes = includeState ? fieldInt(fields, "sconv_state_bytes_per_element", 1n) : 0n;
    const history = includeState ? fieldInt(fields, "sconv_kernel_size", 1n) - 1n : 0n;
    const hidden = includeState ? fieldInt(fields, "hidden_size", 1n) : 0n;
    const checkpoints = checkpointCount(tokens, checkpointInterval);
    const fullState = includeState
      ? sequences * checkpoints * history * (2n * fullHeads * fullDim + 2n * hidden) * stateBytes
      : 0n;
    const slidingState = includeState
      ? sequences * checkpoints * history * (2n * slidingHeads * slidingDim + 2n * hidden) * stateBytes
      : 0n;
    for (let index = 0; index < layerLoopCount(fullLayers); index += 1) {
      const row = { global_kv: fullPerLayer };
      addComponent("global_kv", fullPerLayer);
      if (includeState) {
        row.sconv_state = fullState;
        addComponent("sconv_state", fullState);
      }
      addLayer(index + 1, "Global attention", tokens, null, row, BigInt(index) >= baseFull);
    }
    for (let index = 0; index < layerLoopCount(slidingLayers); index += 1) {
      const row = { sliding_kv: slidingPerLayer };
      addComponent("sliding_kv", slidingPerLayer);
      if (includeState) {
        row.sconv_state = slidingState;
        addComponent("sconv_state", slidingState);
      }
      addLayer(Number(fullLayers) + index + 1, "Sliding attention", slidingTokens, null, row, BigInt(index) >= baseSliding);
    }
    equation(
      "global_kv", "Bglobal = T × S × Lglobal × Hkv × 2 × D × bkv",
      `${tokens} × ${sequences} × ${fullLayers} × ${fullHeads} × 2 × ${fullDim} × ${formatPrecision(precisionHalfBytes)}`,
      components.global_kv || 0n,
    );
    equation(
      "sliding_kv", "Bswa = min(T, W) × S × Lswa × Hkv × 2 × D × bkv",
      `${slidingTokens} × ${sequences} × ${slidingLayers} × ${slidingHeads} × 2 × ${slidingDim} × ${formatPrecision(precisionHalfBytes)}`,
      components.sliding_kv || 0n,
    );
    if (includeState) {
      equation(
        "sconv_state", "Bsconv = S × C(T) × Σlayer[(K − 1) × (2HkvD + 2Dmodel) × bstate]",
        `${sequences} × ${checkpoints} checkpoints across ${fullLayers + slidingLayers} layers`,
        components.sconv_state || 0n,
      );
    }
  } else if (formula === "qwen_linear_full_hybrid") {
    const fullLayers = fieldInt(fields, "full_attention_layers", 0n);
    const heads = fieldInt(fields, "num_key_value_heads", 1n);
    const dim = fieldInt(fields, "head_dim", 1n);
    const fullPerLayer = elementBytes(tokens * sequences * 2n * heads * dim, precisionHalfBytes);
    for (let index = 0; index < layerLoopCount(fullLayers); index += 1) {
      addComponent("full_kv", fullPerLayer);
      addLayer(index + 1, "Full attention", tokens, null, { full_kv: fullPerLayer });
    }
    equation(
      "full_kv", "Bfull = T × S × Lfull × 2 × Hkv × D × bkv",
      `${tokens} × ${sequences} × ${fullLayers} × 2 × ${heads} × ${dim} × ${formatPrecision(precisionHalfBytes)}`,
      components.full_kv || 0n,
    );
    if (includeState) {
      const linearLayers = fieldInt(fields, "linear_attention_layers", 0n);
      const convElements = fieldInt(fields, "linear_conv_kernel_dim", 1n) * (
        2n * fieldInt(fields, "linear_num_key_heads", 1n) * fieldInt(fields, "linear_key_head_dim", 1n)
        + fieldInt(fields, "linear_num_value_heads", 1n) * fieldInt(fields, "linear_value_head_dim", 1n)
      );
      const recurrentElements = fieldInt(fields, "linear_num_value_heads", 1n)
        * fieldInt(fields, "linear_key_head_dim", 1n)
        * fieldInt(fields, "linear_value_head_dim", 1n);
      const convBytes = fieldInt(fields, "linear_conv_state_bytes_per_element", 1n);
      const recurrentBytes = fieldInt(fields, "linear_recurrent_state_bytes_per_element", 1n);
      const checkpoints = checkpointCount(tokens, checkpointInterval);
      const statePerLayer = sequences * checkpoints * (
        convElements * convBytes + recurrentElements * recurrentBytes
      );
      for (let index = 0; index < layerLoopCount(linearLayers); index += 1) {
        addComponent("linear_state", statePerLayer);
        addLayer(
          Number(fullLayers) + index + 1, "GDN state",
          `${checkpoints} checkpoint${checkpoints === 1n ? "" : "s"}`,
          null, { linear_state: statePerLayer },
        );
      }
      equation(
        "linear_state", "Blinear = S × C(T) × Llinear × (Econv × bconv + Erecurrent × brecurrent)",
        `${sequences} × ${checkpoints} × ${linearLayers} × (${convElements} × ${convBytes} + ${recurrentElements} × ${recurrentBytes})`,
        components.linear_state,
      );
    }
  } else if (formula === "mixed_full_sliding_gqa") {
    const kvHeads = () => fieldInt(fields, "num_key_value_heads", 1n);
    const fullHeads = optionalFieldInt(fields, "num_global_key_value_heads", kvHeads);
    const fullKey = optionalFieldInt(fields, "global_head_dim", () => fieldInt(fields, "head_dim", 1n));
    const fullValue = optionalFieldInt(fields, "global_v_head_dim", () => optionalFieldInt(fields, "v_head_dim", fullKey));
    const slidingHeads = optionalFieldInt(
      fields, "swa_num_key_value_heads",
      () => optionalFieldInt(fields, "sliding_num_key_value_heads", kvHeads),
    );
    const slidingKey = optionalFieldInt(
      fields, "swa_head_dim",
      () => optionalFieldInt(fields, "sliding_head_dim", () => fieldInt(fields, "head_dim", 1n)),
    );
    const slidingValue = optionalFieldInt(
      fields, "swa_v_head_dim",
      () => optionalFieldInt(fields, "sliding_v_head_dim", () => optionalFieldInt(fields, "v_head_dim", slidingKey)),
    );
    const fullLayers = fieldInt(fields, "full_attention_layers", 0n);
    const slidingLayers = fieldInt(fields, "sliding_attention_layers", 0n);
    const window = fieldInt(fields, "sliding_window", 1n);
    const slidingTokens = minBigInt(tokens, window);
    const fullPerLayer = elementBytes(tokens * sequences * fullHeads * (fullKey + fullValue), precisionHalfBytes);
    const slidingPerLayer = elementBytes(slidingTokens * sequences * slidingHeads * (slidingKey + slidingValue), precisionHalfBytes);
    for (let index = 0; index < layerLoopCount(fullLayers); index += 1) {
      addComponent("full_kv", fullPerLayer);
      addLayer(index + 1, "Full attention", tokens, null, { full_kv: fullPerLayer });
    }
    for (let index = 0; index < layerLoopCount(slidingLayers); index += 1) {
      addComponent("sliding_kv", slidingPerLayer);
      addLayer(Number(fullLayers) + index + 1, "Sliding attention", slidingTokens, null, { sliding_kv: slidingPerLayer });
    }
    equation(
      "full_kv", "Bfull = T × S × Lfull × Hfull × (Dk + Dv) × bkv",
      `${tokens} × ${sequences} × ${fullLayers} × ${fullHeads} × (${fullKey} + ${fullValue}) × ${formatPrecision(precisionHalfBytes)}`,
      components.full_kv || 0n,
    );
    equation(
      "sliding_kv", "Bswa = min(T, W) × S × Lswa × Hswa × (Dk + Dv) × bkv",
      `${slidingTokens} × ${sequences} × ${slidingLayers} × ${slidingHeads} × (${slidingKey} + ${slidingValue}) × ${formatPrecision(precisionHalfBytes)}`,
      components.sliding_kv || 0n,
    );
  } else if (formula === "minimax_msa") {
    const mainLayers = fieldInt(fields, "num_hidden_layers", 1n);
    const heads = fieldInt(fields, "num_key_value_heads", 1n);
    const dim = fieldInt(fields, "head_dim", 1n);
    const mainPerLayer = elementBytes(tokens * sequences * 2n * heads * dim, precisionHalfBytes);
    const sparseLayers = fieldInt(fields, "sparse_attention_layers", 0n);
    const indexDim = sparseLayers > 0n ? fieldInt(fields, "index_head_dim", 1n) : 0n;
    const indexPerLayer = elementBytes(tokens * sequences * indexDim, indexerPrecisionHalfBytes);
    const count = layerLoopCount(mainLayers > sparseLayers ? mainLayers : sparseLayers);
    for (let index = 0; index < count; index += 1) {
      const row = {};
      if (BigInt(index) < mainLayers) {
        row.main_kv = mainPerLayer;
        addComponent("main_kv", mainPerLayer);
      }
      if (BigInt(index) < sparseLayers) {
        row.indexer = indexPerLayer;
        addComponent("indexer", indexPerLayer);
      }
      addLayer(index + 1, BigInt(index) < sparseLayers ? "Sparse attention" : "Full attention", tokens, null, row);
    }
    equation(
      "main_kv", "Bmain = T × S × L × 2 × Hkv × D × bkv",
      `${tokens} × ${sequences} × ${mainLayers} × 2 × ${heads} × ${dim} × ${formatPrecision(precisionHalfBytes)}`,
      components.main_kv,
    );
    equation(
      "indexer", "Bidx = T × S × Lsparse × Didx × bidx",
      `${tokens} × ${sequences} × ${sparseLayers} × ${indexDim} × ${formatPrecision(indexerPrecisionHalfBytes)}`,
      components.indexer || 0n,
    );
  } else if (formula === "deepseek_v4_hybrid") {
    const baseLayers = fieldInt(fields, "num_hidden_layers", 1n);
    const ratios = ratioFields(fields.compress_ratios);
    if (ratios.length < layerLoopCount(baseLayers)) {
      throw new KvInputError(`compress_ratios needs at least ${baseLayers} entries.`);
    }
    const active = includeDraft ? ratios : ratios.slice(0, layerLoopCount(baseLayers));
    const window = fieldInt(fields, "sliding_window", 1n);
    const dim = fieldInt(fields, "head_dim", 1n);
    const indexDim = active.some(ratio => ratio === 4n) ? fieldInt(fields, "index_head_dim", 1n) : 0n;
    let windowElements = 0n;
    let compressedElements = 0n;
    let indexElements = 0n;
    active.forEach((ratio, index) => {
      const windowRowElements = window * dim * sequences;
      const compressedEntries = ratio > 0n ? tokens / ratio : 0n;
      const compressedRowElements = compressedEntries * dim * sequences;
      const indexEntries = ratio === 4n ? tokens / 4n : 0n;
      const indexRowElements = indexEntries * indexDim * sequences;
      const mainBytes = elementBytes(windowRowElements + compressedRowElements, precisionHalfBytes);
      const row = { main_kv: mainBytes };
      addComponent("main_kv", mainBytes);
      if (indexRowElements > 0n) {
        const indexBytes = elementBytes(indexRowElements, indexerPrecisionHalfBytes);
        row.indexer = indexBytes;
        addComponent("indexer", indexBytes);
      }
      windowElements += windowRowElements;
      compressedElements += compressedRowElements;
      indexElements += indexRowElements;
      addLayer(
        index + 1,
        ratio === 4n ? "Compressed + indexer" : ratio > 0n ? "Compressed attention" : "Window only",
        `${window} window + ${compressedEntries} compressed`,
        ratio,
        row,
        BigInt(index) >= baseLayers,
      );
    });
    equation(
      "main_kv", "Bmain = S × bkv × [Lactive × W × D + Σ floor(T / ri) × D]",
      `${formatPrecision(precisionHalfBytes)} × (${windowElements} window elements + ${compressedElements} compressed elements)`,
      components.main_kv,
    );
    equation(
      "indexer", "Bidx = S × bidx × Σri=4 floor(T / 4) × Didx",
      `${formatPrecision(indexerPrecisionHalfBytes)} × ${indexElements} index elements`,
      components.indexer || 0n,
    );
  } else if (formula === "deepseek_v41_csa2") {
    if (![1, 4].includes(precisionHalfBytes) || indexerPrecisionHalfBytes !== precisionHalfBytes) {
      throw new KvInputError("V4.1 supports the released mixed layout or BF16 reference layout.");
    }
    const released = precisionHalfBytes === 1;
    const main = released ? 288n : 1024n;
    const indexer = released ? 68n : 256n;
    const swa = released ? 528n : 1024n;
    const window = minBigInt(tokens, 128n);
    const tailPolicy = settings.encoderTailPolicy ?? "recompute";
    if (!["recompute", "retain"].includes(tailPolicy)) throw new KvInputError("Invalid Encoder tail output policy.");
    const encoderOwners = [2, 8, 14];
    const reindex = [24, 28, 32, 36];
    for (let index = 0; index < 40 + (includeDraft ? 3 : 0); index += 1) {
      const row = {};
      const draft = index >= 40;
      const encoder = index < 20;
      const owns = encoderOwners.includes(index) || index === 20;
      const entries = encoder ? tokens / 2n : tokens;
      const owner = encoder ? encoderOwners.filter((value) => value <= index).at(-1) : 20;
      const role = owns ? "Full" : reindex.includes(index) ? "Reindex" : owner == null ? "SWA only" : "Reuse";
      if (owns) {
        row[encoder ? "encoder_main_kv" : "decoder_main_kv"] = entries * sequences * main;
        row[encoder ? "encoder_indexer" : "decoder_indexer"] = entries * sequences * indexer;
      }
      if (includeState || draft) row[draft ? "draft_swa" : encoder ? "encoder_swa" : "decoder_swa"] = window * sequences * swa;
      if (includeState && encoderOwners.includes(index)) row.compression_tail = (tokens % 2n) * sequences * 4096n;
      if (includeState && tailPolicy === "retain" && index === 19) row.encoder_outputs = window * sequences * 40976n;
      for (const [key, value] of Object.entries(row)) addComponent(key, value);
      addLayer(index, draft ? "Draft SWA" : `${encoder ? "Encoder" : "Decoder"} · ${role} · owner ${owner ?? "none"}`, owns ? entries : window, owns && encoder ? 2 : null, row, draft);
    }
    if (includeState && tokens > 0n) {
      reservedComponents.compression_tail = sequences * 24576n;
      reservedComponents.swa = sequences * 40n * 128n * swa;
    }
    if (includeDraft && tokens > 0n) reservedComponents.draft_swa = sequences * 3n * 128n * swa;
    equation("global_kv", "Bglobal = S × (main entry + index entry) × [3 × floor(T/2) + T]",
      `${sequences} × (${main} + ${indexer}) × [3 × ${tokens / 2n} + ${tokens}]`,
      sequences * (main + indexer) * (3n * (tokens / 2n) + tokens));
    if (includeState) {
      equation("sliding_kv", "Bswa = S × 40 × min(T,128) × SWA entry", `${sequences} × 40 × ${window} × ${swa}`, sequences * 40n * window * swa);
      equation("compression_tail", "Btail = S × 3 × 2 × 512 × 4 × (T mod 2)", `${sequences} × 12288 × ${tokens % 2n}`, components.compression_tail);
      if (tailPolicy === "retain") equation("encoder_outputs", "Bout = S × min(T,128) × (4 × 5120 × 2 + 4 × 4)", `${sequences} × ${window} × 40976`, components.encoder_outputs);
    }
  } else {
    throw new KvInputError(`Unsupported calculator formula: ${formula || "missing"}.`);
  }

  const totalBytes = Object.values(components).reduce((sum, value) => sum + value, 0n);
  equations.push({
    component: "total",
    expression: "Btotal = Σ Bcomponent",
    substitution: Object.entries(components).map(([name, value]) => `${COMPONENT_LABELS[name] || name} ${value} B`).join(" + "),
    bytes: totalBytes,
  });
  return { formula, totalBytes, components, equations, layers, reservedComponents };
}

export function describeKvResult(model, settings, result) {
  const v41 = result.formula === "deepseek_v41_csa2";
  const maxTokens = BigInt(model.kv_calculator.max_context_tokens || 0);
  const contextWarning = maxTokens > 0n && settings.tokens > maxTokens
    ? `Tokens exceed this model's configured context (${formatInteger(maxTokens)}). The nominal formula is still evaluated.` : "";
  const kvPrecision = v41 && settings.precisionHalfBytes === 1
    ? "Released · FP4 global + FP8 SWA + scales"
    : `${PRECISIONS.find(p => p.halfBytes === settings.precisionHalfBytes)?.label || "KV"} · ${formatPrecision(settings.precisionHalfBytes)}`;
  const indexerPrecision = v41 && settings.precisionHalfBytes === 1
    ? "FP4 + scales · 68 B / entry"
    : `${PRECISIONS.find(p => p.halfBytes === settings.indexerPrecisionHalfBytes)?.label || "Indexer"} · ${formatPrecision(settings.indexerPrecisionHalfBytes ?? settings.precisionHalfBytes)}`;
  const scope = v41 ? `Held / retained cache state at this length; this is not complete recovery peak memory. ${settings.precisionHalfBytes === 1 ? "Global main / index: 288 / 68 B per entry (FP4 + scales); SWA: 528 B (FP8 + scales)." : "Global main / index: 1024 / 256 B per entry; SWA: 1024 B (BF16)."} Compression tail: FP32; retained h: BF16; pre_mix: FP32. Known buffer reserves (alternative to effective payload, never added to it): ${Object.entries(result.reservedComponents).map(([key, bytes]) => `${labelForField(key)} ${formatBytes(bytes)}`).join("; ") || "none"}.` : "Nominal attention cache and enabled state capacity at this sequence length.";
  return { contextWarning, kvPrecision, indexerPrecision, scope, semanticClass: model.kv_calculator.semantic_class || "—" };
}

function renderResult(model, settings, fields, result, explanations = describeKvResult(model, settings, result)) {
  byId("kvError").hidden = true;
  byId("kvResultContent").hidden = false;
  byId("kvTotalBinary").textContent = formatBinary(result.totalBytes, 3, "GiB");
  byId("kvTotalDecimal").textContent = formatDecimal(result.totalBytes, 3, "GB");
  byId("kvTotalBytes").textContent = `${formatInteger(result.totalBytes)} bytes exact`;
  byId("kvPerToken").textContent = formatBytesRatio(result.totalBytes, settings.tokens * settings.sequences);
  byId("kvPerSequence").textContent = formatBytesRatio(result.totalBytes, settings.sequences);
  byId("kvFormulaBadge").textContent = result.formula === "qwen_linear_full_hybrid"
    ? "GDN / Linear Attention Hybrid"
    : result.formula;
  byId("kvSemanticBadge").textContent = model.kv_calculator.semantic_class;
  const scope = byId("kvCapacityScope");
  scope.hidden = result.formula !== "deepseek_v41_csa2";
  scope.textContent = explanations.scope;
  renderComponents(result.components);
  renderEquations(result.equations);
  renderModelFacts(model, fields);
  document.querySelector(".kv-layer-note").textContent = result.formula === "deepseek_v41_csa2"
    ? "V4.1 uses zero-based layer numbers. Full owns global KV; Reindex and Reuse share the listed owner. Draft rows contain their separate SWA."
    : "Calculation layers expand the capacity equation; they do not claim the exact decoder-layer ordering or implementation type.";
  renderLayers(result.layers);
  byId("kvContextWarning").textContent = explanations.contextWarning;
  byId("kvContextWarning").hidden = !explanations.contextWarning;
}

function renderComponents(components) {
  const container = byId("kvComponentGrid");
  container.replaceChildren();
  for (const [name, bytes] of Object.entries(components)) {
    const article = document.createElement("article");
    article.className = "kv-component-card";
    const label = document.createElement("span");
    label.textContent = COMPONENT_LABELS[name] || labelForField(name);
    const value = document.createElement("strong");
    value.textContent = formatBytes(bytes);
    const exact = document.createElement("small");
    exact.textContent = `${formatInteger(bytes)} bytes`;
    article.append(label, value, exact);
    container.append(article);
  }
}

function renderEquations(equations) {
  const container = byId("kvFormulaList");
  container.replaceChildren();
  for (const item of equations) {
    const article = document.createElement("article");
    article.className = "kv-formula-card";
    const heading = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = item.component === "total"
      ? "Total"
      : COMPONENT_LABELS[item.component] || labelForField(item.component);
    const value = document.createElement("span");
    value.textContent = formatBytes(item.bytes);
    heading.append(label, value);
    const expression = document.createElement("code");
    expression.textContent = item.expression;
    const substitution = document.createElement("p");
    substitution.textContent = `${item.substitution} = ${formatInteger(item.bytes)} bytes`;
    article.append(heading, expression, substitution);
    container.append(article);
  }
}

function renderModelFacts(model, fields) {
  const container = byId("kvModelFacts");
  container.replaceChildren();
  const calculator = model.kv_calculator;
  const facts = [
    ["Model", model.model_name || model.label || model.model_id],
    ["Family", model.family || "Unknown"],
    ["Formula", calculator.formula_id],
    ["Semantic class", calculator.semantic_class],
    ["Max context tokens", calculator.max_context_tokens],
    ["Draft support", calculator.draft_support],
    ...Object.entries(fields),
  ];
  for (const [key, value] of facts) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = labelForField(key);
    const detail = document.createElement("dd");
    detail.textContent = formatFieldValue(value);
    row.append(term, detail);
    container.append(row);
  }
}

function renderLayers(layers) {
  const body = byId("kvLayerBody");
  body.replaceChildren();
  for (const layer of layers) {
    const row = document.createElement("tr");
    const index = document.createElement("td");
    index.textContent = String(layer.index);
    const type = document.createElement("td");
    type.textContent = `${layer.type}${layer.draft ? " · draft" : ""}`;
    const tokenSpan = document.createElement("td");
    tokenSpan.textContent = String(layer.tokenSpan);
    const ratio = document.createElement("td");
    ratio.textContent = layer.ratio === null ? "—" : String(layer.ratio);
    const components = document.createElement("td");
    components.textContent = Object.entries(layer.byComponent)
      .map(([name, bytes]) => `${COMPONENT_LABELS[name] || name}: ${formatBytes(bytes)}`)
      .join(" · ");
    const total = document.createElement("td");
    total.textContent = formatBytes(layer.totalBytes);
    row.append(index, type, tokenSpan, ratio, components, total);
    body.append(row);
  }
  byId("kvLayerCount").textContent = `${layers.length} calculation layers`;
}

function renderError(message) {
  byId("kvResultContent").hidden = true;
  const error = byId("kvError");
  error.hidden = false;
  error.textContent = message;
}

function clearInputErrors() {
  for (const input of document.querySelectorAll("#kvSizeView [aria-invalid=true]")) {
    input.removeAttribute("aria-invalid");
  }
}

function inputBigInt(id, label, minimum) {
  const input = byId(id);
  try {
    return requiredBigInt(input.value, label, minimum);
  } catch (error) {
    input.setAttribute("aria-invalid", "true");
    throw error;
  }
}

function requiredBigInt(value, label, minimum = 0n) {
  const text = String(value ?? "").trim();
  if (!/^\d+$/.test(text)) throw new KvInputError(`${label} must be an integer.`);
  const parsed = BigInt(text);
  if (parsed < minimum) throw new KvInputError(`${label} must be at least ${minimum}.`);
  return parsed;
}

function requiredSmallInt(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new KvInputError(`${label} is invalid.`);
  return parsed;
}

function fieldInt(fields, key, minimum = 0n) {
  if (!(key in fields)) throw new KvInputError(`${labelForField(key)} is required.`);
  return requiredBigInt(fields[key], labelForField(key), minimum);
}

function optionalFieldInt(fields, key, fallback = 0n) {
  if (!(key in fields) || fields[key] === null || fields[key] === "") return typeof fallback === "function" ? fallback() : fallback;
  return requiredBigInt(fields[key], labelForField(key), 0n);
}

function ratioFields(value) {
  const values = Array.isArray(value)
    ? value
    : String(value ?? "").split(/[\s,]+/).filter(Boolean);
  if (!values.length) throw new KvInputError("compress_ratios must not be empty.");
  return values.map((item) => requiredBigInt(item, "Compress ratio", 0n));
}

function layerLoopCount(value) {
  if (value > 4096n) throw new KvInputError("Layer count must not exceed 4,096.");
  return Number(value);
}

function checkpointCount(tokens, interval) {
  if (tokens === 0n) return 0n;
  return interval === null ? 1n : (tokens + interval - 1n) / interval;
}

function elementBytes(elements, precisionHalfBytes) {
  const numerator = elements * BigInt(precisionHalfBytes);
  if (numerator % 2n !== 0n) {
    throw new KvInputError("The selected parameters produce a fractional byte count; adjust dimensions or precision.");
  }
  return numerator / 2n;
}

function precisionHalfBytes(id) {
  const precision = PRECISIONS.find((item) => item.id === id);
  if (!precision) throw new KvInputError("Choose a supported precision.");
  return precision.halfBytes;
}

function fieldPrecisionId(value) {
  return PRECISIONS.some((item) => item.id === value) ? value : "";
}

function defaultIncludeState(fields, formula) {
  if (!STATE_FORMULAS.has(formula)) return false;
  if ("default_include_linear_attention_state" in fields) {
    return Boolean(fields.default_include_linear_attention_state);
  }
  if ("default_include_sconv_state" in fields) return Boolean(fields.default_include_sconv_state);
  return true;
}

function supportsDraft(model, fields) {
  if (fields.disable_draft_kv_cache === true) return false;
  const formula = model.kv_calculator.formula_id;
  if (formula === "deepseek_v4_hybrid") {
    const ratios = Array.isArray(fields.compress_ratios) ? fields.compress_ratios : String(fields.compress_ratios ?? "").split(/[\s,]+/).filter(Boolean);
    return ratios.length > Number(fields.num_hidden_layers || 0);
  }
  return [
    "num_nextn_predict_layers", "num_mtp_modules",
    "draft_full_attention_layers", "draft_sliding_attention_layers",
  ].some((key) => Number(fields[key] || 0) > 0);
}

function formatPrecision(halfBytes) {
  return halfBytes % 2 === 0 ? `${halfBytes / 2} B` : `${halfBytes / 2} B`;
}

function formatBytes(bytes) {
  const units = [
    [1024n ** 4n, "TiB"], [1024n ** 3n, "GiB"], [1024n ** 2n, "MiB"],
    [1024n, "KiB"], [1n, "B"],
  ];
  const [divisor, label] = units.find(([value]) => bytes >= value) || units.at(-1);
  return `${formatScaled(bytes, divisor, divisor === 1n ? 0 : 3)} ${label}`;
}

function formatBinary(bytes, digits, unit) {
  const divisors = { KiB: 1024n, MiB: 1024n ** 2n, GiB: 1024n ** 3n, TiB: 1024n ** 4n };
  return `${formatScaled(bytes, divisors[unit], digits)} ${unit}`;
}

function formatDecimal(bytes, digits, unit) {
  const divisors = { KB: 1000n, MB: 1000n ** 2n, GB: 1000n ** 3n, TB: 1000n ** 4n };
  return `${formatScaled(bytes, divisors[unit], digits)} ${unit}`;
}

function formatBytesRatio(bytes, denominator) {
  if (denominator <= 0n) return "—";
  const units = [
    [1024n ** 3n, "GiB"], [1024n ** 2n, "MiB"], [1024n, "KiB"], [1n, "B"],
  ];
  const [divisor, label] = units.find(([value]) => bytes >= denominator * value) || units.at(-1);
  return `${formatFraction(bytes, denominator * divisor, 3)} ${label}`;
}

function formatScaled(value, divisor, digits) {
  return formatFraction(value, divisor, digits);
}

function formatFraction(numerator, denominator, digits) {
  const scale = 10n ** BigInt(digits);
  const rounded = (numerator * scale + denominator / 2n) / denominator;
  const integer = rounded / scale;
  if (!digits) return formatInteger(integer);
  const fraction = String(rounded % scale).padStart(digits, "0").replace(/0+$/, "");
  return fraction ? `${formatInteger(integer)}.${fraction}` : formatInteger(integer);
}

function formatInteger(value) {
  const text = String(value);
  return text.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function labelForField(key) {
  const explicit = {
    model: "Model",
    family: "Family",
    formula: "Formula",
    "semantic class": "Semantic class",
    "max context tokens": "Max context tokens",
    "draft support": "Draft support",
    compress_ratios: "Compress ratios",
  };
  return explicit[key] || String(key).replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatFieldValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined || value === "") return "—";
  return typeof value === "number" ? formatInteger(BigInt(value)) : String(value);
}

function minBigInt(left, right) {
  return left < right ? left : right;
}

function cloneValue(value) {
  return JSON.parse(JSON.stringify(value));
}

function byId(id) {
  return document.getElementById(id);
}
