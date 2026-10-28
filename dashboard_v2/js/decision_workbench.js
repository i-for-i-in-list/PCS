import {
  cancelRecommendationJob,
  compareExperiments,
  createRecommendationJob,
  deleteExperiment,
  getExperiment,
  getExperiments,
  getRecommendationJob,
  updateExperiment,
} from "./api.js";
import {
  initializeEncoderScenarios, applyEncoderScenarios, readEncoderScenarios, supportsEncoderScenarios,
  applyStatePolicyEditor,
  estimateBootstrapState,
  readStatePolicyEditor,
  renderStatePolicyEditor,
} from "./state_policies.js";

const LAST_VIEW_KEY = "dashboard-v3-last-view";
const ACTIVE_RECOMMENDATION_KEY = "dashboard-v3-active-recommendation-job";
const ACTIVE_RECOMMENDATION_PAYLOAD_KEY = "dashboard-v3-active-recommendation-payload";
const POLL_INTERVAL_MS = 750;
const WORKSPACES = ["explore", "recommend", "experiments", "kvSize", "architecture"];
const WORKLOAD_COPY = Object.freeze({
  synthetic_chat: {
    title: "Chat",
    eyebrow: "Conversational reuse",
    description: "Repeated system prompts and popular conversation prefixes create concentrated reuse.",
  },
  synthetic_rag: {
    title: "RAG",
    eyebrow: "Retrieved context",
    description: "More distinct document contexts create a broader, flatter prefix population.",
  },
  synthetic_agent: {
    title: "Agent",
    eyebrow: "Long tool trajectories",
    description: "Long prompts and outputs model multi-step agent sessions with moderate prefix diversity.",
  },
});
const CUSTOM_WORKLOAD_MODE = "custom";
const WORKLOAD_CONTROL_IDS = new Set([
  "recommendSeqLensInput",
  "recommendHitRatesInput",
  "recommendQpsInput",
  "recommendRequestCountInput",
  "recommendGenerationLenInput",
  "recommendPrefixGroupCountInput",
  "recommendZipfInput",
  "recommendSeedInput",
  "recommendArrivalFixed",
  "recommendArrivalPoisson",
]);
const WORKLOAD_SCALAR_CONTROLS = Object.freeze({
  request_count: { id: "recommendRequestCountInput", integer: true, min: 0 },
  generation_len: { id: "recommendGenerationLenInput", integer: true, min: 0 },
  prefix_group_count: { id: "recommendPrefixGroupCountInput", integer: true, min: 1 },
  prefix_zipf_alpha: { id: "recommendZipfInput", integer: false, min: 0 },
  seed: { id: "recommendSeedInput", integer: true, min: null },
});
const WORKLOAD_AXIS_CONTROLS = Object.freeze({
  seq_len: { id: "recommendSeqLensInput", integer: true, min: Number.MIN_VALUE },
  prefix_ratio: { id: "recommendHitRatesInput", integer: false, min: 0, max: 1 },
  qps: { id: "recommendQpsInput", integer: false, min: Number.MIN_VALUE },
});
const RECOMMEND_FIELD_LABELS = Object.freeze({
  recommendSeqLensInput: "Prompt length",
  recommendHitRatesInput: "Prefix ratio",
  recommendQpsInput: "QPS",
  recommendRequestCountInput: "Request count",
  recommendGenerationLenInput: "Generation length",
  recommendPrefixGroupCountInput: "Prefix groups",
  recommendZipfInput: "Zipf alpha",
  recommendSeedInput: "Seed",
  recommendArrivalFixed: "Arrival model",
  constraintTtft: "Dependency TTFT saving",
});
const RESOURCE_CONSTRAINTS = Object.freeze([
  { id: "constraintRead", key: "max_read_avg_GBps", storageKey: "sequential_read_GBps", unit: "GB/s" },
  { id: "constraintWrite", key: "max_write_avg_GBps", storageKey: "sequential_write_GBps", unit: "GB/s" },
  { id: "constraintMissWindow", key: "max_aggregate_dependency_required_GBps", storageKey: "sequential_read_GBps", unit: "GB/s" },
  { id: "constraintIops", key: "max_aggregate_dependency_chunked_iops", storageKey: "random_read_iops", unit: "IOPS" },
  { id: "constraintCapacity", key: "max_retain_all_capacity_GiB", storageKey: "logical_capacity_GiB", unit: "GiB" },
]);
const OBJECTIVE_EXPLANATIONS = Object.freeze({
  balanced_headroom: "Full coverage → worst resource headroom → worst TTFT margin",
  max_coverage: "Coverage → worst resource headroom → average storage pressure",
  min_storage_pressure: "Full coverage → average storage pressure → worst TTFT margin",
  max_dependency_ttft_margin: "Full coverage → worst dependency TTFT margin → worst resource headroom",
});
let activeWorkspace = null;
let workspaceAnimationToken = 0;

export function initializeDecisionWorkbench({ config, onOpenExplore }) {
  const state = {
    config,
    onOpenExplore,
    recommendation: null,
    selectedCandidateId: null,
    resultFingerprint: null,
    isDirty: false,
    activeJobId: null,
    pollToken: 0,
    experiments: [],
    selectedExperimentIds: new Set(),
    workloadMode: "preset",
    lastWorkloadPresetPath: config.defaults.workload_config || "",
  };
  populateRecommendationControls(state);
  wireNavigation(state);
  wireRecommendation(state);
  wireExperiments(state);
  switchWorkspace(safeStorageGet(localStorage, LAST_VIEW_KEY) || "explore", false);
  refreshRecommendationEstimate(state);
  refreshExperiments(state);
  restoreRecommendationJob(state);
  window.addEventListener("dashboard-history-changed", () => refreshExperiments(state));
  return {
    showExplore: () => switchWorkspace("explore"),
    refreshExperiments: () => refreshExperiments(state),
  };
}

function wireNavigation(state) {
  const buttons = [...document.querySelectorAll("[data-workspace-target]")];
  document.querySelector(".workspace-nav").style.setProperty("--workspace-count", buttons.length);
  const topbar = document.querySelector(".topbar");
  const architectureView = byId("architectureView");
  const updateArchitectureHeight = () => {
    architectureView.style.setProperty("--topbar-height", `${topbar.getBoundingClientRect().height}px`);
  };
  updateArchitectureHeight();
  new ResizeObserver(updateArchitectureHeight).observe(topbar, { box: "border-box" });
  buttons.forEach((button, index) => {
    button.addEventListener("click", () => switchWorkspace(button.dataset.workspaceTarget));
    button.addEventListener("keydown", (event) => {
      const delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (!delta) return;
      event.preventDefault();
      const target = buttons[(index + delta + buttons.length) % buttons.length];
      target.focus();
      target.click();
    });
  });
}

function switchWorkspace(name, persist = true) {
  const target = WORKSPACES.includes(name) ? name : "explore";
  const previous = activeWorkspace;
  const previousIndex = WORKSPACES.indexOf(previous);
  const targetIndex = WORKSPACES.indexOf(target);
  for (const button of document.querySelectorAll("[data-workspace-target]")) {
    const active = button.dataset.workspaceTarget === target;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
  const nav = document.querySelector(".workspace-nav");
  if (nav) {
    nav.dataset.activeIndex = String(targetIndex);
    nav.style.setProperty("--workspace-index", targetIndex);
  }
  const targetView = byId(`${target}View`);
  for (const workspace of WORKSPACES) {
    const view = byId(`${workspace}View`);
    view.classList.remove("workspace-enter", "from-left", "from-right");
    delete view.dataset.animationToken;
    view.hidden = workspace !== target;
  }
  activeWorkspace = target;
  if (target === "architecture") {
    const frame = byId("architectureFrame");
    // Initialize only after the panel is visible so the original viewer can
    // measure its viewport, and retain the document on subsequent tab visits.
    if (!frame.hasAttribute("src") && !frame.hasAttribute("srcdoc")) {
      if (frame.dataset.srcdoc !== undefined) {
        frame.srcdoc = frame.dataset.srcdoc;
        delete frame.dataset.srcdoc;
      } else {
        frame.src = frame.dataset.src;
      }
    }
  }
  if (previous && previous !== target && targetView) {
    const token = ++workspaceAnimationToken;
    void targetView.offsetWidth;
    targetView.dataset.animationToken = String(token);
    targetView.classList.add("workspace-enter", targetIndex > previousIndex ? "from-right" : "from-left");
    window.setTimeout(() => {
      if (targetView.dataset.animationToken === String(token)) {
        targetView.classList.remove("workspace-enter", "from-left", "from-right");
        delete targetView.dataset.animationToken;
      }
    }, 220);
  }
  if (persist) safeStorageSet(localStorage, LAST_VIEW_KEY, target);
  if (target === "experiments") window.dispatchEvent(new CustomEvent("dashboard-experiments-visible"));
}

function populateRecommendationControls(state) {
  const { config } = state;
  populateCheckList("recommendModelList", config.models, (item) => ({
    checked: !item.error && (config.defaults.recommendation_model_configs || []).includes(item.path),
    disabled: Boolean(item.error),
    title: item.model_name || item.label,
    detail: item.error || `${item.recovery_capability?.display_attention || item.formula_id || item.attention || "unknown"} · ${item.family || "unknown family"} · ${(item.evidence_statuses || ["unspecified"]).join("/")}`,
  }));
  populateRecommendationModelFilters(state);
  filterRecommendationModels(state);
  populateCheckList("recommendRuntimeList", config.runtimes, (item) => ({
    checked: !item.error && (config.defaults.recommendation_runtime_configs || []).includes(item.path),
    disabled: Boolean(item.error),
    title: item.label,
    detail: item.error || "",
  }));
  populateCheckList("recommendStorageList", config.storages, (item) => ({
    checked: !item.error,
    disabled: Boolean(item.error),
    title: item.label,
    detail: item.error || "",
  }));
  populateSelect(byId("recommendWorkloadSelect"), config.workloads, config.defaults.workload_config);
  applyRecommendationModelingOptions(config.defaults.modeling_options || {}, state);
  state.lastWorkloadPresetPath = byId("recommendWorkloadSelect").value
    || state.lastWorkloadPresetPath
    || config.workloads[0]?.path
    || "";
  populateWorkloadCards(state);
  const objective = byId("recommendObjective");
  objective.replaceChildren();
  for (const item of config.recommendation?.objectives || []) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    objective.append(option);
  }
  applyRecommendationWorkloadDefaults(state);
  byId("recommendAnalysisMode").value = config.defaults.analysis_mode || "bandwidth_envelope";
  applyRecommendationMode();
  renderRecommendationLifecyclePreview(state);
  renderConstraintDefaults(state);
  byId("recommendStatus").textContent = "Ready · recommendation runs in the background.";
}

function populateWorkloadCards(state) {
  const container = byId("recommendWorkloadCards");
  container.replaceChildren();
  const preferredOrder = new Map([
    ["synthetic_chat", 0],
    ["synthetic_rag", 1],
    ["synthetic_agent", 2],
  ]);
  const workloads = [...state.config.workloads].sort((left, right) => (
    (preferredOrder.get(configName(left.path)) ?? 99)
    - (preferredOrder.get(configName(right.path)) ?? 99)
    || String(left.label).localeCompare(String(right.label))
  ));
  workloads.forEach((workload, index) => {
    const copy = workloadCopy(workload);
    const defaults = workloadDefaults(workload);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "workload-preset-card";
    button.dataset.workloadPath = workload.path;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", "false");
    button.tabIndex = -1;
    button.innerHTML = `
      <span class="workload-card-eyebrow">${escapeText(copy.eyebrow)}</span>
      <strong>${escapeText(copy.title)}</strong>
      <span class="workload-card-copy">${escapeText(copy.description)}</span>
      <span class="workload-card-stats">${escapeText(formatTokenCount(defaults.seq_len))} prompt · ${escapeText(formatTokenCount(defaults.generation_len))} generated · ${escapeText(formatNumber(Number(defaults.qps)))} QPS</span>`;
    button.addEventListener("click", () => selectWorkloadPreset(state, workload.path));
    wireWorkloadCardKeyboard(container, button);
    container.append(button);
    if (index === 0 && !byId("recommendWorkloadSelect").value) {
      byId("recommendWorkloadSelect").value = workload.path;
    }
  });
  const custom = document.createElement("button");
  custom.type = "button";
  custom.className = "workload-preset-card workload-custom-card";
  custom.dataset.workloadMode = CUSTOM_WORKLOAD_MODE;
  custom.setAttribute("role", "radio");
  custom.setAttribute("aria-checked", "false");
  custom.tabIndex = -1;
  custom.innerHTML = `
    <span class="workload-card-eyebrow">User-defined envelope</span>
    <strong>Custom</strong>
    <span class="workload-card-copy">Start with blank workload fields and define the traffic envelope explicitly.</span>
    <span class="workload-card-stats">Blank fields · no additional config file</span>`;
  custom.addEventListener("click", () => selectCustomWorkload(state));
  wireWorkloadCardKeyboard(container, custom);
  container.append(custom);
  renderWorkloadPreset(state);
}

function wireWorkloadCardKeyboard(container, button) {
  button.addEventListener("keydown", (event) => {
    const delta = ["ArrowRight", "ArrowDown"].includes(event.key)
      ? 1
      : ["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 0;
    if (!delta) return;
    event.preventDefault();
    const cards = [...container.querySelectorAll("[role=radio]")];
    const current = cards.indexOf(button);
    const next = cards[(current + delta + cards.length) % cards.length];
    next.focus();
    next.click();
  });
}

function selectWorkloadPreset(state, path) {
  state.workloadMode = "preset";
  state.lastWorkloadPresetPath = path;
  byId("recommendWorkloadSelect").value = path;
  applyRecommendationWorkloadDefaults(state);
  refreshRecommendationEstimate(state);
  updateRecommendationFreshness(state);
}

function selectCustomWorkload(state) {
  if (state.workloadMode === CUSTOM_WORKLOAD_MODE) return;
  state.workloadMode = CUSTOM_WORKLOAD_MODE;
  ensureWorkloadBasePath(state);
  clearCustomWorkloadFields();
  renderWorkloadPreset(state);
  refreshRecommendationEstimate(state);
  updateRecommendationFreshness(state);
}

function ensureWorkloadBasePath(state) {
  const paths = new Set(state.config.workloads.map((item) => item.path));
  const current = byId("recommendWorkloadSelect").value;
  const fallback = paths.has(state.lastWorkloadPresetPath)
    ? state.lastWorkloadPresetPath
    : state.config.defaults.workload_config || state.config.workloads[0]?.path || "";
  const path = paths.has(current) ? current : fallback;
  byId("recommendWorkloadSelect").value = path;
  state.lastWorkloadPresetPath = path;
}

function clearCustomWorkloadFields() {
  for (const spec of Object.values(WORKLOAD_AXIS_CONTROLS)) byId(spec.id).value = "";
  for (const spec of Object.values(WORKLOAD_SCALAR_CONTROLS)) byId(spec.id).value = "";
  byId("recommendArrivalFixed").checked = false;
  byId("recommendArrivalPoisson").checked = false;
}

function renderWorkloadPreset(state) {
  const selectedPath = byId("recommendWorkloadSelect").value;
  const workload = state.config.workloads.find((item) => item.path === selectedPath);
  byId("recommendWorkloadCards").querySelectorAll("[role=radio]").forEach((card) => {
    const selected = state.workloadMode === CUSTOM_WORKLOAD_MODE
      ? card.dataset.workloadMode === CUSTOM_WORKLOAD_MODE
      : card.dataset.workloadPath === selectedPath;
    card.classList.toggle("active", selected);
    card.setAttribute("aria-checked", String(selected));
    card.tabIndex = selected ? 0 : -1;
  });
  if (state.workloadMode === CUSTOM_WORKLOAD_MODE) {
    renderCustomWorkloadSummary();
    return;
  }
  if (!workload) {
    byId("recommendWorkloadSummary").replaceChildren();
    return;
  }
  const copy = workloadCopy(workload);
  const defaults = workloadDefaults(workload);
  const hit = Math.max(0, Math.min(1, Number(defaults.prefix_ratio) || 0));
  byId("recommendWorkloadSummary").innerHTML = `
    <div class="workload-flow">
      <span><strong>${escapeText(formatTokenCount(defaults.seq_len))}</strong> prompt tokens</span>
      <div class="reuse-bar" role="img" aria-label="${escapeText(formatNumber(hit * 100))} percent reusable prefix">
        <span style="width:${hit * 100}%"></span>
      </div>
      <span><strong>${escapeText(formatNumber(hit * 100))}%</strong> reused · ${escapeText(formatNumber((1 - hit) * 100))}% miss</span>
      <span><strong>${escapeText(formatTokenCount(defaults.generation_len))}</strong> generated</span>
    </div>
    <p><strong>${escapeText(copy.title)} baseline.</strong> ${escapeText(formatCompact(defaults.prefix_group_count))} prefix groups with Zipf ${escapeText(formatNumber(Number(defaults.prefix_zipf_alpha)))}. Selecting a preset resets every field below; comma-separated coverage axes can then extend it.</p>`;
}

function renderCustomWorkloadSummary() {
  const summary = byId("recommendWorkloadSummary");
  const draft = readWorkloadDraft();
  if (!draft.complete) {
    summary.innerHTML = `
      <div class="workload-custom-empty">
        <strong>Custom envelope is waiting for valid inputs.</strong>
        <span>Complete every workload field and select at least one arrival model to preview the envelope.</span>
      </div>`;
    return;
  }
  const scenarios = draft.seq_len.length
    * draft.prefix_ratio.length
    * draft.qps.length
    * draft.arrival_model.length;
  summary.innerHTML = `
    <div class="workload-custom-metrics">
      <span><small>Prompt</small><strong>${escapeText(formatAxisCoverage(draft.seq_len, formatTokenCount))}</strong></span>
      <span><small>Reuse</small><strong>${escapeText(formatAxisCoverage(draft.prefix_ratio, (value) => `${formatNumber(value * 100)}%`))}</strong></span>
      <span><small>Generated</small><strong>${escapeText(formatTokenCount(draft.generation_len))}</strong></span>
      <span><small>Scenarios</small><strong>${escapeText(formatCompact(scenarios))}</strong></span>
    </div>
    <p><strong>User-defined envelope.</strong> ${escapeText(formatCompact(draft.request_count))} requests · ${escapeText(formatCompact(draft.prefix_group_count))} prefix groups · Zipf ${escapeText(formatNumber(draft.prefix_zipf_alpha))} · ${escapeText(draft.arrival_model.join(" + "))} arrival.</p>`;
}

function formatAxisCoverage(values, formatter) {
  if (values.length === 1) return formatter(values[0]);
  return `${formatter(values[0])}–${formatter(values.at(-1))} · ${values.length} values`;
}

function workloadDefaults(workload) {
  const overrides = workload?.workload_overrides || {};
  return {
    request_count: workload?.preset_defaults?.request_count ?? workload?.request_count ?? overrides.request_count ?? 0,
    qps: workload?.preset_defaults?.qps ?? workload?.qps ?? overrides.qps ?? 0,
    seq_len: workload?.preset_defaults?.seq_len ?? workload?.seq_len ?? 0,
    generation_len: workload?.preset_defaults?.generation_len ?? overrides.generation_len ?? 0,
    prefix_ratio: workload?.preset_defaults?.prefix_ratio ?? workload?.prefix_ratio ?? 0,
    prefix_group_count: workload?.preset_defaults?.prefix_group_count ?? overrides.prefix_group_count ?? 1,
    prefix_zipf_alpha: workload?.preset_defaults?.prefix_zipf_alpha ?? overrides.prefix_zipf_alpha ?? 0,
    arrival_model: workload?.preset_defaults?.arrival_model ?? "fixed",
    seed: workload?.preset_defaults?.seed ?? overrides.seed ?? 1,
  };
}

function workloadCopy(workload) {
  const key = configName(workload?.path);
  return WORKLOAD_COPY[key] || {
    title: workload?.label || key,
    eyebrow: "Synthetic preset",
    description: "A repository-defined synthetic workload profile.",
  };
}

function readWorkloadDraft() {
  const draft = {};
  for (const [key, spec] of Object.entries(WORKLOAD_AXIS_CONTROLS)) {
    draft[key] = parseDraftList(spec);
  }
  for (const [key, spec] of Object.entries(WORKLOAD_SCALAR_CONTROLS)) {
    draft[key] = parseDraftScalar(spec);
  }
  draft.arrival_model = [];
  if (byId("recommendArrivalFixed").checked) draft.arrival_model.push("fixed");
  if (byId("recommendArrivalPoisson").checked) draft.arrival_model.push("poisson");
  draft.complete = [
    ...Object.keys(WORKLOAD_AXIS_CONTROLS),
    ...Object.keys(WORKLOAD_SCALAR_CONTROLS),
  ].every((key) => draft[key] != null) && draft.arrival_model.length > 0;
  return draft;
}

function parseDraftList({ id, integer, min, max = null }) {
  const text = byId(id).value.trim();
  if (!text) return null;
  const tokens = text.split(",").map((value) => value.trim()).filter(Boolean);
  const values = tokens.map(Number);
  if (
    !values.length
    || values.some((value) => !Number.isFinite(value))
    || (integer && values.some((value) => !Number.isInteger(value)))
    || values.some((value) => value < min)
    || (max != null && values.some((value) => value > max))
  ) {
    return null;
  }
  return [...new Set(values)].sort((left, right) => left - right);
}

function parseDraftScalar({ id, integer, min }) {
  const text = byId(id).value.trim();
  if (!text) return null;
  const value = Number(text);
  if (
    !Number.isFinite(value)
    || (integer && !Number.isInteger(value))
    || (min != null && value < min)
  ) {
    return null;
  }
  return value;
}

function matchingWorkloadPreset(state) {
  const draft = readWorkloadDraft();
  if (
    !draft.complete
    || draft.seq_len.length !== 1
    || draft.prefix_ratio.length !== 1
    || draft.qps.length !== 1
    || draft.arrival_model.length !== 1
  ) {
    return null;
  }
  return state.config.workloads.find((workload) => {
    const defaults = workloadDefaults(workload);
    return (
      draft.seq_len[0] === Number(defaults.seq_len)
      && draft.prefix_ratio[0] === Number(defaults.prefix_ratio)
      && draft.qps[0] === Number(defaults.qps)
      && draft.request_count === Number(defaults.request_count)
      && draft.generation_len === Number(defaults.generation_len)
      && draft.prefix_group_count === Number(defaults.prefix_group_count)
      && draft.prefix_zipf_alpha === Number(defaults.prefix_zipf_alpha)
      && draft.seed === Number(defaults.seed)
      && draft.arrival_model[0] === defaults.arrival_model
    );
  }) || null;
}

function synchronizeWorkloadMode(state) {
  const matching = matchingWorkloadPreset(state);
  if (matching) {
    state.workloadMode = "preset";
    state.lastWorkloadPresetPath = matching.path;
    byId("recommendWorkloadSelect").value = matching.path;
  } else {
    state.workloadMode = CUSTOM_WORKLOAD_MODE;
    ensureWorkloadBasePath(state);
  }
  renderWorkloadPreset(state);
}

function formatTokenCount(value) {
  const number = Number(value);
  if (Number.isInteger(number) && number >= 1024 && number % 1024 === 0) {
    return `${formatNumber(number / 1024)}K`;
  }
  return formatCompact(number);
}

function populateCheckList(id, items, describe) {
  const container = byId(id);
  container.replaceChildren();
  for (const item of items) {
    const spec = describe(item);
    const label = document.createElement("label");
    label.className = "model-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = item.path;
    input.checked = spec.checked;
    input.disabled = spec.disabled;
    input.title = spec.detail;
    const copy = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = spec.title;
    const small = document.createElement("small");
    small.textContent = spec.detail;
    copy.append(strong, small);
    label.append(input, copy);
    container.append(label);
  }
}

function populateRecommendationModelFilters(state) {
  for (const [id, field, allLabel] of [
    ["recommendModelFamily", "family", "All families"],
    ["recommendModelFormula", "formula_id", "All formulas"],
  ]) {
    const select = byId(id);
    select.replaceChildren();
    select.append(new Option(allLabel, ""));
    const values = [...new Set(state.config.models.map((item) => item[field]).filter(Boolean))].sort();
    for (const value of values) select.append(new Option(value, value));
  }
  const evidence = byId("recommendModelEvidence");
  evidence.replaceChildren();
  evidence.append(new Option("All evidence", ""));
  const statuses = [...new Set(
    state.config.models.flatMap((item) => item.evidence_statuses || []),
  )].sort();
  for (const value of statuses) evidence.append(new Option(value, value));
}

function filterRecommendationModels(state) {
  const search = byId("recommendModelSearch").value.trim().toLowerCase();
  const family = byId("recommendModelFamily").value;
  const formula = byId("recommendModelFormula").value;
  const evidence = byId("recommendModelEvidence").value;
  const byPath = new Map(state.config.models.map((item) => [item.path, item]));
  const visibleInputs = [];
  for (const label of byId("recommendModelList").querySelectorAll(".model-option")) {
    const input = label.querySelector("input");
    const model = byPath.get(input?.value);
    const haystack = `${model?.model_name || ""} ${model?.label || ""} ${model?.model_id || ""}`.toLowerCase();
    const visible = Boolean(model)
      && (!search || haystack.includes(search))
      && (!family || model.family === family)
      && (!formula || model.formula_id === formula)
      && (!evidence || (model.evidence_statuses || []).includes(evidence));
    label.hidden = !visible;
    if (visible && input && !input.disabled) visibleInputs.push(input);
  }
  const allSelected = visibleInputs.length > 0 && visibleInputs.every((input) => input.checked);
  const button = byId("recommendSelectVisibleModels");
  button.disabled = visibleInputs.length === 0;
  button.textContent = allSelected ? "Clear visible" : "Select visible";
}

function toggleVisibleRecommendationModels(state) {
  const visibleInputs = [...byId("recommendModelList").querySelectorAll(
    ".model-option:not([hidden]) input:not(:disabled)",
  )];
  if (!visibleInputs.length) return;
  const select = !visibleInputs.every((input) => input.checked);
  for (const input of visibleInputs) input.checked = select;
  filterRecommendationModels(state);
  refreshRecommendationStatePolicyEditor(state);
  renderConstraintDefaults(state);
  refreshRecommendationEstimate(state);
  updateRecommendationFreshness(state);
}

function wireRecommendation(state) {
  initializeEncoderScenarios(byId("recommendEncoderScenarios"), () => { refreshRecommendationEstimate(state); updateRecommendationFreshness(state); });
  const inputs = document.querySelectorAll(
    "#recommendView input:not([data-recommend-model-filter]), "
    + "#recommendView select:not([data-recommend-model-filter]), "
    + "#recommendView textarea:not([data-recommend-model-filter])",
  );
  inputs.forEach((input) => input.addEventListener("input", () => {
    if (input.id === "recommendWorkloadSelect") applyRecommendationWorkloadDefaults(state);
    if (WORKLOAD_CONTROL_IDS.has(input.id)) synchronizeWorkloadMode(state);
    if (input.dataset.constraintCustom && input.checked) {
      initializeCustomConstraint(state, input.dataset.constraintCustom);
    }
    renderConstraintDefaults(state);
    renderRecommendationLifecyclePreview(state);
    refreshRecommendationEstimate(state);
    updateRecommendationFreshness(state);
    if (input.id === "recommendObjective" && state.recommendation && !state.isDirty) {
      rerankRecommendation(state, input.value);
      renderRecommendation(state);
    }
    if (input.closest("#recommendModelList")) filterRecommendationModels(state);
    if (input.closest("#recommendModelList") || input.id === "recommendSeqLensInput") {
      refreshRecommendationStatePolicyEditor(state);
    }
    if (input.id === "recommendAnalysisMode") applyRecommendationMode();
  }));
  byId("recommendModelSearch").addEventListener("input", () => filterRecommendationModels(state));
  for (const id of ["recommendModelFamily", "recommendModelFormula", "recommendModelEvidence"]) {
    byId(id).addEventListener("change", () => filterRecommendationModels(state));
  }
  byId("recommendSelectVisibleModels").addEventListener(
    "click",
    () => toggleVisibleRecommendationModels(state),
  );
  byId("runRecommendationButton").addEventListener("click", () => executeRecommendation(state));
  byId("cancelRecommendationButton").addEventListener("click", () => cancelRecommendation(state));
  byId("openRecommendationInExplore").addEventListener("click", () => openSelectedCandidate(state));
}

function applyRecommendationMode() {
  const replay = byId("recommendAnalysisMode").value === "cache_replay";
  byId("recommendCacheCapacityField").hidden = !replay;
  for (const id of ["constraintWrite", "constraintIops", "constraintCapacity"]) {
    byId(id).closest(".constraint-item").hidden = !replay;
  }
  byId("constraintRead").closest(".constraint-item").querySelector(".constraint-title span").textContent = replay
    ? "Read GB/s ≤" : "Optimistic GB/s ≤";
  byId("constraintMissWindow").closest(".constraint-item").querySelector(".constraint-title span").textContent = replay
    ? "Aggregate dependency GB/s ≤" : "Break-even GB/s ≤";
  byId("recommendLifecyclePreview").hidden = !replay;
}

function applyRecommendationWorkloadDefaults(state) {
  const workload = state.config.workloads.find(
    (item) => item.path === byId("recommendWorkloadSelect").value,
  );
  const values = workload
    ? workloadDefaults(workload)
    : {
      ...state.config.defaults.workload_overrides,
      qps: state.config.defaults.workload_overrides?.qps,
      seq_len: state.config.defaults.seq_len?.[0],
      prefix_ratio: state.config.defaults.prefix_ratio?.[0],
      arrival_model: state.config.defaults.arrival_model?.[0],
    };
  if (workload) {
    state.workloadMode = "preset";
    state.lastWorkloadPresetPath = workload.path;
  }
  const mapping = {
    recommendRequestCountInput: "request_count",
    recommendGenerationLenInput: "generation_len",
    recommendPrefixGroupCountInput: "prefix_group_count",
    recommendZipfInput: "prefix_zipf_alpha",
    recommendSeedInput: "seed",
  };
  for (const [id, key] of Object.entries(mapping)) byId(id).value = values[key] ?? "";
  byId("recommendSeqLensInput").value = values.seq_len ?? "";
  byId("recommendHitRatesInput").value = values.prefix_ratio ?? "";
  byId("recommendQpsInput").value = values.qps ?? "";
  byId("recommendArrivalFixed").checked = values.arrival_model !== "poisson";
  byId("recommendArrivalPoisson").checked = values.arrival_model === "poisson";
  renderWorkloadPreset(state);
}

function applyRecommendationPayloadToControls(state, payload) {
  const analysisMode = payload.analysis_mode || "bandwidth_envelope";
  byId("recommendAnalysisMode").value = analysisMode;
  const design = payload.design_space || {};
  const workload = payload.workload_envelope || {};
  applyEncoderScenarios(byId("recommendEncoderScenarios"), workload);
  setCheckedValues("recommendModelList", design.model_configs || []);
  setCheckedValues("recommendRuntimeList", design.runtime_configs || []);
  setCheckedValues("recommendStorageList", design.storage_configs || []);
  filterRecommendationModels(state);
  byId("recommendBlockSizesInput").value = (design.block_size_tokens || []).join(", ");
  const maxIo = design.max_io_bytes || [];
  byId("recommendMaxIoInput").value = maxIo.length === 1 && maxIo[0] == null
    ? ""
    : maxIo.map((value) => value ?? "None").join(", ");
  byId("recommendWorkloadSelect").value = workload.workload_config || "";
  ensureWorkloadBasePath(state);
  byId("recommendSeqLensInput").value = (workload.seq_len || []).join(", ");
  byId("recommendHitRatesInput").value = (workload.prefix_ratio || []).join(", ");
  byId("recommendQpsInput").value = (workload.qps || []).join(", ");
  byId("recommendArrivalFixed").checked = (workload.arrival_model || []).includes("fixed");
  byId("recommendArrivalPoisson").checked = (workload.arrival_model || []).includes("poisson");
  byId("recommendWriteMisses").checked = Boolean(payload.write_miss_blocks);
  applyRecommendationModelingOptions(
    payload.modeling_options || state.config.defaults.modeling_options || {},
    state,
  );
  const overrides = payload.workload_overrides || {};
  const mapping = {
    recommendRequestCountInput: "request_count",
    recommendGenerationLenInput: "generation_len",
    recommendPrefixGroupCountInput: "prefix_group_count",
    recommendZipfInput: "prefix_zipf_alpha",
    recommendSeedInput: "seed",
  };
  for (const [id, key] of Object.entries(mapping)) byId(id).value = overrides[key] ?? "";
  byId("recommendWorkloadShape").value = overrides.workload_shape || "shared_prefix_groups";
  synchronizeWorkloadMode(state);
  const constraints = payload.constraints || {};
  byId("recommendUseStorageLimits").checked = Boolean(constraints.use_storage_limits);
  for (const spec of RESOURCE_CONSTRAINTS) {
    const value = spec.id === "constraintRead" && analysisMode === "bandwidth_envelope"
      ? constraints.max_optimistic_overlap_GBps
      : spec.id === "constraintMissWindow" && analysisMode === "bandwidth_envelope"
        ? constraints.max_dependency_aware_break_even_GBps
        : constraints[spec.key];
    byId(`${spec.id}Custom`).checked = value != null;
    byId(spec.id).value = value ?? "";
  }
  const ttftUs = constraints.min_dependency_ttft_savings_us_p50;
  byId("constraintTtftEnabled").checked = ttftUs != null;
  byId("constraintTtft").value = ttftUs == null ? "0" : String(Number(ttftUs) / 1000);
  renderConstraintDefaults(state);
  byId("recommendObjective").value = payload.objective || "balanced_headroom";
  if (payload.cache_replay_config?.total_capacity_bytes != null) {
    byId("recommendCacheCapacityGiB").value = Number(payload.cache_replay_config.total_capacity_bytes) / (1024 ** 3);
  }
  applyRecommendationMode();
  renderRecommendationLifecyclePreview(state);
  refreshRecommendationEstimate(state);
}

function setCheckedValues(containerId, values) {
  const selected = new Set(values);
  byId(containerId).querySelectorAll("input").forEach((input) => {
    if (!input.disabled) input.checked = selected.has(input.value);
  });
}

function initializeCustomConstraint(state, id) {
  const input = byId(id);
  if (input.value.trim()) return;
  const spec = RESOURCE_CONSTRAINTS.find((item) => item.id === id);
  const values = spec ? selectedStorageLimits(state, spec) : [];
  if (values.length) input.value = String(Math.min(...values.map((item) => item.value)));
}

function renderConstraintDefaults(state) {
  const useStorageLimits = byId("recommendUseStorageLimits").checked;
  for (const spec of RESOURCE_CONSTRAINTS) {
    const custom = byId(`${spec.id}Custom`).checked;
    const input = byId(spec.id);
    const item = input.closest(".constraint-item");
    const effective = byId(`${spec.id}Effective`);
    const breakdown = byId(`${spec.id}Breakdown`);
    const storageValues = selectedStorageLimits(state, spec);
    input.hidden = !custom;
    input.disabled = !custom;
    item.classList.toggle("is-custom", custom);
    item.classList.toggle("is-inactive", !custom && !useStorageLimits);
    if (custom) {
      const value = Number(input.value);
      effective.textContent = Number.isFinite(value)
        ? `Custom · ${formatConstraintValue(value, spec.unit)}`
        : "Custom override";
      breakdown.textContent = "Applies to every selected Storage candidate.";
      breakdown.title = breakdown.textContent;
      continue;
    }
    if (!useStorageLimits) {
      effective.textContent = "Inactive";
      breakdown.textContent = "Enable Storage limits or turn on Custom.";
      breakdown.title = breakdown.textContent;
      continue;
    }
    if (!storageValues.length) {
      effective.textContent = "Unavailable";
      breakdown.textContent = "No selected Storage exposes this threshold.";
      breakdown.title = breakdown.textContent;
      continue;
    }
    const numbers = storageValues.map((itemValue) => itemValue.value);
    const minimum = Math.min(...numbers);
    const maximum = Math.max(...numbers);
    effective.textContent = minimum === maximum
      ? formatConstraintValue(minimum, spec.unit)
      : `${formatConstraintValue(minimum, spec.unit)} – ${formatConstraintValue(maximum, spec.unit)}`;
    breakdown.textContent = storageValues
      .map((itemValue) => `${itemValue.label}: ${formatConstraintValue(itemValue.value, spec.unit)}`)
      .join(" · ");
    breakdown.title = breakdown.textContent;
  }
  const ttftEnabled = byId("constraintTtftEnabled").checked;
  byId("constraintTtft").disabled = !ttftEnabled;
  byId("constraintTtft").closest(".constraint-item").classList.toggle("is-inactive", !ttftEnabled);
}

function selectedStorageLimits(state, spec) {
  const selected = new Set(
    [...byId("recommendStorageList").querySelectorAll("input:checked")]
      .map((input) => input.value),
  );
  return state.config.storages.flatMap((storage) => {
    const value = Number(storage.thresholds?.[spec.storageKey]);
    return selected.has(storage.path) && Number.isFinite(value)
      ? [{ label: storage.label || configName(storage.path), value }]
      : [];
  });
}

function formatConstraintValue(value, unit) {
  if (!Number.isFinite(Number(value))) return `Missing ${unit}`;
  const formatted = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(Number(value));
  return unit ? `${formatted} ${unit}` : formatted;
}

function recommendationPayload(state) {
  const arrivals = [];
  if (byId("recommendArrivalFixed").checked) arrivals.push("fixed");
  if (byId("recommendArrivalPoisson").checked) arrivals.push("poisson");
  const seqLengths = parseNumbers(
    byId("recommendSeqLensInput").value,
    true,
    "Prompt length",
    "recommendSeqLensInput",
  );
  const hitRates = parseNumbers(
    byId("recommendHitRatesInput").value,
    false,
    "Prefix ratio",
    "recommendHitRatesInput",
  );
  const qpsValues = parseNumbers(
    byId("recommendQpsInput").value,
    false,
    "QPS",
    "recommendQpsInput",
  );
  const workloadOverrides = {
    request_count: requiredInput("recommendRequestCountInput", { integer: true, min: 0 }),
    generation_len: requiredInput("recommendGenerationLenInput", { integer: true, min: 0 }),
    prefix_group_count: requiredInput("recommendPrefixGroupCountInput", { integer: true, min: 1 }),
    prefix_zipf_alpha: requiredInput("recommendZipfInput", { min: 0 }),
    seed: requiredInput("recommendSeedInput", { integer: true }),
    workload_shape: byId("recommendWorkloadShape").value,
  };
  if (!arrivals.length) {
    throw validationError("Select at least one arrival model.", "recommendArrivalFixed");
  }
  const analysisMode = byId("recommendAnalysisMode").value;
  const payload = {
    analysis_mode: analysisMode,
    design_space: {
      model_configs: selectedValues("recommendModelList", "model"),
      runtime_configs: selectedValues("recommendRuntimeList", "runtime"),
      storage_configs: selectedValues("recommendStorageList", "storage"),
      block_size_tokens: parseNumbers(byId("recommendBlockSizesInput").value, true, "block sizes"),
      max_io_bytes: parseOptionalIntegers(byId("recommendMaxIoInput").value),
    },
    workload_envelope: {
      workload_config: byId("recommendWorkloadSelect").value
        || state.lastWorkloadPresetPath
        || state.config.defaults.workload_config,
      seq_len: seqLengths,
      prefix_ratio: hitRates,
      arrival_model: arrivals,
      qps: qpsValues,
    },
    workload_overrides: workloadOverrides,
    constraints: {
      use_storage_limits: byId("recommendUseStorageLimits").checked,
      max_read_avg_GBps: analysisMode === "cache_replay" ? customConstraintValue("constraintRead") : null,
      max_write_avg_GBps: customConstraintValue("constraintWrite"),
      max_aggregate_dependency_required_GBps: analysisMode === "cache_replay" ? customConstraintValue("constraintMissWindow") : null,
      max_aggregate_dependency_chunked_iops: customConstraintValue("constraintIops"),
      max_retain_all_capacity_GiB: customConstraintValue("constraintCapacity"),
      max_optimistic_overlap_GBps: analysisMode === "bandwidth_envelope" ? customConstraintValue("constraintRead") : null,
      max_dependency_aware_break_even_GBps: analysisMode === "bandwidth_envelope" ? customConstraintValue("constraintMissWindow") : null,
      min_dependency_ttft_savings_us_p50: byId("constraintTtftEnabled").checked
        ? requiredInput("constraintTtft") * 1000
        : null,
    },
    objective: byId("recommendObjective").value || "balanced_headroom",
    write_miss_blocks: byId("recommendWriteMisses").checked,
    modeling_options: recommendationModelingOptions(),
  };
  if (supportsEncoderScenarios(state.config.models, payload.design_space.model_configs)) Object.assign(payload.workload_envelope, readEncoderScenarios(byId("recommendEncoderScenarios")));
  if (analysisMode === "cache_replay") {
    const capacityGiB = requiredInput("recommendCacheCapacityGiB", { min: 0 });
    payload.cache_replay_config = {
      source_path: state.config.defaults.cache_config,
      total_capacity_bytes: Math.round(capacityGiB * (1024 ** 3)),
      shard_count: 1,
      placement_seed: 0,
      warmup_fraction: 0,
    };
  }
  return payload;
}

function recommendationModelingOptions() {
  const decodeText = byId("recommendDecodeStep").value.trim();
  const decodeStep = decodeText === ""
    ? null
    : requiredInput("recommendDecodeStep", { positive: true });
  const statePolicies = readStatePolicyEditor(byId("recommendStatePolicyEditor"));
  return {
    footprint_view: byId("recommendFootprintView").value,
    state_policies: statePolicies.state_policies,
    model_state_policies: statePolicies.model_state_policies,
    recovery_assumption: byId("recommendRecoveryAssumption").value,
    checkpoint_interval_tokens: requiredInput("recommendCheckpointInterval", { integer: true, min: 1 }),
    retain_prompt_end: byId("recommendRetainPromptEnd").checked,
    sparse_access_policy: {
      restore_mode: byId("recommendSparseRestore").value,
      locality_scenario: byId("recommendSparseLocality").value,
      reuse_scope: byId("recommendSparseReuse").value,
      storage_layout: byId("recommendSparseLayout").value,
      query_granularity_tokens: requiredInput("recommendSparseQueryChunk", { integer: true, min: 1 }),
      seed: requiredInput("recommendSparseSeed", { integer: true }),
    },
    include_draft_cache: byId("recommendDraftCache").checked,
    draft_execution: byId("recommendDraftExecution").value,
    decode_state_writes: byId("recommendDecodeWrites").checked || decodeStep != null,
    decode_write_step_us: decodeStep,
  };
}

function applyRecommendationModelingOptions(options, state) {
  const sparse = options.sparse_access_policy || {};
  byId("recommendFootprintView").value = String(options.footprint_view || "prefix-store").replaceAll("_", "-");
  byId("recommendRecoveryAssumption").value = options.recovery_assumption || "model_default";
  byId("recommendCheckpointInterval").value = options.checkpoint_interval_tokens ?? 2048;
  byId("recommendRetainPromptEnd").checked = Boolean(options.retain_prompt_end);
  applyStatePolicyEditor(byId("recommendStatePolicyEditor"), options);
  byId("recommendSparseRestore").value = sparse.restore_mode || "full_restore";
  byId("recommendSparseLocality").value = sparse.locality_scenario || "window_global";
  byId("recommendSparseReuse").value = sparse.reuse_scope || "request_union";
  byId("recommendSparseLayout").value = sparse.storage_layout || "all_kv_heads_together";
  byId("recommendSparseQueryChunk").value = sparse.query_granularity_tokens ?? 1;
  byId("recommendSparseSeed").value = sparse.seed ?? 0;
  byId("recommendDraftCache").checked = Boolean(options.include_draft_cache);
  byId("recommendDraftExecution").value = options.draft_execution || "disabled";
  byId("recommendDecodeWrites").checked = Boolean(options.decode_state_writes);
  byId("recommendDecodeStep").value = options.decode_write_step_us ?? "";
  if (state) refreshRecommendationStatePolicyEditor(state);
}

function refreshRecommendationStatePolicyEditor(state) {
  byId("recommendEncoderScenarios").hidden = !supportsEncoderScenarios(state.config.models, [...byId("recommendModelList").querySelectorAll("input:checked")].map((input) => input.value));
  if (!state?.config || !byId("recommendStatePolicyEditor")) return;
  let seqLengths = [32768];
  try {
    seqLengths = parseNumbers(
      byId("recommendSeqLensInput").value,
      true,
      "Prompt length",
      "recommendSeqLensInput",
    );
  } catch (_error) { /* Keep the previous preview while the axis is incomplete. */ }
  renderStatePolicyEditor(byId("recommendStatePolicyEditor"), {
    models: state.config.models,
    selectedPaths: [...byId("recommendModelList").querySelectorAll("input:checked")].map((input) => input.value),
    seqLengths,
    prefix: "recommendPolicy",
    onChange: () => {
      renderConstraintDefaults(state);
      renderRecommendationLifecyclePreview(state);
      refreshRecommendationEstimate(state);
      updateRecommendationFreshness(state);
    },
  });
}

function renderRecommendationLifecyclePreview(state) {
  const container = byId("recommendLifecyclePreview");
  if (!container || byId("recommendAnalysisMode").value !== "cache_replay") {
    if (container) container.replaceChildren();
    return;
  }
  try {
    const normal = requiredInput("recommendRequestCountInput", { integer: true, min: 0 });
    const groups = requiredInput("recommendPrefixGroupCountInput", { integer: true, min: 1 });
    const hitRates = parseNumbers(byId("recommendHitRatesInput").value, false, "Prefix ratio", "recommendHitRatesInput");
    const seqLengths = parseNumbers(byId("recommendSeqLensInput").value, true, "Prompt length", "recommendSeqLensInput");
    const bootstrap = byId("recommendWorkloadShape").value === "prompt_lineage_bootstrap"
      && hitRates.some((value) => value > 0) ? groups : 0;
    const selected = [...byId("recommendModelList").querySelectorAll("input:checked")].map((input) => input.value);
    const estimate = estimateBootstrapState(byId("recommendStatePolicyEditor"), {
      models: state.config.models,
      selectedPaths: selected,
      seqLengths,
      prefixRatios: hitRates,
      prefixGroups: bootstrap,
      globalPolicy: {
        mode: byId("recommendRecoveryAssumption").value,
        interval: Number(byId("recommendCheckpointInterval").value) || 2048,
        retain: byId("recommendRetainPromptEnd").checked,
      },
    });
    container.innerHTML = `<strong>Cache lifecycle</strong><span>Bootstrap ${bootstrap.toLocaleString()}</span><span>Warmup 0</span><span>Measured ${normal.toLocaleString()}</span><span>Total ${(bootstrap + normal).toLocaleString()}</span>${bootstrap && estimate.bytes ? `<span>Max bootstrap state ${formatNumber(estimate.bytes / (1024 ** 3))} GiB · ${escapeText(estimate.model)}</span>` : ""}`;
  } catch (_error) {
    container.textContent = "Lifecycle preview is waiting for valid workload inputs.";
  }
}

function customConstraintValue(id) {
  return byId(`${id}Custom`).checked ? requiredInput(id, { min: 0 }) : null;
}

function selectedValues(containerId, label) {
  const values = [...byId(containerId).querySelectorAll("input:checked")].map((input) => input.value);
  if (!values.length) throw new Error(`Select at least one ${label}.`);
  return values;
}

function recommendationEvaluationFingerprint(payload) {
  if (!payload) return null;
  const evaluation = { ...payload };
  delete evaluation.objective;
  if (evaluation.cache_replay_config == null) delete evaluation.cache_replay_config;
  // The server pins selected config files in result identity. The browser form
  // has no digest controls, so exclude that evidence only from dirty-state
  // comparison while retaining it in API requests and saved experiments.
  delete evaluation.config_digests;
  return stableSerialize(evaluation);
}

function updateRecommendationFreshness(state) {
  if (!state.recommendation) return;
  try {
    state.isDirty = (
      recommendationEvaluationFingerprint(recommendationPayload(state))
      !== state.resultFingerprint
    );
  } catch (_error) {
    state.isDirty = true;
  }
  renderRecommendationDecisionBanner(state);
}

function recommendationEstimate(payload, config) {
  const maxSeq = Math.max(...payload.workload_envelope.seq_len);
  let computeDesigns = 0;
  let applicableDesigns = 0;
  for (const runtimePath of payload.design_space.runtime_configs) {
    const runtime = config.runtimes.find((item) => item.path === runtimePath);
    for (const modelPath of payload.design_space.model_configs) {
      const model = runtime?.model_compatibility?.[modelPath];
      if (!model?.compatible) continue;
      const constraints = model.constraints || {};
      if (constraints.max_context_tokens && maxSeq > constraints.max_context_tokens) continue;
      const multiple = Number(constraints.block_size_multiple || 1);
      const blocks = payload.design_space.block_size_tokens.filter((value) => value % multiple === 0);
      const designs = blocks.length * payload.design_space.max_io_bytes.length;
      computeDesigns += designs;
      const variants = supportsEncoderScenarios(config.models, [modelPath])
        ? (payload.workload_envelope.encoder_state?.length || 1) * (payload.workload_envelope.encoder_tail_policy?.length || 1) : 1;
      applicableDesigns += designs * variants;
    }
  }
  const scenarios = payload.workload_envelope.seq_len.length
    * payload.workload_envelope.prefix_ratio.length
    * payload.workload_envelope.arrival_model.length
    * payload.workload_envelope.qps.length;
  return {
    computeDesigns,
    computeCases: applicableDesigns * scenarios,
    applicableDesigns,
    candidates: computeDesigns * payload.design_space.storage_configs.length,
    scenarios,
  };
}

function recommendationPreflight(payload, config) {
  const constraints = payload.constraints;
  const storagePaths = payload.design_space.storage_configs;
  const storageIopsActive = constraints.use_storage_limits && storagePaths.some((path) => {
    const storage = config.storages.find((item) => item.path === path);
    return Number.isFinite(Number(storage?.thresholds?.random_read_iops));
  });
  const explicit = [
    constraints.max_read_avg_GBps,
    constraints.max_write_avg_GBps,
    constraints.max_aggregate_dependency_required_GBps,
    constraints.max_aggregate_dependency_chunked_iops,
    constraints.max_optimistic_overlap_GBps,
    constraints.max_dependency_aware_break_even_GBps,
    constraints.max_retain_all_capacity_GiB,
    constraints.min_dependency_ttft_savings_us_p50,
  ].some((value) => value != null);
  const storageConstraint = constraints.use_storage_limits && storagePaths.some((path) => {
    const thresholds = config.storages.find((item) => item.path === path)?.thresholds || {};
    return ["sequential_read_GBps", "sequential_write_GBps", "random_read_iops", "logical_capacity_GiB"]
      .some((key) => Number.isFinite(Number(thresholds[key])));
  });
  if (!explicit && !storageConstraint) throw new Error("Enable at least one recommendation constraint.");

  const iopsActive = constraints.max_aggregate_dependency_chunked_iops != null || storageIopsActive;
  const maxIo = payload.design_space.max_io_bytes;
  if (iopsActive && maxIo.every((value) => value == null)) {
    throw new Error("max IO bytes is required when a read IOPS constraint is active.");
  }
  return iopsActive && maxIo.some((value) => value == null)
    ? ["Candidates without max IO bytes will fail the read IOPS constraint."]
    : [];
}

function refreshRecommendationEstimate(state) {
  clearRecommendationFieldErrors();
  try {
    const payload = recommendationPayload(state);
    const estimate = recommendationEstimate(payload, state.config);
    const warnings = recommendationPreflight(payload, state.config);
    const limit = state.config.limits.max_recommendation_compute_cases || 1200;
    if (!estimate.computeCases) throw new Error("No compatible design remains.");
    byId("recommendCaseEstimate").textContent = `${estimate.computeCases.toLocaleString()} compute cases · ${estimate.candidates.toLocaleString()} candidates`;
    byId("recommendMath").textContent = (
      `${estimate.applicableDesigns.toLocaleString()} ${payload.workload_envelope.encoder_state ? "design / state combinations" : "compatible designs"} × `
      + `${estimate.scenarios.toLocaleString()} workload scenarios = `
      + `${estimate.computeCases.toLocaleString()} Demand Cases · `
      + `${payload.design_space.storage_configs.length.toLocaleString()} Storage profile`
      + `${payload.design_space.storage_configs.length === 1 ? "" : "s"} → `
      + `${estimate.candidates.toLocaleString()} candidates.`
    );
    byId("runRecommendationButton").disabled = Boolean(state.activeJobId) || estimate.computeCases > limit;
    setRecommendationStatus(
      state,
      estimate.computeCases > limit
        ? `Grid exceeds ${limit.toLocaleString()} actual Demand Cases.`
        : warnings[0] || "Ready · recommendation runs in the background.",
      estimate.computeCases > limit,
    );
  } catch (error) {
    markRecommendationFieldError(error);
    byId("recommendCaseEstimate").textContent = "Invalid design space";
    byId("recommendMath").textContent = `Grid unavailable · ${error.message}`;
    byId("runRecommendationButton").disabled = true;
    setRecommendationStatus(state, error.message, true);
  }
}

async function executeRecommendation(state) {
  let payload;
  try {
    payload = recommendationPayload(state);
    recommendationPreflight(payload, state.config);
    const estimate = recommendationEstimate(payload, state.config);
    if (estimate.computeCases > state.config.limits.max_recommendation_compute_cases) {
      throw new Error("Recommendation grid is above the compute-case limit.");
    }
  } catch (error) {
    markRecommendationFieldError(error, true);
    setRecommendationStatus(state, error.message, true);
    return;
  }
  setRecommendationBusy(state, true);
  setRecommendationStatus(state, "Starting recommendation job…");
  try {
    const job = await createRecommendationJob(payload);
    state.activeJobId = job.job_id;
    const request = job.request || payload;
    applyRecommendationPayloadToControls(state, request);
    storeRecommendationJob(job.job_id, request);
    await pollRecommendation(state, job.job_id);
  } catch (error) {
    clearRecommendationJob(state);
    setRecommendationBusy(state, false);
    const conflict = error.status === 409 && error.payload?.active_job_id;
    setRecommendationStatus(
      state,
      conflict
        ? `Compute engine busy with a different ${error.payload.active_job_kind || "compute"} job.`
        : `Recommendation failed: ${error.message}`,
      true,
    );
  }
}

async function pollRecommendation(state, jobId) {
  const token = ++state.pollToken;
  while (token === state.pollToken && state.activeJobId === jobId) {
    let job;
    try {
      job = await getRecommendationJob(jobId);
    } catch (error) {
      clearRecommendationJob(state);
      setRecommendationBusy(state, false);
      setRecommendationStatus(
        state,
        error.status === 404 ? "Recommendation job no longer exists after server restart." : error.message,
        true,
      );
      return;
    }
    const percent = job.total_cases ? job.completed_cases / job.total_cases * 100 : 0;
    setRecommendationStatus(
      state,
      `${job.status} · ${job.completed_cases.toLocaleString()} / ${job.total_cases.toLocaleString()} cases · ${formatNumber(percent)}%${job.cancel_requested ? " · cancel requested" : ""}`,
    );
    if (job.status === "completed") {
      state.recommendation = job.result;
      state.selectedCandidateId = job.result.recommended_candidate_id;
      state.resultFingerprint = recommendationEvaluationFingerprint(job.result.request);
      state.isDirty = false;
      clearRecommendationJob(state);
      setRecommendationBusy(state, false);
      renderRecommendation(state);
      setRecommendationStatus(state, `Completed · ranked ${job.result.candidate_count.toLocaleString()} candidates.`);
      window.dispatchEvent(new CustomEvent("dashboard-history-changed"));
      return;
    }
    if (job.status === "failed" || job.status === "canceled") {
      clearRecommendationJob(state);
      setRecommendationBusy(state, false);
      setRecommendationStatus(
        state,
        job.status === "canceled"
          ? state.recommendation
            ? "Recommendation canceled; previous result retained."
            : "Recommendation canceled; no result was produced."
          : `Recommendation failed: ${job.error || "unknown error"}`,
        job.status === "failed",
      );
      return;
    }
    await delay(POLL_INTERVAL_MS);
  }
}

async function cancelRecommendation(state) {
  if (!state.activeJobId) return;
  byId("cancelRecommendationButton").disabled = true;
  try {
    await cancelRecommendationJob(state.activeJobId);
    setRecommendationStatus(state, "Cancel requested · waiting for the current Case.");
  } catch (error) {
    setRecommendationStatus(state, `Cancel failed: ${error.message}`, true);
    byId("cancelRecommendationButton").disabled = false;
  }
}

async function restoreRecommendationJob(state) {
  const jobId = safeStorageGet(sessionStorage, ACTIVE_RECOMMENDATION_KEY);
  if (!jobId) return;
  const storedPayload = safeJsonParse(
    safeStorageGet(sessionStorage, ACTIVE_RECOMMENDATION_PAYLOAD_KEY),
  );
  try {
    const job = await getRecommendationJob(jobId);
    const request = job.request || storedPayload;
    if (request) applyRecommendationPayloadToControls(state, request);
    state.activeJobId = jobId;
    setRecommendationBusy(state, true);
    await pollRecommendation(state, jobId);
  } catch (error) {
    clearRecommendationJob(state);
    setRecommendationBusy(state, false);
    setRecommendationStatus(
      state,
      error.status === 404
        ? "Recommendation job no longer exists after server restart."
        : `Recommendation restore failed: ${error.message}`,
      true,
    );
  }
}

function storeRecommendationJob(jobId, payload) {
  safeStorageSet(sessionStorage, ACTIVE_RECOMMENDATION_KEY, jobId);
  safeStorageSet(sessionStorage, ACTIVE_RECOMMENDATION_PAYLOAD_KEY, JSON.stringify(payload));
}

function clearRecommendationJob(state) {
  state.activeJobId = null;
  try {
    sessionStorage.removeItem(ACTIVE_RECOMMENDATION_KEY);
    sessionStorage.removeItem(ACTIVE_RECOMMENDATION_PAYLOAD_KEY);
  } catch (_error) {
    // Current-page polling remains valid when session storage is unavailable.
  }
}

function setRecommendationBusy(state, busy) {
  byId("recommendControlsFieldset").disabled = busy;
  byId("runRecommendationButton").disabled = busy;
  byId("runRecommendationButton").textContent = busy ? "Evaluating…" : "Recommend a design";
  byId("cancelRecommendationButton").hidden = !busy;
  byId("cancelRecommendationButton").disabled = false;
  if (!busy) refreshRecommendationEstimate(state);
}

function setRecommendationStatus(_state, message, error = false) {
  const node = byId("recommendStatus");
  node.textContent = message;
  node.classList.toggle("error", error);
}

function rerankRecommendation(state, objective) {
  const result = state.recommendation;
  result.objective = objective;
  const evaluated = result.candidates.filter((candidate) => candidate.evaluation_status !== "not_evaluated");
  const unevaluated = result.candidates.filter((candidate) => candidate.evaluation_status === "not_evaluated");
  evaluated.sort((left, right) => compareCandidates(left, right, objective));
  evaluated.forEach((candidate, index) => { candidate.rank = index + 1; });
  unevaluated.forEach((candidate) => { candidate.rank = null; });
  result.candidates = [...evaluated, ...unevaluated];
  result.recommended_candidate_id = evaluated[0]?.candidate_id || null;
  state.selectedCandidateId = result.recommended_candidate_id;
  const recommended = evaluated[0];
  if (result.summary && recommended) {
    result.summary.recommended_candidate_id = recommended.candidate_id;
    result.summary.recommended_coverage_ratio = recommended.coverage_ratio;
    result.summary.recommended_worst_headroom = recommended.worst_headroom;
    result.summary.recommended_binding_constraint = recommended.binding_constraint;
  } else if (result.summary) {
    result.summary.recommended_candidate_id = null;
    result.summary.recommended_coverage_ratio = null;
    result.summary.recommended_worst_headroom = null;
    result.summary.recommended_binding_constraint = null;
  }
}

function compareCandidates(left, right, objective) {
  const full = (candidate) => candidate.fully_feasible ? 0 : 1;
  const desc = (value) => Number.isNaN(Number(value)) ? Infinity : -Number(value);
  const asc = (value) => Number.isNaN(Number(value)) ? Infinity : Number(value);
  let a;
  let b;
  if (objective === "max_coverage") {
    a = [desc(left.coverage_ratio), full(left), desc(left.worst_headroom), asc(left.mean_storage_pressure)];
    b = [desc(right.coverage_ratio), full(right), desc(right.worst_headroom), asc(right.mean_storage_pressure)];
  } else if (objective === "min_storage_pressure") {
    a = [full(left), desc(left.coverage_ratio), asc(left.mean_storage_pressure), desc(left.min_ttft_margin_us)];
    b = [full(right), desc(right.coverage_ratio), asc(right.mean_storage_pressure), desc(right.min_ttft_margin_us)];
  } else if (objective === "max_dependency_ttft_margin") {
    a = [full(left), desc(left.coverage_ratio), desc(left.min_ttft_margin_us), desc(left.worst_headroom)];
    b = [full(right), desc(right.coverage_ratio), desc(right.min_ttft_margin_us), desc(right.worst_headroom)];
  } else {
    a = [full(left), desc(left.coverage_ratio), desc(left.worst_headroom), desc(left.min_ttft_margin_us)];
    b = [full(right), desc(right.coverage_ratio), desc(right.worst_headroom), desc(right.min_ttft_margin_us)];
  }
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return left.candidate_id.localeCompare(right.candidate_id);
}

function renderRecommendation(state) {
  const result = state.recommendation;
  if (!result) return;
  byId("recommendEmpty").hidden = true;
  byId("recommendResult").hidden = false;
  const selected = selectedCandidate(state) || result.candidates[0];
  if (!selected) return;
  state.selectedCandidateId = selected.candidate_id;
  const design = selected.design;
  const partial = !selected.fully_feasible;
  const notEvaluated = selected.evaluation_status === "not_evaluated";
  const noFullyFeasible = Number(result.fully_feasible_candidate_count || 0) === 0;
  const hero = document.querySelector(".recommend-hero");
  hero.dataset.feasibility = partial ? "partial" : "feasible";
  byId("recommendedDesignLabel").textContent = notEvaluated ? "Not evaluated" : noFullyFeasible
    ? "Best partial candidate"
    : partial ? "Selected partial candidate" : "Recommended design";
  byId("openRecommendationInExplore").textContent = partial
    ? "Inspect partial candidate in Explore"
    : "Open in Explore";
  byId("recommendedDesign").textContent = `${design.model_name || configName(design.model_config)} · ${configName(design.runtime_config)}`;
  byId("recommendedContext").textContent = `${design.storage_config ? configName(design.storage_config) : "storage"} · block ${design.block_size_tokens} · max IO ${design.max_io_bytes ?? "None"} · objective ${result.objective}${selected.heterogeneous_state_policy ? " · Heterogeneous checkpoint policy" : ""}`;
  const metrics = [
    ["Coverage", notEvaluated ? "Not evaluated" : `${formatNumber(selected.coverage_ratio * 100)}%`],
    ["Worst resource headroom", notEvaluated ? "Not evaluated" : formatRatio(selected.worst_headroom)],
    ["Binding resource", selected.binding_resource_constraint || selected.binding_constraint || "None"],
    ["Worst TTFT margin", notEvaluated ? "Not evaluated" : formatTtftMargin(selected.min_ttft_margin_us)],
  ];
  byId("recommendMetricGrid").innerHTML = metrics.map(([label, value]) => (
    `<article class="recommend-metric"><span>${escapeText(label)}</span><strong title="${escapeText(value)}">${escapeText(value)}</strong></article>`
  )).join("");
  const warning = byId("recommendHistoryWarning");
  warning.hidden = !result.history_error;
  warning.textContent = result.history_error || "";
  renderRecommendationDecisionBanner(state);
  renderRecommendationEvidence(state, selected);
  renderCandidateTable(state);
  renderBoundaries(state);
}

function renderRecommendationEvidence(state, candidate) {
  const container = byId("recommendWhyEvidence");
  const resource = candidate.binding_resource_evidence;
  const ttft = candidate.worst_ttft_evidence;
  const objective = state.recommendation?.objective || "balanced_headroom";
  const cards = [];
  if (resource) {
    const unit = constraintUnit(resource.constraint);
    const utilization = Number(resource.utilization);
    cards.push(`
      <article>
        <span>Binding resource</span>
        <strong>${escapeText(resource.constraint || candidate.binding_resource_constraint || "Resource limit")}</strong>
        <p>${escapeText(formatEvidenceValue(resource.value, unit))} demand ÷ ${escapeText(formatEvidenceValue(resource.limit, unit))} limit = ${escapeText(Number.isFinite(utilization) ? formatNumber(utilization * 100) + "%" : "unbounded pressure")}.</p>
        <small>${escapeText(formatRatio(resource.headroom))} headroom · ${escapeText(formatScenario(resource.scenario))}</small>
      </article>`);
  } else {
    cards.push(`
      <article>
        <span>Binding resource</span>
        <strong>${escapeText(candidate.binding_resource_constraint || candidate.binding_constraint || "No active resource constraint")}</strong>
        <p>${candidate.binding_resource_constraint || candidate.binding_constraint ? "Exact value/limit evidence is unavailable in this older saved result." : "This candidate was ranked without an active Storage resource limit."}</p>
      </article>`);
  }
  if (ttft) {
    cards.push(`
      <article>
        <span>Worst TTFT boundary</span>
        <strong>${escapeText(formatMillisecondsFromMicroseconds(ttft.margin))} margin</strong>
        <p>${escapeText(formatMillisecondsFromMicroseconds(ttft.value))} hidden budget − ${escapeText(formatMillisecondsFromMicroseconds(ttft.limit))} required minimum.</p>
        <small>${escapeText(formatScenario(ttft.scenario))}</small>
      </article>`);
  } else {
    cards.push(`
      <article>
        <span>Worst TTFT boundary</span>
        <strong>${escapeText(formatTtftMargin(candidate.min_ttft_margin_us))}</strong>
        <p>${candidate.worst_ttft_scenario ? "Exact value/limit evidence is unavailable in this older saved result." : "The TTFT constraint is disabled for this recommendation."}</p>
        <small>${escapeText(formatScenario(candidate.worst_ttft_scenario))}</small>
      </article>`);
  }
  cards.push(`
    <article>
      <span>Ranking objective</span>
      <strong>${escapeText(objective)}</strong>
      <p>${escapeText(OBJECTIVE_EXPLANATIONS[objective] || "Rank candidates using the selected objective.")}</p>
      <small>${candidate.design.encoder_scenarios_supported ? "Common prefix intervals require every selected state combination to pass; no values are interpolated. Rated-bandwidth TTFT uses one shared read channel without queueing." : "Coverage uses exact sampled scenarios; no feasibility values are interpolated."}</small>
    </article>`);
  container.innerHTML = cards.join("");
}

function constraintUnit(key) {
  return RESOURCE_CONSTRAINTS.find((item) => item.key === key)?.unit || "";
}

function formatEvidenceValue(value, unit) {
  if (value === null || value === undefined || value === "") return "Missing";
  const number = Number(value);
  if (number === Infinity) return `∞${unit ? ` ${unit}` : ""}`;
  if (number === -Infinity) return `−∞${unit ? ` ${unit}` : ""}`;
  return Number.isFinite(number) ? formatConstraintValue(number, unit) : "Missing";
}

function renderRecommendationDecisionBanner(state) {
  const banner = byId("recommendDecisionBanner");
  const result = state.recommendation;
  if (!result) {
    banner.hidden = true;
    banner.textContent = "";
    return;
  }
  const selected = selectedCandidate(state) || result.candidates?.[0];
  const messages = [];
  if (state.isDirty) {
    messages.push("Showing previous recommendation; controls changed. Rerun to refresh the decision.");
  }
  if (selected?.evaluation_status === "not_evaluated") {
    messages.push("This design contains an unsupported or unevaluated case and is excluded from performance ranking.");
  } else if (selected && !selected.fully_feasible) {
    const failed = selected.scenario_count - selected.feasible_scenario_count;
    const failures = selected.constraint_failures
      .map((item) => `${item.constraint} ×${item.count}`)
      .join(", ");
    messages.push(
      `${formatNumber(selected.coverage_ratio * 100)}% sampled coverage; `
      + `${failed} of ${selected.scenario_count} scenarios fail`
      + `${failures ? ` (${failures})` : ""}.`,
    );
  }
  for (const message of result.warnings || []) {
    if (!messages.includes(message)) messages.push(message);
  }
  banner.hidden = !messages.length;
  banner.innerHTML = messages.map((message, index) => (
    `${index === 0 ? "<strong>Decision notice.</strong> " : ""}${escapeText(message)}`
  )).join("<br>");
}

function renderCandidateTable(state) {
  const rows = state.recommendation.candidates.slice(0, 5);
  byId("recommendCandidateTable").innerHTML = `
    <table>
      <thead><tr><th>Rank</th><th>Design</th><th>Coverage / status</th><th>Headroom</th><th>Binding / failures</th><th></th></tr></thead>
      <tbody>${rows.map((candidate) => {
        const design = candidate.design;
        const failures = candidate.constraint_failures.map((item) => (
          `${item.constraint} ×${item.count}${item.missing_count ? ` (${item.missing_count} missing)` : ""}`
        )).join(", ");
        const selected = candidate.candidate_id === state.selectedCandidateId;
        const label = design.model_name || configName(design.model_config);
        return `<tr data-candidate-id="${escapeText(candidate.candidate_id)}" data-selected="${selected}" aria-selected="${selected}">
          <td>${candidate.rank ?? "—"}</td>
          <td>${escapeText(label)}<br><small>${escapeText(configName(design.runtime_config))} · ${escapeText(configName(design.storage_config))}</small></td>
          <td><span class="feasible-chip ${candidate.fully_feasible ? "" : "partial"}">${candidate.evaluation_status === "not_evaluated" ? "—" : `${formatNumber(candidate.coverage_ratio * 100)}%`}</span><span class="candidate-status">${candidate.evaluation_status === "not_evaluated" ? "Not evaluated" : candidate.fully_feasible ? "Fully feasible" : "Partial"}</span></td>
          <td>${candidate.evaluation_status === "not_evaluated" ? "—" : escapeText(formatRatio(candidate.worst_headroom))}</td>
          <td>${escapeText(failures || candidate.binding_resource_constraint || candidate.binding_constraint || "None")}</td>
          <td><button type="button" aria-label="Select ${escapeText(label)} candidate" aria-pressed="${selected}" data-select-candidate="${escapeText(candidate.candidate_id)}">Select</button></td>
        </tr>`;
      }).join("")}</tbody>
    </table>`;
  byId("recommendCandidateTable").querySelectorAll("[data-select-candidate]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedCandidateId = button.dataset.selectCandidate;
      renderRecommendation(state);
    });
  });
}

function renderBoundaries(state) {
  const candidate = selectedCandidate(state);
  const boundaries = candidate?.sampled_hit_rate_boundaries || [];
  byId("recommendBoundaryCaption").textContent = boundaries.some((item) => item.non_monotonic)
    ? "Non-monotonic feasibility detected; multiple exact sampled intervals are shown."
    : "Intervals join adjacent feasible samples only; no values are interpolated.";
  byId("recommendBoundaries").innerHTML = boundaries.length ? boundaries.map((boundary) => {
    const intervals = boundary.intervals.length
      ? boundary.intervals.map((item) => item.start === item.end ? formatNumber(item.start) : `${formatNumber(item.start)}–${formatNumber(item.end)}`).join(", ")
      : "No feasible sampled hit rate";
    return `<div class="boundary-row"><strong>seq ${formatCompact(boundary.seq_len)} · ${formatNumber(boundary.qps)} QPS · ${escapeText(boundary.arrival_model)}</strong><span>${escapeText(intervals)}${boundary.non_monotonic ? " · non-monotonic" : ""}</span></div>`;
  }).join("") : '<div class="empty-state">No sampled boundary evidence.</div>';
}

function selectedCandidate(state) {
  return state.recommendation?.candidates.find(
    (candidate) => candidate.candidate_id === state.selectedCandidateId,
  );
}

function openSelectedCandidate(state) {
  const candidate = selectedCandidate(state);
  const recommendation = state.recommendation;
  if (!candidate || !recommendation) return;
  const request = recommendation.request;
  const qps = candidate.worst_resource_scenario?.qps
    ?? candidate.worst_ttft_scenario?.qps
    ?? candidate.worst_scenario?.qps
    ?? request.workload_envelope.qps[0];
  const rowIds = new Set(candidate.row_ids);
  const rows = recommendation.rows.filter(
    (row) => rowIds.has(row.id) && Number(row.qps) === Number(qps),
  );
  const payload = {
    model_configs: [candidate.design.model_config],
    runtime_config: candidate.design.runtime_config,
    workload_config: request.workload_envelope.workload_config,
    storage_config: candidate.design.storage_config,
    axes: {
      seq_len: request.workload_envelope.seq_len,
      prefix_ratio: request.workload_envelope.prefix_ratio,
      block_size_tokens: [candidate.design.block_size_tokens],
      arrival_model: request.workload_envelope.arrival_model,
      max_io_bytes: [candidate.design.max_io_bytes],
    },
    workload_overrides: { ...request.workload_overrides, qps },
    write_miss_blocks: request.write_miss_blocks,
    modeling_options: request.modeling_options,
  };
  if (candidate.design.encoder_scenarios_supported) {
    for (const key of ["encoder_state", "encoder_tail_policy"]) payload.axes[key] = request.workload_envelope[key];
  }
  state.onOpenExplore({
    payload,
    result: {
      case_count: rows.length,
      rows,
      summary: {},
      warnings: recommendation.warnings || [],
      request: payload,
      experiment_id: recommendation.experiment_id,
      decision_context: {
        source_kind: "recommendation",
        candidate_id: candidate.candidate_id,
        fully_feasible: candidate.fully_feasible,
        coverage_ratio: candidate.coverage_ratio,
        feasible_scenario_count: candidate.feasible_scenario_count,
        scenario_count: candidate.scenario_count,
        constraint_failures: candidate.constraint_failures,
        constraints: request.constraints,
        has_row_evaluations: rows.every((row) => Boolean(row.recommendation_evaluation)),
      },
    },
  });
  switchWorkspace("explore");
}

function wireExperiments(state) {
  byId("refreshExperimentsButton").addEventListener("click", () => refreshExperiments(state));
  byId("experimentKindFilter").addEventListener("change", () => refreshExperiments(state));
  byId("experimentPinnedFilter").addEventListener("change", () => refreshExperiments(state));
  byId("selectAllExperiments").addEventListener("change", (event) => {
    const visibleIds = state.experiments.map((item) => item.id);
    if (event.target.checked) {
      visibleIds.forEach((id) => state.selectedExperimentIds.add(id));
    } else {
      visibleIds.forEach((id) => state.selectedExperimentIds.delete(id));
    }
    renderExperiments(state);
  });
  byId("deleteSelectedExperiments").addEventListener(
    "click",
    () => removeSelectedExperiments(state),
  );
  let searchTimer = null;
  byId("experimentSearch").addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => refreshExperiments(state), 180);
  });
  byId("compareExperimentsButton").addEventListener("click", () => compareSelected(state));
  window.addEventListener("dashboard-experiments-visible", () => refreshExperiments(state));
}

async function refreshExperiments(state) {
  try {
    const response = await getExperiments({
      kind: byId("experimentKindFilter").value,
      search: byId("experimentSearch").value.trim(),
      pinned: byId("experimentPinnedFilter").checked ? "true" : "",
    });
    state.experiments = response.experiments || [];
    const visibleIds = new Set(state.experiments.map((item) => item.id));
    state.selectedExperimentIds = new Set(
      [...state.selectedExperimentIds].filter((id) => visibleIds.has(id)),
    );
    renderExperiments(state);
  } catch (error) {
    byId("experimentListCaption").textContent = `History unavailable: ${error.message}`;
  }
}

function renderExperiments(state) {
  byId("experimentListCaption").textContent = `${state.experiments.length.toLocaleString()} saved runs · pinned runs do not count toward retention`;
  const list = byId("experimentList");
  if (!state.experiments.length) {
    list.innerHTML = '<div class="empty-state">No matching experiments yet.</div>';
  } else {
    list.innerHTML = state.experiments.map((item) => `
      <article class="experiment-row ${item.pinned ? "is-pinned" : ""} ${state.selectedExperimentIds.has(item.id) ? "is-selected" : ""}" data-experiment-id="${escapeText(item.id)}">
        <label class="experiment-select">
          <input type="checkbox" data-select-experiment="${escapeText(item.id)}" aria-label="Select ${escapeText(item.name)}" ${state.selectedExperimentIds.has(item.id) ? "checked" : ""}>
        </label>
        <span class="kind-chip">${escapeText(item.kind)}</span>
        <div><h3 title="${escapeText(item.name)}">${escapeText(item.name)}</h3><p>${escapeText(new Date(item.created_at * 1000).toLocaleString())} · ${Number(item.case_count).toLocaleString()} cases · ${escapeText(item.fingerprint.slice(0, 10))}</p></div>
        <div class="experiment-actions">
          <button type="button" data-action="open">Open</button>
          <button type="button" data-action="pin">${item.pinned ? "Unpin" : "Pin"}</button>
          <button type="button" data-action="rename">Rename</button>
          <button type="button" data-action="delete">Delete</button>
        </div>
      </article>`).join("");
    list.querySelectorAll(".experiment-row").forEach((row) => {
      const id = row.dataset.experimentId;
      row.querySelector("[data-select-experiment]").addEventListener("change", (event) => {
        if (event.target.checked) state.selectedExperimentIds.add(id);
        else state.selectedExperimentIds.delete(id);
        row.classList.toggle("is-selected", event.target.checked);
        syncExperimentSelectionControls(state);
      });
      row.querySelector('[data-action="open"]').addEventListener("click", () => openExperiment(state, id));
      row.querySelector('[data-action="pin"]').addEventListener("click", () => togglePin(state, id));
      row.querySelector('[data-action="rename"]').addEventListener("click", () => renameExperiment(state, id));
      row.querySelector('[data-action="delete"]').addEventListener("click", () => removeExperiment(state, id));
    });
  }
  syncExperimentSelectionControls(state);
  populateExperimentSelect(byId("compareBaseline"), state.experiments, 1);
  populateExperimentSelect(byId("compareCandidate"), state.experiments, 0);
}

function syncExperimentSelectionControls(state) {
  const visibleIds = state.experiments.map((item) => item.id);
  const selectedCount = visibleIds.filter((id) => state.selectedExperimentIds.has(id)).length;
  const selectAll = byId("selectAllExperiments");
  selectAll.disabled = visibleIds.length === 0;
  selectAll.checked = visibleIds.length > 0 && selectedCount === visibleIds.length;
  selectAll.indeterminate = selectedCount > 0 && selectedCount < visibleIds.length;
  byId("experimentSelectionCount").textContent = `${selectedCount.toLocaleString()} selected`;
  const deleteButton = byId("deleteSelectedExperiments");
  deleteButton.disabled = selectedCount === 0;
  deleteButton.textContent = selectedCount
    ? `Delete selected (${selectedCount.toLocaleString()})`
    : "Delete selected";
}

function populateExperimentSelect(select, experiments, preferredIndex) {
  const previous = select.value;
  select.replaceChildren();
  for (const item of experiments) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.pinned ? "★ " : ""}${item.name}`;
    select.append(option);
  }
  if (experiments.some((item) => item.id === previous)) select.value = previous;
  else if (experiments[preferredIndex]) select.value = experiments[preferredIndex].id;
}

async function openExperiment(state, id) {
  try {
    const experiment = await getExperiment(id);
    const storedResult = {
      ...experiment.result,
      experiment_id: experiment.id,
    };
    if (experiment.kind === "recommendation") {
      state.recommendation = storedResult;
      state.selectedCandidateId = storedResult.recommended_candidate_id;
      applyRecommendationPayloadToControls(state, storedResult.request);
      state.resultFingerprint = recommendationEvaluationFingerprint(storedResult.request);
      state.isDirty = false;
      renderRecommendation(state);
      openSelectedCandidate(state);
      return;
    }
    state.onOpenExplore({ payload: experiment.request, result: storedResult });
    switchWorkspace("explore");
  } catch (error) {
    byId("experimentListCaption").textContent = `Open failed: ${error.message}`;
  }
}

async function togglePin(state, id) {
  const item = state.experiments.find((experiment) => experiment.id === id);
  if (!item) return;
  await mutateExperiment(state, id, { pinned: !item.pinned });
}

async function renameExperiment(state, id) {
  const item = state.experiments.find((experiment) => experiment.id === id);
  const name = window.prompt("Experiment name", item?.name || "");
  if (name == null) return;
  await mutateExperiment(state, id, { name });
}

async function mutateExperiment(state, id, patch) {
  try {
    await updateExperiment(id, patch);
    await refreshExperiments(state);
  } catch (error) {
    byId("experimentListCaption").textContent = `Update failed: ${error.message}`;
  }
}

async function removeExperiment(state, id) {
  const item = state.experiments.find((experiment) => experiment.id === id);
  if (!window.confirm(`Delete “${item?.name || id}”? This cannot be undone.`)) return;
  try {
    await deleteExperiment(id);
    state.selectedExperimentIds.delete(id);
    await refreshExperiments(state);
  } catch (error) {
    byId("experimentListCaption").textContent = `Delete failed: ${error.message}`;
  }
}

async function removeSelectedExperiments(state) {
  const visibleIds = new Set(state.experiments.map((item) => item.id));
  const ids = [...state.selectedExperimentIds].filter((id) => visibleIds.has(id));
  if (!ids.length) return;
  const noun = ids.length === 1 ? "run" : "runs";
  if (!window.confirm(`Delete ${ids.length} selected ${noun}? This cannot be undone.`)) return;

  const button = byId("deleteSelectedExperiments");
  button.disabled = true;
  button.textContent = `Deleting ${ids.length.toLocaleString()}…`;
  let deletedCount = 0;
  const failures = [];
  for (const id of ids) {
    try {
      await deleteExperiment(id);
      state.selectedExperimentIds.delete(id);
      deletedCount += 1;
    } catch (error) {
      failures.push(error.message);
    }
  }
  await refreshExperiments(state);
  byId("experimentListCaption").textContent = failures.length
    ? `${deletedCount.toLocaleString()} deleted · ${failures.length.toLocaleString()} failed.`
    : `${deletedCount.toLocaleString()} selected ${noun} deleted.`;
}

async function compareSelected(_state) {
  const baseline = byId("compareBaseline").value;
  const candidate = byId("compareCandidate").value;
  if (!baseline || !candidate) return;
  try {
    const result = await compareExperiments(baseline, candidate);
    renderComparison(result);
  } catch (error) {
    byId("compareResult").innerHTML = `<div class="result-state-banner">${escapeText(error.message)}</div>`;
  }
}

function renderComparison(result) {
  const stats = [
    ["Exact matches", result.matched_cases],
    ["Added", result.added_cases],
    ["Removed", result.removed_cases],
  ];
  const summaryChanges = Object.entries(result.summary_delta || {}).filter(([, delta]) => (
    delta.baseline_state !== delta.candidate_state || delta.absolute !== 0
  ));
  byId("compareResult").innerHTML = `
    <div class="compare-summary">${stats.map(([label, value]) => `<div class="compare-stat"><span>${label}</span><strong>${Number(value).toLocaleString()}</strong></div>`).join("")}</div>
    <div class="diff-list">
      <h3>Request changes · ${result.request_diff.length}</h3>
      ${result.request_diff.slice(0, 30).map((item) => `<div class="diff-row"><span>${escapeText(item.path)}</span><span>${escapeText(formatAny(item.baseline))}</span><span>→ ${escapeText(formatAny(item.candidate))}</span></div>`).join("") || '<div class="empty-state">Requests are identical.</div>'}
      <h3>Summary changes · ${summaryChanges.length}</h3>
      ${summaryChanges.slice(0, 30).map(([key, delta]) => `<div class="diff-row"><span>${escapeText(key)}</span><span>${escapeText(formatAny(delta.baseline))}</span><span>→ ${escapeText(formatAny(delta.candidate))}${delta.percent == null ? "" : ` (${formatNumber(delta.percent)}%)`}</span></div>`).join("") || '<div class="empty-state">No summary change.</div>'}
    </div>`;
}

function populateSelect(select, options, selected) {
  select.replaceChildren();
  for (const item of options) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = item.label;
    option.disabled = Boolean(item.error);
    option.selected = item.path === selected;
    select.append(option);
  }
}

function parseNumbers(text, integer, label, fieldId = null) {
  const tokens = String(text).split(",").map((value) => value.trim()).filter(Boolean);
  const values = tokens.map((value) => Number(value));
  if (!values.length || values.some((value) => !Number.isFinite(value))) {
    throw validationError(
      `${label} must contain comma-separated ${integer ? "integers" : "numbers"}.`,
      fieldId,
    );
  }
  if (integer && values.some((value) => !Number.isInteger(value))) {
    throw validationError(`${label} must contain integers.`, fieldId);
  }
  if (values.some((value) => value <= 0) && !label.toLowerCase().includes("hit")) {
    throw validationError(`${label} values must be positive.`, fieldId);
  }
  if (
    label.toLowerCase().includes("hit")
    && values.some((value) => value < 0 || value > 1)
  ) {
    throw validationError("Prefix ratios must stay within [0, 1].", fieldId);
  }
  return [...new Set(values)].sort((a, b) => a - b);
}

function parseOptionalIntegers(text) {
  if (!String(text).trim()) return [null];
  const values = String(text).split(",").map((value) => value.trim()).filter(Boolean).map((value) => {
    if (["none", "null"].includes(value.toLowerCase())) return null;
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
      throw new Error("max IO bytes must contain positive integers or None.");
    }
    return parsed;
  });
  if (!values.length) return [null];
  return [...new Set(values)].sort((left, right) => (
    left == null ? -1 : right == null ? 1 : left - right
  ));
}

function requiredInput(id, { integer = false, min = null } = {}) {
  const text = byId(id).value.trim();
  const label = RECOMMEND_FIELD_LABELS[id] || id;
  if (!text) throw validationError(`${label} is required.`, id);
  const value = Number(text);
  if (!Number.isFinite(value)) throw validationError(`${label} must be numeric.`, id);
  if (integer && !Number.isInteger(value)) {
    throw validationError(`${label} must be an integer.`, id);
  }
  if (min != null && value < min) {
    throw validationError(`${label} must be at least ${min}.`, id);
  }
  return value;
}

function validationError(message, fieldId = null) {
  const error = new Error(message);
  error.fieldId = fieldId;
  return error;
}

function clearRecommendationFieldErrors() {
  byId("recommendView").querySelectorAll('[aria-invalid="true"]').forEach((input) => {
    input.removeAttribute("aria-invalid");
  });
}

function markRecommendationFieldError(error, focus = false) {
  if (!error?.fieldId) return;
  const input = byId(error.fieldId);
  if (!input) return;
  input.setAttribute("aria-invalid", "true");
  if (focus) input.focus();
}

function optionalInput(id, nonNegative = false) {
  const text = byId(id).value.trim();
  if (!text) return null;
  const value = Number(text);
  if (!Number.isFinite(value)) throw new Error(`${id} must be numeric or blank.`);
  if (nonNegative && value < 0) throw new Error(`${id} must be non-negative.`);
  return value;
}

function formatScenario(value) {
  if (!value) return "None";
  return `seq ${formatCompact(value.seq_len)} · hit ${formatNumber(Number(value.prefix_ratio) * 100)}% · ${formatNumber(value.qps)} QPS · ${value.arrival_model || "unknown"}${value.encoder_state == null ? "" : ` · Encoder ${value.encoder_state} · output ${value.encoder_tail_policy}`}`;
}

function formatRatio(value) {
  if (value === null || value === undefined || value === "") return "Missing";
  const number = Number(value);
  if (number === Infinity) return "Unbounded";
  if (number === -Infinity) return "−∞";
  if (!Number.isFinite(number)) return "Missing";
  return `${formatNumber(number * 100)}%`;
}

function formatTtftMargin(value) {
  return formatMillisecondsFromMicroseconds(value);
}

function formatMillisecondsFromMicroseconds(value) {
  if (value === null || value === undefined || value === "") return "Missing";
  const number = Number(value);
  if (number === Infinity) return "Unbounded";
  if (number === -Infinity) return "−∞ ms";
  return Number.isFinite(number) ? `${formatNumber(number / 1000)} ms` : "Missing";
}

function formatNumber(value) {
  const number = Number(value);
  if (number === Infinity) return "∞";
  if (number === -Infinity) return "−∞";
  if (!Number.isFinite(number)) return "Missing";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(number);
}

function formatCompact(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(number)
    : String(value);
}

function formatAny(value) {
  if (value === null || value === undefined) return "Missing";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function configName(path) {
  return String(path || "unknown").split("/").at(-1)?.replace(/\.(yaml|yml|json)$/i, "") || "unknown";
}

function escapeText(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function safeStorageGet(storage, key) {
  try { return storage.getItem(key); } catch (_error) { return null; }
}

function safeStorageSet(storage, key, value) {
  try { storage.setItem(key, value); } catch (_error) { /* local-only convenience */ }
}

function safeJsonParse(value) {
  try { return value ? JSON.parse(value) : null; } catch (_error) { return null; }
}

function stableSerialize(value) {
  if (Array.isArray(value)) return `[${value.map(stableSerialize).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableSerialize(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function byId(id) {
  return document.getElementById(id);
}
