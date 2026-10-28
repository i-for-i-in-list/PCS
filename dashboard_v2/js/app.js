import {
  cancelSweepJob, createSweepJob, downloadCsv, downloadJson, getCase,
  getConfigs, getSweepJob, runSweep,
} from "./api.js";
import { renderCaseDetail } from "./case_detail.js";
import { renderCachePanel } from "./cache_panel.js";
import { renderCurves } from "./curves.js";
import { initializeDecisionWorkbench } from "./decision_workbench.js";
import { renderHeatmap } from "./heatmap.js";
import { initializeKvSize } from "./kv_size.js";
import {
  initializeEncoderScenarios, applyEncoderScenarios, readEncoderScenarios, supportsEncoderScenarios, scenarioCaseCount,
  applyStatePolicyEditor,
  estimateBootstrapState,
  readStatePolicyEditor,
  renderStatePolicyEditor,
} from "./state_policies.js";
import {
  METRICS, classifyValue, formatCompact, formatMetric, formatMetricForRow, formatNumber,
  riskCounts, seriesKey, storageThreshold, summarizeRows,
  METRIC_GROUPS, PEAK_MEASURES, peakMetricKey, metricUnavailableReason, hasCacheRow, axisValue, axisLabel, scaleLabel,
} from "./metrics.js";
import { createTableState, ensureSelectedPage, renderResultsTable } from "./table.js";

const state = {
  config: null,
  computeConfig: null,
  workloadConfig: null,
  workloadOverrides: {},
  sweep: null,
  rows: [],
  activeMetric: "optimistic_overlap_GBps_p99",
  activeMetricGroup: null,
  groupMetrics: Object.fromEntries(Object.entries(METRIC_GROUPS).map(([key, group]) => [key, group.defaultMetric])),
  peakMeasure: "read_GBps",
  peakWindow: "1ms",
  viewMode: "curves",
  xMode: "hit-linear",
  curveXMode: "hit-linear",
  heatmapXMode: "hit-linear",
  axisNotice: "",
  heatmapScale: "global",
  thresholdsEnabled: false,
  smoothGuides: false,
  hiddenSeries: new Set(),
  selectedRow: null,
  caseDetail: null,
  detailTab: "overview",
  heatmapSlice: null,
  table: createTableState(),
  requestToken: 0,
  activePreset: "quick_compare",
  resultFingerprint: null,
  isDirty: false,
  selectedModelsByRuntime: new Map(),
  activeJobId: null,
  jobPollToken: 0,
  busy: false,
  exploreMode: "simple",
  resultView: "charts",
};

const ACTIVE_JOB_KEY = "dashboard-v2-active-sweep-job";
const ACTIVE_JOB_PAYLOAD_KEY = "dashboard-v2-active-sweep-payload";
const EXPLORE_MODE_KEY = "dashboard-v3-explore-mode";
const JOB_POLL_INTERVAL_MS = 750;
let decisionWorkbench = null;

document.addEventListener("DOMContentLoaded", init);

async function init() {
  wireEvents();
  setExploreMode(readExploreMode(), { persist: false, reformat: false });
  renderSummary();
  renderTable();
  renderChart();
  try {
    setStatus("Loading local configurations…");
    state.config = await getConfigs();
    populateConfigControls();
    applyPreset("quick_compare");
    decisionWorkbench = initializeDecisionWorkbench({
      config: state.config,
      onOpenExplore: ({ payload, result }) => {
        applyPayloadToControls(payload);
        acceptSweep(result, payload);
      },
    });
    initializeKvSize({ config: state.config });
    setStatus("Ready · Quick Compare is configured for 20 cases.");
    await restoreActiveJob();
  } catch (error) {
    setStatus(`Configuration error: ${error.message}`, true);
  }
}

function wireEvents() {
  window.addEventListener("dashboard-theme-changed", refreshChartTheme);
  initializeEncoderScenarios(byId("encoderScenarios"), handleDataChange);
  byId("simpleModeButton").addEventListener("click", () => setExploreMode("simple"));
  byId("expertModeButton").addEventListener("click", () => setExploreMode("expert"));
  byId("openModelDialogButton").addEventListener("click", () => {
    renderModelList();
    byId("modelDialog").showModal();
    byId("modelSearchInput").focus();
  });
  for (const id of ["closeModelDialogButton", "modelDialogDoneButton"]) {
    byId(id).addEventListener("click", () => byId("modelDialog").close());
  }
  byId("modelDialog").addEventListener("click", (event) => {
    if (event.target === byId("modelDialog")) byId("modelDialog").close();
  });
  byId("preflightKeepDefault").addEventListener("click", () => {
    byId("recoveryAssumptionSelect").value = "model_default";
    setStatus("Keeping model-default checkpoint behavior. The warning remains part of this experiment's evidence.");
    handleDataChange();
  });
  byId("preflightPeriodic").addEventListener("click", () => {
    byId("recoveryAssumptionSelect").value = "periodic";
    byId("checkpointIntervalInput").value = "2048";
    handleDataChange();
  });
  byId("preflightExact").addEventListener("click", () => {
    byId("recoveryAssumptionSelect").value = "assume_exact_at_resident";
    handleDataChange();
  });
  byId("preflightAddFullHit").addEventListener("click", addFullHitControlPoint);
  byId("runButton").addEventListener("click", executeSweep);
  byId("cancelSweepButton").addEventListener("click", cancelActiveSweep);
  byId("exportJsonButton").addEventListener("click", () => downloadJson("dashboard-v2-sweep.json", state.sweep));
  byId("exportCsvButton").addEventListener("click", () => downloadCsv("dashboard-v2-sweep.csv", state.rows));
  byId("quickPresetButton").addEventListener("click", () => applyPreset("quick_compare"));
  byId("fullPresetButton").addEventListener("click", () => applyPreset("full_tail"));
  byId("analysisModeSelect").addEventListener("change", () => {
    applyAnalysisMode();
    handleDataChange();
  });
  byId("selectAllModelsButton").addEventListener("click", selectAllModels);
  byId("modelSearchInput").addEventListener("input", renderModelList);
  for (const id of ["modelFamilyFilter", "modelFormulaFilter", "modelEvidenceFilter"]) {
    byId(id).addEventListener("change", renderModelList);
  }

  for (const id of ["seqLensInput", "hitRatesInput", "blockSizesInput", "maxIoInput", "arrivalFixed", "arrivalPoisson"]) {
    byId(id).addEventListener("input", () => {
      if (id === "seqLensInput") refreshStatePolicyEditor();
      handleDataChange();
    });
  }
  for (const id of [
    "requestCountInput", "qpsInput", "generationLenInput", "prefixGroupCountInput",
    "prefixZipfAlphaInput", "seedInput", "writeMissBlocksInput", "workloadShapeSelect",
  ]) byId(id).addEventListener("input", handleDataChange);
  for (const id of [
    "footprintViewSelect", "recoveryAssumptionSelect", "checkpointIntervalInput",
    "retainPromptEndInput", "sparseRestoreSelect",
    "sparseLocalitySelect", "sparseReuseSelect", "sparseLayoutSelect",
    "sparseQueryChunkInput", "sparseSeedInput", "includeDraftCacheInput",
    "draftExecutionSelect", "decodeStateWritesInput", "decodeWriteStepInput",
  ]) byId(id).addEventListener("input", handleDataChange);
  byId("runtimeSelect").addEventListener("change", () => {
    applyComputeToControls();
    renderModelList();
    handleDataChange();
  });
  byId("computeTflopsInput").addEventListener("input", handleDataChange);
  byId("storageSelect").addEventListener("change", handleDataChange);
  byId("cacheReplayToggle").addEventListener("change", () => {
    byId("analysisModeSelect").value = byId("cacheReplayToggle").checked
      ? "cache_replay" : "bandwidth_envelope";
    applyAnalysisMode();
    handleDataChange();
  });
  byId("cacheConfigSelect").addEventListener("change", () => {
    applyCachePreset();
    handleDataChange();
  });
  for (const id of [
    "cacheCapacityGiBInput", "cacheShardCountInput", "cachePlacementSeedInput",
    "cacheWarmupFractionInput",
  ]) byId(id).addEventListener("input", handleDataChange);

  document.querySelectorAll(".metric-tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeMetricGroup = button.dataset.metricGroup || null;
      if (!state.activeMetricGroup) state.activeMetric = button.dataset.metric;
      refreshChartSelection();
    });
  });
  byId("groupMetricSelect").addEventListener("change", (event) => {
    if (state.activeMetricGroup === "peak") state.peakMeasure = event.target.value;
    else state.groupMetrics[state.activeMetricGroup] = event.target.value;
    refreshChartSelection();
  });
  byId("peakWindowSelect").addEventListener("change", (event) => {
    state.peakWindow = event.target.value;
    refreshChartSelection();
  });
  byId("viewModeSelect").addEventListener("change", (event) => {
    if (state.viewMode === "curves") state.curveXMode = state.xMode;
    else state.heatmapXMode = state.xMode;
    state.viewMode = event.target.value;
    state.xMode = state.viewMode === "curves" ? state.curveXMode : state.heatmapXMode;
    byId("heatmapFilters").hidden = state.viewMode !== "heatmap";
    byId("smoothGuideControl").hidden = state.viewMode === "heatmap";
    byId("chartTitle").textContent = state.viewMode === "heatmap" ? "Sweep Heatmap" : "Sweep Curves";
    document.querySelector(".legend-actions").hidden = state.viewMode === "heatmap";
    renderChart();
  });
  byId("xModeSelect").addEventListener("change", (event) => {
    state.xMode = event.target.value;
    if (state.viewMode === "curves") state.curveXMode = state.xMode;
    else state.heatmapXMode = state.xMode;
    state.axisNotice = "";
    state.hiddenSeries.clear();
    renderChart();
  });
  byId("smoothGuidesToggle").addEventListener("change", (event) => {
    state.smoothGuides = event.target.checked;
    renderChart();
  });
  byId("heatmapScaleSelect").addEventListener("change", (event) => {
    state.heatmapScale = event.target.value;
    renderChart();
  });
  byId("thresholdToggle").addEventListener("change", (event) => {
    state.thresholdsEnabled = event.target.checked;
    byId("thresholdNote").hidden = !state.thresholdsEnabled;
    renderSummary();
    renderChart();
  });
  for (const id of ["heatmapModelSelect", "heatmapBlockSelect", "heatmapArrivalSelect", "heatmapIoSelect", "heatmapCacheCapacitySelect", "heatmapEncoderStateSelect", "heatmapEncoderTailSelect"]) {
    byId(id).addEventListener("change", readHeatmapSliceFromControls);
  }
  byId("legendShowAll").addEventListener("click", () => { state.hiddenSeries.clear(); renderChart(); });
  byId("legendHideAll").addEventListener("click", () => {
    state.rows.forEach((row) => state.hiddenSeries.add(seriesKey(row, state.xMode)));
    renderChart();
  });
  byId("tableFilter").addEventListener("input", (event) => {
    state.table.filter = event.target.value;
    state.table.page = 1;
    renderTable();
  });
  document.querySelectorAll("[data-detail-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.detailTab = button.dataset.detailTab;
      document.querySelectorAll("[data-detail-tab]").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
      renderDetail();
    });
  });
  document.querySelectorAll("[data-result-view]").forEach((button) => {
    button.addEventListener("click", () => setExploreResultView(button.dataset.resultView));
  });
  wireTabKeyboard(document.querySelectorAll(".metric-tab"));
  wireTabKeyboard(document.querySelectorAll("[data-detail-tab]"));
  wireTabKeyboard(document.querySelectorAll("[data-result-view]"));
  byId("runStorageButton").addEventListener("click", runStorageSimulation);
  byId("showInTableButton").addEventListener("click", () => {
    setExploreResultView("cases");
    focusSelectedTableRow();
  });
  byId("openDetailsButton").addEventListener("click", () => {
    setExploreResultView("cases");
    queueMicrotask(() => {
      byId("exploreInspectorPanel").scrollIntoView({ behavior: "smooth", block: "start" });
      byId("detailContent").focus({ preventScroll: true });
    });
  });
}

function populateConfigControls() {
  populateSelect(byId("runtimeSelect"), state.config.runtimes, state.config.defaults.runtime_config);
  applyComputeToControls();
  populateSelect(byId("storageSelect"), state.config.storages, state.config.defaults.storage_config);
  populateSelect(byId("cacheConfigSelect"), state.config.caches || [], state.config.defaults.cache_config);
  state.selectedModelsByRuntime.set(
    state.config.defaults.runtime_config,
    new Set(state.config.defaults.model_configs),
  );
  populateModelFilters();
  renderModelList();
  applyWorkloadToControls(state.config.defaults.workload_config);
  byId("writeMissBlocksInput").checked = Boolean(state.config.defaults.write_miss_blocks);
  applyModelingOptionsToControls(state.config.defaults.modeling_options || {});
  byId("analysisModeSelect").value = state.config.defaults.analysis_mode || "bandwidth_envelope";
  applyAnalysisMode();
  applyCachePreset();
  setExploreMode(state.exploreMode, { persist: false, reformat: true });
}

function readExploreMode() {
  try {
    return localStorage.getItem(EXPLORE_MODE_KEY) === "expert" ? "expert" : "simple";
  } catch (_error) {
    return "simple";
  }
}

function setExploreMode(mode, { persist = true, reformat = true } = {}) {
  const normalized = mode === "expert" ? "expert" : "simple";
  let prefixRatios = null;
  if (reformat && byId("hitRatesInput").value.trim()) {
    try { prefixRatios = parseNumberList(byId("hitRatesInput").value, "prefix_ratio"); }
    catch (_error) { prefixRatios = null; }
  }
  state.exploreMode = normalized;
  byId("exploreView").dataset.exploreMode = normalized;
  for (const [id, value] of [["simpleModeButton", "simple"], ["expertModeButton", "expert"]]) {
    const active = value === normalized;
    byId(id).classList.toggle("active", active);
    byId(id).setAttribute("aria-pressed", String(active));
  }
  const noneOption = byId("recoveryAssumptionSelect").querySelector('[value="none"]');
  if (noneOption) noneOption.hidden = normalized === "simple";
  if (prefixRatios) byId("hitRatesInput").value = formatPrefixAxis(prefixRatios, normalized);
  if (persist) {
    try { localStorage.setItem(EXPLORE_MODE_KEY, normalized); }
    catch (_error) { /* The mode still applies for the current page. */ }
  }
  if (state.config) {
    applyAnalysisMode();
    updateAdvancedOverrideBadge();
  }
}

function formatPrefixAxis(values, mode = state.exploreMode) {
  return values.map((value) => {
    if (mode === "simple") return `${Number((Number(value) * 100).toPrecision(8))}%`;
    return String(Number(Number(value).toPrecision(10)));
  }).join(", ");
}

function updateAdvancedOverrideBadge() {
  const badge = byId("advancedOverrideBadge");
  if (!state.config) { badge.hidden = true; return; }
  const reasons = [];
  try {
    const payload = formPayload();
    const defaults = state.config.defaults || {};
    if (stableSerialize(payload.axes.block_size_tokens) !== stableSerialize(defaults.block_size_tokens || [])) reasons.push("block axis");
    if (stableSerialize(payload.axes.max_io_bytes) !== stableSerialize(defaults.max_io_bytes || [])) reasons.push("max IO axis");
    if (stableSerialize(payload.axes.arrival_model) !== stableSerialize(defaults.arrival_model || [])) reasons.push("arrival axis");
    if (payload.storage_config !== defaults.storage_config) reasons.push("reference storage");

    const workload = state.config.workloads.find((item) => item.path === payload.workload_config);
    const workloadDefaults = workload?.workload_overrides || defaults.workload_overrides || {};
    if (stableSerialize(payload.workload_overrides) !== stableSerialize(workloadDefaults)) reasons.push("workload overrides");

    if (payload.cache_replay_config) {
      const cacheDefaults = state.config.caches.find((item) => item.path === defaults.cache_config) || {};
      if (payload.cache_replay_config.source_path !== defaults.cache_config
          || payload.cache_replay_config.shard_count !== Number(cacheDefaults.shard_count ?? 1)
          || payload.cache_replay_config.placement_seed !== Number(cacheDefaults.placement_seed ?? 0)
          || payload.cache_replay_config.warmup_fraction !== Number(cacheDefaults.warmup_fraction ?? 0)) {
        reasons.push("cache lifecycle");
      }
    }

    const currentModeling = { ...(payload.modeling_options || {}) };
    const defaultModeling = { ...(defaults.modeling_options || {}) };
    for (const key of ["recovery_assumption", "checkpoint_interval_tokens"]) {
      delete currentModeling[key];
      delete defaultModeling[key];
    }
    if (stableSerialize(currentModeling) !== stableSerialize(defaultModeling)) reasons.push("modeling policies");
    if (payload.modeling_options?.recovery_assumption === "none") reasons.push("no checkpoint state");
  } catch (_error) {
    badge.hidden = true;
    return;
  }
  badge.hidden = !reasons.length;
  badge.textContent = reasons.length
    ? `${reasons.length} advanced override${reasons.length === 1 ? "" : "s"} active`
    : "";
  badge.title = reasons.join(", ");
}

function applyAnalysisMode() {
  const replay = byId("analysisModeSelect").value === "cache_replay";
  byId("workloadSection").hidden = !replay;
  byId("cacheReplayDisclosure").closest("section").hidden = !replay;
  byId("cacheReplayToggle").checked = replay;
  byId("cacheReplayFields").disabled = !replay;
  if (replay) byId("cacheReplayDisclosure").open = true;
  byId("retainPromptEndField").hidden = byId("recoveryAssumptionSelect").value !== "periodic";
  byId("analysisModeHelp").textContent = replay
    ? "Replay a finite request stream through the capacity-aware radix cache."
    : "Assume a resident prefix and calculate isolated restore boundaries.";
  const selectedModels = [...(state.selectedModelsByRuntime.get(byId("runtimeSelect").value) || [])]
    .map((path) => state.config?.models.find((item) => item.path === path))
    .filter(Boolean);
  const stateful = selectedModels.filter((model) => model.recovery_capability?.has_blocking_state);
  const recoverySection = byId("recoveryScenarioSection");
  const hasEncoder = selectedModels.some((model) => model.recovery_capability?.encoder_scenarios_supported);
  recoverySection.classList.toggle("no-stateful-models", !stateful.length && !hasEncoder);
  byId("recoveryScenarioHelp").textContent = hasEncoder ? "Global cache reuse and model-specific state recovery." : "Stateful models need a checkpoint at or before the cached boundary.";
  recoverySection.hidden = !stateful.length && !hasEncoder && state.exploreMode === "simple";
  byId("recoveryAssumptionSelect").closest("label").hidden = hasEncoder && !stateful.length;
  byId("stateCapabilitySummary").textContent = stateful.length
    ? stateful.map((model) => {
      const targets = (model.recovery_capability?.state_policy_targets || []).map((target) => (
        `${target.name}: ${checkpointPolicyLabel(target.default_policy)}`
      ));
      return `${model.model_name || model.label} · ${targets.join(", ")}`;
    }).join(" · ")
    : hasEncoder ? "V4.1 keeps global KV reuse independent of Encoder recovery. Select the execution-side state and output scenarios above."
      : "Selected models use token-state recovery; checkpoint controls are not on their critical path.";
  byId("checkpointIntervalField").hidden = (hasEncoder && !stateful.length) || byId("recoveryAssumptionSelect").value !== "periodic";
  renderRecoveryPreflight(stateful);
  renderCacheLifecyclePreview();
}

function renderRecoveryPreflight(statefulModels) {
  const container = byId("recoveryPreflight");
  container.hidden = true;
  if (!state.config || byId("recoveryAssumptionSelect").value !== "model_default") return;
  let prefixRatios;
  let overrides;
  try {
    prefixRatios = parseNumberList(byId("hitRatesInput").value, "prefix_ratio");
    overrides = readStatePolicyEditor(byId("statePolicyEditor"));
  } catch (_error) {
    return;
  }
  const partial = prefixRatios.filter((value) => value < 1);
  if (!partial.length) return;
  const affected = statefulModels.filter((model) => (
    (model.recovery_capability?.state_policy_targets || []).some((target) => {
      const value = overrides.model_state_policies?.[model.path]?.[target.name]
        ?? overrides.state_policies?.[target.name]
        ?? target.default_policy;
      return isPromptEndOnlyPolicy(value);
    })
  ));
  if (!affected.length) return;

  const names = affected.map((model) => model.model_name || model.label).join(", ");
  const points = partial.map((value) => `${formatNumber(value * 100)}%`).join(", ");
  byId("recoveryPreflightTitle").textContent = `${names}: partial prefix hits cannot resume with model defaults`;
  byId("recoveryPreflightBody").textContent = `${partial.length} sampled point${partial.length === 1 ? "" : "s"} (${points}) stop before the prompt-end state snapshot. Prefix objects may be resident, but the common recovery boundary is token zero, so the cached prefix is replayed.`;
  container.hidden = false;
}

function isPromptEndOnlyPolicy(value) {
  if (typeof value === "string") {
    return ["prompt_end", "terminal", "terminal_only"].includes(value);
  }
  return value?.storage_form === "snapshot" && value.interval_tokens == null;
}

function checkpointPolicyLabel(value) {
  if (!value) return "model default";
  if (typeof value === "string") {
    if (isPromptEndOnlyPolicy(value)) return "prompt end only";
    if (value.startsWith("periodic")) return value.replace("periodic+prompt_end", "periodic + prompt end");
    return value.replaceAll("_", " ");
  }
  if (value.storage_form === "snapshot") {
    return value.interval_tokens == null
      ? "prompt end only"
      : `every ${Number(value.interval_tokens).toLocaleString()} tokens${value.retain_prompt_end ? " + prompt end" : ""}`;
  }
  return String(value.storage_form || "model default").replaceAll("_", " ");
}

function addFullHitControlPoint() {
  let values;
  try { values = parseNumberList(byId("hitRatesInput").value, "prefix_ratio"); }
  catch (error) { setStatus(error.message, true); return; }
  if (!values.includes(1)) values.push(1);
  values.sort((left, right) => left - right);
  byId("hitRatesInput").value = formatPrefixAxis(values, state.exploreMode);
  setStatus("Added a 100% full-hit control point. Partial-hit recovery behavior is unchanged.");
  handleDataChange();
}

function applyCachePreset() {
  if (!state.config) return;
  const selected = (state.config.caches || []).find(
    (option) => option.path === byId("cacheConfigSelect").value,
  );
  if (!selected) return;
  byId("cacheCapacityGiBInput").value = formatGiBInput(selected.total_capacity_bytes);
  byId("cacheShardCountInput").value = selected.shard_count ?? 1;
  byId("cachePlacementSeedInput").value = selected.placement_seed ?? 0;
  byId("cacheWarmupFractionInput").value = selected.warmup_fraction ?? 0;
  byId("cachePolicyValue").textContent = selected.policy || "radix_leaf_lru";
}

function renderModelList() {
  const modelList = byId("modelList");
  modelList.replaceChildren();
  const runtime = currentRuntimeOption();
  const compatibility = runtime?.model_compatibility || {};
  let selected = state.selectedModelsByRuntime.get(runtime?.path);
  if (!selected) {
    selected = new Set(
      (state.config.defaults.model_configs || [])
        .filter((path) => compatibility[path]?.compatible),
    );
    state.selectedModelsByRuntime.set(runtime?.path, selected);
  }
  const search = byId("modelSearchInput").value.trim().toLowerCase();
  const family = byId("modelFamilyFilter").value;
  const formula = byId("modelFormulaFilter").value;
  const evidence = byId("modelEvidenceFilter").value;
  const visibleModels = state.config.models.filter((model) => {
    const haystack = `${model.model_name || ""} ${model.label || ""} ${model.model_id || ""} ${model.recovery_capability?.display_attention || ""}`.toLowerCase();
    return (!search || haystack.includes(search))
      && (!family || model.family === family)
      && (!formula || model.formula_id === formula)
      && (!evidence || (model.evidence_statuses || []).includes(evidence));
  }).sort((left, right) => (
    Number(selected.has(right.path)) - Number(selected.has(left.path))
    || String(left.model_name || left.label).localeCompare(String(right.model_name || right.label))
  ));
  for (const model of visibleModels) {
    const runtimeModel = compatibility[model.path] || {
      compatible: !model.error,
      error: model.error,
      model_name: model.model_name,
      attention: model.attention,
      cache_bytes_per_token: model.cache_bytes_per_token,
      constraints: model.constraints,
    };
    const label = document.createElement("label");
    label.className = "model-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = model.path;
    input.checked = Boolean(runtimeModel.compatible && selected.has(model.path));
    input.disabled = !runtimeModel.compatible;
    input.title = runtimeModel.error || "";
    input.addEventListener("change", () => {
      if (input.checked) selected.add(model.path);
      else selected.delete(model.path);
      renderSelectedModelSummary();
      refreshStatePolicyEditor();
      handleDataChange();
    });
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = runtimeModel.model_name || model.model_name || model.label;
    const detail = document.createElement("small");
    const constraints = runtimeModel.constraints || {};
    detail.textContent = runtimeModel.compatible
      ? `${model.recovery_capability?.display_attention || model.formula_id || runtimeModel.attention || "unknown"} · ${model.family || "unknown family"} · evidence ${(model.evidence_statuses || ["unspecified"]).join("/")} · ${formatCompact(runtimeModel.cache_bytes_per_token || 0)} B/token${constraints.max_context_tokens ? ` · max ${formatCompact(constraints.max_context_tokens)}` : ""}${constraints.block_size_multiple > 1 ? ` · block ×${constraints.block_size_multiple}` : ""}`
      : `Unavailable · ${runtimeModel.error || "incompatible runtime"}`;
    copy.append(title, detail);
    label.append(input, copy);
    modelList.append(label);
  }
  if (!visibleModels.length) {
    const empty = document.createElement("p");
    empty.className = "field-message";
    empty.textContent = "No models match the current filters.";
    modelList.append(empty);
  }
  renderSelectedModelSummary();
  refreshStatePolicyEditor();
}

function renderSelectedModelSummary() {
  if (!state.config) return;
  const container = byId("selectedModelSummary");
  container.replaceChildren();
  const selectedPaths = state.selectedModelsByRuntime.get(byId("runtimeSelect").value) || new Set();
  const compatibility = currentRuntimeOption()?.model_compatibility || {};
  const selected = [...selectedPaths]
    .map((path) => state.config.models.find((item) => item.path === path))
    .filter((model) => model && compatibility[model.path]?.compatible !== false)
    .sort((left, right) => String(left.model_name || left.label).localeCompare(String(right.model_name || right.label)));
  if (!selected.length) {
    const empty = document.createElement("span");
    empty.className = "selected-model-empty";
    empty.textContent = "No compatible models selected.";
    container.append(empty);
  } else {
    for (const model of selected) {
      const chip = document.createElement("span");
      chip.className = "selected-model-chip";
      chip.textContent = model.model_name || model.label;
      chip.title = `${model.recovery_capability?.display_attention || model.formula_id || "unknown"} · ${model.family || "unknown family"}`;
      container.append(chip);
    }
  }
  byId("modelDialogSelectionCount").textContent = `${selected.length.toLocaleString()} selected`;
}

function populateModelFilters() {
  const dimensions = [
    ["modelFamilyFilter", "family", "All families"],
    ["modelFormulaFilter", "formula_id", "All formulas"],
  ];
  for (const [id, field, label] of dimensions) {
    const select = byId(id);
    select.replaceChildren();
    const all = document.createElement("option");
    all.value = "";
    all.textContent = label;
    select.append(all);
    const values = [...new Set(state.config.models.map((item) => item[field]).filter(Boolean))].sort();
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.append(option);
    }
  }
  const evidence = byId("modelEvidenceFilter");
  evidence.replaceChildren();
  const allEvidence = document.createElement("option");
  allEvidence.value = "";
  allEvidence.textContent = "All evidence";
  evidence.append(allEvidence);
  const statuses = [...new Set(state.config.models.flatMap((item) => item.evidence_statuses || []))].sort();
  for (const value of statuses) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    evidence.append(option);
  }
}

function populateSelect(select, options, selected) {
  select.replaceChildren();
  for (const option of options) {
    const node = document.createElement("option");
    node.value = option.path;
    node.textContent = option.label;
    node.disabled = Boolean(option.error);
    node.selected = option.path === selected;
    select.append(node);
  }
}

function applyComputeToControls(computeConfig = null) {
  state.computeConfig = computeConfig;
  const effective = computeConfig || currentRuntimeOption()?.compute_config || state.config.defaults.compute_config;
  byId("computeTflopsInput").value = effective?.effective_tflops?.default ?? 2000;
}

function applyWorkloadToControls(path, overrides = {}) {
  // Keep the original workload source and unshown overrides when reopening a run.
  state.workloadConfig = path || state.config.defaults.workload_config;
  state.workloadOverrides = { ...overrides };
  const selected = state.config.workloads.find((option) => option.path === state.workloadConfig);
  const defaults = selected?.workload_overrides
    || (state.workloadConfig === state.config.defaults.workload_config ? state.config.defaults.workload_overrides : {})
    || {};
  const values = { ...defaults, ...state.workloadOverrides };
  const mapping = {
    requestCountInput: "request_count", qpsInput: "qps", generationLenInput: "generation_len",
    prefixGroupCountInput: "prefix_group_count", prefixZipfAlphaInput: "prefix_zipf_alpha", seedInput: "seed",
  };
  for (const [id, key] of Object.entries(mapping)) byId(id).value = values[key] ?? "";
  byId("workloadShapeSelect").value = values.workload_shape || "shared_prefix_groups";
}

function applyPreset(name) {
  applyEncoderScenarios(byId("encoderScenarios"));
  if (!state.config) return;
  const preset = state.config.presets?.[name];
  if (!preset) return;
  const axes = preset.axes;
  byId("seqLensInput").value = axes.seq_len.join(", ");
  byId("hitRatesInput").value = formatPrefixAxis(axes.prefix_ratio, state.exploreMode);
  byId("blockSizesInput").value = axes.block_size_tokens.join(", ");
  byId("maxIoInput").value = axes.max_io_bytes.map((value) => value ?? "").join(", ");
  byId("arrivalFixed").checked = axes.arrival_model.includes("fixed");
  byId("arrivalPoisson").checked = axes.arrival_model.includes("poisson");
  if (Array.isArray(preset.model_configs)) {
    const runtimePath = byId("runtimeSelect").value;
    const compatible = currentRuntimeOption()?.model_compatibility || {};
    state.selectedModelsByRuntime.set(
      runtimePath,
      new Set(preset.model_configs.filter((path) => compatible[path]?.compatible)),
    );
    renderModelList();
  }
  state.xMode = name === "full_tail" ? "miss-log" : "hit-linear";
  state.curveXMode = state.xMode;
  state.heatmapXMode = state.xMode;
  byId("xModeSelect").value = state.xMode;
  handleDataChange();
}

function selectAllModels() {
  const inputs = [...document.querySelectorAll("#modelList input:not(:disabled)")];
  const shouldSelect = inputs.some((input) => !input.checked);
  inputs.forEach((input) => { input.checked = shouldSelect; });
  const selected = state.selectedModelsByRuntime.get(byId("runtimeSelect").value) || new Set();
  for (const input of inputs) {
    if (input.checked) selected.add(input.value);
    else selected.delete(input.value);
  }
  state.selectedModelsByRuntime.set(byId("runtimeSelect").value, selected);
  renderSelectedModelSummary();
  refreshStatePolicyEditor();
  handleDataChange();
}

function modelingOptionsFromControls() {
  const decodeStepText = byId("decodeWriteStepInput").value.trim();
  const decodeStep = decodeStepText === ""
    ? null
    : requiredNumber("decodeWriteStepInput", { positive: true });
  const statePolicies = readStatePolicyEditor(byId("statePolicyEditor"));
  return {
    footprint_view: byId("footprintViewSelect").value,
    state_policies: statePolicies.state_policies,
    model_state_policies: statePolicies.model_state_policies,
    recovery_assumption: byId("recoveryAssumptionSelect").value,
    checkpoint_interval_tokens: requiredNumber("checkpointIntervalInput", { integer: true, min: 1 }),
    retain_prompt_end: byId("retainPromptEndInput").checked,
    sparse_access_policy: {
      restore_mode: byId("sparseRestoreSelect").value,
      locality_scenario: byId("sparseLocalitySelect").value,
      reuse_scope: byId("sparseReuseSelect").value,
      storage_layout: byId("sparseLayoutSelect").value,
      query_granularity_tokens: requiredNumber("sparseQueryChunkInput", { integer: true, min: 1 }),
      seed: requiredNumber("sparseSeedInput", { integer: true }),
    },
    include_draft_cache: byId("includeDraftCacheInput").checked,
    draft_execution: byId("draftExecutionSelect").value,
    decode_state_writes: byId("decodeStateWritesInput").checked || decodeStep != null,
    decode_write_step_us: decodeStep,
  };
}

function applyModelingOptionsToControls(options) {
  const sparse = options.sparse_access_policy || {};
  byId("footprintViewSelect").value = String(options.footprint_view || "prefix-store").replaceAll("_", "-");
  byId("recoveryAssumptionSelect").value = options.recovery_assumption || "model_default";
  byId("checkpointIntervalInput").value = options.checkpoint_interval_tokens ?? 2048;
  byId("retainPromptEndInput").checked = Boolean(options.retain_prompt_end);
  applyStatePolicyEditor(byId("statePolicyEditor"), options);
  byId("sparseRestoreSelect").value = sparse.restore_mode || "full_restore";
  byId("sparseLocalitySelect").value = sparse.locality_scenario || "window_global";
  byId("sparseReuseSelect").value = sparse.reuse_scope || "request_union";
  byId("sparseLayoutSelect").value = sparse.storage_layout || "all_kv_heads_together";
  byId("sparseQueryChunkInput").value = sparse.query_granularity_tokens ?? 1;
  byId("sparseSeedInput").value = sparse.seed ?? 0;
  byId("includeDraftCacheInput").checked = Boolean(options.include_draft_cache);
  byId("draftExecutionSelect").value = options.draft_execution || "disabled";
  byId("decodeStateWritesInput").checked = Boolean(options.decode_state_writes);
  byId("decodeWriteStepInput").value = options.decode_write_step_us ?? "";
  refreshStatePolicyEditor();
}

function refreshStatePolicyEditor() {
  if (!state.config || !byId("statePolicyEditor")) return;
  let seqLengths = [32768];
  try { seqLengths = parseNumberList(byId("seqLensInput").value, "seq_len", true); }
  catch (_error) { /* Keep a stable preview while the user edits the axis. */ }
  const selected = [...(state.selectedModelsByRuntime.get(byId("runtimeSelect").value) || [])];
  byId("encoderScenarios").hidden = !supportsEncoderScenarios(state.config.models, selected);
  renderStatePolicyEditor(byId("statePolicyEditor"), {
    models: state.config.models,
    selectedPaths: selected,
    seqLengths,
    prefix: "explorePolicy",
    onChange: handleDataChange,
  });
}

function renderCacheLifecyclePreview() {
  const container = byId("cacheLifecyclePreview");
  if (!container || byId("analysisModeSelect").value !== "cache_replay") {
    if (container) container.replaceChildren();
    return;
  }
  try {
    const normal = requiredNumber("requestCountInput", { integer: true, min: 0 });
    const groups = requiredNumber("prefixGroupCountInput", { integer: true, min: 1 });
    const warmupFraction = requiredNumber("cacheWarmupFractionInput", { min: 0 });
    const hitRates = parseNumberList(byId("hitRatesInput").value, "prefix_ratio");
    const bootstrap = byId("workloadShapeSelect").value === "prompt_lineage_bootstrap"
      && hitRates.some((value) => value > 0) ? groups : 0;
    const warmup = Math.floor(normal * warmupFraction);
    const measured = normal - warmup;
    const seqLengths = parseNumberList(byId("seqLensInput").value, "seq_len", true);
    const selected = [...(state.selectedModelsByRuntime.get(byId("runtimeSelect").value) || [])];
    const estimate = estimateBootstrapState(byId("statePolicyEditor"), {
      models: state.config.models,
      selectedPaths: selected,
      seqLengths,
      prefixRatios: hitRates,
      prefixGroups: bootstrap,
      globalPolicy: {
        mode: byId("recoveryAssumptionSelect").value,
        interval: Number(byId("checkpointIntervalInput").value) || 2048,
        retain: byId("retainPromptEndInput").checked,
      },
    });
    const estimateText = bootstrap && estimate.bytes
      ? `<span>Max bootstrap state ${formatCompact(estimate.bytes / (1024 ** 3))} GiB · ${escapeText(estimate.model)}</span>`
      : "";
    container.innerHTML = `<strong>Cache lifecycle</strong><span>Bootstrap ${bootstrap.toLocaleString()}</span><span>Warmup ${warmup.toLocaleString()}</span><span>Measured ${measured.toLocaleString()}</span><span>Total ${(bootstrap + normal).toLocaleString()}</span>${estimateText}`;
  } catch (_error) {
    container.textContent = "Lifecycle preview is waiting for valid workload inputs.";
  }
}

function formPayload() {
  const axes = {
    seq_len: parseNumberList(byId("seqLensInput").value, "seq_len", true),
    prefix_ratio: parseNumberList(byId("hitRatesInput").value, "prefix_ratio"),
    block_size_tokens: parseNumberList(byId("blockSizesInput").value, "block_size_tokens", true),
    arrival_model: [byId("arrivalFixed"), byId("arrivalPoisson")].filter((input) => input.checked).map((input) => input.value),
    max_io_bytes: parseOptionalIntList(byId("maxIoInput").value),
  };
  const selectedForRuntime = state.selectedModelsByRuntime.get(byId("runtimeSelect").value) || new Set();
  const compatibility = currentRuntimeOption()?.model_compatibility || {};
  const modelConfigs = [...selectedForRuntime].filter((path) => compatibility[path]?.compatible !== false);
  if (!modelConfigs.length) throw new Error("Select at least one model.");
  if (!axes.arrival_model.length) throw new Error("Select at least one arrival model.");
  if (supportsEncoderScenarios(state.config.models, modelConfigs)) Object.assign(axes, readEncoderScenarios(byId("encoderScenarios")));
  validateAxes(modelConfigs, axes);
  const workloadOverrides = {
    ...state.workloadOverrides,
    request_count: requiredNumber("requestCountInput", { integer: true, min: 0 }),
    qps: requiredNumber("qpsInput", { positive: true }),
    generation_len: requiredNumber("generationLenInput", { integer: true, min: 0 }),
    prefix_group_count: requiredNumber("prefixGroupCountInput", { integer: true, min: 1 }),
    prefix_zipf_alpha: requiredNumber("prefixZipfAlphaInput", { min: 0 }),
    seed: requiredNumber("seedInput", { integer: true }),
    workload_shape: byId("workloadShapeSelect").value,
  };
  const payload = {
    analysis_mode: byId("analysisModeSelect").value,
    model_configs: modelConfigs,
    runtime_config: byId("runtimeSelect").value,
    workload_config: state.workloadConfig,
    storage_config: byId("storageSelect").value,
    axes,
    workload_overrides: workloadOverrides,
    write_miss_blocks: byId("writeMissBlocksInput").checked,
    modeling_options: modelingOptionsFromControls(),
  };
  const computeTflops = requiredNumber("computeTflopsInput", { positive: true, label: "Compute (TFLOPS)" });
  const baseCompute = currentRuntimeOption()?.compute_config || state.config.defaults.compute_config;
  const baseTflops = Number(baseCompute?.effective_tflops?.default);
  if (!Number.isFinite(baseTflops) || baseTflops <= 0) {
    throw new Error("Compute configuration is unavailable. Refresh this page; if this message remains, restart the local dashboard server.");
  }
  // An unchanged legacy run keeps its original request and cache configuration.
  if (state.computeConfig || computeTflops !== baseTflops) {
    const compute = state.computeConfig || baseCompute;
    payload.compute_config = {
      ...compute,
      effective_tflops: { ...compute.effective_tflops, default: computeTflops },
    };
  }
  if (payload.analysis_mode === "cache_replay") {
    const capacitiesGiB = parseNumberList(
      byId("cacheCapacityGiBInput").value,
      "cache capacity GiB",
    );
    if (capacitiesGiB.some((value) => value < 0)) {
      throw new Error("cache capacity GiB must contain non-negative values.");
    }
    const capacityBytes = capacitiesGiB.map((value) => {
      const bytes = Math.round(value * (1024 ** 3));
      if (!Number.isSafeInteger(bytes) || bytes < 0) {
        throw new Error("cache capacity GiB is outside the safe integer range.");
      }
      return bytes;
    });
    const shardCount = requiredNumber("cacheShardCountInput", { integer: true, min: 1 });
    const placementSeed = requiredNumber("cachePlacementSeedInput", { integer: true });
    const warmupFraction = requiredNumber("cacheWarmupFractionInput", { min: 0 });
    if (warmupFraction >= 1) {
      throw new Error("cache warmup fraction must stay within [0, 1). ");
    }
    const preset = (state.config.caches || []).find(
      (option) => option.path === byId("cacheConfigSelect").value,
    );
    if (!preset || preset.error) throw new Error("Select a valid cache preset.");
    payload.cache_replay_config = {
      source_path: preset.path,
      name: preset.name || "radix-leaf-lru",
      policy: "radix_leaf_lru",
      shard_count: shardCount,
      placement_seed: placementSeed,
      warmup_fraction: warmupFraction,
    };
    payload.axes.cache_total_capacity_bytes = [...new Set(capacityBytes)].sort((a, b) => a - b);
  }
  const totalCases = scenarioCaseCount(payload.model_configs, payload.axes, state.config.models);
  if (totalCases > state.config.limits.max_sweep_cases) {
    throw new Error(`Sweep has ${totalCases} cases; limit is ${state.config.limits.max_sweep_cases}.`);
  }
  return payload;
}

function validateAxes(modelConfigs, axes) {
  if (axes.prefix_ratio.some((value) => value < 0 || value > 1)) throw new Error("prefix_ratio must stay within [0, 1].");
  if ([...axes.seq_len, ...axes.block_size_tokens].some((value) => value <= 0)) throw new Error("seq_len and block_size_tokens must be positive.");
  const compatibility = currentRuntimeOption()?.model_compatibility || {};
  for (const path of modelConfigs) {
    const model = state.config.models.find((item) => item.path === path);
    const runtimeModel = compatibility[path];
    if (runtimeModel && !runtimeModel.compatible) {
      throw new Error(`${model?.label || path} is incompatible with the selected runtime.`);
    }
    const constraints = runtimeModel?.constraints || model?.constraints || {};
    const modelName = runtimeModel?.model_name || model?.model_name || model?.label || path;
    if (constraints.max_context_tokens && axes.seq_len.some((value) => value > constraints.max_context_tokens)) {
      throw new Error(`${modelName}: seq_len exceeds max_context_tokens=${constraints.max_context_tokens}.`);
    }
    if (constraints.minimum_block_size_tokens > 1 && axes.block_size_tokens.some((value) => value < constraints.minimum_block_size_tokens)) {
      throw new Error(`${modelName}: block size must be at least ${constraints.minimum_block_size_tokens}.`);
    }
    if (constraints.block_size_multiple > 1 && axes.block_size_tokens.some((value) => value % constraints.block_size_multiple !== 0)) {
      throw new Error(`${modelName}: block size must be a multiple of ${constraints.block_size_multiple} (native ${constraints.native_block_size_tokens}).`);
    }
  }
  const count = scenarioCaseCount(modelConfigs, axes, state.config.models);
  if (count > state.config.limits.max_sweep_cases) throw new Error(`Sweep has ${count} cases; limit is ${state.config.limits.max_sweep_cases}.`);
}

function updateEstimate() {
  if (!state.config) return;
  try {
    const payload = formPayload();
    const count = scenarioCaseCount(payload.model_configs, payload.axes, state.config.models);
    byId("caseEstimate").textContent = `${count.toLocaleString()} ${count === 1 ? "case" : "cases"}`;
    byId("caseFormula").textContent = caseFormula(payload, count);
    byId("validationText").textContent = "";
    byId("cacheValidationText").textContent = "";
    byId("runButton").disabled = state.busy || Boolean(state.activeJobId);
  } catch (error) {
    byId("caseEstimate").textContent = "Invalid";
    byId("caseFormula").textContent = "Check highlighted inputs to continue.";
    byId("validationText").textContent = error.message;
    byId("cacheValidationText").textContent = byId("cacheReplayToggle").checked ? error.message : "";
    byId("runButton").disabled = true;
  }
}

function handleDataChange() {
  applyAnalysisMode();
  updateEstimate();
  syncPresetIndicator();
  updateAdvancedOverrideBadge();
  updateResultFreshness();
}

async function executeSweep() {
  let payload;
  try { payload = formPayload(); }
  catch (error) { setStatus(error.message, true); return; }
  const caseCount = scenarioCaseCount(payload.model_configs, payload.axes, state.config.models);
  if (caseCount > (state.config.limits.sync_sweep_cases || 100)) {
    await executeBackgroundSweep(payload);
    return;
  }
  setBusy(true);
  setStatus("Running demand sweep…");
  try {
    const response = await runSweep(payload);
    acceptSweep(response, payload);
  } catch (error) {
    console.error(error);
    setStatus(`Sweep failed: ${error.message}`, true);
    updateResultFreshness();
  } finally {
    setBusy(false);
  }
}

async function executeBackgroundSweep(payload) {
  setBusy(true, true);
  setStatus("Starting background sweep…");
  try {
    const job = await createSweepJob(payload);
    const authoritativePayload = job.request || payload;
    applyPayloadToControls(authoritativePayload);
    state.activeJobId = job.job_id;
    storeActiveJob(job.job_id, authoritativePayload);
    setBusy(true, true);
    await pollSweepJob(job.job_id, authoritativePayload);
  } catch (error) {
    clearActiveJob();
    setBusy(false);
    const message = error.status === 409
      ? "Another, different compute request is already running. This Sweep was not attached to it."
      : `Sweep failed: ${error.message}`;
    setStatus(message, true);
    updateResultFreshness();
  }
}

async function pollSweepJob(jobId, payload) {
  const token = ++state.jobPollToken;
  while (token === state.jobPollToken && state.activeJobId === jobId) {
    let job;
    try {
      job = await getSweepJob(jobId);
    } catch (error) {
      clearActiveJob();
      setBusy(false);
      setStatus(error.status === 404 ? "Background sweep no longer exists." : `Sweep status failed: ${error.message}`, true);
      updateResultFreshness();
      return;
    }
    const percent = job.total_cases ? (job.completed_cases / job.total_cases) * 100 : 0;
    setStatus(`${job.status === "queued" ? "Queued" : "Running"} · ${job.completed_cases.toLocaleString()} / ${job.total_cases.toLocaleString()} cases · ${formatNumber(percent)}%${job.cancel_requested ? " · cancel requested" : ""}`);
    if (job.status === "completed") {
      clearActiveJob();
      setBusy(false);
      acceptSweep(job.result, payload);
      return;
    }
    if (job.status === "failed" || job.status === "canceled") {
      clearActiveJob();
      setBusy(false);
      setStatus(job.status === "canceled" ? "Background sweep canceled." : `Sweep failed: ${job.error || "unknown error"}`, job.status === "failed");
      updateResultFreshness();
      return;
    }
    await delay(JOB_POLL_INTERVAL_MS);
  }
}

async function cancelActiveSweep() {
  if (!state.activeJobId) return;
  byId("cancelSweepButton").disabled = true;
  try {
    await cancelSweepJob(state.activeJobId);
    setStatus("Cancel requested · waiting for the current case to finish.");
  } catch (error) {
    setStatus(`Cancel failed: ${error.message}`, true);
    byId("cancelSweepButton").disabled = false;
  }
}

function acceptSweep(response, fallbackPayload) {
  state.sweep = response;
  state.rows = response.rows || [];
  state.axisNotice = "";
  syncMetricControls();
  syncXAxisControl();
  state.resultFingerprint = payloadFingerprint(response.request || fallbackPayload);
  state.hiddenSeries.clear();
  state.selectedRow = null;
  state.caseDetail = null;
  state.detailTab = "overview";
  state.resultView = "charts";
  state.table = createTableState();
  byId("tableFilter").value = "";
  syncHeatmapSlice(state.rows[0] || null);
  renderAll();
  updateResultFreshness();
  if (response.history_error) {
    setStatus(`Completed ${state.rows.length.toLocaleString()} cases · ${response.history_error}`, true);
  } else {
    setStatus(`Completed ${state.rows.length.toLocaleString()} cases${response.experiment_id ? " · saved to Experiments" : ""}.`);
  }
  if (response.experiment_id || response.history_error) {
    window.dispatchEvent(new CustomEvent("dashboard-history-changed"));
  }
}

function refreshChartTheme() {
  if (!state.rows.length) return;
  const active = document.activeElement;
  const focusContainer = [byId("chartFrame"), byId("chartLegend")].find((node) => node.contains(active));
  const focusable = "button, [tabindex]";
  const focusIndex = focusContainer ? [...focusContainer.querySelectorAll(focusable)].indexOf(active) : -1;
  const legendScroll = byId("chartLegend").scrollTop;
  // Repaint computed colors only. CSS variables update Cache and detail bars
  // directly; all experiment, selection, filter and dirty state stays in place.
  renderChart();
  byId("chartLegend").scrollTop = legendScroll;
  if (focusIndex >= 0) focusContainer.querySelectorAll(focusable)[focusIndex]?.focus({ preventScroll: true });
}

function renderAll() {
  renderSummary();
  renderChart();
  renderCache();
  renderTable();
  renderDetail();
  renderSelectedCaseBar();
}

function renderCache() {
  renderCachePanel({
    panel: byId("cachePanel"),
    summary: byId("cacheSummaryGrid"),
    hitChart: byId("cacheHitChart"),
    churnChart: byId("cacheChurnChart"),
    caption: byId("cachePanelCaption"),
    tooltip: byId("chartTooltip"),
    rows: state.rows,
    selectedId: state.selectedRow?.id,
    onSelect: (row, element) => selectRow(row, focusKindForElement(element)),
  });
  syncExploreResultTabs();
}

function renderSummary() {
  const grid = byId("summaryGrid");
  grid.replaceChildren();
  const summary = summarizeRows(state.rows);
  for (const cardData of summary.cards) {
    const { metric, def, source, value } = cardData;
    const card = document.createElement("article");
    card.className = "metric-card";
    card.dataset.tone = def.tone;
    card.dataset.state = classifyValue(value);
    card.tabIndex = source ? 0 : -1;
    card.setAttribute("role", source ? "button" : "group");
    const label = document.createElement("span");
    label.textContent = def.shortLabel;
    const strong = document.createElement("strong");
    strong.textContent = source ? formatMetricForRow(source, metric) : "–";
    const model = document.createElement("small");
    model.textContent = source ? `${source.id} · ${source.model}` : "No source case";
    const context = document.createElement("em");
    context.textContent = source ? `seq ${formatCompact(source.seq_len)} · prefix ${formatNumber(source.prefix_ratio * 100)}%` : "Run a sweep";
    card.append(label, strong, model, context);
    if (source) {
      card.addEventListener("click", () => selectRow(source));
      card.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectRow(source); } });
    }
    grid.append(card);
  }

  const threshold = activeThreshold();
  const risks = riskCounts(state.rows, state.thresholdsEnabled ? state.activeMetric : null, threshold);
  const strip = byId("riskStrip");
  strip.replaceChildren();
  const items = [
    ["Cached, not resumable", risks.unrecoverable, "warning"],
    ["No TTFT gain", risks.zero, "warning"],
    ["Mixed break-even", risks.mixed, "warning"],
    ["No finite break-even", risks.noFinite, "danger"],
    ["Negative TTFT", risks.negative, "danger"],
    ["True ±Infinity", risks.infinite, "danger"],
  ].filter(([, value]) => Number(value) > 0);
  if (state.sweep?.decision_context) {
    items.push(["Constraint failures", risks.recommendationFailed, risks.recommendationFailed ? "warning" : ""]);
  }
  items.push(["Over device proxy", state.thresholdsEnabled ? risks.exceeded : "Off", risks.exceeded ? "warning" : ""]);
  if (items.length === 1 && !state.thresholdsEnabled) items.unshift(["Flagged cases", 0, ""]);
  for (const [label, value, tone] of items) {
    const item = document.createElement("div");
    item.className = `risk-pill ${tone}`;
    item.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    strip.append(item);
  }
}

function renderChart() {
  syncMetricControls();
  syncXAxisControl();
  const metricDefinition = METRICS[state.activeMetric];
  document.querySelector(".chart-panel")?.classList.toggle("has-results", Boolean(state.rows.length));
  byId("metricDescription").textContent = metricDefinition.description;
  const notices = [state.axisNotice, metricUnavailableReason(state.activeMetric, state.rows)];
  if (state.activeMetricGroup && state.activeMetricGroup !== "peak" && METRIC_GROUPS[state.activeMetricGroup].metrics.some((key) => METRICS[key].cacheOnly) && !state.rows.some(hasCacheRow)) {
    notices.push("Cache-only metrics are disabled until Cache Replay results are available.");
  }
  const availability = byId("metricAvailability");
  availability.textContent = notices.filter(Boolean).join(" ");
  availability.hidden = !availability.textContent;
  const activeMetricTab = document.querySelector('.metric-tab[aria-selected="true"]');
  if (activeMetricTab) byId("chartFrame").setAttribute("aria-labelledby", activeMetricTab.id);
  const common = {
    frame: byId("chartFrame"), legend: byId("chartLegend"), tooltip: byId("chartTooltip"),
    rows: state.rows, metric: state.activeMetric, xMode: state.xMode, selectedId: state.selectedRow?.id,
    onSelect: (row, element) => selectRow(row, focusKindForElement(element)),
  };
  if (state.viewMode === "heatmap") {
    const result = renderHeatmap({
      ...common,
      globalRows: state.rows,
      slice: state.heatmapSlice || emptySlice(),
      scaleMode: state.heatmapScale,
    });
    byId("chartTitle").textContent = `${metricDefinition.shortLabel} heatmap`;
    const cacheSlice = state.rows.some((row) => row.metrics?.cache_total_capacity_bytes !== undefined)
      ? "/cache capacity"
      : "";
    byId("chartCaption").textContent = `${result.cellCount} cases · ${result.rowCount} context lengths × ${result.columnCount} prefix ratios · fixed model/block/arrival/max IO${cacheSlice} · ${state.heatmapScale} color scale.`;
  } else {
    const result = renderCurves({
      ...common,
      hiddenSeries: state.hiddenSeries,
      threshold: state.thresholdsEnabled ? activeThreshold() : NaN,
      smoothGuides: state.smoothGuides,
      onToggleSeries: (key) => { state.hiddenSeries.has(key) ? state.hiddenSeries.delete(key) : state.hiddenSeries.add(key); renderChart(); },
      onOnlySeries: (key, keys) => { state.hiddenSeries = new Set(keys.filter((item) => item !== key)); renderChart(); },
    });
    const xTitle = state.xMode === "hit-linear" ? "Prefix ratio" : state.xMode === "miss-log" ? "candidate miss ratio" : state.xMode === "cache-capacity" ? "Cache capacity" : "Context length";
    byId("chartTitle").textContent = `${metricDefinition.shortLabel} vs ${xTitle}`;
    if (!state.rows.length) {
      byId("chartCaption").textContent = "Run a Sweep to display research-style sensitivity curves.";
      return;
    }
    const sampleCount = result.sampleCountMin === result.sampleCountMax
      ? `${result.sampleCountMin} actual points/series`
      : `${result.sampleCountMin}–${result.sampleCountMax} actual points/series`;
    const yScale = result.zeroOnly ? "zero-value view" : scaleLabel(metricDefinition);
    const seriesCount = result.visibleSeriesCount === result.seriesCount
      ? `${result.seriesCount} series`
      : `${result.visibleSeriesCount} of ${result.seriesCount} series visible`;
    const runtimePath = state.sweep?.request?.runtime_config;
    const runtime = runtimePath ? ` · ${configName(runtimePath)} runtime` : "";
    byId("chartCaption").textContent = `${sampleCount} · ${seriesCount} · ${yScale}${runtime} · ${axisLabel(state.xMode)} · ${state.smoothGuides ? "PCHIP visual guide; markers are observed cases" : "straight segments between observed cases"}.${result.missingAxisCount ? ` ${result.missingAxisCount} cases omitted: missing or inapplicable horizontal-axis value.` : ""}`;
  }
}

function refreshChartSelection() {
  syncMetricControls();
  renderSummary();
  renderChart();
  renderSelectedCaseBar();
}

function syncMetricControls() {
  const group = state.activeMetricGroup;
  byId("metricGroupControls").hidden = !group;
  byId("peakWindowControl").hidden = group !== "peak";
  if (group) {
    state.activeMetric = group === "peak"
      ? peakMetricKey(state.peakMeasure, state.peakWindow)
      : state.groupMetrics[group];
    if (METRICS[state.activeMetric].cacheOnly && !state.rows.some(hasCacheRow)) {
      state.activeMetric = METRIC_GROUPS[group].defaultMetric;
    }
    const options = group === "peak"
      ? Object.entries(PEAK_MEASURES)
      : METRIC_GROUPS[group].metrics.map((key) => [key, METRICS[key].label]);
    byId("groupMetricSelect").replaceChildren(...options.map(([key, label]) => {
      const option = document.createElement("option");
      const metricKey = group === "peak" ? peakMetricKey(key, state.peakWindow) : key;
      option.value = key;
      option.disabled = Boolean(METRICS[metricKey].cacheOnly && !state.rows.some(hasCacheRow));
      option.textContent = `${label}${option.disabled ? " — requires Cache Replay" : ""}`;
      option.selected = key === (group === "peak" ? state.peakMeasure : state.activeMetric);
      return option;
    }));
    byId("peakWindowSelect").value = state.peakWindow;
  }
  for (const button of document.querySelectorAll(".metric-tab")) {
    const active = group ? button.dataset.metricGroup === group : button.dataset.metric === state.activeMetric;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
}

function syncXAxisControl() {
  const capacityAvailable = state.rows.some((row) => Number.isFinite(axisValue(row, "cache-capacity")));
  if (!capacityAvailable && state.curveXMode === "cache-capacity") {
    state.curveXMode = "hit-linear";
    state.hiddenSeries.clear();
    state.axisNotice = "Cache capacity requires Cache Replay results. The curve axis was reset to Prefix Ratio.";
  }
  state.xMode = state.viewMode === "curves" ? state.curveXMode : state.heatmapXMode;
  for (const option of byId("xModeSelect").options) {
    option.disabled = option.value === "cache-capacity"
      ? state.viewMode === "heatmap" || !capacityAvailable
      : option.value === "context-length" && state.viewMode === "heatmap";
  }
  byId("xModeSelect").value = state.xMode;
}

function renderTable() {
  renderResultsTable({
    container: byId("resultsTable"), caption: byId("tableCaption"), rows: state.rows,
    selectedId: state.selectedRow?.id, state: state.table,
    onSelect: (row) => selectRow(row, "table"),
    onStateChange: renderTable,
    thresholdMetric: state.thresholdsEnabled ? state.activeMetric : null,
    threshold: state.thresholdsEnabled ? activeThreshold() : NaN,
  });
}

function renderDetail() {
  byId("detailCaption").textContent = state.selectedRow ? `${state.selectedRow.id} · ${state.selectedRow.model}${state.selectedRow.encoder_state == null ? "" : ` · Encoder ${state.selectedRow.encoder_state} · output ${state.selectedRow.encoder_tail_policy}`}` : "Select a chart point, heatmap cell, or table row.";
  const cacheEnabled = Boolean(state.selectedRow?.case?.cache_replay_config);
  byId("detailCacheTab").hidden = !cacheEnabled;
  if (!cacheEnabled && state.detailTab === "cache") state.detailTab = "overview";
  const multiShard = Number(state.selectedRow?.case?.cache_replay_config?.shard_count || 1) !== 1;
  byId("runStorageButton").disabled = !state.selectedRow || !cacheEnabled || multiShard || state.isDirty || state.busy || Boolean(state.activeJobId);
  byId("runStorageButton").title = multiShard
    ? "SSD simulation requires cache shard_count=1."
    : "";
  renderCaseDetail({ container: byId("detailContent"), detail: state.caseDetail, activeTab: state.detailTab, onStatus: setStatus });
  const tab = document.querySelector(`[data-detail-tab="${state.detailTab}"]`);
  if (tab) byId("detailContent").setAttribute("aria-labelledby", tab.id);
}

async function selectRow(row, focusKind = null) {
  if (!row) return;
  state.selectedRow = row;
  ensureSelectedPage(state.rows, state.table, row.id);
  syncHeatmapSlice(row);
  renderSummary();
  renderChart();
  renderCache();
  renderTable();
  renderSelectedCaseBar();
  restoreSelectionFocus(row.id, focusKind);
  state.caseDetail = null;
  renderDetail();
  const token = ++state.requestToken;
  setStatus(`Loading ${row.id} details…`);
  try {
    const detail = await getCase(row.case, false);
    if (token !== state.requestToken) return;
    state.caseDetail = detail;
    renderDetail();
    setStatus(`${row.id} details loaded.`);
  } catch (error) {
    if (token === state.requestToken) setStatus(`Case detail failed: ${error.message}`, true);
  }
}

function renderSelectedCaseBar() {
  const bar = byId("selectedCaseBar");
  if (!state.selectedRow) {
    bar.hidden = true;
    return;
  }
  const row = state.selectedRow;
  const value = formatMetricForRow(row, state.activeMetric);
  const hitContext = row.metrics?.resident_prefix_ratio === undefined
    ? `assumed resident ${formatNumber(row.prefix_ratio * 100)}%`
    : `configured ${formatNumber(row.prefix_ratio * 100)}% · shareable ${formatNumber(row.metrics.shareable_prefix_ratio * 100)}% · resident ${formatNumber(row.metrics.resident_prefix_ratio * 100)}% · recovery ${formatNumber(row.metrics.recovery_boundary_ratio * 100)}%`;
  byId("selectedCaseSummary").textContent = `${row.id} · ${row.model}${row.encoder_state == null ? "" : ` · Encoder ${row.encoder_state} · output ${row.encoder_tail_policy}`} · seq ${formatCompact(row.seq_len)} · ${hitContext} · ${METRICS[state.activeMetric].shortLabel} ${value}`;
  bar.hidden = false;
}

function focusKindForElement(element) {
  if (element?.classList?.contains("heat-cell")) return "heatmap";
  if (element?.classList?.contains("point-hit-target")) return "curve";
  if (element?.classList?.contains("cache-hit-point") || element?.classList?.contains("cache-churn-point")) return "cache";
  return null;
}

function restoreSelectionFocus(caseId, focusKind) {
  if (!focusKind) return;
  const selectors = {
    curve: `.point-hit-target[data-case-id="${caseId}"]`,
    heatmap: `.heat-cell[data-case-id="${caseId}"]`,
    cache: `.cache-hit-point[data-case-id="${caseId}"]`,
    table: `.results-table tr[data-case-id="${caseId}"]`,
  };
  queueMicrotask(() => document.querySelector(selectors[focusKind])?.focus({ preventScroll: true }));
}

function focusSelectedTableRow() {
  if (!state.selectedRow) return;
  ensureSelectedPage(state.rows, state.table, state.selectedRow.id);
  renderTable();
  const row = document.querySelector(`.results-table tr[data-case-id="${state.selectedRow.id}"]`);
  row?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
  row?.focus({ preventScroll: true });
}

function setExploreResultView(view, { focus = false } = {}) {
  const cacheAvailable = hasCacheResults();
  const requested = view === "cache" && !cacheAvailable ? "charts" : view;
  const normalized = ["charts", "cache", "cases"].includes(requested)
    ? requested
    : "charts";
  state.resultView = normalized;
  for (const button of document.querySelectorAll("[data-result-view]")) {
    const active = button.dataset.resultView === normalized;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
  for (const panel of document.querySelectorAll("[data-result-panel]")) {
    panel.hidden = panel.dataset.resultPanel !== normalized;
  }
  if (focus) document.querySelector(`[data-result-view="${normalized}"]`)?.focus();
}

function syncExploreResultTabs() {
  const cacheAvailable = hasCacheResults();
  byId("exploreCacheTab").hidden = !cacheAvailable;
  if (!cacheAvailable && state.resultView === "cache") state.resultView = "charts";
  setExploreResultView(state.resultView);
}

function hasCacheResults() {
  return state.rows.some((row) => (
    row.analysis_mode === "cache_replay" || Boolean(row.case?.cache_replay_config)
  ));
}

function caseFormula(payload, total) {
  const axes = payload.axes || {};
  const terms = [
    [scenarioCaseCount(payload.model_configs || [], { encoder_state: axes.encoder_state || ["unavailable"], encoder_tail_policy: axes.encoder_tail_policy || ["recompute"] }, state.config.models), axes.encoder_state ? "model / state combinations" : "models"],
    [axes.seq_len?.length || 0, "context"],
    [axes.prefix_ratio?.length || 0, "prefix"],
    [axes.block_size_tokens?.length || 0, "block"],
    [axes.arrival_model?.length || 0, "arrival"],
    [axes.max_io_bytes?.length || 0, "max IO"],
  ];
  if (Array.isArray(axes.cache_total_capacity_bytes)) {
    terms.push([axes.cache_total_capacity_bytes.length, "capacity"]);
  }
  return `${terms.map(([count, label]) => `${count} ${label}`).join(" × ")} = ${total.toLocaleString()}`;
}

async function runStorageSimulation() {
  if (!state.selectedRow || state.isDirty || state.busy || state.activeJobId) return;
  const row = state.selectedRow;
  if (Number(row.case?.cache_replay_config?.shard_count || 1) !== 1) {
    setStatus("SSD simulation blocked: cache shard_count must be 1; multiple cache owners cannot be collapsed into one SSD.", true);
    return;
  }
  const token = ++state.requestToken;
  byId("runStorageButton").disabled = true;
  setStatus(`Validating and simulating SSD for ${row.id}…`);
  try {
    const detail = await getCase(row.case, true);
    if (token !== state.requestToken) return;
    state.caseDetail = detail;
    state.detailTab = "timing";
    document.querySelectorAll("[data-detail-tab]").forEach((item) => item.setAttribute("aria-selected", String(item.dataset.detailTab === "timing")));
    renderDetail();
    setStatus(`${row.id} SSD simulation complete.`);
  } catch (error) {
    if (token === state.requestToken) setStatus(`SSD simulation blocked: ${error.message}`, true);
  } finally {
    byId("runStorageButton").disabled = !state.selectedRow || state.selectedRow.analysis_mode !== "cache_replay" || state.isDirty || state.busy || Boolean(state.activeJobId);
  }
}

function syncHeatmapSlice(row) {
  if (!row) { state.heatmapSlice = null; populateHeatmapControls(); return; }
  state.heatmapSlice = {
    encoder_state: row.encoder_state ?? null, encoder_tail_policy: row.encoder_tail_policy ?? null,
    model: row.model, block_size_tokens: row.block_size_tokens,
    arrival_model: row.arrival_model, max_io_bytes: row.max_io_bytes,
    cache_total_capacity_bytes: row.metrics?.cache_total_capacity_bytes ?? null,
  };
  populateHeatmapControls();
}

function populateHeatmapControls() {
  const slice = state.heatmapSlice || emptySlice();
  const specs = [
    ["heatmapModelSelect", "model"],
    ["heatmapEncoderStateSelect", "encoder_state", "heatmapEncoderStateControl"],
    ["heatmapEncoderTailSelect", "encoder_tail_policy", "heatmapEncoderTailControl"],
    ["heatmapBlockSelect", "block_size_tokens"],
    ["heatmapArrivalSelect", "arrival_model"], ["heatmapIoSelect", "max_io_bytes"],
    ["heatmapCacheCapacitySelect", "cache_total_capacity_bytes", "heatmapCacheCapacityControl"],
  ];
  // Each field is scoped to the preceding choices. In particular, V4.1's
  // encoder scenarios must never remain active when switching to another model.
  let candidates = state.rows;
  for (const [id, key, controlId] of specs) {
    const valueFor = (row) => (key === "cache_total_capacity_bytes" ? row.metrics?.[key] : row[key]) ?? null;
    const values = [...new Set(candidates.map(valueFor))]
      .sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
    const selected = values.find((value) => String(value) === String(slice[key] ?? null));
    slice[key] = selected !== undefined ? selected : (values[0] ?? null);
    if (controlId) byId(controlId).hidden = !values.some((value) => value !== null);
    const select = byId(id);
    select.replaceChildren();
    for (const value of values) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = value === null ? "None"
        : key === "cache_total_capacity_bytes" ? `${formatGiBInput(value)} GiB`
          : String(value);
      option.selected = String(value) === String(slice[key]);
      select.append(option);
    }
    candidates = candidates.filter((row) => String(valueFor(row)) === String(slice[key]));
  }
  state.heatmapSlice = state.rows.length ? slice : null;
}

function readHeatmapSliceFromControls() {
  state.heatmapSlice = {
    encoder_state: byId("heatmapEncoderStateSelect").value === "null" ? null : byId("heatmapEncoderStateSelect").value,
    encoder_tail_policy: byId("heatmapEncoderTailSelect").value === "null" ? null : byId("heatmapEncoderTailSelect").value,
    model: byId("heatmapModelSelect").value,
    block_size_tokens: Number(byId("heatmapBlockSelect").value),
    arrival_model: byId("heatmapArrivalSelect").value,
    max_io_bytes: byId("heatmapIoSelect").value === "null" ? null : Number(byId("heatmapIoSelect").value),
    cache_total_capacity_bytes: byId("heatmapCacheCapacityControl").hidden
      ? null
      : Number(byId("heatmapCacheCapacitySelect").value),
  };
  populateHeatmapControls();
  renderChart();
}

function activeThreshold() {
  if (!state.config) return NaN;
  const storagePath = state.sweep?.request?.storage_config || byId("storageSelect").value;
  const storage = state.config.storages.find((item) => item.path === storagePath);
  return storageThreshold(state.activeMetric, storage);
}

function setBusy(busy, background = false) {
  state.busy = busy;
  byId("runButton").disabled = busy;
  byId("runButton").textContent = busy ? "Running…" : "Run Sweep";
  byId("cancelSweepButton").hidden = !(busy && background);
  byId("cancelSweepButton").disabled = false;
  if (!busy) updateEstimate();
  updateActionAvailability();
}

function setStatus(message, error = false) {
  const node = byId("statusText");
  node.textContent = message;
  node.classList.toggle("error", error);
}

function currentRuntimeOption() {
  return state.config?.runtimes.find((item) => item.path === byId("runtimeSelect").value);
}

function syncPresetIndicator() {
  if (!state.config) return;
  let axes = null;
  try { axes = formPayload().axes; }
  catch (_error) { axes = null; }
  let active = "custom";
  for (const name of ["quick_compare", "full_tail"]) {
    if (axes && stableSerialize(axes) === stableSerialize(state.config.presets?.[name]?.axes)) {
      active = name;
      break;
    }
  }
  state.activePreset = active;
  byId("quickPresetButton").classList.toggle("active", active === "quick_compare");
  byId("fullPresetButton").classList.toggle("active", active === "full_tail");
  byId("customPresetBadge").hidden = active !== "custom";
}

function updateResultFreshness() {
  let fingerprint = null;
  try { fingerprint = payloadFingerprint(formPayload()); }
  catch (_error) { fingerprint = null; }
  state.isDirty = Boolean(state.sweep && fingerprint !== state.resultFingerprint);
  const banner = byId("resultStateBanner");
  const messages = [];
  if (state.isDirty) {
    const request = state.sweep?.request || {};
    messages.push(`<strong>Showing previous results.</strong> ${escapeText(configName(request.runtime_config))} · ${escapeText(configName(request.workload_config))} · ${state.rows.length.toLocaleString()} cases. Controls changed; rerun to unlock export and SSD simulation.`);
  }
  const context = state.sweep?.decision_context;
  if (context) {
    const failedScenarios = Math.max(0, Number(context.scenario_count || 0) - Number(context.feasible_scenario_count || 0));
    const failureLabels = (context.constraint_failures || []).map((item) => item.constraint).filter(Boolean);
    const coverage = Number.isFinite(Number(context.coverage_ratio))
      ? `${formatNumber(Number(context.coverage_ratio) * 100)}%`
      : "unknown";
    messages.push(
      `<strong>Recommendation source · ${escapeText(context.candidate_id || "candidate")}.</strong> Coverage ${coverage} · ${failedScenarios.toLocaleString()} failed scenarios${failureLabels.length ? ` · ${escapeText(failureLabels.join(", "))}` : ""}.`,
    );
    if (!context.has_row_evaluations) {
      messages.push("<strong>Compatibility note.</strong> This older experiment has no per-case recommendation annotations; generic numeric risk is shown, but historical feasibility is not inferred.");
    }
  }
  for (const warning of state.sweep?.warnings || []) {
    messages.push(escapeText(warning));
  }
  if (!messages.length) {
    banner.hidden = true;
    banner.textContent = "";
  } else {
    banner.hidden = false;
    banner.innerHTML = messages.map((message) => `<div>${message}</div>`).join("");
  }
  updateActionAvailability();
}

function updateActionAvailability() {
  const locked = state.isDirty || state.busy || Boolean(state.activeJobId);
  const hasRows = state.rows.length > 0;
  const multiShard = Number(state.selectedRow?.case?.cache_replay_config?.shard_count || 1) !== 1;
  const cacheEnabled = state.selectedRow?.analysis_mode === "cache_replay";
  byId("exportJsonButton").disabled = !hasRows || locked;
  byId("exportCsvButton").disabled = !hasRows || locked;
  byId("runStorageButton").disabled = !state.selectedRow || !cacheEnabled || locked || multiShard;
}

function payloadFingerprint(payload) {
  if (!payload || typeof payload !== "object") return stableSerialize(payload);
  const comparable = { ...payload };
  // Digests are server-derived identity evidence. They deliberately affect the
  // persisted experiment fingerprint, but are not editable form state and must
  // not make a freshly returned result look stale in the browser.
  delete comparable.config_digests;
  return stableSerialize(comparable);
}

function stableSerialize(value) {
  if (Array.isArray(value)) return `[${value.map(stableSerialize).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableSerialize(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function storeActiveJob(jobId, payload) {
  try {
    sessionStorage.setItem(ACTIVE_JOB_KEY, jobId);
    sessionStorage.setItem(ACTIVE_JOB_PAYLOAD_KEY, JSON.stringify(payload));
  } catch (_error) {
    // Job polling still works in the current page when session storage is unavailable.
  }
}

function clearActiveJob() {
  state.activeJobId = null;
  state.jobPollToken += 1;
  try {
    sessionStorage.removeItem(ACTIVE_JOB_KEY);
    sessionStorage.removeItem(ACTIVE_JOB_PAYLOAD_KEY);
  } catch (_error) {
    // No-op when session storage is unavailable.
  }
}

async function restoreActiveJob() {
  let jobId = null;
  let storedPayload = null;
  try {
    jobId = sessionStorage.getItem(ACTIVE_JOB_KEY);
    storedPayload = JSON.parse(sessionStorage.getItem(ACTIVE_JOB_PAYLOAD_KEY) || "null");
  } catch (_error) {
    return;
  }
  if (!jobId) return;
  try {
    const job = await getSweepJob(jobId);
    const payload = job.request || storedPayload;
    if (!payload) throw new Error("The background job has no recoverable request.");
    state.activeJobId = jobId;
    applyPayloadToControls(payload);
    storeActiveJob(jobId, payload);
    setBusy(true, true);
    await pollSweepJob(jobId, payload);
  } catch (error) {
    clearActiveJob();
    setBusy(false);
    setStatus(error.status === 404 ? "Background sweep no longer exists." : `Sweep recovery failed: ${error.message}`, true);
  }
}

function applyPayloadToControls(payload) {
  byId("analysisModeSelect").value = payload.analysis_mode || (payload.cache_replay_config ? "cache_replay" : "bandwidth_envelope");
  byId("runtimeSelect").value = payload.runtime_config;
  applyComputeToControls(payload.compute_config);
  state.selectedModelsByRuntime.set(payload.runtime_config, new Set(payload.model_configs || []));
  renderModelList();
  applyWorkloadToControls(payload.workload_config, payload.workload_overrides);
  byId("storageSelect").value = payload.storage_config;
  const axes = payload.axes || {};
  applyEncoderScenarios(byId("encoderScenarios"), axes);
  byId("seqLensInput").value = (axes.seq_len || []).join(", ");
  byId("hitRatesInput").value = formatPrefixAxis(axes.prefix_ratio || [], state.exploreMode);
  byId("blockSizesInput").value = (axes.block_size_tokens || []).join(", ");
  byId("maxIoInput").value = (axes.max_io_bytes || []).map((value) => value ?? "").join(", ");
  byId("arrivalFixed").checked = (axes.arrival_model || []).includes("fixed");
  byId("arrivalPoisson").checked = (axes.arrival_model || []).includes("poisson");
  byId("writeMissBlocksInput").checked = Boolean(payload.write_miss_blocks);
  applyModelingOptionsToControls(
    payload.modeling_options || state.config.defaults.modeling_options || {},
  );
  const cache = payload.cache_replay_config;
  const cacheEnabled = byId("analysisModeSelect").value === "cache_replay";
  byId("cacheReplayToggle").checked = cacheEnabled;
  byId("cacheReplayFields").disabled = !cacheEnabled;
  if (cacheEnabled) {
    byId("cacheReplayDisclosure").open = true;
    byId("cacheConfigSelect").value = cache.source_path || state.config.defaults.cache_config;
    byId("cacheCapacityGiBInput").value = (
      axes.cache_total_capacity_bytes || [cache.total_capacity_bytes || 0]
    ).map(formatGiBInput).join(", ");
    byId("cacheShardCountInput").value = cache.shard_count ?? 1;
    byId("cachePlacementSeedInput").value = cache.placement_seed ?? 0;
    byId("cacheWarmupFractionInput").value = cache.warmup_fraction ?? 0;
    byId("cachePolicyValue").textContent = cache.policy || "radix_leaf_lru";
  } else {
    byId("cacheConfigSelect").value = state.config.defaults.cache_config;
    applyCachePreset();
  }
  applyAnalysisMode();
  handleDataChange();
}

function wireTabKeyboard(buttonsLike) {
  const buttons = [...buttonsLike];
  buttons.forEach((button, index) => {
    button.tabIndex = button.getAttribute("aria-selected") === "true" ? 0 : -1;
    button.addEventListener("keydown", (event) => {
      const available = buttons.filter((item) => !item.hidden && !item.disabled);
      const availableIndex = available.indexOf(button);
      let target = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") target = available[(availableIndex + 1) % available.length];
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") target = available[(availableIndex - 1 + available.length) % available.length];
      if (event.key === "Home") target = available[0];
      if (event.key === "End") target = available.at(-1);
      if (!target) return;
      event.preventDefault();
      buttons.forEach((item) => { item.tabIndex = item === target ? 0 : -1; });
      target.focus();
      target.click();
    });
    button.addEventListener("click", () => buttons.forEach((item) => { item.tabIndex = item === button ? 0 : -1; }));
  });
}

function configName(path) {
  return String(path || "unknown").split("/").at(-1)?.replace(/\.(yaml|yml|json)$/i, "") || "unknown";
}

function escapeText(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function parseNumberList(text, label, integer = false) {
  const values = String(text).split(",").map((value) => value.trim()).filter(Boolean).map((token) => {
    if (label === "prefix_ratio" && token.endsWith("%")) {
      return Number(token.slice(0, -1).trim()) / 100;
    }
    return Number(token);
  });
  if (
    !values.length
    || values.some((value) => !Number.isFinite(value) || (integer && !Number.isInteger(value)))
  ) {
    throw new Error(`${label} must contain valid comma-separated ${integer ? "integers" : "numbers"}.`);
  }
  return [...new Set(values)].sort((a, b) => a - b);
}

function parseOptionalIntList(text) {
  if (!String(text).trim()) return [null];
  const tokens = String(text).split(",").map((value) => value.trim());
  const values = tokens.map((value) => {
    if (!value || ["none", "null"].includes(value.toLowerCase())) return null;
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
      throw new Error("max_io_bytes must contain positive comma-separated integers or None.");
    }
    return parsed;
  });
  return [...new Set(values)].sort((a, b) => {
    if (a === null) return -1;
    if (b === null) return 1;
    return a - b;
  });
}

function requiredNumber(
  id,
  { integer = false, min = null, positive = false, label = id } = {},
) {
  const raw = byId(id).value.trim();
  const value = Number(raw);
  if (!raw || !Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new Error(`${label} must be ${integer ? "an integer" : "a number"}.`);
  }
  if (positive && value <= 0) throw new Error(`${label} must be positive.`);
  if (min != null && value < min) throw new Error(`${label} must be at least ${min}.`);
  return value;
}

function emptySlice() {
  return { model: "", block_size_tokens: "", arrival_model: "", max_io_bytes: null, cache_total_capacity_bytes: null };
}

function formatGiBInput(bytes) {
  const value = Number(bytes) / (1024 ** 3);
  return Number.isFinite(value) ? String(Number(value.toPrecision(12))) : "0";
}

function byId(id) { return document.getElementById(id); }
