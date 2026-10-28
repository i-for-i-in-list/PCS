const MIB = 1024 ** 2;
const GIB = 1024 ** 3;

const ENCODER_SCENARIOS = Object.freeze({
  encoder_state: [["unavailable", "Unavailable"], ["available", "Available"]],
  encoder_tail_policy: [["recompute", "Recompute when missing"], ["retain", "Retain and reuse"]],
});

export function initializeEncoderScenarios(container, onChange) {
  for (const [key, options] of Object.entries(ENCODER_SCENARIOS)) {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "encoder-scenario-field";
    const legend = document.createElement("legend");
    legend.textContent = key === "encoder_state" ? "Encoder state" : "Encoder tail output";
    fieldset.append(legend);
    options.forEach(([value, label], index) => {
      const item = document.createElement("label");
      item.className = "checkbox-line";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.dataset.encoderScenario = key;
      input.value = value;
      input.checked = index === 0;
      input.addEventListener("change", () => {
        if (!fieldset.querySelector("input:checked")) input.checked = true;
        onChange?.();
      });
      item.append(input, document.createTextNode(label));
      fieldset.append(item);
    });
    container.append(fieldset);
  }
  const note = document.createElement("p");
  note.className = "boundary-note";
  note.textContent = "V4.1 only · select one or both per group. Available assumes SWA and the required compression tail are ready at the hit boundary. Retain assumes a valid matching output snapshot is already available. These are fixed scenarios.";
  container.append(note);
}

export function readEncoderScenarios(container) {
  return Object.fromEntries(Object.entries(ENCODER_SCENARIOS).map(([key, options]) => [
    key, options.map(([value]) => value).filter((value) =>
      container.querySelector(`[data-encoder-scenario="${key}"][value="${value}"]`)?.checked),
  ]));
}

export function applyEncoderScenarios(container, values = {}) {
  for (const [key, options] of Object.entries(ENCODER_SCENARIOS)) {
    const chosen = values[key] ?? [options[0][0]];
    const selection = new Set(Array.isArray(chosen) ? chosen : [chosen]);
    container.querySelectorAll(`[data-encoder-scenario="${key}"]`).forEach((input) => {
      input.checked = selection.has(input.value);
    });
  }
}

export function supportsEncoderScenarios(models, paths) {
  return paths.some((path) => models.find((model) => model.path === path)?.recovery_capability?.encoder_scenarios_supported);
}

export function scenarioCaseCount(paths, axes, models) {
  const base = Object.entries(axes).filter(([key]) => !(key in ENCODER_SCENARIOS))
    .reduce((count, [, values]) => count * values.length, 1);
  const variants = (axes.encoder_state?.length || 1) * (axes.encoder_tail_policy?.length || 1);
  return base * paths.reduce((count, path) => count + (supportsEncoderScenarios(models, [path]) ? variants : 1), 0);
}

export function renderStatePolicyEditor(container, {
  models,
  selectedPaths,
  seqLengths,
  onChange,
  prefix = "statePolicy",
}) {
  const current = container.dataset.ready === "true"
    ? readStatePolicyEditor(container)
    : (container._policyValue || { state_policies: {}, model_state_policies: {} });
  container._policyValue = current;
  container.replaceChildren();

  const selected = selectedPaths.map((path) => models.find((item) => item.path === path)).filter(Boolean);
  const groups = new Map();
  for (const model of selected) {
    for (const target of model.recovery_capability?.state_policy_targets || []) {
      if (!groups.has(target.name)) groups.set(target.name, []);
      groups.get(target.name).push({ model, target });
    }
  }
  if (!groups.size) {
    const empty = document.createElement("div");
    empty.className = "boundary-note";
    empty.textContent = "Selected models expose no configurable checkpoint state.";
    container.append(empty);
    container.dataset.ready = "true";
    return;
  }

  const title = document.createElement("div");
  title.className = "state-policy-heading";
  title.innerHTML = "<strong>State group overrides</strong><small>Inherit the global scenario unless a group needs a different persistence policy.</small>";
  container.append(title);

  for (const [name, entries] of groups) {
    container.append(policyRow({
      prefix,
      kind: "group",
      name,
      label: `${name} · ${entries.length} model${entries.length === 1 ? "" : "s"}`,
      entries,
      value: current.state_policies?.[name],
      seqLengths,
    }));
  }

  const details = document.createElement("details");
  details.className = "state-policy-exceptions";
  const summary = document.createElement("summary");
  summary.innerHTML = "<span><strong>Model exceptions</strong><small>Optional per-model overrides and token-history conversions</small></span>";
  details.append(summary);
  for (const model of selected) {
    for (const target of model.recovery_capability?.state_policy_targets || []) {
      details.append(policyRow({
        prefix,
        kind: "model",
        name: target.name,
        modelPath: model.path,
        label: `${model.model_name || model.label} · ${target.name}`,
        entries: [{ model, target }],
        value: current.model_state_policies?.[model.path]?.[target.name],
        seqLengths,
      }));
    }
  }
  container.append(details);
  container.dataset.ready = "true";

  container.querySelectorAll("select,input").forEach((input) => {
    input.addEventListener("input", () => {
      syncRow(input.closest(".state-policy-row"), seqLengths);
      container._policyValue = readStatePolicyEditor(container);
      onChange?.();
    });
  });
}

export function readStatePolicyEditor(container) {
  const state_policies = {};
  const model_state_policies = {};
  for (const row of container.querySelectorAll(".state-policy-row")) {
    const policy = policyFromRow(row);
    if (!policy) continue;
    const name = row.dataset.target;
    if (row.dataset.kind === "group") state_policies[name] = policy;
    else {
      const path = row.dataset.modelPath;
      if (!model_state_policies[path]) model_state_policies[path] = {};
      model_state_policies[path][name] = policy;
    }
  }
  return { state_policies, model_state_policies };
}

export function applyStatePolicyEditor(container, options) {
  container._policyValue = {
    state_policies: structuredClone(options?.state_policies || {}),
    model_state_policies: structuredClone(options?.model_state_policies || {}),
  };
  container.dataset.ready = "false";
}

export function estimateBootstrapState(container, {
  models,
  selectedPaths,
  seqLengths,
  prefixRatios,
  prefixGroups,
  globalPolicy,
}) {
  const configured = readStatePolicyEditor(container);
  const prefixTokens = Math.floor(
    Math.max(0, ...(seqLengths || [0])) * Math.max(0, ...(prefixRatios || [0])),
  );
  let maximum = { bytes: 0, model: "", checkpoints: 0 };
  for (const path of selectedPaths) {
    const model = models.find((item) => item.path === path);
    if (!model) continue;
    let perGroupBytes = 0;
    let checkpoints = 0;
    for (const target of model.recovery_capability?.state_policy_targets || []) {
      const override = configured.model_state_policies?.[path]?.[target.name]
        ?? configured.state_policies?.[target.name];
      let policy = normalizePolicy(override || target.default_policy);
      if (!override && target.global_assumption_applicable && globalPolicy?.mode !== "model_default") {
        if (globalPolicy.mode === "periodic") policy = { mode: "periodic", interval: globalPolicy.interval, retain: globalPolicy.retain };
        else if (globalPolicy.mode === "none") policy = { mode: "none", interval: 2048, retain: false };
        else policy = { mode: "prompt_end", interval: 2048, retain: true };
      }
      const estimate = bytesForPolicy(policy, target, prefixTokens);
      perGroupBytes += estimate.bytes;
      checkpoints += estimate.checkpoints;
    }
    const total = perGroupBytes * Math.max(0, Number(prefixGroups) || 0);
    if (total > maximum.bytes) maximum = {
      bytes: total,
      model: model.model_name || model.label || path,
      checkpoints: checkpoints * Math.max(0, Number(prefixGroups) || 0),
    };
  }
  return maximum;
}

function policyRow({ prefix, kind, name, modelPath = "", label, entries, value, seqLengths }) {
  const row = document.createElement("div");
  row.className = "state-policy-row";
  row.dataset.kind = kind;
  row.dataset.target = name;
  row.dataset.modelPath = modelPath;
  row._entries = entries;

  const copy = document.createElement("div");
  copy.className = "state-policy-copy";
  const strong = document.createElement("strong");
  strong.textContent = label;
  const size = entries.map(({ target }) => Number(target.checkpoint_bytes) || 0);
  const defaults = [...new Set(entries.map(({ target }) => policyLabel(target.default_policy)))];
  const small = document.createElement("small");
  small.textContent = `${formatBytes(Math.min(...size))}${size.length > 1 && Math.max(...size) !== Math.min(...size) ? `–${formatBytes(Math.max(...size))}` : ""} / checkpoint · default ${defaults.join(" / ")}`;
  copy.append(strong, small);

  const controls = document.createElement("div");
  controls.className = "state-policy-controls";
  const mode = document.createElement("select");
  mode.className = "state-policy-mode";
  mode.id = `${prefix}-${kind}-${safeId(modelPath)}-${safeId(name)}`;
  const supported = new Set(entries.flatMap(({ target }) => target.supported_storage_forms || []));
  addOption(mode, "inherit", "Inherit");
  addOption(mode, "prompt_end", "Prompt end");
  addOption(mode, "periodic", "Periodic");
  addOption(mode, "none", "None");
  if (supported.has("token_history")) addOption(mode, "token_history", "Token history");
  const normalized = normalizePolicy(value);
  mode.value = normalized.mode;

  const interval = document.createElement("input");
  interval.className = "state-policy-interval";
  interval.type = "number";
  interval.min = "1";
  interval.step = "1";
  interval.value = String(normalized.interval || 2048);
  interval.setAttribute("aria-label", `${label} checkpoint interval`);

  const retainLabel = document.createElement("label");
  retainLabel.className = "checkbox-line state-policy-retain";
  const retain = document.createElement("input");
  retain.type = "checkbox";
  retain.checked = normalized.retain;
  retainLabel.append(retain, " prompt end");
  controls.append(mode, interval, retainLabel);

  const preview = document.createElement("div");
  preview.className = "state-policy-preview";
  row.append(copy, controls, preview);
  syncRow(row, seqLengths);
  return row;
}

function syncRow(row, seqLengths) {
  if (!row) return;
  const mode = row.querySelector(".state-policy-mode").value;
  const periodic = mode === "periodic";
  row.querySelector(".state-policy-interval").hidden = !periodic;
  row.querySelector(".state-policy-retain").hidden = !periodic;
  const preview = row.querySelector(".state-policy-preview");
  if (mode === "inherit") {
    preview.textContent = "Uses the next policy level.";
    return;
  }
  const values = [];
  for (const { target } of row._entries || []) {
    for (const tokens of seqLengths?.length ? seqLengths : [32768]) {
      values.push(policyBytes(row, target, Number(tokens) || 0));
    }
  }
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const conversion = (row._entries || []).some(({ target }) => (
    target.default_policy?.storage_form === "token_history" && mode !== "token_history"
  ));
  preview.textContent = `${conversion ? "Storage-form conversion · " : ""}${formatBytes(minimum)}${maximum !== minimum ? `–${formatBytes(maximum)}` : ""} state / sequence`;
  preview.classList.toggle("risk", conversion);
}

function policyBytes(row, target, tokens) {
  const mode = row.querySelector(".state-policy-mode").value;
  if (mode === "none" || mode === "inherit") return 0;
  if (mode === "token_history") return tokens * Number(target.token_history_bytes_per_token || 0);
  const checkpoint = Number(target.checkpoint_bytes || 0);
  if (mode === "prompt_end") return tokens > 0 ? checkpoint : 0;
  const interval = Math.max(1, Number(row.querySelector(".state-policy-interval").value) || 1);
  const retain = row.querySelector(".state-policy-retain input").checked;
  const regular = Math.floor(tokens / interval);
  const extra = retain && tokens % interval !== 0 ? 1 : 0;
  return (regular + extra) * checkpoint;
}

function bytesForPolicy(policy, target, tokens) {
  if (policy.mode === "none" || policy.mode === "inherit") return { bytes: 0, checkpoints: 0 };
  if (policy.mode === "token_history") return {
    bytes: tokens * Number(target.token_history_bytes_per_token || 0),
    checkpoints: 0,
  };
  const checkpoint = Number(target.checkpoint_bytes || 0);
  if (policy.mode === "prompt_end") return {
    bytes: tokens > 0 ? checkpoint : 0,
    checkpoints: tokens > 0 ? 1 : 0,
  };
  const interval = Math.max(1, Number(policy.interval) || 1);
  const count = Math.floor(tokens / interval) + (policy.retain && tokens % interval !== 0 ? 1 : 0);
  return { bytes: count * checkpoint, checkpoints: count };
}

function policyFromRow(row) {
  const mode = row.querySelector(".state-policy-mode").value;
  if (mode === "inherit") return null;
  if (mode === "prompt_end") return "prompt_end";
  if (mode === "none") return "none";
  if (mode === "token_history") return "token_history";
  const interval = Number(row.querySelector(".state-policy-interval").value);
  if (!Number.isInteger(interval) || interval <= 0) throw new Error(`${row.dataset.target} interval must be a positive integer.`);
  return `${row.querySelector(".state-policy-retain input").checked ? "periodic+prompt_end" : "periodic"}:${interval}`;
}

function normalizePolicy(value) {
  if (!value) return { mode: "inherit", interval: 2048, retain: false };
  if (typeof value === "string") {
    if (value.startsWith("periodic+prompt_end:")) return { mode: "periodic", interval: Number(value.split(":")[1]), retain: true };
    if (value.startsWith("periodic:")) return { mode: "periodic", interval: Number(value.split(":")[1]), retain: false };
    if (["terminal", "terminal_only", "prompt_end"].includes(value)) return { mode: "prompt_end", interval: 2048, retain: true };
    if (["none", "zero"].includes(value)) return { mode: "none", interval: 2048, retain: false };
    if (value === "token_history") return { mode: "token_history", interval: 2048, retain: false };
  }
  const storage = value.storage_form;
  if (storage === "none") return { mode: "none", interval: 2048, retain: false };
  if (storage === "token_history") return { mode: "token_history", interval: 2048, retain: false };
  if (value.interval_tokens == null) return { mode: "prompt_end", interval: 2048, retain: true };
  return { mode: "periodic", interval: Number(value.interval_tokens), retain: Boolean(value.retain_prompt_end) };
}

function policyLabel(value) {
  const policy = normalizePolicy(value);
  if (policy.mode === "periodic") return `${policy.retain ? "periodic+end" : "periodic"}:${policy.interval}`;
  return policy.mode.replaceAll("_", " ");
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes >= GIB) return `${trim(bytes / GIB)} GiB`;
  if (bytes >= MIB) return `${trim(bytes / MIB)} MiB`;
  if (bytes >= 1024) return `${trim(bytes / 1024)} KiB`;
  return `${trim(bytes)} B`;
}

function trim(value) { return Number(value.toPrecision(4)).toLocaleString(); }
function safeId(value) { return String(value || "all").replace(/[^a-z0-9]+/gi, "-"); }
function addOption(select, value, label) { select.append(new Option(label, value)); }
