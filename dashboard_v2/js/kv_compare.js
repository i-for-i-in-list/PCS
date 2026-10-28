export const KV_GROUPS = Object.freeze([
  { id: "kv", label: "Main model KV" },
  { id: "state", label: "Main model State" },
  { id: "indexer", label: "Main model Indexer" },
  { id: "draft", label: "Draft" },
  { id: "other", label: "Other" },
]);

const KV_COMPONENTS = new Set(["kv", "mla", "global_kv", "sliding_kv", "full_kv", "main_kv", "encoder_main_kv", "decoder_main_kv", "encoder_swa", "decoder_swa"]);
const STATE_COMPONENTS = new Set(["kda_state", "linear_state", "sconv_state", "compression_tail", "encoder_outputs"]);
const INDEXER_COMPONENTS = new Set(["indexer", "encoder_indexer", "decoder_indexer"]);
const FORMULA_COMPONENTS = {
  standard_gqa: ["kv"], mla: ["mla"], dsa_mla: ["mla", "indexer"],
  kimi_kda_mla_hybrid: ["mla", "kda_state"],
  inkling_hybrid: ["global_kv", "sliding_kv", "sconv_state"],
  qwen_linear_full_hybrid: ["full_kv", "linear_state"],
  mixed_full_sliding_gqa: ["full_kv", "sliding_kv"],
  minimax_msa: ["main_kv", "indexer"], deepseek_v4_hybrid: ["main_kv", "indexer"],
  deepseek_v41_csa2: ["encoder_main_kv", "decoder_main_kv", "encoder_indexer", "decoder_indexer", "encoder_swa", "decoder_swa", "compression_tail", "encoder_outputs", "draft_swa"],
};

export function configurationKey(value) {
  const canonical = item => {
    if (typeof item === "bigint") return String(item);
    if (typeof item === "number" && Number.isInteger(item)) return String(item);
    if (typeof item === "string") {
      const text = item.trim();
      return /^\d+$/.test(text) ? String(BigInt(text)) : text;
    }
    if (Array.isArray(item)) return item.map(canonical);
    if (item && typeof item === "object") return Object.fromEntries(Object.keys(item).sort().map(key => [key, canonical(item[key])]));
    return item;
  };
  return JSON.stringify(canonical(value));
}

export function aggregateKvGroups(result) {
  const groups = Object.fromEntries(KV_GROUPS.map(group => [group.id, 0n]));
  for (const layer of result.layers) {
    for (const [component, bytes] of Object.entries(layer.byComponent)) {
      const group = layer.draft ? "draft" : KV_COMPONENTS.has(component) ? "kv"
        : STATE_COMPONENTS.has(component) ? "state" : INDEXER_COMPONENTS.has(component) ? "indexer" : "other";
      groups[group] += bytes;
    }
  }
  if (Object.values(groups).reduce((sum, bytes) => sum + bytes, 0n) !== result.totalBytes) {
    throw new Error("Comparison groups do not match the calculated total.");
  }
  return groups;
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

function nameOf(snapshot) {
  return snapshot.model.model_name || snapshot.model.label || snapshot.model.model_id;
}

function checkpointLabel(snapshot, integer) {
  if (!snapshot.applicability.checkpoint) return "—";
  if (!snapshot.settings.includeState) return "Off";
  return snapshot.settings.checkpointInterval === null ? "Prompt-end snapshot"
    : `Every ${integer(snapshot.settings.checkpointInterval)} tokens`;
}

function componentValue(snapshot, component) {
  if (component in snapshot.result.components) return snapshot.result.components[component];
  if (!(FORMULA_COMPONENTS[snapshot.result.formula] || []).includes(component)) return undefined;
  if ((STATE_COMPONENTS.has(component) || ["encoder_swa", "decoder_swa"].includes(component)) && !snapshot.settings.includeState) return "Off";
  if (component === "encoder_outputs" && snapshot.settings.encoderTailPolicy !== "retain") return "Off";
  if (component === "draft_swa" && !snapshot.settings.includeDraft) return "Off";
  if (INDEXER_COMPONENTS.has(component) && !snapshot.applicability.indexer) return undefined;
  return 0n;
}

export function initializeKvCompare({ capture, format, componentLabels }) {
  const byId = id => document.getElementById(id);
  let snapshots = [];
  let nextId = 1;
  let comparing = false;
  let inputError = null;

  const announce = message => { byId("kvCompareStatus").textContent = message; };
  function updateInput(error) {
    inputError = error;
    const reason = error || (snapshots.length >= 4 ? "Comparison is limited to 4 configurations. Remove one to add another." : "");
    byId("kvAddCompare").disabled = Boolean(reason);
    byId("kvAddCompareReason").textContent = reason;
    byId("kvAddCompareReason").hidden = !reason;
  }
  function showCalculator(focus = true) {
    comparing = false;
    byId("kvCompareView").hidden = true;
    byId("kvCalculator").hidden = false;
    if (focus) (byId("kvAddCompare").disabled ? byId("kvModel") : byId("kvAddCompare")).focus();
  }
  function showComparison() {
    if (snapshots.length < 2) return;
    comparing = true;
    renderComparison();
    byId("kvCalculator").hidden = true;
    byId("kvCompareView").hidden = false;
    byId("kvCompareHeading").focus();
  }
  function renderList() {
    byId("kvCompareTray").hidden = snapshots.length === 0;
    byId("kvCompareCount").textContent = `${snapshots.length} / 4 configurations`;
    byId("kvRunCompare").disabled = snapshots.length < 2;
    byId("kvCompareCards").replaceChildren();
    snapshots.forEach((snapshot, index) => {
      const card = element("article", "kv-compare-card");
      card.dataset.snapshotId = snapshot.id;
      card.dataset.totalBytes = String(snapshot.result.totalBytes);
      const heading = element("div", "kv-compare-card-heading");
      heading.append(element("h3", "", `#${snapshot.id} · ${nameOf(snapshot)}`));
      const remove = element("button", "kv-compare-remove", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove configuration ${snapshot.id}: ${nameOf(snapshot)}`);
      remove.addEventListener("click", () => {
        snapshots = snapshots.filter(item => item.id !== snapshot.id);
        renderList();
        announce(`Removed configuration ${snapshot.id}. ${snapshots.length} remaining.`);
        if (snapshots.length < 2 && comparing) showCalculator();
        else {
          if (comparing) renderComparison();
          const buttons = byId("kvCompareCards").querySelectorAll("button");
          (buttons[Math.min(index, buttons.length - 1)] || (byId("kvAddCompare").disabled ? byId("kvModel") : byId("kvAddCompare"))).focus();
        }
      });
      heading.append(remove);
      card.append(heading,
        element("p", "", `${format.integer(snapshot.settings.tokens)} tokens · ${format.integer(snapshot.settings.sequences)} sequence(s)`),
        element("p", "", snapshot.explanations.kvPrecision));
      if (snapshot.applicability.checkpoint) card.append(element("p", "", checkpointLabel(snapshot, format.integer)));
      card.append(element("strong", "kv-compare-card-total", format.bytes(snapshot.result.totalBytes)));
      if (snapshot.custom) card.append(element("span", "kv-custom-badge", "Custom parameters"));
      byId("kvCompareCards").append(card);
    });
    updateInput(inputError);
  }

  function renderChart() {
    const chart = byId("kvCompareChart");
    chart.replaceChildren();
    const max = snapshots.reduce((value, item) => item.result.totalBytes > value ? item.result.totalBytes : value, 0n);
    const legend = element("div", "kv-compare-legend");
    for (const group of KV_GROUPS) {
      const item = element("span");
      item.append(element("i", `kv-group-${group.id}`), document.createTextNode(group.label));
      legend.append(item);
    }
    chart.append(legend);
    for (const snapshot of snapshots) {
      const row = element("div", "kv-compare-bar-row");
      row.dataset.snapshotId = snapshot.id;
      row.append(element("strong", "kv-compare-bar-label", `#${snapshot.id} · ${nameOf(snapshot)}`));
      const track = element("div", "kv-compare-bar-track");
      track.setAttribute("role", "img");
      track.setAttribute("aria-label", KV_GROUPS.map(group => `${group.label}: ${format.bytes(snapshot.groups[group.id])}`).join("; "));
      for (const group of KV_GROUPS) {
        const bytes = snapshot.groups[group.id];
        const segment = element("span", `kv-compare-segment kv-group-${group.id}`);
        segment.dataset.group = group.id;
        segment.dataset.bytes = String(bytes);
        // The Number conversion is bounded by 100; arbitrarily large totals
        // never enter floating point arithmetic.
        segment.style.width = `${max === 0n ? 0 : Number(bytes * 1000000n / max) / 10000}%`;
        segment.title = `${group.label}: ${format.bytes(bytes)} (${format.integer(bytes)} bytes)`;
        track.append(segment);
      }
      row.append(track, element("strong", "kv-compare-bar-total", format.bytes(snapshot.result.totalBytes)));
      chart.append(row);
    }
    const axis = element("div", "kv-compare-axis");
    for (const fraction of [0n, 1n, 2n, 3n, 4n]) axis.append(element("span", "", format.bytes(max * fraction / 4n)));
    chart.append(axis);
  }

  function table(rows, label) {
    const scroll = element("div", "kv-compare-table-scroll");
    scroll.tabIndex = 0;
    scroll.setAttribute("role", "region");
    scroll.setAttribute("aria-label", label);
    const node = element("table", "kv-compare-table");
    node.style.minWidth = `${210 + snapshots.length * 235}px`;
    const head = element("thead");
    const headings = element("tr");
    const corner = element("th", "", label);
    corner.scope = "col";
    headings.append(corner);
    for (const snapshot of snapshots) {
      const th = element("th", "", `#${snapshot.id} · ${nameOf(snapshot)}`);
      th.scope = "col";
      headings.append(th);
    }
    head.append(headings);
    const body = element("tbody");
    for (const row of rows) {
      const tr = element("tr");
      tr.dataset.compareRow = row.key;
      if (row.section) {
        const th = element("th", "kv-compare-section", row.label);
        th.colSpan = snapshots.length + 1;
        th.scope = "colgroup";
        tr.append(th);
      } else {
        const values = snapshots.map(row.get);
        const different = row.parameter && new Set(values.map(configurationKey)).size > 1;
        if (different) tr.classList.add("kv-parameter-different");
        const th = element("th", "", row.label);
        th.scope = "row";
        if (different) th.append(element("small", "kv-difference-label", "Different"));
        tr.append(th);
        values.forEach((value, index) => {
          const display = value === undefined || value === null ? "—" : row.render ? row.render(value, snapshots[index]) : String(value);
          const td = element("td", "", display);
          if (row.bytes && typeof value === "bigint") {
            td.dataset.bytes = String(value);
            td.title = `${format.integer(value)} bytes exact`;
          }
          tr.append(td);
        });
      }
      body.append(tr);
    }
    node.append(head, body);
    scroll.append(node);
    return scroll;
  }

  function renderComparison() {
    renderChart();
    const section = (key, label) => ({ key, label, section: true });
    const bytes = (key, label, get) => ({ key, label, get, bytes: true, render: v => typeof v === "bigint" ? format.bytes(v) : v });
    const parameter = (key, label, get, render = format.field) => ({ key, label, get, render, parameter: true });
    const rows = [
      section("capacity", "Capacity summary"),
      bytes("total", "Total nominal cache", s => s.result.totalBytes),
      { key: "per_sequence", label: "Per sequence", get: s => s, render: s => format.ratio(s.result.totalBytes, s.settings.sequences) },
      { key: "per_token", label: "Average per token / sequence", get: s => s, render: s => format.ratio(s.result.totalBytes, s.settings.tokens * s.settings.sequences) },
      section("conditions", "Comparison settings"),
      parameter("tokens", "Tokens per sequence", s => s.settings.tokens, format.integer),
      parameter("sequences", "Sequences", s => s.settings.sequences, format.integer),
      parameter("precision", "KV precision / layout", s => s.explanations.kvPrecision),
      parameter("indexer_precision", "Indexer precision / layout", s => s.applicability.indexer ? s.explanations.indexerPrecision : undefined),
      parameter("include_state", "Attention state", s => s.applicability.state ? s.settings.includeState : undefined, v => v ? "On" : "Off"),
      parameter("include_draft", "Draft / MTP cache", s => s.applicability.draft ? s.settings.includeDraft : undefined, v => v ? "On" : "Off"),
      parameter("checkpoint", "State checkpoint policy", s => s.applicability.checkpoint ? s.settings.includeState ? s.settings.checkpointInterval === null ? "Prompt-end snapshot" : "Fixed interval" : "Off" : undefined),
      parameter("checkpoint_interval", "Checkpoint interval (tokens)", s => s.applicability.checkpoint && s.settings.includeState ? s.settings.checkpointInterval : undefined, format.integer),
      parameter("encoder_tail", "Encoder tail output", s => s.applicability.encoderTail ? s.settings.includeState ? s.settings.encoderTailPolicy : "Off" : undefined),
      section("groups", "Chart group capacity"),
    ];
    for (const group of KV_GROUPS) rows.push(bytes(`group_${group.id}`, group.label, s => {
      if (["state", "draft"].includes(group.id)) return s.applicability[group.id]
        ? s.settings[group.id === "state" ? "includeState" : "includeDraft"] ? s.groups[group.id] : "Off" : undefined;
      if (group.id === "indexer" && !s.applicability.indexer) return undefined;
      if (group.id === "other" && !s.result.layers.some(layer => !layer.draft && Object.keys(layer.byComponent).some(key => !KV_COMPONENTS.has(key) && !STATE_COMPONENTS.has(key) && !INDEXER_COMPONENTS.has(key)))) return undefined;
      return s.groups[group.id];
    }));
    rows.push(section("components", "Original components · may include Draft; breakdowns are not additive"));
    const components = [...new Set(snapshots.flatMap(s => [...Object.keys(s.result.components), ...(FORMULA_COMPONENTS[s.result.formula] || [])]))];
    for (const component of components) rows.push(bytes(`component_${component}`, componentLabels[component] || format.fieldLabel(component), s => componentValue(s, component)));
    byId("kvCompareTables").replaceChildren(table(rows, "Configuration comparison"));

    const advanced = byId("kvCompareAdvanced");
    const facts = [
      parameter("model", "Model", s => nameOf(s)),
      parameter("family", "Family", s => s.model.family),
      parameter("formula", "Formula", s => s.result.formula),
      parameter("semantic_class", "Semantic class", s => s.explanations.semanticClass),
      parameter("context_limit", "Configured context limit", s => s.model.kv_calculator.max_context_tokens),
      parameter("draft_support", "Draft support", s => s.model.kv_calculator.draft_support),
    ];
    const keys = [...new Set(snapshots.flatMap(s => Object.keys(s.fields)))].sort();
    for (const key of keys) facts.push(parameter(`field_${key}`, format.fieldLabel(key), s => {
      const value = s.fields[key];
      return key === "compress_ratios" && typeof value === "string" ? value.split(/[\s,]+/).filter(Boolean) : value;
    }));
    advanced.replaceChildren(table(facts, "Model and advanced parameters"));

    const notes = byId("kvCompareNotes");
    notes.replaceChildren();
    for (const snapshot of snapshots) {
      const article = element("article", "kv-compare-note");
      article.append(element("h3", "", `#${snapshot.id} · ${nameOf(snapshot)}`));
      if (snapshot.explanations.contextWarning) article.append(element("p", "boundary-note", snapshot.explanations.contextWarning));
      article.append(element("p", "", snapshot.explanations.scope), element("p", "", `Semantic class: ${snapshot.explanations.semanticClass}`));
      notes.append(article);
    }
  }

  byId("kvAddCompare").addEventListener("click", () => {
    if (snapshots.length >= 4) return;
    try {
      const snapshot = structuredClone(capture());
      const key = configurationKey(snapshot.effective);
      const existing = snapshots.find(item => item.key === key);
      if (existing) {
        announce(`Already added · configuration ${existing.id}.`);
        return;
      }
      snapshot.key = key;
      snapshot.id = nextId++;
      snapshots.push(snapshot);
      renderList();
      announce(`Added configuration ${snapshot.id}: ${nameOf(snapshot)}. ${snapshots.length} of 4.`);
    } catch (error) {
      updateInput(error.message || String(error));
      announce(error.message || String(error));
    }
  });
  byId("kvRunCompare").addEventListener("click", showComparison);
  byId("kvBackCalculator").addEventListener("click", () => showCalculator());
  byId("kvClearCompare").addEventListener("click", () => {
    snapshots = [];
    renderList();
    showCalculator();
    announce("Comparison cleared.");
  });
  renderList();
  return { updateInput };
}
