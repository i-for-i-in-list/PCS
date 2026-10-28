import { copyText, downloadJson } from "./api.js";
import {
  classifyValue, escapeHtml, formatCompact, formatMetric, formatMetricForRow,
  formatNumber, VALUE_STATES,
} from "./metrics.js";
import {
  recoveryOutcome, resultSemantics, semanticMetricExplanation, ttftSavingsOutcome,
} from "./result_semantics.js";

const COMPONENT_COLORS = Object.freeze({
  KV: "var(--chart-series-0, #2868c7)", RoPE: "var(--chart-series-3, #7155bd)", Indexer: "var(--chart-series-2, #c16a00)", SWA: "var(--chart-series-1, #087568)",
  CSA: "var(--chart-series-5, #2c8ea6)", HCA: "var(--chart-series-4, #c33d63)", Other: "var(--chart-series-11, #697684)",
});

export function renderCaseDetail({ container, detail, activeTab, onStatus }) {
  container.replaceChildren();
  if (!detail) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Select a chart point, heatmap cell, or table row to inspect a Case.";
    container.append(empty);
    return;
  }
  const evaluation = detail.demand_report?.evaluation || detail.row?.evaluation || {};
  if (["not_evaluated", "unsupported"].includes(evaluation.status)) {
    const notice = document.createElement("div");
    notice.className = "empty-state";
    notice.textContent = evaluation.message || "Complete first-token path not evaluated: final hidden/logits or output-head work is not modeled. Known suffix work and restore bytes remain available.";
    container.append(notice);
    if (evaluation.aborted && activeTab !== "raw") return;
    if (activeTab === "timing") return;
  }
  if (activeTab === "recovery") renderRecovery(container, detail);
  else if (activeTab === "components") renderComponents(container, detail);
  else if (activeTab === "timing") renderTiming(container, detail);
  else if (activeTab === "cache") renderCache(container, detail);
  else if (activeTab === "raw") renderRaw(container, detail, onStatus);
  else renderOverview(container, detail);
}

function renderCache(container, detail) {
  const cache = detail.demand_report?.cache_replay;
  if (!cache) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "This Case was analyzed without capacity-aware prefix-cache replay.";
    container.append(empty);
    return;
  }
  const hit = cache.hit_rate || {};
  const capacity = cache.capacity || {};
  const admission = cache.admission || {};
  const eviction = cache.eviction || {};
  const tree = cache.tree || {};
  const requests = cache.requests || {};
  const phases = cache.phases || {};
  const lifecycle = cache.lifecycle || {};

  container.append(detailSectionTitle("Cache lifecycle"));
  const lifecycleGrid = document.createElement("div");
  lifecycleGrid.className = "detail-grid cache-detail-grid";
  for (const [label, value] of [
    ["Bootstrap requests", requests.bootstrap],
    ["Warmup requests", requests.warmup],
    ["Measured requests", requests.measured],
    ["Total requests", requests.total],
  ]) lifecycleGrid.append(metricCard(label, value, ""));
  container.append(lifecycleGrid);

  if (Object.keys(phases).length) {
    const phaseGrid = document.createElement("div");
    phaseGrid.className = "detail-grid cache-detail-grid";
    for (const name of ["bootstrap", "warmup", "measured"]) {
      const phase = phases[name] || {};
      const phaseHit = phase.hit_rate || {};
      const phaseAdmission = phase.admission || {};
      const phaseEviction = phase.eviction || {};
      const card = document.createElement("div");
      card.className = "detail-metric";
      card.innerHTML = `<span>${escapeHtml(name)}</span><strong>${formatNumber(Number(phaseHit.actual_block || 0) * 100)}% block hit</strong><small>${formatBytes(phaseAdmission.inserted_bytes)} inserted · ${formatBytes(phaseEviction.bytes)} evicted</small>`;
      phaseGrid.append(card);
    }
    container.append(phaseGrid);
  }

  if (Object.keys(lifecycle).length) {
    const snapshotGrid = document.createElement("div");
    snapshotGrid.className = "detail-grid cache-detail-grid";
    for (const name of ["cold_initial", "bootstrap_drained", "measurement_start", "measured_peak", "final_drained"]) {
      const snapshot = lifecycle[name];
      if (!snapshot) continue;
      const card = byteMetricCard(name.replaceAll("_", " "), snapshot.used_bytes);
      const small = document.createElement("small");
      small.textContent = `${formatBytes(snapshot.checkpoint_attachment_bytes)} checkpoints`;
      card.append(small);
      snapshotGrid.append(card);
    }
    container.append(snapshotGrid);
  }

  container.append(detailSectionTitle("Configured, shareable and resident reuse"));
  const hitGrid = document.createElement("div");
  hitGrid.className = "detail-grid cache-detail-grid";
  const hits = [
    ["Configured prefix", hit.configured_prefix],
    ["Object-shareable prefix", hit.shareable_prefix],
    ["Resident prefix", hit.resident_prefix],
    ["Candidate block hit", hit.candidate_block],
    ["Actual block hit", hit.actual_block],
    ["Candidate byte hit", hit.candidate_byte],
    ["Actual byte hit", hit.actual_byte],
  ];
  for (const [label, value] of hits) hitGrid.append(metricCard(label, value, "ratio"));
  container.append(hitGrid);

  container.append(detailSectionTitle("Residency and protection"));
  const residencyGrid = document.createElement("div");
  residencyGrid.className = "detail-grid cache-detail-grid";
  const residency = [
    ["Configured capacity", capacity.total_capacity_bytes],
    ["Initial used", capacity.initial_used_bytes],
    ["Peak used", capacity.peak_used_bytes],
    ["Final used", capacity.final_used_bytes],
    ["Peak evictable", capacity.peak_evictable_bytes],
    ["Final evictable", capacity.final_evictable_bytes],
    ["Peak protected", capacity.peak_protected_bytes],
    ["Final protected", capacity.final_protected_bytes],
  ];
  for (const [label, value] of residency) residencyGrid.append(byteMetricCard(label, value));
  container.append(residencyGrid);

  container.append(detailSectionTitle("Admission, eviction and radix tree"));
  const activityGrid = document.createElement("div");
  activityGrid.className = "detail-grid cache-detail-grid";
  const activity = [
    ["Inserted", admission.inserted_blocks, admission.inserted_bytes],
    ["Deduplicated", admission.deduplicated_insert_blocks, admission.deduplicated_insert_bytes],
    ["Rejected", admission.rejected_blocks, admission.rejected_bytes],
    ["Evicted", eviction.blocks, eviction.bytes],
    ["Self-evicted", eviction.self_eviction_blocks, eviction.self_eviction_bytes],
  ];
  for (const [label, blocks, bytes] of activity) {
    const card = byteMetricCard(label, bytes);
    const small = document.createElement("small");
    small.textContent = `${Number(blocks || 0).toLocaleString()} blocks`;
    card.append(small);
    activityGrid.append(card);
  }
  activityGrid.append(metricCard("Radix nodes", tree.node_count, ""));
  activityGrid.append(metricCard("Leaves", tree.leaf_count, ""));
  activityGrid.append(metricCard("Max depth", tree.max_depth, ""));
  container.append(activityGrid);

  const shardTitle = detailSectionTitle("Shard residency");
  container.append(shardTitle);
  const scroll = document.createElement("div");
  scroll.className = "table-scroll cache-shard-scroll";
  const table = document.createElement("table");
  table.className = "results-table cache-shard-table";
  table.setAttribute("aria-label", "Cache shard residency and radix tree state");
  const columns = [
    ["Shard", "shard", (value) => String(value)],
    ["Capacity", "capacity_bytes", formatBytes],
    ["Final used", "used_bytes", formatBytes],
    ["Evictable", "evictable_bytes", formatBytes],
    ["Protected", "protected_bytes", formatBytes],
    ["Nodes", "node_count", (value) => Number(value || 0).toLocaleString()],
    ["Leaves", "leaf_count", (value) => Number(value || 0).toLocaleString()],
    ["Max depth", "max_depth", (value) => Number(value || 0).toLocaleString()],
  ];
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const [label] of columns) {
    const cell = document.createElement("th");
    cell.textContent = label;
    headRow.append(cell);
  }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const shard of capacity.shards || []) {
    const row = document.createElement("tr");
    for (const [, key, formatter] of columns) {
      const cell = document.createElement("td");
      cell.textContent = formatter(shard[key]);
      row.append(cell);
    }
    body.append(row);
  }
  table.append(head, body);
  scroll.append(table);
  container.append(scroll);

  const note = document.createElement("div");
  note.className = "boundary-note cache-semantics-note";
  note.innerHTML = "<strong>Eviction byte semantics</strong>Eviction bytes are logical serialized cache capacity released by leaf-LRU. They are not SSD delete, trim, garbage-collection or write-amplification IO.";
  container.append(note);
}

function renderRecovery(container, detail) {
  const recovery = detail.demand_report?.recovery || {};
  const phased = recovery.execution_scope === "phased";
  if (phased) renderExecutionPhases(container, detail, false);
  const funnel = recovery.funnel || {};
  const ratios = recovery.ratios || {};
  const bytes = recovery.bytes || {};
  const compute = recovery.compute || {};
  const effectivePolicies = detail.demand_report?.effective_state_policies || {};
  if (Object.keys(effectivePolicies).length) {
    container.append(detailSectionTitle("Effective checkpoint policies"));
    const policyGrid = document.createElement("div");
    policyGrid.className = "detail-grid";
    for (const [name, value] of Object.entries(effectivePolicies)) {
      const policy = value.policy || {};
      const cadence = policy.storage_form === "snapshot"
        ? (policy.interval_tokens == null ? "prompt end" : `every ${Number(policy.interval_tokens).toLocaleString()} tokens${policy.retain_prompt_end ? " + end" : ""}`)
        : policy.storage_form;
      const card = document.createElement("div");
      card.className = "detail-metric";
      card.innerHTML = `<span>${escapeHtml(name)}</span><strong>${escapeHtml(cadence)}</strong><small>${escapeHtml(value.source || "model_default")} · ${Number(value.checkpoint_count || 0).toLocaleString()} checkpoints · ${formatBytes(value.state_bytes)}</small>`;
      policyGrid.append(card);
    }
    container.append(policyGrid);
  }
  container.append(detailSectionTitle("Recovery funnel"));
  const stages = [
    ["Configured", funnel.configured_prefix_tokens?.p50, ratios.configured_prefix?.p50],
    ["Shareable", funnel.shareable_prefix_tokens?.p50, ratios.shareable_prefix?.p50],
    ["Resident", funnel.resident_prefix_tokens?.p50, ratios.resident_prefix?.p50],
    [phased ? "Encoder replay start" : "Recovery", funnel.recovery_boundary_tokens?.p50, phased ? null : ratios.recovery_boundary?.p50],
    ["Replay", funnel.replay_tokens?.p50, null],
    ["Suffix", funnel.suffix_tokens?.p50, null],
  ];
  const flow = document.createElement("div");
  flow.className = "recovery-funnel";
  for (const [label, tokens, ratio] of stages) {
    const card = document.createElement("div");
    card.className = "detail-metric";
    card.innerHTML = `<span>${escapeHtml(label)}</span><strong>${formatCompact(Number(tokens) || 0)} tokens</strong><small>${ratio == null ? "compute phase" : `${formatNumber(Number(ratio) * 100)}% of input`}</small>`;
    flow.append(card);
  }
  container.append(flow);

  container.append(detailSectionTitle("Resident versus planned restore"));
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  for (const [label, value] of [
    ["Resident cache", bytes.resident_cache?.p50],
    ["Planned restore", bytes.planned_restore?.p50],
    ["Replay gate", bytes.replay_gate?.p50],
    ["Blocking checkpoint", bytes.blocking_checkpoint?.p50],
    ["Optional restore", bytes.optional_restore?.p50],
    ["Stranded resident", bytes.stranded_resident?.p50],
  ]) grid.append(byteMetricCard(label, value));
  grid.append(metricCard("Net saved FLOPs", compute.net_saved_flops?.p50, ""));
  grid.append(metricCard("Net saved ratio", compute.net_saved_flops_ratio?.p50, "ratio"));
  container.append(grid);

  const note = document.createElement("div");
  note.className = "boundary-note";
  const causes = Object.entries(recovery.causes || {}).map(
    ([name, count]) => `${formatRecoveryReason(name)} (${Number(count).toLocaleString()})`,
  );
  note.innerHTML = `<strong>Why boundaries differ</strong>${escapeHtml(causes.join(" · ") || "No recovery loss in this case.")}`;
  container.append(note);
}

function detailSectionTitle(text) {
  const title = document.createElement("h3");
  title.className = "timing-section-title cache-detail-title";
  title.textContent = text;
  return title;
}

function byteMetricCard(label, bytes) {
  const card = document.createElement("div");
  card.className = "detail-metric";
  const title = document.createElement("span");
  title.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = formatBytes(bytes);
  card.append(title, strong);
  return card;
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (Math.abs(bytes) >= 1024 ** 3) return `${formatNumber(bytes / (1024 ** 3))} GiB`;
  if (Math.abs(bytes) >= 1024 ** 2) return `${formatNumber(bytes / (1024 ** 2))} MiB`;
  if (Math.abs(bytes) >= 1024) return `${formatNumber(bytes / 1024)} KiB`;
  return `${formatNumber(bytes)} B`;
}

function renderOverview(container, detail) {
  const row = detail.row;
  const metrics = row.metrics || {};
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  grid.append(
    rowMetricCard("Optimistic BW p99", row, "optimistic_overlap_GBps_p99"),
    rowMetricCard("Break-even BW p99", row, "dependency_aware_break_even_GBps_p99"),
    metricCard("Read", metrics.read_avg_GBps, "GB/s"),
    metricCard("Write", metrics.write_avg_GBps, "GB/s"),
    metricCard("Dependency IOPS", metrics.aggregate_dependency_chunked_iops, "IOPS"),
    metricCard("Capacity", metrics.retain_all_capacity_GiB, "GiB"),
    metricCard("Replay tokens", metrics.replay_tokens_p50, ""),
    byteMetricCard("Stranded resident", metrics.stranded_resident_bytes_p50),
    rowMetricCard("TTFT saving upper bound", row, "dependency_ttft_savings_us_p50"),
  );
  container.append(grid);

  const recovery = recoveryOutcome(row);
  const ttft = ttftSavingsOutcome(row);
  const note = document.createElement("div");
  note.className = `detail-note${ttft === "negative" ? " risk" : ""}${recovery === "resident_not_recoverable" ? " warning" : ""}`;
  if (recovery === "resident_not_recoverable") {
    const recoverySemantics = resultSemantics(row).recovery || {};
    const affected = Number(recoverySemantics.status_counts?.resident_not_recoverable || 0);
    const total = Number(recoverySemantics.total_request_count || 0);
    const count = total ? ` ${affected.toLocaleString()} of ${total.toLocaleString()} requests are affected.` : "";
    note.textContent = `Cached prefix objects are resident, but no common checkpoint exists at the hit boundary. The prefix must be replayed and resident bytes are stranded.${count}`;
  } else if (ttft === "negative") {
    note.textContent = "Risk: the dependency-aware cache path is slower than the no-cache path.";
  } else if (ttft === "zero") {
    note.textContent = "Boundary: the cache path has no dependency-aware TTFT gain.";
  } else {
    note.textContent = `${semanticMetricExplanation(row, "dependency_aware_break_even_GBps_p99") || `Break-even status: ${metrics.dependency_aware_break_even_status || "finite"}.`} Bandwidth is a model boundary; TTFT is not measured end-to-end service latency.`;
  }
  container.append(note);

  const decode = document.createElement("div");
  decode.className = "boundary-note";
  const decodeSummary = detail.demand_report?.decode_write_summary || {};
  decode.innerHTML = `<strong>Decode boundary</strong>${decodeSummary.enabled
    ? `KV/state writes enabled · ${escapeHtml(decodeSummary.schedule || "burst_at_completion")} · ${Number(decodeSummary.generated_token_count || 0).toLocaleString()} generated tokens. `
    : "Decode KV/state writes disabled. "}This is not a complete TPOT, scheduler or token-kernel latency model.`;
  container.append(decode);
}

function renderComponents(container, detail) {
  renderFootprintAndEvidence(container, detail);
  renderIoAccounting(container, detail);
  const capacity = detail.demand_report?.capacity || {};
  const measures = [
    ["Reusable prefix", capacity.reusable_prefix_bytes_by_component],
    ["Unique write", capacity.cumulative_unique_write_bytes_by_component],
    ["Retain-all", capacity.retain_all_upper_bound_bytes_by_component],
  ];
  const chart = document.createElement("div");
  chart.className = "component-chart";
  const usedCategories = new Set();
  for (const [label, values] of measures) {
    const aggregate = aggregateComponents(values || {});
    Object.keys(aggregate).forEach((key) => usedCategories.add(key));
    const total = Object.values(aggregate).reduce((sum, value) => sum + value, 0);
    const group = document.createElement("div");
    group.className = "bar-group";
    const title = document.createElement("h3");
    title.textContent = `${label} · ${formatMetric(total / (1024 ** 3), "GiB")}`;
    const bar = document.createElement("div");
    bar.className = "stacked-bar";
    const valuesList = document.createElement("div");
    valuesList.className = "component-values";
    for (const [category, value] of Object.entries(aggregate)) {
      if (!value) continue;
      const percentage = value / Math.max(1, total) * 100;
      const segment = document.createElement("div");
      segment.className = "stack-segment";
      segment.style.width = `${percentage}%`;
      segment.style.background = COMPONENT_COLORS[category];
      const description = `${category}: ${formatMetric(value / (1024 ** 3), "GiB")} (${formatNumber(percentage)}%)`;
      segment.title = description;
      segment.setAttribute("role", "img");
      segment.setAttribute("aria-label", description);
      segment.textContent = (value / total) > .12 ? category : "";
      bar.append(segment);
      const valueLabel = document.createElement("span");
      valueLabel.textContent = description;
      valuesList.append(valueLabel);
    }
    group.append(title, bar, valuesList);
    chart.append(group);
  }
  const legend = document.createElement("div");
  legend.className = "component-legend";
  for (const category of usedCategories) {
    const item = document.createElement("span");
    item.style.setProperty("--swatch", COMPONENT_COLORS[category]);
    item.textContent = category;
    legend.append(item);
  }
  chart.append(legend);
  container.append(chart);
  const note = document.createElement("p");
  note.className = "detail-note";
  note.textContent = "Stacked bars group reported cache components into KV, RoPE, Indexer, SWA, CSA, and HCA. Width represents serialized-capacity share.";
  container.append(note);
}

function renderFootprintAndEvidence(container, detail) {
  const report = detail.demand_report || {};
  const models = report.models || {};
  const model = models[detail.row?.model] || Object.values(models)[0] || {};
  const selected = report.selected_footprint_view
    || detail.row?.case?.modeling_options?.footprint_view
    || "prefix-store";
  const views = [
    ["calculator-reference", "Calculator reference", model.calculator_reference, "total_bytes"],
    ["active-execution", "Active execution", model.active_execution_footprint, "serialized_bytes"],
    ["prefix-store", "Prefix Store", model.prefix_store_footprint, "serialized_bytes"],
    ["replay-resident", "Replay resident", report.cache_replay?.capacity, "final_used_bytes"],
  ];
  container.append(detailSectionTitle("Footprint scopes"));
  const grid = document.createElement("div");
  grid.className = "detail-grid footprint-grid";
  for (const [key, label, value, byteKey] of views) {
    const card = byteMetricCard(label, value?.[byteKey]);
    card.classList.toggle("is-selected", key === selected);
    const semantic = document.createElement("small");
    semantic.textContent = key === "calculator-reference"
      ? value?.semantic_class || "calculator convention"
      : key.replaceAll("-", " ");
    card.append(semantic);
    grid.append(card);
  }
  container.append(grid);

  const evidence = model.evidence || {};
  const entries = Object.entries(evidence);
  if (!entries.length) return;
  container.append(detailSectionTitle("Evidence and assumptions"));
  const evidenceList = document.createElement("div");
  evidenceList.className = "evidence-list";
  for (const [metric, record] of entries) {
    const item = document.createElement("article");
    item.className = "evidence-item";
    const heading = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = metric.replaceAll("_", " ");
    const badge = document.createElement("span");
    badge.className = `evidence-badge evidence-${String(record?.status || "unsupported").replaceAll("_", "-")}`;
    badge.textContent = record?.status || "unsupported";
    heading.append(title, badge);
    item.append(heading);
    for (const source of record?.sources || []) {
      const link = document.createElement("a");
      link.href = source;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = source;
      item.append(link);
    }
    for (const assumption of record?.assumptions || []) {
      const note = document.createElement("small");
      const field = assumption.field ? `${assumption.field}: ` : "";
      note.textContent = `${field}${assumption.value ?? ""}${assumption.rationale ? ` · ${assumption.rationale}` : ""}`;
      item.append(note);
    }
    evidenceList.append(item);
  }
  container.append(evidenceList);
}

function renderTiming(container, detail) {
  const report = detail.demand_report || {};
  const ttft = report.ttft || {};
  const metrics = detail.row?.metrics || {};
  if (ttft.execution_phases?.length) renderExecutionPhases(container, detail, true);
  const timing = [
    ["No-cache TTFT", ttft.no_cache_ttft_us?.p50],
    ["Optimistic zero-latency floor", ttft.optimistic_overlap_estimate_us?.p50],
    ["Dependency zero-latency floor", ttft.dependency_aware_estimate_us?.p50],
    ...(ttft.execution_phases?.length ? [] : [["Replay compute", ttft.replay_compute_time_us?.p50], ["Suffix compute", ttft.suffix_compute_time_us?.p50]]),
    ["Cache overhead", ttft.cache_path_overhead_us?.p50],
    ["Dependency TTFT saving upper bound", ttft.dependency_ttft_savings_us?.p50 ?? metrics.dependency_ttft_savings_us_p50],
  ];
  const chart = document.createElement("div");
  chart.className = "timing-chart";
  const ttftTitle = document.createElement("h3");
  ttftTitle.className = "timing-section-title";
  ttftTitle.textContent = "TTFT boundary · signed symlog";
  chart.append(ttftTitle);
  const maxAbs = Math.max(1, ...timing.map(([, value]) => Math.abs(Number(value) || 0)));
  for (const [label, value] of timing) chart.append(signedTimelineRow(label, value, maxAbs, "µs"));

  const readWindows = report.read?.peak_windows || {};
  const writeWindows = report.write?.peak_windows || {};
  const windows = ["1ms", "10ms", "100ms", "1s"];
  const peakValues = windows.flatMap((window) => [readWindows[window]?.GBps, writeWindows[window]?.GBps]);
  const peakMax = Math.max(1, ...peakValues.map((value) => Number(value) || 0));
  const peakTitle = document.createElement("h3");
  peakTitle.className = "timing-section-title";
  peakTitle.textContent = "Peak windows · offered demand · log magnitude";
  chart.append(peakTitle);
  for (const window of windows) {
    chart.append(logTimelineRow(`Read ${window}`, readWindows[window]?.GBps, peakMax, "GB/s"));
    chart.append(logTimelineRow(`Write ${window}`, writeWindows[window]?.GBps, peakMax, "GB/s"));
  }
  container.append(chart);

  if (detail.storage_report) renderStorageTiming(container, detail.storage_report, maxAbs);
}

function renderExecutionPhases(container, detail, timing) {
  const report = detail.demand_report || {};
  const phases = report.ttft?.execution_phases || report.recovery?.execution_phases || [];
  container.append(detailSectionTitle(timing ? "CED execution phases · per-request reference" : "Global reuse and Encoder recovery"));
  const notes = document.createElement("p");
  notes.className = "boundary-note";
  const sources = report.recovery?.state_sources || {};
  const sourceText = Object.entries(sources).map(([name, counts]) => `${name.replaceAll("_", " ")}: ${Object.entries(counts).map(([source, count]) => `${source} (${count})`).join(", ")}`).join(" · ");
  notes.textContent = timing
    ? "Each request follows the ordered stages below. Compute and reads overlap using first-use dependencies; fixed overhead and first-token cost are added once. Displayed phase P50 values are descriptive and must not be added to produce total P50."
    : `Global KV reuse: ${formatCompact(report.recovery?.funnel?.resident_prefix_tokens?.p50 || 0)} tokens (P50). ${sourceText}. Bounded Encoder replay is a PCS approximation; completed global entries stay unchanged. Retain assumes a valid matching boundary snapshot is already on the execution device. Output-only replay preserves the valid SWA / compression tail.`;
  if (!timing) {
    const start = Math.max(0, Number(detail.row?.seq_len || 0) - 128);
    const hit = Number(report.recovery?.funnel?.resident_prefix_tokens?.p50 || 0);
    notes.textContent += ` Required old output at the P50 hit boundary: ${start < hit ? `[${start}, ${hit})` : "none"}.`;
  }
  container.append(notes);
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  for (const phase of phases) {
    const card = document.createElement("div");
    card.className = "detail-metric";
    const title = document.createElement("span");
    title.textContent = phase.name.replaceAll("_", " ");
    const value = document.createElement("strong");
    value.textContent = timing ? `${formatNumber(phase.compute_us?.p50 || 0)} µs` : `[${phase.token_start?.p50 || 0}, ${phase.token_end?.p50 || 0})`;
    const note = document.createElement("small");
    note.textContent = `layers ${(phase.layer_ranges || []).map(([start, end]) => `[${start}, ${end})`).join(", ")} · ${formatCompact(phase.flops?.p50 || 0)} FLOPs · ${phase.tokens?.p50 || 0} tokens · reads: ${(phase.read_components || []).join(", ") || "none"}${phase.approximate ? " · bounded approximation" : ""}`;
    card.append(title, value, note);
    grid.append(card);
  }
  container.append(grid);
}

function renderIoAccounting(container, detail) {
  const values = detail.demand_report?.io_accounting;
  if (!values) return;
  container.append(detailSectionTitle("Entry payload, object IO and SSD IO · all requests"));
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  const physical = detail.storage_report?.io_accounting;
  for (const [name, value] of [
    ["New global entry payload", values.generated_global_payload_bytes],
    ["Logical object reads", values.logical_read_payload_bytes],
    ["Logical object writes", values.logical_write_payload_bytes],
    ...(physical ? [["Physical SSD reads", physical.physical_read_payload_bytes], ["Physical SSD writes", physical.physical_write_payload_bytes]] : []),
  ]) grid.append(byteMetricCard(name, value));
  container.append(grid);
  const note = document.createElement("p");
  note.className = "boundary-note";
  note.textContent = `New entries: Encoder ${values.generated_global_entries?.encoder || 0}, Decoder ${values.generated_global_entries?.decoder || 0}. ` + "New entry payload is B(N) − B(H). Object versions can carry old entries in a rewritten partial block; SSD transfer sizes follow existing splitting and page handling. These quantities need not be equal. " + (physical ? "Physical values are simulated transfers, not allocated page capacity." : "Run SSD simulation in Cache Replay to obtain physical IO.");
  container.append(note);
}

function renderStorageTiming(container, storage) {
  const block = document.createElement("div");
  block.className = "ssd-block";
  const heading = document.createElement("h3");
  heading.textContent = "Simulated SSD timing";
  block.append(heading);
  const ttft = storage.ttft || {};
  const values = [
    ["Optimistic TTFT", ttft.optimistic_overlap_estimate_us?.p50, "µs"],
    ["Dependency TTFT", ttft.dependency_aware_estimate_us?.p50, "µs"],
    ["No-overlap TTFT", ttft.no_compute_io_overlap_estimate_us?.p50, "µs"],
    ["Dependency saving", ttft.dependency_ttft_savings_us?.p50, "µs"],
    ["Replay gate", ttft.replay_gate_elapsed_us?.p50, "µs"],
    ["Hidden miss ratio", ttft.hidden_restore_miss_ratio, "ratio"],
    ["Achieved throughput", storage.throughput?.achieved_GBps, "GB/s"],
  ];
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  for (const [label, value, unit] of values) grid.append(metricCard(label, value, unit));
  block.append(grid);
  container.append(block);
}

function renderRaw(container, detail, onStatus) {
  const toolbar = document.createElement("div");
  toolbar.className = "raw-toolbar";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "Copy JSON";
  const download = document.createElement("button");
  download.type = "button";
  download.textContent = "Download JSON";
  const json = JSON.stringify(detail, jsonReplacer, 2);
  copy.addEventListener("click", async () => {
    try { await copyText(json); onStatus?.("Case JSON copied."); }
    catch (error) { onStatus?.(`Copy failed: ${error.message}`, true); }
  });
  download.addEventListener("click", () => downloadJson(`${detail.row?.id || "case"}.json`, detail));
  toolbar.append(copy, download);
  const disclosure = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Raw JSON is collapsed by default · click to expand";
  const pre = document.createElement("pre");
  pre.className = "json-view";
  pre.textContent = json;
  disclosure.append(summary, pre);
  container.append(toolbar, disclosure);
}

function metricCard(label, value, unit) {
  const card = document.createElement("div");
  const state = classifyValue(value);
  card.className = `detail-metric${state === VALUE_STATES.NEGATIVE ? " negative" : ""}`;
  const title = document.createElement("span");
  title.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = unit === "ratio" ? (Number.isFinite(Number(value)) ? formatNumber(Number(value) * 100) + "%" : "Missing") : formatMetric(value, unit);
  card.append(title, strong);
  return card;
}

function rowMetricCard(label, row, metric) {
  const card = document.createElement("div");
  const state = classifyValue(row?.metrics?.[metric]);
  card.className = `detail-metric${state === VALUE_STATES.NEGATIVE ? " negative" : ""}`;
  const title = document.createElement("span");
  title.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = formatMetricForRow(row, metric);
  card.append(title, strong);
  return card;
}

function formatRecoveryReason(code) {
  const labels = {
    cold_start: "Cold start has no resident prefix",
    checkpoint_not_resident_or_tail_replay: "Checkpoint is not resident at the prefix boundary",
    checkpoint_not_resident_or_too_sparse: "No usable checkpoint exists at the resident boundary",
    missing_checkpoint_state: "Required checkpoint state is missing",
    replay_from_zero: "Recovery must replay from token zero",
  };
  return labels[code] || String(code).replaceAll("_", " ");
}

function signedTimelineRow(label, value, maximum, unit) {
  const row = document.createElement("div");
  row.className = "timeline-row";
  row.setAttribute("aria-label", `${label}: ${formatMetric(value, unit)}`);
  const name = document.createElement("span");
  name.textContent = label;
  const track = document.createElement("div");
  track.className = "timeline-track signed-track";
  track.setAttribute("aria-hidden", "true");
  const fill = document.createElement("div");
  const negative = Number(value) < 0;
  fill.className = `timeline-fill ${negative ? "negative" : "positive"}`;
  const ratio = Math.log1p(Math.abs(Number(value) || 0)) / Math.log1p(Math.max(1, maximum));
  fill.style.width = `${Math.max(0, Math.min(50, ratio * 50))}%`;
  track.append(fill);
  const output = document.createElement("span");
  output.className = "timeline-value";
  output.textContent = formatMetric(value, unit);
  row.append(name, track, output);
  return row;
}

function logTimelineRow(label, value, maximum, unit) {
  const row = document.createElement("div");
  row.className = "timeline-row";
  row.setAttribute("aria-label", `${label}: ${formatMetric(value, unit)}`);
  const name = document.createElement("span");
  name.textContent = label;
  const track = document.createElement("div");
  track.className = "timeline-track log-track";
  track.setAttribute("aria-hidden", "true");
  const fill = document.createElement("div");
  fill.className = "timeline-fill";
  const ratio = Math.log1p(Math.max(0, Number(value) || 0)) / Math.log1p(Math.max(1, maximum));
  fill.style.width = `${Math.max(0, Math.min(100, ratio * 100))}%`;
  track.append(fill);
  const output = document.createElement("span");
  output.className = "timeline-value";
  output.textContent = formatMetric(value, unit);
  row.append(name, track, output);
  return row;
}

function aggregateComponents(values) {
  const result = { KV: 0, RoPE: 0, Indexer: 0, SWA: 0, CSA: 0, HCA: 0, Other: 0 };
  for (const [name, value] of Object.entries(values)) result[componentCategory(name)] += Number(value) || 0;
  return Object.fromEntries(Object.entries(result).filter(([, value]) => value > 0));
}

function componentCategory(name) {
  const value = name.toLowerCase();
  if (value.includes("rope")) return "RoPE";
  if (value.includes("index")) return "Indexer";
  if (value.includes("swa")) return "SWA";
  if (value.includes("csa")) return "CSA";
  if (value.includes("hca")) return "HCA";
  if (value.includes("kv") || value.includes("latent") || value.includes("main")) return "KV";
  return "Other";
}

function jsonReplacer(_key, value) {
  if (value === Infinity) return "Infinity";
  if (value === -Infinity) return "-Infinity";
  if (Number.isNaN(value)) return "NaN";
  return value;
}
