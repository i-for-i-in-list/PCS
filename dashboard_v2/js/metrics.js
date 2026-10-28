import {
  RESULT_STATES,
  resultRiskCounts,
  riskForResultRow,
  riskReasonsForResultRow,
  semanticMetricExplanation,
  semanticMetricLabel,
  semanticStateForMetric,
} from "./result_semantics.js";

export const VALUE_STATES = Object.freeze({
  NOT_EVALUATED: RESULT_STATES.NOT_EVALUATED,
  POSITIVE: "positive",
  NEGATIVE: "negative",
  ZERO: "zero",
  MISSING: "missing",
  INFINITE: "infinite",
  CACHED_NOT_RESUMABLE: RESULT_STATES.CACHED_NOT_RESUMABLE,
  NO_FINITE_BREAK_EVEN: RESULT_STATES.NO_FINITE_BREAK_EVEN,
  NO_OVERLAP_WINDOW: RESULT_STATES.NO_OVERLAP_WINDOW,
  MIXED: RESULT_STATES.MIXED,
});

const PRIMARY_DEFINITIONS = {
  optimistic_overlap_GBps_p99: {
    label: "Optimistic fully-hidden bandwidth p99",
    shortLabel: "Optimistic BW",
    description: "Bandwidth required to hide blocking restore behind replay plus suffix compute under the optimistic overlap bound.",
    unit: "GB/s",
    scale: "base2",
    summary: "max",
    tone: "restore",
    threshold: "sequential_read_GBps",
  },
  read_avg_GBps: {
    label: "Offered read bandwidth",
    shortLabel: "Read avg",
    description: "Shows the workload’s average offered read bandwidth across all requests, not achieved device throughput. It answers: how much sustained aggregate read capacity does the storage system need?",
    unit: "GB/s",
    scale: "base2",
    summary: "max",
    tone: "read",
    threshold: "sequential_read_GBps",
  },
  write_avg_GBps: {
    label: "Offered write bandwidth",
    shortLabel: "Write avg",
    description: "Shows the workload’s average offered write bandwidth from persisting cache blocks created by misses when write_miss_blocks is enabled. It answers: how much sustained write pressure does cache population create as the miss rate changes?",
    unit: "GB/s",
    scale: "base2",
    summary: "max",
    tone: "write",
    threshold: "sequential_write_GBps",
  },
  aggregate_dependency_chunked_iops: {
    label: "Aggregate dependency IOPS",
    shortLabel: "Dependency IOPS",
    description: "Peak aggregate chunked IOPS over dependency-aware break-even windows.",
    unit: "IOPS",
    scale: "base2",
    summary: "max",
    tone: "iops",
    threshold: "random_read_iops",
  },
  dependency_ttft_savings_us_p50: {
    label: "Dependency-aware TTFT saving upper bound p50",
    shortLabel: "TTFT upper bound",
    description: "No-cache TTFT minus the dependency-aware zero-restore-latency floor. Use rated recommendation timing or SSD simulation for realizable savings.",
    unit: "ms",
    sourceUnit: "µs",
    scale: "symlog",
    summary: "min",
    tone: "ttft",
    threshold: null,
  },
};

export const PRIMARY_METRIC_KEYS = Object.freeze(Object.keys(PRIMARY_DEFINITIONS));
export const PEAK_WINDOWS = Object.freeze(["1ms", "10ms", "100ms", "1s"]);
export const PEAK_MEASURES = Object.freeze({
  read_GBps: "Read peak bandwidth",
  write_GBps: "Write peak bandwidth",
  read_chunked_iops: "Read peak IOPS",
  write_chunked_iops: "Write peak IOPS",
  read_to_avg: "Read bandwidth peak / average",
  write_to_avg: "Write bandwidth peak / average",
});

function metric(label, unit, description, options = {}) {
  return { label, shortLabel: label, unit, description, scale: "base2", summary: "max", threshold: null, ...options };
}

const EXTRA_DEFINITIONS = {
  retain_all_capacity_GiB: metric("Retain-all capacity upper bound", "GiB", "Serialized capacity upper bound if all reported objects are retained; this is not measured cache residency."),
  reusable_capacity_GiB: metric("Reusable prefix capacity", "GiB", "Total serialized capacity of reusable prefix objects in this Case."),
  cache_peak_used_bytes: metric("Peak cache occupancy", "GiB", "Maximum total resident cache bytes during measured replay, including protected objects.", { sourceUnit: "bytes", cacheOnly: true }),
  cache_final_used_bytes: metric("Final cache occupancy", "GiB", "Total resident cache bytes at the end of replay.", { sourceUnit: "bytes", cacheOnly: true }),
  cache_peak_protected_bytes: metric("Peak protected cache", "GiB", "Maximum protected resident bytes during measured replay; these bytes cannot yet be evicted.", { sourceUnit: "bytes", cacheOnly: true }),
  shareable_prefix_ratio: metric("Object-shareable prefix ratio", "%", "Object-aligned shareable prefix fraction: request P50 in Bandwidth Envelope; aggregate token fraction in Cache Replay.", { scale: "linear", domain: [0, 1] }),
  resident_prefix_ratio: metric("Resident prefix ratio", "%", "Resident prefix fraction: assumed request P50 in Bandwidth Envelope; aggregate measured token fraction in Cache Replay. Residency alone does not guarantee recovery.", { scale: "linear", domain: [0, 1] }),
  recovery_boundary_ratio: metric("Recovery boundary ratio p50", "%", "Request P50 recovery boundary as a fraction of input length. For V4.1 this is the Encoder replay start, not the fraction of reusable global KV.", { scale: "linear", domain: [0, 1] }),
  cache_actual_block_hit_rate: metric("Actual block hit rate", "%", "Actual hit blocks divided by all cacheable blocks across measured replay requests, not just the candidate prefix blocks.", { scale: "linear", domain: [0, 1], cacheOnly: true }),
  cache_actual_byte_hit_rate: metric("Actual byte hit rate", "%", "Actual hit bytes divided by all cacheable bytes across measured replay requests, not just the candidate prefix bytes.", { scale: "linear", domain: [0, 1], cacheOnly: true }),
  replay_tokens_p50: metric("Replay tokens p50", "tokens", "Request P50 tokens recomputed to reach a usable recovery boundary.", { scale: "linear" }),
  planned_restore_bytes_p50: metric("Planned restore p50", "GiB", "Request P50 logical bytes planned for restoration; this is not total cache capacity or physical SSD traffic.", { sourceUnit: "bytes" }),
  stranded_resident_bytes_p50: metric("Stranded resident p50", "GiB", "Request P50 resident bytes that cannot contribute to recovery under the selected state policy.", { sourceUnit: "bytes" }),
  net_saved_flops_p50: metric("Net saved FLOPs p50", "TFLOPs", "Request P50 net computation saved relative to no-cache prefill. One TFLOP is 10^12 operations, not a throughput rate; negative values mean additional computation.", { sourceUnit: "FLOPs", scale: "symlog" }),
  net_saved_flops_ratio_p50: metric("Net compute saving p50", "%", "Request P50 net saved FLOPs divided by that request's no-cache prefill FLOPs. Negative values mean additional computation.", { scale: "linear", domain: [0, 1], diverging: true }),
  replay_compute_time_us_p50: metric("Replay compute time p50", "ms", "Request P50 modeled replay compute time. It excludes storage queueing and is not measured service latency.", { sourceUnit: "µs" }),
  suffix_compute_time_us_p50: metric("Suffix compute time p50", "ms", "Request P50 compute required by new suffix tokens, including mandatory conversion and algorithm finalization; excludes state restoration and output head. Zero for an empty suffix.", { sourceUnit: "µs" }),
};

export function peakMetricKey(measure, window) {
  const split = measure.indexOf("_");
  return `${measure.slice(0, split)}_peak_${window}_${measure.slice(split + 1)}`;
}

for (const window of PEAK_WINDOWS) {
  for (const [measure, label] of Object.entries(PEAK_MEASURES)) {
    const direction = measure.startsWith("read_") ? "read" : "write";
    const iops = measure.endsWith("chunked_iops");
    const ratio = measure.endsWith("to_avg");
    EXTRA_DEFINITIONS[peakMetricKey(measure, window)] = metric(
      `${label} · ${window}`, ratio ? "×" : iops ? "IOPS" : "GB/s",
      ratio
        ? `Maximum ${window} offered ${direction} bandwidth divided by average ${direction} bandwidth over the Case. Undefined when average demand is zero.`
        : `Maximum ${window} offered ${direction} ${iops ? "chunked IOPS (requires Max IO)" : "bandwidth"}; this is demand, not achieved device throughput.`,
      { peakWindow: window, peakMeasure: measure, requiresChunking: iops, threshold: ratio ? null : iops ? `random_${direction}_iops` : `sequential_${direction}_GBps` },
    );
  }
}

export const METRICS = Object.freeze({ ...PRIMARY_DEFINITIONS, ...EXTRA_DEFINITIONS });
export const EXTRA_METRIC_KEYS = Object.freeze(Object.keys(EXTRA_DEFINITIONS));
export const METRIC_GROUPS = Object.freeze({
  capacity: { defaultMetric: "retain_all_capacity_GiB", metrics: ["retain_all_capacity_GiB", "reusable_capacity_GiB", "cache_peak_used_bytes", "cache_final_used_bytes", "cache_peak_protected_bytes"] },
  recovery: { defaultMetric: "recovery_boundary_ratio", metrics: ["shareable_prefix_ratio", "resident_prefix_ratio", "recovery_boundary_ratio", "cache_actual_block_hit_rate", "cache_actual_byte_hit_rate", "replay_tokens_p50", "planned_restore_bytes_p50", "stranded_resident_bytes_p50"] },
  compute: { defaultMetric: "net_saved_flops_ratio_p50", metrics: ["net_saved_flops_p50", "net_saved_flops_ratio_p50", "replay_compute_time_us_p50", "suffix_compute_time_us_p50"] },
  peak: { defaultMetric: "read_peak_1ms_GBps", metrics: Object.keys(EXTRA_DEFINITIONS).filter((key) => EXTRA_DEFINITIONS[key].peakWindow) },
});

export function hasCacheRow(row) {
  return row.analysis_mode === "cache_replay" || Boolean(row.case?.cache_replay_config);
}

export function metricUnavailableReason(metric, rows) {
  const definition = METRICS[metric];
  if (definition?.cacheOnly && !rows.some(hasCacheRow)) return "Requires Cache Replay results.";
  if (!rows.length) return "";
  if (rows.some((row) => metricValueState(row, metric) !== VALUE_STATES.MISSING)) return "";
  if (definition?.requiresChunking && rows.every((row) => row.max_io_bytes == null && row.case?.max_io_bytes == null)) {
    return "IOPS requires Max IO chunking. Set Max IO and rerun the Sweep.";
  }
  if (definition?.peakMeasure?.endsWith("to_avg")) {
    const direction = definition.peakMeasure.split("_")[0];
    if (rows.every((row) => numericValue(row.metrics?.[`${direction}_avg_GBps`]) === 0)) return "Peak / average is undefined because average demand is zero.";
  }
  return "This result has no values for this metric. Rerun the Sweep with the required reporting options; older experiments are not recalculated.";
}

export function scaleLabel(definition) {
  return definition.scale === "linear" ? "linear scale" : definition.scale === "symlog" ? "signed symlog scale" : "base-2 scale";
}

export function numericValue(value) {
  if (typeof value === "number") return value;
  if (value === "Infinity" || value === "+Infinity" || value === "∞") return Infinity;
  if (value === "-Infinity" || value === "-∞") return -Infinity;
  if (value === null || value === undefined || value === "" || value === "NaN") return NaN;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? NaN : parsed;
}

export function classifyValue(value) {
  const number = numericValue(value);
  if (Number.isNaN(number)) return VALUE_STATES.MISSING;
  if (!Number.isFinite(number)) return VALUE_STATES.INFINITE;
  if (number < 0) return VALUE_STATES.NEGATIVE;
  if (number === 0) return VALUE_STATES.ZERO;
  return VALUE_STATES.POSITIVE;
}

export function metricValueState(row, metric) {
  if (row?.evaluation?.aborted) return VALUE_STATES.NOT_EVALUATED;
  if (METRICS[metric]?.cacheOnly && !hasCacheRow(row)) return VALUE_STATES.MISSING;
  return semanticStateForMetric(row, metric) || classifyValue(row?.metrics?.[metric]);
}

export function formatMetric(value, metricOrUnit = "") {
  const def = METRICS[metricOrUnit];
  const unit = def?.unit || metricOrUnit;
  const sourceUnit = def?.sourceUnit || unit;
  const state = classifyValue(value);
  if (state === VALUE_STATES.MISSING) return "Missing";
  if (state === VALUE_STATES.INFINITE) return numericValue(value) < 0 ? "−∞" : "∞";
  const number = numericValue(value);
  if (sourceUnit === "bytes") return `${formatDisplayNumber(number / (1024 ** 3))} GiB`;
  if (sourceUnit === "FLOPs") return `${formatDisplayNumber(number / 1e12)} TFLOPs`;
  if (unit === "%") return `${formatDisplayNumber(number * 100)}%`;
  if (unit === "tokens") return `${formatCompact(number)} tokens`;
  if (unit === "×") return `${formatDisplayNumber(number)}×`;
  if (sourceUnit === "µs") return formatDuration(number);
  if (unit === "IOPS") return `${formatCompact(number)} IOPS`;
  if (unit === "GiB") return `${formatNumber(number)} GiB`;
  if (unit === "GB/s") return `${formatNumber(number)} GB/s`;
  return formatNumber(number);
}

export function formatMetricForRow(row, metric) {
  const semantic = semanticMetricLabel(row, metric);
  const formatKey = METRICS[metric]
    ? metric
    : metric.includes("GBps") ? "GB/s"
      : metric.includes("iops") || metric.includes("IOPS") ? "IOPS"
        : metric;
  if (!semantic) return formatMetric(row?.metrics?.[metric], formatKey);
  if (semanticStateForMetric(row, metric) === VALUE_STATES.MIXED) {
    return `${formatMetric(row?.metrics?.[metric], formatKey)} · ${semantic}`;
  }
  return semantic;
}

export function formatAxisMetric(value, metric) {
  const state = classifyValue(value);
  if (state === VALUE_STATES.MISSING) return "Missing";
  if (state === VALUE_STATES.INFINITE) return numericValue(value) < 0 ? "−∞" : "∞";
  const number = numericValue(value);
  const definition = METRICS[metric];
  if (definition?.sourceUnit === "bytes") return formatDisplayNumber(number / (1024 ** 3));
  if (definition?.sourceUnit === "FLOPs") return formatDisplayNumber(number / 1e12);
  if (definition?.unit === "%") return `${formatDisplayNumber(number * 100)}%`;
  if (definition?.unit === "×") return `${formatDisplayNumber(number)}×`;
  if (METRICS[metric]?.sourceUnit === "µs") return formatMilliseconds(number / 1_000);
  if (METRICS[metric]?.scale === "base2") {
    const sign = number < 0 ? "−" : "";
    const magnitude = Math.abs(number);
    if (magnitude >= 1024) return `${sign}${formatNumber(magnitude / 1024)}k`;
    if (magnitude >= 1) return `${sign}${formatNumber(magnitude)}`;
    return `${sign}${Number(magnitude.toPrecision(6))}`;
  }
  return formatCompact(number);
}

function formatDisplayNumber(number) {
  return number !== 0 && Math.abs(number) < .0001 ? number.toExponential(2) : formatNumber(number);
}

export function formatDuration(microseconds) {
  const abs = Math.abs(microseconds);
  const sign = microseconds < 0 ? "−" : "";
  return `${sign}${formatMilliseconds(abs / 1_000)} ms`;
}

function formatMilliseconds(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(value);
}

export function formatCompact(value) {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${formatNumber(value / 1_000_000)}M`;
  if (abs >= 1_000) return `${formatNumber(value / 1_000)}K`;
  return formatNumber(value);
}

export function formatNumber(value) {
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "−∞";
  const abs = Math.abs(value);
  let digits = 2;
  if (abs >= 100) digits = 0;
  else if (abs >= 10) digits = 1;
  else if (abs > 0 && abs < .01) digits = 4;
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value);
}

export function missRatioScore(hitRate) {
  const hit = Number(hitRate);
  if (hit >= 1) return Infinity;
  return -Math.log10(Math.max(1 - hit, 1e-12));
}

export function xValue(row, mode) {
  return mode === "hit-linear" ? Number(row.prefix_ratio) : missRatioScore(row.prefix_ratio);
}

export function formatXAxis(value, mode) {
  if (mode === "cache-capacity") return formatDisplayNumber(Number(value) / (1024 ** 3));
  if (mode === "context-length") return formatCompact(Number(value));
  if (mode === "hit-linear") {
    const rounded = Math.round(Number(value) * 1_000_000) / 1_000_000;
    return Object.is(rounded, -0) ? "0" : String(rounded);
  }
  if (!Number.isFinite(value)) return "full hit";
  if (Math.abs(value - Math.round(value)) < .001) return `10⁻${Math.round(value)}`;
  return `10⁻${formatNumber(value)}`;
}

export function axisValue(row, mode) {
  if (mode === "cache-capacity") return hasCacheRow(row) ? numericValue(row.metrics?.cache_total_capacity_bytes ?? row.case?.cache_replay_config?.total_capacity_bytes) : NaN;
  if (mode === "context-length") return numericValue(row.seq_len);
  return numericValue(row.prefix_ratio);
}

export function axisLabel(mode) {
  if (mode === "cache-capacity") return "Cache capacity (GiB, log₂ scale)";
  if (mode === "context-length") return "Context length (tokens, log₂ scale)";
  return mode === "miss-log" ? "Miss ratio (log₁₀ scale; farther right = smaller miss)" : "Prefix ratio";
}

export function seriesIdentity(row, mode = "hit-linear") {
  const identity = { ...(row.case || {}) };
  for (const key of ["id", "case_identity"]) delete identity[key];
  for (const key of ["model", "seq_len", "prefix_ratio", "block_size_tokens", "arrival_model", "max_io_bytes", "analysis_mode", "encoder_state", "encoder_tail_policy", "qps", "request_count", "workload_shape"]) {
    if (row[key] !== undefined && identity[key] === undefined) identity[key] = row[key];
  }
  if (identity.cache_replay_config) identity.cache_replay_config = { ...identity.cache_replay_config };
  if (row.metrics?.cache_total_capacity_bytes !== undefined) {
    identity.cache_replay_config = { ...(identity.cache_replay_config || {}), total_capacity_bytes: row.metrics.cache_total_capacity_bytes };
  }
  if (mode === "cache-capacity") {
    if (identity.cache_replay_config) delete identity.cache_replay_config.total_capacity_bytes;
  } else if (mode === "context-length") delete identity.seq_len;
  else delete identity.prefix_ratio;
  return identity;
}

export function stableIdentity(value) {
  if (Array.isArray(value)) return value.map(stableIdentity);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableIdentity(value[key])]));
  return value;
}

export function seriesKey(row, mode = "hit-linear") {
  return JSON.stringify(stableIdentity(seriesIdentity(row, mode)));
}

export function seriesShortLabel(row) {
  const cache = row.metrics?.cache_total_capacity_bytes === undefined
    ? ""
    : ` · Cache ${formatNumber(Number(row.metrics.cache_total_capacity_GiB) || 0)} GiB`;
  return `${row.model} · L${formatCompact(row.seq_len)} · B${row.block_size_tokens} · ${row.arrival_model} · IO ${row.max_io_bytes ?? "∞"}${cache}${row.encoder_state == null ? "" : ` · Encoder ${row.encoder_state} · output ${row.encoder_tail_policy}`}`;
}

export function riskForRow(row) {
  return riskForResultRow(row);
}

export function riskReasonsForRow(row, metric = null, threshold = NaN) {
  return riskReasonsForResultRow(row, metric, threshold);
}

export function storageThreshold(metric, storageOption) {
  const key = METRICS[metric]?.threshold;
  return key ? numericValue(storageOption?.thresholds?.[key]) : NaN;
}

export function summarizeRows(rows) {
  const cards = PRIMARY_METRIC_KEYS.map((metric) => {
    const def = METRICS[metric];
    const candidates = rows.filter((row) => classifyValue(row.metrics?.[metric]) !== VALUE_STATES.MISSING);
    const source = candidates.reduce((best, row) => {
      if (!best) return row;
      const value = numericValue(row.metrics[metric]);
      const bestValue = numericValue(best.metrics[metric]);
      return def.summary === "min" ? (value < bestValue ? row : best) : (value > bestValue ? row : best);
    }, null);
    return { metric, def, source, value: source?.metrics?.[metric] };
  });
  return { cards, risks: riskCounts(rows) };
}

export function riskCounts(rows, metric = null, threshold = NaN) {
  return resultRiskCounts(rows, metric, threshold);
}

export function tooltipHtml(row, metric) {
  const def = METRICS[metric];
  const miss = Math.max(0, 1 - Number(row.prefix_ratio));
  const cacheRows = row.metrics?.resident_prefix_ratio !== undefined && row.analysis_mode === "cache_replay";
  const hitRows = cacheRows
    ? `<dt>Configured</dt><dd>${formatNumber(row.prefix_ratio * 100)}%</dd>
    <dt>Shareable</dt><dd>${formatNumber(Number(row.metrics.shareable_prefix_ratio) * 100)}%</dd>
    <dt>Resident</dt><dd>${formatNumber(Number(row.metrics.resident_prefix_ratio) * 100)}%</dd>
    <dt>Recovery</dt><dd>${formatNumber(Number(row.metrics.recovery_boundary_ratio) * 100)}%</dd>
    <dt>Cache capacity</dt><dd>${formatNumber(Number(row.metrics.cache_total_capacity_GiB) || 0)} GiB</dd>`
    : `<dt>Assumed resident</dt><dd>${formatNumber(row.prefix_ratio * 100)}%</dd>`;
  const semanticState = semanticStateForMetric(row, metric);
  const semanticLabel = semanticMetricLabel(row, metric);
  const semanticExplanation = semanticMetricExplanation(row, metric);
  const valueRows = semanticLabel
    ? `<dt>Status</dt><dd>${escapeHtml(semanticLabel)}</dd>
    <dt>Raw value</dt><dd>${escapeHtml(formatMetric(row.metrics?.[metric], metric))}</dd>`
    : `<dt>${escapeHtml(def.shortLabel)}</dt><dd>${escapeHtml(formatMetric(row.metrics?.[metric], metric))}</dd>`;
  return `<strong>${escapeHtml(row.id)} · ${escapeHtml(def.label)}</strong><dl>
    <dt>Model</dt><dd>${escapeHtml(row.model)}</dd>
    ${row.encoder_state == null ? "" : `<dt>Encoder state</dt><dd>${escapeHtml(row.encoder_state)}</dd><dt>Encoder output</dt><dd>${escapeHtml(row.encoder_tail_policy)}</dd>`}
    <dt>seq_len</dt><dd>${formatCompact(row.seq_len)}</dd>
    ${hitRows}
    <dt>Miss ratio</dt><dd>${miss === 0 ? "0" : miss.toExponential(2)}</dd>
    <dt>Block</dt><dd>${row.block_size_tokens}</dd>
    <dt>Arrival</dt><dd>${escapeHtml(row.arrival_model)}</dd>
    <dt>Max IO</dt><dd>${row.max_io_bytes ?? "None"}</dd>
    ${valueRows}
  </dl>${def.peakWindow ? `<p>Window: ${escapeHtml(def.peakWindow)} · offered demand</p>` : ""}${metric === "recovery_boundary_ratio" && row.formula_id === "deepseek_v41_csa2" ? "<p>V4.1: Encoder replay start; global KV reuse is reported separately.</p>" : ""}${semanticState ? `<p class="tooltip-semantic">${escapeHtml(semanticExplanation)}</p>` : ""}`;
}

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}
