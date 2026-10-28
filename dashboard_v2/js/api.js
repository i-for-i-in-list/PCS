import { semanticsForCsv } from "./result_semantics.js";
import { EXTRA_METRIC_KEYS } from "./metrics.js";

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    cache: "no-store",
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`Server returned ${response.status} without valid JSON.`);
  }
  if (!response.ok) {
    const error = new Error(payload?.error || `Request failed (${response.status}).`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return normalizeResponse(payload);
}

export function getConfigs() {
  return request("/api/configs");
}

export function runSweep(payload) {
  return request("/api/sweep", { method: "POST", body: JSON.stringify(payload) });
}

export function createSweepJob(payload) {
  return request("/api/sweep/jobs", { method: "POST", body: JSON.stringify(payload) });
}

export function getSweepJob(jobId) {
  return request(`/api/sweep/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelSweepJob(jobId) {
  return request(`/api/sweep/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
}

export function createRecommendationJob(payload) {
  return request("/api/recommendations/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getRecommendationJob(jobId) {
  return request(`/api/recommendations/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelRecommendationJob(jobId) {
  return request(`/api/recommendations/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
}

export function getExperiments(params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== null && value !== undefined && value !== ""),
  );
  return request(`/api/experiments${query.size ? `?${query}` : ""}`);
}

export function getExperiment(experimentId) {
  return request(`/api/experiments/${encodeURIComponent(experimentId)}`);
}

export function updateExperiment(experimentId, patch) {
  return request(`/api/experiments/${encodeURIComponent(experimentId)}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function deleteExperiment(experimentId) {
  return request(`/api/experiments/${encodeURIComponent(experimentId)}`, {
    method: "DELETE",
  });
}

export function compareExperiments(baselineId, candidateId) {
  return request("/api/experiments/compare", {
    method: "POST",
    body: JSON.stringify({ baseline_id: baselineId, candidate_id: candidateId }),
  });
}

export function getCase(casePayload, includeStorage = false) {
  return request("/api/case", {
    method: "POST",
    body: JSON.stringify({ case: casePayload, include_storage: includeStorage }),
  });
}

export function normalizeResponse(value) {
  if (Array.isArray(value)) return value.map(normalizeResponse);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, normalizeResponse(item)]));
  }
  if (value === "Infinity") return Infinity;
  if (value === "-Infinity") return -Infinity;
  if (value === "NaN") return NaN;
  return value;
}

export function downloadJson(filename, payload) {
  downloadBlob(filename, JSON.stringify(payload, jsonReplacer, 2), "application/json");
}

export function downloadCsv(filename, rows) {
  const columns = [
    "id", "analysis_mode", "model", "attention_display", "formula_id", "seq_len",
    "encoder_state", "encoder_tail_policy", "generated_global_payload_bytes", "logical_read_payload_bytes", "logical_write_payload_bytes",
    "prefix_ratio", "workload_shape", "block_size_tokens", "arrival_model", "max_io_bytes",
    "configured_prefix_ratio", "shareable_prefix_ratio", "resident_prefix_ratio",
    "recovery_boundary_ratio", "configured_prefix_tokens_p50", "shareable_prefix_tokens_p50",
    "resident_prefix_tokens_p50", "recovery_boundary_tokens_p50", "replay_tokens_p50",
    "suffix_tokens_p50", "resident_cache_bytes_p50", "planned_restore_bytes_p50",
    "replay_gate_bytes_p50", "suffix_gate_bytes_p50", "blocking_checkpoint_bytes_p50",
    "stranded_resident_bytes_p50", "net_saved_flops_p50", "net_saved_flops_ratio_p50",
    "optimistic_overlap_GBps_p50", "optimistic_overlap_GBps_p99",
    "optimistic_overlap_GBps_max", "optimistic_overlap_status",
    "dependency_aware_break_even_GBps_p50", "dependency_aware_break_even_GBps_p99",
    "dependency_aware_break_even_GBps_max", "dependency_aware_break_even_status",
    "no_cache_ttft_us_p50", "optimistic_ttft_us_p50", "dependency_aware_ttft_us_p50",
    "no_overlap_ttft_us_p50", "no_overlap_ttft_status",
    "dependency_ttft_savings_us_p50",
    "read_avg_GBps", "write_avg_GBps", "aggregate_dependency_required_GBps",
    "aggregate_dependency_chunked_iops", "retain_all_capacity_GiB", "cache_total_capacity_GiB",
    "cache_eviction_GiB", "capacity_bottleneck_status",
  ];
  if (rows.some((row) => row.metrics?.cache_total_capacity_bytes !== undefined)) {
    columns.push(
      "cache_actual_block_hit_rate",
      "cache_actual_byte_hit_rate",
      "cache_total_capacity_bytes",
      "cache_inserted_bytes",
      "cache_deduplicated_insert_bytes",
      "cache_eviction_bytes",
      "cache_self_eviction_bytes",
      "cache_rejected_bytes",
      "cache_peak_protected_bytes",
      "cache_final_used_bytes",
      "cache_radix_node_count",
      "cache_radix_leaf_count",
      "cache_radix_max_depth",
    );
  }
  columns.push(
    "evaluation_status", "evaluation_reason", "full_hit_finalization_status_counts", "compute_algorithms",
    "recovery_assumption",
    "recovery_outcome",
    "recovery_status_counts",
    "recovery_reason_counts",
    "recovery_policy_summary",
    "ttft_savings_outcome",
    "ttft_savings_status_counts",
    "optimistic_overlap_finite_request_count",
    "optimistic_overlap_total_request_count",
    "optimistic_overlap_status_counts",
    "dependency_break_even_finite_request_count",
    "dependency_break_even_total_request_count",
    "dependency_break_even_status_counts",
  );
  columns.push(...EXTRA_METRIC_KEYS.filter((key) => !columns.includes(key)));
  const lines = [columns.join(",")];
  for (const row of rows) {
    const flat = { ...row, ...(row.metrics || {}), ...semanticsForCsv(row) };
    lines.push(columns.map((column) => csvCell(flat[column])).join(","));
  }
  downloadBlob(filename, lines.join("\n"), "text/csv;charset=utf-8");
}

export async function copyText(value) {
  await navigator.clipboard.writeText(value);
}

function downloadBlob(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  if (
    value === Infinity || value === -Infinity || Number.isNaN(value)
    || value === "Infinity" || value === "+Infinity"
    || value === "-Infinity" || value === "NaN"
  ) return "";
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function jsonReplacer(_key, value) {
  if (value === Infinity) return "Infinity";
  if (value === -Infinity) return "-Infinity";
  if (Number.isNaN(value)) return "NaN";
  return value;
}
