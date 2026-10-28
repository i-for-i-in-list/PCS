export const RESULT_STATES = Object.freeze({
  NOT_EVALUATED: "not-evaluated",
  CACHED_NOT_RESUMABLE: "cached-not-resumable",
  NO_FINITE_BREAK_EVEN: "no-finite-break-even",
  NO_OVERLAP_WINDOW: "no-overlap-window",
  MIXED: "mixed",
});

const OPTIMISTIC_METRICS = new Set([
  "optimistic_overlap_GBps_p50",
  "optimistic_overlap_GBps_p99",
  "optimistic_overlap_GBps_max",
]);
const DEPENDENCY_METRICS = new Set([
  "dependency_aware_break_even_GBps_p50",
  "dependency_aware_break_even_GBps_p99",
  "dependency_aware_break_even_GBps_max",
]);

export function resultSemantics(row) {
  return row?.result_semantics || fallbackResultSemantics(row || {});
}

export function recoveryOutcome(row) {
  return resultSemantics(row).recovery?.dominant_status || "missing";
}

export function ttftSavingsOutcome(row) {
  return resultSemantics(row).ttft_savings?.dominant_status || "missing";
}

export function thresholdSemantics(row, kind) {
  return resultSemantics(row)[kind] || emptyThresholdSemantics();
}

export function semanticStateForMetric(row, metric) {
  const evaluation = row?.evaluation || resultSemantics(row).evaluation || {};
  if (["not_evaluated", "unsupported"].includes(evaluation.status)
      && (evaluation.aborted || OPTIMISTIC_METRICS.has(metric) || DEPENDENCY_METRICS.has(metric)
          || /^(optimistic_ttft|dependency_aware_ttft|no_overlap_ttft|dependency_ttft|aggregate_dependency|net_saved_flops)/.test(metric))) {
    return RESULT_STATES.NOT_EVALUATED;
  }
  const numeric = Number(row?.metrics?.[metric]);
  const recovery = recoveryOutcome(row);
  if (OPTIMISTIC_METRICS.has(metric)) {
    const semantics = thresholdSemantics(row, "optimistic_overlap");
    if (recovery === "resident_not_recoverable") {
      return RESULT_STATES.CACHED_NOT_RESUMABLE;
    }
    if (semantics.status === "no_overlap_window") {
      return RESULT_STATES.NO_OVERLAP_WINDOW;
    }
    if (semantics.status === "mixed") {
      return Number.isFinite(numeric)
        ? RESULT_STATES.MIXED
        : dominantThresholdLane(semantics);
    }
  }
  if (DEPENDENCY_METRICS.has(metric)) {
    const semantics = thresholdSemantics(row, "dependency_break_even");
    if (recovery === "resident_not_recoverable") {
      return RESULT_STATES.CACHED_NOT_RESUMABLE;
    }
    if (["no_compute_saving", "no_replay_slack"].includes(semantics.status)) {
      return RESULT_STATES.NO_FINITE_BREAK_EVEN;
    }
    if (semantics.status === "no_overlap_window") {
      return RESULT_STATES.NO_OVERLAP_WINDOW;
    }
    if (semantics.status === "mixed") {
      return Number.isFinite(numeric)
        ? RESULT_STATES.MIXED
        : dominantThresholdLane(semantics);
    }
  }
  return null;
}

export function semanticLaneLabel(state) {
  if (state === RESULT_STATES.NOT_EVALUATED) return "Not evaluated";
  if (state === RESULT_STATES.CACHED_NOT_RESUMABLE) return "Cached, not resumable";
  if (state === RESULT_STATES.NO_FINITE_BREAK_EVEN) return "No finite break-even";
  if (state === RESULT_STATES.NO_OVERLAP_WINDOW) return "No overlap window";
  return state;
}

export function semanticMetricLabel(row, metric) {
  if (metric === "dependency_ttft_savings_us_p50" && ttftSavingsOutcome(row) === "zero") {
    return "No TTFT gain";
  }
  const state = semanticStateForMetric(row, metric);
  if (state === RESULT_STATES.NOT_EVALUATED) return "Not evaluated";
  if (state === RESULT_STATES.CACHED_NOT_RESUMABLE) return "N/A · cached, not resumable";
  if (state === RESULT_STATES.NO_FINITE_BREAK_EVEN) return "No finite break-even";
  if (state === RESULT_STATES.NO_OVERLAP_WINDOW) return "No overlap window";
  if (state === RESULT_STATES.MIXED) {
    const kind = OPTIMISTIC_METRICS.has(metric) ? "optimistic_overlap" : "dependency_break_even";
    return mixedThresholdLabel(thresholdSemantics(row, kind));
  }
  return null;
}

export function semanticMetricExplanation(row, metric) {
  const state = semanticStateForMetric(row, metric);
  const semantics = resultSemantics(row);
  if (state === RESULT_STATES.NOT_EVALUATED) {
    const evaluation = row?.evaluation || semantics.evaluation || {};
    return evaluation.message || "Full-hit first-token finalization is not modeled. Complete-request performance is not evaluated; known suffix work and restore bytes remain available.";
  }
  if (state === RESULT_STATES.CACHED_NOT_RESUMABLE) {
    const recovery = semantics.recovery || {};
    const affected = Number(recovery.status_counts?.resident_not_recoverable || 0);
    const total = Number(recovery.total_request_count || 0);
    const count = total ? ` ${affected.toLocaleString()} of ${total.toLocaleString()} requests are affected.` : "";
    return `Prefix objects are resident, but no common checkpoint exists at the hit boundary.${count}`;
  }
  if (state === RESULT_STATES.NO_FINITE_BREAK_EVEN) {
    return "The cache path has no positive dependency-aware compute saving at any finite restore bandwidth.";
  }
  if (state === RESULT_STATES.NO_OVERLAP_WINDOW) {
    return "Restore has no compute window in which it can be fully hidden.";
  }
  if (state === RESULT_STATES.MIXED) {
    const kind = OPTIMISTIC_METRICS.has(metric) ? "optimistic_overlap" : "dependency_break_even";
    return mixedThresholdLabel(thresholdSemantics(row, kind));
  }
  return "";
}

export function riskForResultRow(row) {
  if (["not_evaluated", "unsupported"].includes((row?.evaluation || resultSemantics(row).evaluation || {}).status)) return "not-evaluated";
  const metrics = row?.metrics || {};
  if (Object.values(metrics).some((value) => numericState(value) === "infinite")) return "infinite";
  if (recoveryOutcome(row) === "resident_not_recoverable") return "unrecoverable";
  const dependency = thresholdSemantics(row, "dependency_break_even");
  if (dependency.status === "mixed"
      && (dependency.finite_required_count == null || Number(dependency.finite_required_count) > 0)) return "mixed";
  if (["no_compute_saving", "no_replay_slack"].includes(dependency.status)
      || (dependency.status === "mixed" && dependency.finite_required_count != null
        && Number(dependency.finite_required_count) === 0)) {
    return "no-break-even";
  }
  const ttft = ttftSavingsOutcome(row);
  if (ttft === "negative") return "negative";
  if (row?.recommendation_evaluation?.feasible === false) return "constraint";
  if (ttft === "zero") return "zero";
  return "normal";
}

export function riskReasonsForResultRow(row, metric = null, threshold = NaN) {
  const reasons = [];
  const semantics = resultSemantics(row);
  const caseEvaluation = row?.evaluation || semantics.evaluation || {};
  if (["not_evaluated", "unsupported"].includes(caseEvaluation.status)) {
    reasons.push(caseEvaluation.message || "Complete first-token path not evaluated");
  }
  const recovery = semantics.recovery || {};
  if (recovery.dominant_status === "resident_not_recoverable") {
    reasons.push(countedLabel(
      "Cached but not resumable",
      recovery.status_counts?.resident_not_recoverable,
      recovery.total_request_count,
    ));
  }
  const ttft = semantics.ttft_savings || {};
  if (ttft.dominant_status === "negative") reasons.push("Cache path is slower");
  if (ttft.dominant_status === "zero") reasons.push("No TTFT gain");

  const dependency = semantics.dependency_break_even || {};
  if (dependency.status === "mixed"
      && (dependency.finite_required_count == null || Number(dependency.finite_required_count) > 0)) {
    reasons.push(mixedThresholdLabel(dependency));
  } else if (["no_compute_saving", "no_replay_slack"].includes(dependency.status)
      || (dependency.status === "mixed" && dependency.finite_required_count != null
        && Number(dependency.finite_required_count) === 0)) {
    reasons.push("No finite break-even");
  }
  if (Object.values(row?.metrics || {}).some((value) => numericState(value) === "infinite")) {
    reasons.push("Infinite demand");
  }
  const evaluation = row?.recommendation_evaluation;
  if (evaluation?.feasible === false) {
    const labels = [...new Set((evaluation.failures || []).map((item) => item.constraint).filter(Boolean))];
    reasons.push(`Recommendation constraint failed${labels.length ? `: ${labels.join(", ")}` : ""}`);
  }
  if (metric && Number.isFinite(threshold) && Number(row?.metrics?.[metric]) > threshold) {
    reasons.push("Over proxy");
  }
  return reasons;
}

export function resultRiskCounts(rows, metric = null, threshold = NaN) {
  const counts = {
    unrecoverable: 0,
    zero: 0,
    mixed: 0,
    noFinite: 0,
    negative: 0,
    infinite: 0,
    exceeded: 0,
    recommendationFailed: 0,
  };
  for (const row of rows) {
    const risk = riskForResultRow(row);
    if (risk === "unrecoverable") counts.unrecoverable += 1;
    if (ttftSavingsOutcome(row) === "zero") counts.zero += 1;
    if (risk === "mixed") counts.mixed += 1;
    if (risk === "no-break-even") counts.noFinite += 1;
    if (ttftSavingsOutcome(row) === "negative") counts.negative += 1;
    if (risk === "infinite") counts.infinite += 1;
    if (row?.recommendation_evaluation?.feasible === false) counts.recommendationFailed += 1;
    if (metric && Number.isFinite(threshold) && Number(row?.metrics?.[metric]) > threshold) counts.exceeded += 1;
  }
  return counts;
}

export function semanticsForCsv(row) {
  const semantics = resultSemantics(row);
  const recovery = semantics.recovery || {};
  const ttft = semantics.ttft_savings || {};
  const optimistic = semantics.optimistic_overlap || {};
  const dependency = semantics.dependency_break_even || {};
  return {
    evaluation_status: (row?.evaluation || semantics.evaluation || {}).status || "",
    evaluation_reason: (row?.evaluation || semantics.evaluation || {}).reason || "",
    full_hit_finalization_status_counts: stableJson((row?.evaluation || semantics.evaluation || {}).full_hit_finalization_status_counts || {}),
    compute_algorithms: JSON.stringify(row?.compute_algorithms || []),
    recovery_assumption: recovery.assumption
      || row?.case?.modeling_options?.recovery_assumption
      || "model_default",
    recovery_outcome: recovery.dominant_status || "missing",
    recovery_status_counts: stableJson(recovery.status_counts || {}),
    recovery_reason_counts: stableJson(recovery.reason_counts || {}),
    recovery_policy_summary: recovery.policy_summary || "",
    ttft_savings_outcome: ttft.dominant_status || "missing",
    ttft_savings_status_counts: stableJson(ttft.status_counts || {}),
    optimistic_overlap_finite_request_count: optimistic.finite_required_count ?? "",
    optimistic_overlap_total_request_count: optimistic.total_request_count ?? "",
    optimistic_overlap_status_counts: stableJson(optimistic.status_counts || {}),
    dependency_break_even_finite_request_count: dependency.finite_required_count ?? "",
    dependency_break_even_total_request_count: dependency.total_request_count ?? "",
    dependency_break_even_status_counts: stableJson(dependency.status_counts || {}),
  };
}

export function isEffectivelyZero(value, baseline = 0) {
  const number = Number(value);
  const reference = Number(baseline);
  if (!Number.isFinite(number)) return false;
  const tolerance = Math.max(1e-9, Math.abs(Number.isFinite(reference) ? reference : 0) * 1e-12);
  return Math.abs(number) <= tolerance;
}

function fallbackResultSemantics(row) {
  const metrics = row.metrics || {};
  const requestCount = Math.max(1, Number(row.request_count) || 1);
  const residentTokens = Number(metrics.resident_prefix_tokens_p50);
  const residentBytes = Number(metrics.resident_cache_bytes_p50);
  const recoveryBoundary = Number(metrics.recovery_boundary_tokens_p50);
  const replayTokens = Number(metrics.replay_tokens_p50);
  let recoveryStatus = "missing";
  if (Number.isFinite(residentTokens) || Number.isFinite(residentBytes)) {
    if (!(residentTokens > 0) || !(residentBytes > 0)) recoveryStatus = "no_resident_prefix";
    else if (!(recoveryBoundary > 0)) recoveryStatus = "resident_not_recoverable";
    else if (replayTokens > 0) recoveryStatus = "partial_replay";
    else recoveryStatus = "direct_resume";
  }

  const ttftStatus = fallbackTtftStatus(metrics);
  return {
    recovery: {
      dominant_status: recoveryStatus,
      status_counts: { [recoveryStatus]: requestCount },
      reason_counts: {},
      total_request_count: requestCount,
      assumption: row?.case?.modeling_options?.recovery_assumption || "model_default",
      policy_summary: fallbackPolicySummary(row.effective_state_policies || {}),
    },
    ttft_savings: {
      dominant_status: ttftStatus,
      status_counts: { [ttftStatus]: requestCount },
      total_request_count: requestCount,
    },
    optimistic_overlap: fallbackThresholdSemantics(
      metrics.optimistic_overlap_status,
      metrics.optimistic_overlap_GBps_p99,
      requestCount,
    ),
    dependency_break_even: fallbackThresholdSemantics(
      metrics.dependency_aware_break_even_status,
      metrics.dependency_aware_break_even_GBps_p99,
      requestCount,
    ),
  };
}

function fallbackTtftStatus(metrics) {
  const value = Number(metrics.dependency_ttft_savings_us_p50);
  if (!Number.isFinite(value)) return "missing";
  if (isEffectivelyZero(value, metrics.no_cache_ttft_us_p50)) return "zero";
  return value < 0 ? "negative" : "positive";
}

function fallbackThresholdSemantics(status, value, total) {
  const normalized = String(status || (Number.isFinite(Number(value)) ? "finite" : "not_applicable"));
  return {
    status: normalized,
    status_counts: { [normalized]: total },
    finite_required_count: normalized === "finite" ? total : (normalized === "mixed" ? null : 0),
    total_request_count: total,
  };
}

function fallbackPolicySummary(policies) {
  const parts = [];
  for (const [name, value] of Object.entries(policies || {}).sort(([a], [b]) => a.localeCompare(b))) {
    const policy = value?.policy || {};
    let cadence = policy.storage_form || "unknown";
    if (cadence === "snapshot") cadence = policy.interval_tokens == null
      ? "prompt_end"
      : `periodic:${policy.interval_tokens}${policy.retain_prompt_end ? "+prompt_end" : ""}`;
    parts.push(`${name}=${cadence}(${value?.source || "model_default"})`);
  }
  return parts.join(" | ") || "token_state";
}

function dominantThresholdLane(semantics) {
  const counts = semantics.status_counts || {};
  if (Number(counts.no_overlap_window) > 0) return RESULT_STATES.NO_OVERLAP_WINDOW;
  if (Number(counts.no_compute_saving) > 0 || Number(counts.no_replay_slack) > 0) {
    return RESULT_STATES.NO_FINITE_BREAK_EVEN;
  }
  return RESULT_STATES.NO_FINITE_BREAK_EVEN;
}

function mixedThresholdLabel(semantics) {
  const finite = semantics.finite_required_count;
  const total = semantics.total_request_count;
  if (finite != null && Number.isFinite(Number(finite)) && Number(total) > 0) {
    return `Mixed break-even · ${Number(finite).toLocaleString()} / ${Number(total).toLocaleString()} finite`;
  }
  return "Mixed break-even";
}

function countedLabel(label, count, total) {
  if (Number(count) > 0 && Number(total) > 0) {
    return `${label} (${Number(count).toLocaleString()} / ${Number(total).toLocaleString()})`;
  }
  return label;
}

function emptyThresholdSemantics() {
  return {
    status: "not_applicable",
    status_counts: {},
    finite_required_count: 0,
    total_request_count: 0,
  };
}

function numericState(value) {
  const number = Number(value);
  if (value === null || value === undefined || value === "" || Number.isNaN(number)) return "missing";
  if (!Number.isFinite(number)) return "infinite";
  if (number < 0) return "negative";
  if (number === 0) return "zero";
  return "positive";
}

function stableJson(value) {
  const entries = Object.entries(value || {}).sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify(Object.fromEntries(entries));
}
