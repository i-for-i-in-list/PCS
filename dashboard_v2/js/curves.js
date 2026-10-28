import {
  METRICS, VALUE_STATES, classifyValue, formatMetric, formatMetricForRow, formatXAxis, metricValueState,
  numericValue, seriesKey, seriesIdentity, stableIdentity, tooltipHtml, axisValue, axisLabel, scaleLabel,
} from "./metrics.js";
import {
  attachTooltip, buildXAxisLayout, buildYScale, chartColor, drawAxes, svgElement,
} from "./chart_core.js";
import { ttftSavingsOutcome } from "./result_semantics.js";

const VIEWBOX = { width: 1020, height: 540 };
const BOX = { left: 92, right: 994, top: 28, bottom: 474 };
const VISIBLE_SAMPLE_COUNT = 10;
const DASH_PATTERNS = ["", "10 5", "2 5", "11 4 2 4", "1 4", "13 4 3 4 3 4"];
const MARKER_SHAPES = ["circle", "square", "triangle", "diamond"];
const ATTENTION_COLORS = Object.freeze({
  gqa: "#1d5fd1",
  mqa: "#1d5fd1",
  mha: "#1d5fd1",
  mla: "#dc2626",
  dsa: "#059669",
  deepseek_v4: "#7155bd",
});
const FALLBACK_COLORS = [
  "#2868c7", "#087568", "#bd6500", "#7155bd", "#c33d63", "#287f9c",
  "#58851e", "#9c4f16", "#4859b8", "#008a73", "#aa3d32", "#5d6875",
];
const SHADE_VARIANTS = [0, .18, -.14, .31, -.24];

export function renderCurves(options) {
  const {
    frame, legend, tooltip, rows, metric, xMode, hiddenSeries, selectedId,
    threshold, smoothGuides, onSelect, onToggleSeries, onOnlySeries,
  } = options;
  frame.replaceChildren();
  legend.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "Run a sweep to plot curves.";
    frame.append(empty);
    return curveStats([]);
  }

  const plottable = rows.filter((row) => {
    const value = axisValue(row, xMode);
    return Number.isFinite(value) && (xMode === "context-length" ? value > 0 : value >= 0);
  });
  const grouped = groupSeries(plottable, xMode);
  const stats = curveStats(grouped);
  stats.missingAxisCount = rows.length - plottable.length;
  renderLegend(legend, grouped, hiddenSeries, selectedId, onToggleSeries, onOnlySeries, plottable, metric, xMode);
  const visible = grouped.filter((series) => !hiddenSeries.has(series.key));
  const visibleRows = visible.flatMap((series) => series.rows);
  stats.visibleSeriesCount = visible.length;
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = grouped.length ? "All series are hidden. Use Show all to restore them." : "No cases have a valid value for this horizontal axis.";
    frame.append(empty);
    return stats;
  }

  const svg = svgElement("svg", {
    viewBox: `0 0 ${VIEWBOX.width} ${VIEWBOX.height}`,
    role: "group",
    "aria-label": `${METRICS[metric].label} sweep curves`,
  });
  frame.append(svg);

  const semanticLaneStates = visible.flatMap((series) => series.rows.map((row) => metricValueState(row, metric)))
    .filter((state) => isSemanticLaneState(state));
  const plotBox = semanticLaneStates.length ? { ...BOX, left: 160 } : BOX;
  const xScale = buildXAxisLayout(
    visible.flatMap((series) => series.rows.map((row) => axisValue(row, xMode))),
    xMode,
    plotBox.left,
    plotBox.right,
  );
  const yValues = visible.flatMap((series) => series.rows.flatMap((row) => (
    isSemanticLaneState(metricValueState(row, metric))
      ? []
      : [numericValue(row.metrics?.[metric])]
  )));
  const metricDefinition = METRICS[metric];
  stats.zeroOnly = metricDefinition.scale === "base2"
    && visibleRows.every((row) => metricValueState(row, metric) === VALUE_STATES.ZERO);
  const yScale = stats.zeroOnly
    ? zeroOnlyScale(plotBox)
    : buildYScale(yValues, metricDefinition.scale, plotBox.top, plotBox.bottom, semanticLaneStates, metricDefinition);
  const yScaleLabel = stats.zeroOnly ? "zero-value view" : scaleLabel(metricDefinition);
  if (stats.zeroOnly) renderZeroNotice(frame, visible, metric, xMode);
  drawAxes(svg, {
    xScale,
    yScale,
    xFormatter: (value) => formatXAxis(value, xMode),
    yMetric: metric,
    box: plotBox,
    xLabel: axisLabel(xMode),
    yLabel: `${metricDefinition.label} (${metricDefinition.unit}, ${yScaleLabel})`,
    threshold: stats.zeroOnly ? NaN : threshold,
  });

  const points = [];
  for (const series of visible) {
    const orderedRows = [...series.rows].sort(
      (a, b) => axisValue(a, xMode) - axisValue(b, xMode),
    );
    const selectedSeries = orderedRows.some((row) => row.id === selectedId);
    for (const segment of finiteSegments(orderedRows, metric, xMode, xScale, yScale)) {
      if (segment.length < 2) continue;
      svg.append(svgElement("path", {
        d: smoothGuides ? shapePreservingPath(segment) : observedPath(segment),
        class: `series-path${selectedSeries ? " is-highlighted" : ""}`,
        stroke: series.color,
        "stroke-dasharray": series.dash || null,
        "data-series": series.key,
        "data-model": series.model,
        "data-attention": series.attention,
        "data-seq-len": series.seqLen,
        "data-block-size": series.blockSize,
      }));
    }

    const visibleIndexes = visibleSampleIndexes(orderedRows.length);
    orderedRows.forEach((row, index) => {
      const value = row.metrics?.[metric];
      const state = metricValueState(row, metric);
      const x = xScale.xForValue(axisValue(row, xMode));
      const y = isSemanticLaneState(state)
        ? yScale.semanticY(state)
        : yScale(numericValue(value));
      const selected = row.id === selectedId;
      const isSpecial = state !== VALUE_STATES.POSITIVE;
      if (visibleIndexes.has(index) || selected || isSpecial) {
        const marker = createVisibleMarker({
          x, y, state, color: series.color, shape: series.shape, selected,
        });
        marker.style.setProperty("--point-color", series.color);
        marker.setAttribute("aria-hidden", "true");
        svg.append(marker);
        if (state === VALUE_STATES.INFINITE) {
          svg.append(svgElement("text", {
            x, y: y - 9, class: "point-infinity-label", "text-anchor": "middle", "aria-hidden": "true",
          }, numericValue(value) < 0 ? "−∞" : "+∞"));
        }
      }

      const point = svgElement("circle", {
        cx: x,
        cy: y,
        r: 9,
        class: `data-point point-hit-target state-${state}${selected ? " is-selected" : ""}`,
        role: "button",
        tabindex: selected ? "0" : "-1",
        "aria-label": pointAccessibleLabel(row, metric, state, xMode),
        "data-case-id": row.id,
        "data-series": series.key,
        "data-x-value": axisValue(row, xMode),
      });
      point.style.setProperty("--point-color", series.color);
      attachTooltip(point, tooltip, tooltipHtml(row, metric), (element) => onSelect(row, element));
      svg.append(point);
      points.push(point);
    });
  }
  wireRovingPoints(points);
  return stats;
}

function zeroOnlyScale(box) {
  const middle = (box.top + box.bottom) / 2;
  const scale = () => middle;
  return Object.assign(scale, {
    ticks: [],
    specialLanes: [{ key: "zero", label: "Zero", y: middle }],
    plotTop: box.top,
    plotBottom: box.bottom,
    zeroY: middle,
  });
}

function renderZeroNotice(frame, visible, metric, xMode) {
  const rows = visible.flatMap((series) => series.rows);
  const sampleCount = new Set(rows.map((row) => axisValue(row, xMode))).size;
  const notice = document.createElement("div");
  notice.className = "chart-state-note";
  notice.setAttribute("role", "note");
  const title = document.createElement("strong");
  title.textContent = `All ${rows.length} visible cases are ${formatMetric(0, metric)}`;
  const detail = document.createElement("p");
  detail.textContent = `${visible.length} visible series · ${sampleCount} sampled X ${sampleCount === 1 ? "value" : "values"}. Zero is shown in a separate lane outside the log scale.`;
  if (rows.length > sampleCount) {
    detail.textContent += " Zero-valued series overlap; use “only” in the legend to inspect one series.";
  }
  if (metric.startsWith("optimistic_overlap_GBps") && rows.every((row) => ttftSavingsOutcome(row) === "zero")) {
    detail.textContent += " No TTFT gain under the saved recovery assumptions.";
  }
  notice.append(title, detail);
  frame.append(notice);
}

export function observedPath(points) {
  if (!points.length) return "";
  return points
    .map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .join(" ");
}

function pointAccessibleLabel(row, metric, state, xMode) {
  const miss = Math.max(0, 1 - Number(row.prefix_ratio));
  return [
    row.id,
    row.model,
    ...(row.encoder_state == null ? [] : [`Encoder ${row.encoder_state}`, `output ${row.encoder_tail_policy}`]),
    `sequence length ${row.seq_len}`,
    ...(xMode === "cache-capacity" ? [`cache capacity ${axisValue(row, xMode)} bytes`] : []),
    `hit rate ${(Number(row.prefix_ratio) * 100).toLocaleString("en-US", { maximumFractionDigits: 6 })} percent`,
    `miss ratio ${miss === 0 ? "zero" : miss.toExponential(2)}`,
    `${METRICS[metric].label} ${formatMetricForRow(row, metric)}`,
    `state ${state}`,
  ].join(", ");
}

export function shapePreservingPath(points) {
  if (!points.length) return "";
  const commands = [`M${points[0].x.toFixed(2)},${points[0].y.toFixed(2)}`];
  if (points.length === 1) return commands[0];
  if (points.length === 2 || points.some((point, index) => index && point.x <= points[index - 1].x)) {
    return [
      commands[0],
      ...points.slice(1).map((point) => `L${point.x.toFixed(2)},${point.y.toFixed(2)}`),
    ].join(" ");
  }

  const tangents = pchipTangents(points);
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const width = end.x - start.x;
    const low = Math.min(start.y, end.y);
    const high = Math.max(start.y, end.y);
    const cp1 = {
      x: start.x + width / 3,
      y: clamp(start.y + tangents[index] * width / 3, low, high),
    };
    const cp2 = {
      x: end.x - width / 3,
      y: clamp(end.y - tangents[index + 1] * width / 3, low, high),
    };
    commands.push(
      `C${cp1.x.toFixed(2)},${cp1.y.toFixed(2)} ${cp2.x.toFixed(2)},${cp2.y.toFixed(2)} ${end.x.toFixed(2)},${end.y.toFixed(2)}`,
    );
  }
  return commands.join(" ");
}

export function visibleSampleIndexes(length) {
  if (length <= 0) return new Set();
  if (length <= 12) return new Set(Array.from({ length }, (_, index) => index));
  const indexes = new Set();
  for (let index = 0; index < VISIBLE_SAMPLE_COUNT; index += 1) {
    indexes.add(Math.round(index * (length - 1) / (VISIBLE_SAMPLE_COUNT - 1)));
  }
  return indexes;
}

function pchipTangents(points) {
  const count = points.length;
  const widths = [];
  const slopes = [];
  for (let index = 0; index < count - 1; index += 1) {
    const width = points[index + 1].x - points[index].x;
    widths.push(width);
    slopes.push((points[index + 1].y - points[index].y) / width);
  }
  if (count === 2) return [slopes[0], slopes[0]];

  const tangents = new Array(count).fill(0);
  tangents[0] = endpointTangent(widths[0], widths[1], slopes[0], slopes[1]);
  tangents[count - 1] = endpointTangent(
    widths.at(-1), widths.at(-2), slopes.at(-1), slopes.at(-2),
  );
  for (let index = 1; index < count - 1; index += 1) {
    const previous = slopes[index - 1];
    const next = slopes[index];
    if (previous === 0 || next === 0 || Math.sign(previous) !== Math.sign(next)) {
      tangents[index] = 0;
      continue;
    }
    const previousWidth = widths[index - 1];
    const nextWidth = widths[index];
    const firstWeight = 2 * nextWidth + previousWidth;
    const secondWeight = nextWidth + 2 * previousWidth;
    tangents[index] = (firstWeight + secondWeight)
      / (firstWeight / previous + secondWeight / next);
  }
  return tangents;
}

function endpointTangent(endpointWidth, neighborWidth, endpointSlope, neighborSlope) {
  let tangent = ((2 * endpointWidth + neighborWidth) * endpointSlope
    - endpointWidth * neighborSlope) / (endpointWidth + neighborWidth);
  if (Math.sign(tangent) !== Math.sign(endpointSlope)) return 0;
  if (Math.sign(endpointSlope) !== Math.sign(neighborSlope)
    && Math.abs(tangent) > Math.abs(3 * endpointSlope)) {
    tangent = 3 * endpointSlope;
  }
  return tangent;
}

export function finiteSegments(rows, metric, xMode, xScale, yScale) {
  const segments = [];
  let current = [];
  let currentRegion = null;
  const flush = () => {
    if (current.length) segments.push(current);
    current = [];
    currentRegion = null;
  };
  for (const row of rows) {
    const state = metricValueState(row, metric);
    const numericState = classifyValue(row.metrics?.[metric]);
    const isFiniteLineValue = state === VALUE_STATES.POSITIVE
      || state === VALUE_STATES.ZERO
      || (state === VALUE_STATES.MIXED && numericState !== VALUE_STATES.MISSING && numericState !== VALUE_STATES.INFINITE)
      || (["symlog", "linear"].includes(METRICS[metric].scale)
        && state === VALUE_STATES.NEGATIVE);
    const isSemanticFullHit = xMode === "miss-log" && Number(row.prefix_ratio) >= 1;
    const isZeroCapacity = xMode === "cache-capacity" && axisValue(row, xMode) === 0;
    if (!isFiniteLineValue || isSemanticFullHit || isZeroCapacity || !Number.isFinite(axisValue(row, xMode))) {
      flush();
      continue;
    }
    // Zero has its own lane on log axes: connect adjacent zero samples, but
    // never bridge the lane to a positive value or across a missing sample.
    const region = METRICS[metric].scale === "base2" && numericState === VALUE_STATES.ZERO ? "zero" : "numeric";
    if (currentRegion !== region) flush();
    currentRegion = region;
    current.push({
      x: xScale.xForValue(axisValue(row, xMode)),
      y: yScale(numericValue(row.metrics?.[metric])),
    });
  }
  flush();
  return segments;
}

export function groupSeries(rows, xMode = "hit-linear") {
  const blockSizes = [...new Set(rows.map((row) => Number(row.block_size_tokens)))].sort((a, b) => a - b);
  const arrivals = [...new Set(rows.map((row) => String(row.arrival_model)))];
  const maxIoValues = [...new Set(rows.map((row) => row.max_io_bytes ?? null))];
  const dashValues = [...new Set(rows.map((row) => xMode === "context-length" ? Number(row.prefix_ratio) : Number(row.seq_len)))].sort((a, b) => a - b);
  const contextKeys = varyingContextKeys(rows, xMode);
  const modelColors = buildModelColors(rows);
  const groups = new Map();
  for (const row of rows) {
    const key = seriesKey(row, xMode);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const series = [...groups.entries()].map(([key, seriesRows]) => {
    const row = seriesRows[0];
    return {
      key,
      rows: seriesRows,
      color: modelColors.get(row.model),
      dash: DASH_PATTERNS[(Math.max(0, dashValues.indexOf(xMode === "context-length" ? Number(row.prefix_ratio) : Number(row.seq_len))) + (row.encoder_state === "available" ? 1 : 0) + (row.encoder_tail_policy === "retain" ? 2 : 0)) % DASH_PATTERNS.length],
      shape: MARKER_SHAPES[Math.max(0, blockSizes.indexOf(Number(row.block_size_tokens))) % MARKER_SHAPES.length],
      label: compactSeriesLabel(row, {
        showBlock: blockSizes.length > 1,
        showArrival: arrivals.length > 1,
        showMaxIo: maxIoValues.length > 1,
        contextKeys, xMode,
      }),
      model: row.model,
      attention: row.attention || "unknown",
      seqLen: Number(row.seq_len),
      blockSize: Number(row.block_size_tokens),
    };
  }).sort((first, second) => (
    attentionRank(first.attention) - attentionRank(second.attention)
    || first.model.localeCompare(second.model)
    || first.seqLen - second.seqLen
    || first.blockSize - second.blockSize
    || first.key.localeCompare(second.key)
  ));
  // Compact labels for ordinary capacities, exact values if labels collide.
  const labelCounts = new Map();
  for (const item of series) labelCounts.set(item.label, (labelCounts.get(item.label) || 0) + 1);
  for (const item of series) {
    if (labelCounts.get(item.label) > 1) {
      const row = item.rows[0];
      const exact = [];
      if (xMode !== "cache-capacity" && row.metrics?.cache_total_capacity_bytes != null) exact.push(`${row.metrics.cache_total_capacity_bytes} B cache`);
      if (xMode === "cache-capacity" || xMode === "context-length") exact.push(`prefix=${row.prefix_ratio}`);
      item.label += exact.length ? ` / ${exact.join(" / ")}` : ` / ${row.case?.model_config || row.model}`;
    }
  }
  return series;
}

function compactSeriesLabel(row, dimensions) {
  const parts = [row.model];
  if (dimensions.xMode !== "context-length") parts.push(formatSeqLen(row.seq_len));
  if (["context-length", "cache-capacity"].includes(dimensions.xMode)) parts.push(`prefix ${Number((Number(row.prefix_ratio) * 100).toPrecision(12))}%`);
  if (dimensions.xMode !== "cache-capacity" && row.metrics?.cache_total_capacity_bytes != null) parts.push(`Cache ${Number((Number(row.metrics.cache_total_capacity_bytes) / (1024 ** 3)).toPrecision(6))} GiB`);
  if (row.encoder_state != null) parts.push(`Encoder ${row.encoder_state}`, `output ${row.encoder_tail_policy}`);
  if (dimensions.showBlock) parts.push(`B${row.block_size_tokens}`);
  if (dimensions.showArrival) parts.push(row.arrival_model);
  if (dimensions.showMaxIo) parts.push(`IO ${row.max_io_bytes ?? "∞"}`);
  const context = seriesContext(row, dimensions.xMode);
  for (const key of dimensions.contextKeys) {
    const value = context[key];
    parts.push(`${key.replaceAll("_", " ")}: ${typeof value === "string" ? value.split("/").at(-1) : JSON.stringify(stableIdentity(value)) ?? "unset"}`);
  }
  return parts.join(" / ");
}

function seriesContext(row, xMode) {
  const identity = seriesIdentity(row, xMode);
  const basic = new Set(["model", "model_config", "seq_len", "prefix_ratio", "block_size_tokens", "arrival_model", "max_io_bytes", "encoder_state", "encoder_tail_policy", "config_digests"]);
  const context = Object.fromEntries(Object.entries(identity).filter(([key]) => !basic.has(key)));
  if (context.cache_replay_config) {
    context.cache_replay_config = { ...context.cache_replay_config };
    delete context.cache_replay_config.total_capacity_bytes;
  }
  return context;
}

function varyingContextKeys(rows, xMode) {
  const contexts = rows.map((row) => seriesContext(row, xMode));
  const keys = new Set(contexts.flatMap(Object.keys));
  return [...keys].filter((key) => new Set(contexts.map((item) => JSON.stringify(stableIdentity(item[key])))).size > 1);
}

function formatSeqLen(value) {
  const number = Number(value);
  if (number >= 1024 && number % 1024 === 0) return `${number / 1024}k`;
  return String(number);
}

function attentionRank(attention) {
  const family = attentionFamily(attention);
  if (["gqa", "mqa", "mha"].includes(family)) return 0;
  if (family === "mla") return 1;
  if (family === "dsa") return 2;
  if (family === "deepseek_v4") return 3;
  return 4;
}

function buildModelColors(rows) {
  const modelMetadata = new Map();
  for (const row of rows) {
    if (!modelMetadata.has(row.model)) modelMetadata.set(row.model, row.attention || "unknown");
  }
  const familyModels = new Map();
  for (const [model, attention] of modelMetadata) {
    const family = attentionFamily(attention);
    if (!familyModels.has(family)) familyModels.set(family, []);
    familyModels.get(family).push(model);
  }
  for (const models of familyModels.values()) models.sort((a, b) => a.localeCompare(b));

  const colors = new Map();
  for (const [model, attention] of modelMetadata) {
    colors.set(model, modelColor(model, attention));
  }
  return colors;
}

export function modelColor(model, attention) {
  const family = attentionFamily(attention);
  const slot = stableHash(`${attention}:${model}`) % FALLBACK_COLORS.length;
  const base = ATTENTION_COLORS[family]
    ? chartColor(`chart-attention-${family}`, ATTENTION_COLORS[family])
    : chartColor(`chart-series-${slot}`, FALLBACK_COLORS[slot]);
  const shade = SHADE_VARIANTS[stableHash(`${family}:${model}`) % SHADE_VARIANTS.length];
  return shadeHex(base, shade);
}

function attentionFamily(attention) {
  const normalized = String(attention || "").toLowerCase();
  if (["gqa", "mqa", "mha"].includes(normalized)) return normalized;
  if (normalized.includes("deepseek_v4") || normalized.includes("v4")) return "deepseek_v4";
  if (normalized.includes("mla")) return "mla";
  if (normalized.includes("dsa")) return "dsa";
  return normalized || "unknown";
}

function renderLegend(container, seriesList, hiddenSeries, selectedId, onToggle, onOnly, rows, metric, xMode) {
  for (const series of seriesList) {
    const row = document.createElement("div");
    row.className = `legend-row${hiddenSeries.has(series.key) ? " is-hidden" : ""}${series.rows.some((item) => item.id === selectedId) ? " is-selected" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "legend-item";
    button.dataset.series = series.key;
    button.setAttribute("aria-pressed", String(!hiddenSeries.has(series.key)));
    button.append(createLegendSwatch(series, metric));
    const label = document.createElement("span");
    label.textContent = series.label;
    label.title = series.label;
    button.append(label);
    const stateCounts = new Map();
    for (const item of series.rows) {
      const state = metricValueState(item, metric);
      stateCounts.set(state, (stateCounts.get(state) || 0) + 1);
    }
    const cached = stateCounts.get(VALUE_STATES.CACHED_NOT_RESUMABLE) || 0;
    const mixed = stateCounts.get(VALUE_STATES.MIXED) || 0;
    const zeroOnly = stateCounts.get(VALUE_STATES.ZERO) === series.rows.length;
    if (cached || mixed || zeroOnly) {
      const badge = document.createElement("small");
      badge.className = "legend-status-badge";
      badge.textContent = cached
        ? `${cached} cached-not-resumable`
        : mixed ? `${mixed} mixed` : `${series.rows.length} zero-valued cases`;
      button.append(badge);
    }
    button.addEventListener("click", () => onToggle(series.key));
    const only = document.createElement("button");
    only.type = "button";
    only.className = "legend-only";
    only.textContent = "only";
    only.title = "Only show this series";
    only.setAttribute("aria-label", `Only show ${series.label}`);
    only.addEventListener("click", () => onOnly(series.key, seriesList.map((item) => item.key)));
    row.append(button, only);
    container.append(row);
  }
  const semantics = document.createElement("p");
  semantics.className = "boundary-note";
  semantics.textContent = xMode === "context-length"
    ? "Color = model · dash = fixed prefix / Encoder scenario · marker = block. Each curve keeps the other Case inputs fixed."
    : rows.some((row) => row.encoder_state != null)
    ? "Color = model · dash = sequence length / Encoder scenario · marker = block · legend names identify the exact state and output combination."
    : "Color = stable model identity · dash = seq_len · marker = block · hollow = Zero · amber = Mixed · status lanes = non-numeric outcomes · triangle = ±Infinity";
  container.append(semantics);
  const seqCount = new Set(rows.map((row) => Number(row.seq_len))).size;
  const blockCount = new Set(rows.map((row) => Number(row.block_size_tokens))).size;
  if (seqCount > DASH_PATTERNS.length || blockCount > MARKER_SHAPES.length) {
    const warning = document.createElement("p");
    warning.className = "boundary-note encoding-warning";
    warning.textContent = `Encoding capacity exceeded (${seqCount} seq lengths, ${blockCount} block sizes). Reduce the selection or use Heatmap to avoid repeated dash/marker encodings.`;
    container.append(warning);
  }
}

function createLegendSwatch(series, metric) {
  const swatch = svgElement("svg", {
    class: "legend-swatch",
    viewBox: "0 0 42 16",
    "aria-hidden": "true",
  });
  const statusOnly = series.rows.every((row) => isSemanticLaneState(metricValueState(row, metric)));
  const zeroOnly = series.rows.every((row) => metricValueState(row, metric) === VALUE_STATES.ZERO);
  if (!statusOnly) {
    swatch.append(svgElement("line", {
      x1: 2, x2: 40, y1: 8, y2: 8, stroke: series.color,
      "stroke-width": 2.4, "stroke-dasharray": series.dash || null,
    }));
  }
  swatch.append(createShapeMarker({
    x: 21, y: 8, shape: statusOnly ? "diamond" : series.shape, size: statusOnly ? 4 : 3.2,
    attributes: {
      fill: statusOnly ? "var(--chart-semantic, #fff2c9)" : zeroOnly ? "var(--panel)" : series.color,
      stroke: series.color,
      "stroke-width": statusOnly ? 1.7 : 1,
    },
  }));
  return swatch;
}

function wireRovingPoints(points) {
  if (!points.length) return;
  if (!points.some((point) => point.getAttribute("tabindex") === "0")) points[0].setAttribute("tabindex", "0");
  points.forEach((point, index) => {
    point.addEventListener("keydown", (event) => {
      let target = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") target = points[(index + 1) % points.length];
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") target = points[(index - 1 + points.length) % points.length];
      if (event.key === "Home") target = points[0];
      if (event.key === "End") target = points[points.length - 1];
      if (!target) return;
      event.preventDefault();
      points.forEach((item) => item.setAttribute("tabindex", item === target ? "0" : "-1"));
      target.focus();
    });
  });
}

function createVisibleMarker({ x, y, state, color, shape, selected }) {
  if (state === VALUE_STATES.INFINITE) {
    return createShapeMarker({
      x, y, shape: "triangle", size: selected ? 7 : 6,
      attributes: {
        class: `sample-marker state-infinite${selected ? " is-selected" : ""}`,
        fill: color,
        stroke: selected ? "var(--chart-selected, #17212b)" : color,
        "stroke-width": selected ? 2.2 : 1.4,
      },
    });
  }
  if (isSemanticLaneState(state)) {
    return createShapeMarker({
      x, y, shape: state === VALUE_STATES.CACHED_NOT_RESUMABLE ? "diamond" : "square",
      size: selected ? 6 : 4.5,
      attributes: {
        class: `sample-marker state-${state}${selected ? " is-selected" : ""}`,
        fill: "var(--chart-semantic, #fff2c9)",
        stroke: selected ? "var(--chart-selected, #17212b)" : color,
        "stroke-width": selected ? 2.2 : 1.7,
      },
    });
  }
  if (state === VALUE_STATES.MIXED) {
    return createShapeMarker({
      x, y, shape, size: selected ? 5.5 : 4,
      attributes: {
        class: `sample-marker state-mixed${selected ? " is-selected" : ""}`,
        fill: "var(--chart-semantic, #fff2c9)",
        stroke: selected ? "var(--chart-selected, #17212b)" : color,
        "stroke-width": selected ? 2.2 : 1.8,
      },
    });
  }
  return createShapeMarker({
    x, y, shape, size: selected ? 5.5 : 3.2,
    attributes: {
      class: `sample-marker state-${state}${selected ? " is-selected" : ""}`,
      fill: color,
      stroke: selected ? "var(--chart-selected, #17212b)" : color,
      "stroke-width": selected ? 2.2 : 1.2,
    },
  });
}

function isSemanticLaneState(state) {
  return [
    VALUE_STATES.NOT_EVALUATED,
    VALUE_STATES.CACHED_NOT_RESUMABLE,
    VALUE_STATES.NO_FINITE_BREAK_EVEN,
    VALUE_STATES.NO_OVERLAP_WINDOW,
  ].includes(state);
}

function createShapeMarker({ x, y, shape, size, attributes }) {
  if (shape === "square") {
    return svgElement("rect", {
      x: x - size, y: y - size, width: size * 2, height: size * 2, ...attributes,
    });
  }
  if (shape === "triangle") {
    return svgElement("path", {
      d: `M${x},${y - size} L${x - size * .9},${y + size * .78} L${x + size * .9},${y + size * .78} Z`,
      ...attributes,
    });
  }
  if (shape === "diamond") {
    return svgElement("path", {
      d: `M${x},${y - size} L${x - size},${y} L${x},${y + size} L${x + size},${y} Z`,
      ...attributes,
    });
  }
  return svgElement("circle", { cx: x, cy: y, r: size, ...attributes });
}

function curveStats(seriesList) {
  const counts = seriesList.map((series) => series.rows.length);
  return {
    seriesCount: seriesList.length,
    visibleSeriesCount: seriesList.length,
    sampleCountMin: counts.length ? Math.min(...counts) : 0,
    sampleCountMax: counts.length ? Math.max(...counts) : 0,
    sampleCountTotal: counts.reduce((sum, count) => sum + count, 0),
  };
}

function shadeHex(hex, amount) {
  if (!amount) return hex;
  const target = amount > 0 ? 255 : 0;
  const ratio = Math.abs(amount);
  const channels = [1, 3, 5].map((index) => Number.parseInt(hex.slice(index, index + 2), 16));
  const mixed = channels.map((channel) => Math.round(channel + (target - channel) * ratio));
  return `#${mixed.map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

function stableHash(value) {
  let hash = 2166136261;
  for (const character of String(value)) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}
