import {
  METRICS, VALUE_STATES, formatCompact, formatMetric, formatMetricForRow, formatNumber,
  metricValueState, numericValue, tooltipHtml,
} from "./metrics.js";
import {
  addMissingPattern, attachTooltip, svgElement, valueColor,
  valueDomain,
} from "./chart_core.js";
import { ttftSavingsOutcome } from "./result_semantics.js";

const WIDTH = 1020;
const HEIGHT = 540;
const BOX = { left: 96, right: 992, top: 42, bottom: 470 };

export function renderHeatmap(options) {
  const {
    frame, legend, tooltip, rows, globalRows, metric, xMode, slice, selectedId,
    scaleMode, onSelect,
  } = options;
  frame.replaceChildren();
  legend.replaceChildren();
  const sliced = rows.filter((row) => matchesSlice(row, slice));
  if (!sliced.length) {
    renderHeatLegend(legend, METRICS[metric].scale, valueDomain([], METRICS[metric].scale, METRICS[metric]), metric, scaleMode, false);
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "No cases match the selected Heatmap slice.";
    frame.append(empty);
    return { cellCount: 0, rowCount: 0, columnCount: 0 };
  }

  const svg = svgElement("svg", {
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    role: "group",
    "aria-label": `${METRICS[metric].label} heatmap`,
  });
  addMissingPattern(svg);
  frame.append(svg);

  const seqLens = [...new Set(sliced.map((row) => row.seq_len))].sort((a, b) => a - b);
  const hits = [...new Set(sliced.map((row) => row.prefix_ratio))].sort((a, b) => a - b);
  const cellGap = 3;
  const fullHitGap = hits.some((hit) => Number(hit) >= 1) && hits.length > 1 ? 20 : 0;
  const slotWidth = ((BOX.right - BOX.left) - fullHitGap) / Math.max(1, hits.length);
  const cellWidth = Math.max(1, slotWidth - cellGap);
  const xForIndex = (index) => (
    BOX.left + slotWidth * (index + .5)
    + (fullHitGap && Number(hits[index]) >= 1 ? fullHitGap : 0)
  );
  const effectiveTop = seqLens.length === 1 ? 205 : BOX.top;
  const effectiveBottom = seqLens.length === 1 ? 285 : BOX.bottom;
  const band = (effectiveBottom - effectiveTop) / seqLens.length;
  const cellHeight = Math.max(1, band - cellGap);

  const domainRows = scaleMode === "local" ? sliced : (globalRows || rows);
  const domainValues = domainRows.flatMap((row) => isSemanticHeatState(metricValueState(row, metric))
    ? [] : [numericValue(row.metrics?.[metric])]);
  const domain = valueDomain(
    domainValues,
    METRICS[metric].scale,
    METRICS[metric],
  );
  const hasNumericScale = domainValues.some((value) => Number.isFinite(value)
    && (METRICS[metric].scale !== "base2" || value > 0));
  renderHeatLegend(legend, METRICS[metric].scale, domain, metric, scaleMode, hasNumericScale);
  if (metric.startsWith("optimistic_overlap_GBps")
      && sliced.every((row) => metricValueState(row, metric) === VALUE_STATES.ZERO
        && ttftSavingsOutcome(row) === "zero")) {
    const note = document.createElement("p");
    note.className = "boundary-note heatmap-slice-note";
    note.textContent = `All ${sliced.length} cases show 0 GB/s and no TTFT gain under the saved recovery assumptions.`;
    legend.append(note);
  }
  const rowsByCoordinate = new Map(sliced.map((row) => [`${row.seq_len}|${row.prefix_ratio}`, row]));
  const cells = [];

  for (let yIndex = 0; yIndex < seqLens.length; yIndex += 1) {
    const seqLen = seqLens[yIndex];
    const y = effectiveTop + yIndex * band + (band - cellHeight) / 2;
    svg.append(svgElement("text", { x: BOX.left - 12, y: y + cellHeight / 2 + 4, class: "tick-label", "text-anchor": "end" }, formatCompact(seqLen)));
    for (let xIndex = 0; xIndex < hits.length; xIndex += 1) {
      const hit = hits[xIndex];
      const row = rowsByCoordinate.get(`${seqLen}|${hit}`);
      const value = row?.metrics?.[metric];
      const state = row ? metricValueState(row, metric) : VALUE_STATES.MISSING;
      const x = xForIndex(xIndex) - cellWidth / 2;
      const cell = svgElement("rect", {
        x, y, width: cellWidth, height: cellHeight, rx: 4,
        fill: row
          ? (isSemanticHeatState(state) ? "var(--chart-semantic, #fff2c9)" : valueColor(value, METRICS[metric].scale, domain))
          : "url(#missingPattern)",
        class: `heat-cell state-${row ? state : VALUE_STATES.MISSING}${row?.id === selectedId ? " is-selected" : ""}`,
        role: row ? "button" : "img",
        tabindex: row ? (row.id === selectedId ? "0" : "-1") : null,
        "aria-label": row
          ? heatmapAccessibleLabel(row, metric, state)
          : `Missing case at sequence length ${seqLen}, hit rate ${formatNumber(hit * 100)} percent`,
        "data-case-id": row?.id || "",
        "data-grid-row": yIndex,
        "data-grid-column": xIndex,
      });
      if (state === VALUE_STATES.INFINITE) cell.setAttribute("stroke", "var(--heat-negative-high, #c93636)");
      if (isSemanticHeatState(state) || state === VALUE_STATES.MIXED) cell.setAttribute("stroke", "var(--border-warning, #b7791f)");
      if (row) {
        attachTooltip(cell, tooltip, tooltipHtml(row, metric), (element) => onSelect(row, element));
        cells.push(cell);
      }
      svg.append(cell);
      if (state === VALUE_STATES.INFINITE) {
        svg.append(svgElement("text", { x: x + cellWidth / 2, y: y + cellHeight / 2 + 5, class: "heat-infinity", "text-anchor": "middle" }, "∞"));
      }
    }
  }

  const tickStep = Math.max(1, Math.ceil(hits.length / 9));
  hits.forEach((hit, index) => {
    if (index % tickStep !== 0 && index !== hits.length - 1) return;
    const x = xForIndex(index);
    svg.append(svgElement("text", {
      x, y: effectiveBottom + 24, class: "tick-label", "text-anchor": "middle",
    }, heatTickLabel(hit, xMode)));
  });
  if (fullHitGap) {
    const fullIndex = hits.findIndex((hit) => Number(hit) >= 1);
    const dividerX = xForIndex(fullIndex) - slotWidth / 2 - fullHitGap / 2;
    svg.append(svgElement("line", {
      x1: dividerX, x2: dividerX,
      y1: effectiveTop, y2: effectiveBottom, class: "semantic-divider",
    }));
    svg.append(svgElement("text", {
      x: xForIndex(fullIndex), y: effectiveBottom + 10,
      class: "semantic-lane-label", "text-anchor": "middle",
    }, "Full hit"));
  }
  svg.append(svgElement("text", {
    x: (BOX.left + BOX.right) / 2, y: effectiveBottom + 50,
    class: "axis-label", "text-anchor": "middle",
  }, xMode === "hit-linear"
    ? "Prefix ratio · ordered sampled cases"
    : "Miss ratio · ordered sampled cases (not positional scale)"));
  svg.append(svgElement("text", { x: 20, y: (effectiveTop + effectiveBottom) / 2, class: "axis-label", transform: `rotate(-90 20 ${(effectiveTop + effectiveBottom) / 2})`, "text-anchor": "middle" }, "seq_len"));
  if (seqLens.length === 1) {
    svg.append(svgElement("text", { x: BOX.left, y: effectiveTop - 18, class: "axis-label" }, "Single seq_len · heat strip"));
  }
  wireRovingCells(cells, hits.length);
  return { cellCount: sliced.length, rowCount: seqLens.length, columnCount: hits.length };
}

function heatTickLabel(hit, xMode) {
  const value = Number(hit);
  if (xMode === "hit-linear") return `${Number((value * 100).toPrecision(7))}%`;
  if (value >= 1) return "miss=0";
  return `miss ${(1 - value).toExponential(1)}`;
}

function heatmapAccessibleLabel(row, metric, state) {
  const miss = Math.max(0, 1 - Number(row.prefix_ratio));
  return [
    row.id,
    row.model,
    `sequence length ${row.seq_len}`,
    `hit rate ${formatNumber(row.prefix_ratio * 100)} percent`,
    `miss ratio ${miss === 0 ? "zero" : miss.toExponential(2)}`,
    `${METRICS[metric].label} ${formatMetricForRow(row, metric)}`,
    `state ${state}`,
  ].join(", ");
}

function matchesSlice(row, slice) {
  return String(row.encoder_state ?? "null") === String(slice.encoder_state ?? "null")
    && String(row.encoder_tail_policy ?? "null") === String(slice.encoder_tail_policy ?? "null")
    && String(row.model) === String(slice.model)
    && String(row.block_size_tokens) === String(slice.block_size_tokens)
    && String(row.arrival_model) === String(slice.arrival_model)
    && String(row.max_io_bytes ?? "null") === String(slice.max_io_bytes ?? "null")
    && String(row.metrics?.cache_total_capacity_bytes ?? "null")
      === String(slice.cache_total_capacity_bytes ?? "null");
}

function renderHeatLegend(container, scale, domain, metric, scaleMode, hasNumericScale) {
  const title = document.createElement("p");
  title.className = "panel-label";
  title.textContent = `${scaleMode === "local" ? "Local" : "Global"} ${scale === "linear" ? "linear" : scale === "symlog" ? "diverging" : "log₂"} scale`;
  const colorbar = document.createElement("div");
  colorbar.className = `heat-colorbar ${scale === "symlog" || domain.diverging ? "symlog" : "base2"}`;
  if (domain.diverging) {
    const zero = -domain.min / (domain.max - domain.min) * 100;
    colorbar.style.background = `linear-gradient(to right, ${domain.min < 0 ? "var(--heat-negative-high, #c93636)" : "var(--chart-zero, #eeeae2)"}, var(--chart-zero, #eeeae2) ${zero}%, var(--heat-positive-high, #087568))`;
  }
  const labels = document.createElement("div");
  labels.className = "heat-colorbar-labels";
  if (scale === "linear") {
    labels.append(colorLabel(formatMetric(domain.min, metric)));
    if (domain.min < 0 && domain.max > 0) labels.append(colorLabel(formatMetric(0, metric)));
    labels.append(colorLabel(formatMetric(domain.max, metric)));
  } else if (scale === "symlog") {
    labels.append(
      colorLabel(formatMetric(-domain.maxAbs, metric)),
      colorLabel(formatMetric(0, metric)),
      colorLabel(formatMetric(domain.maxAbs, metric)),
    );
  } else {
    labels.append(
      colorLabel(formatMetric(domain.minPositive, metric)),
      colorLabel(formatMetric(domain.maxPositive, metric)),
    );
  }
  const notes = scale === "linear"
    ? [["var(--chart-semantic, #fff2c9)", "Semantic status"], ["var(--chart-infinite, #ffe0e0)", "±Infinity"]]
    : scale === "symlog"
    ? [["var(--chart-zero, #eeeae2)", "Zero"], ["var(--chart-semantic, #fff2c9)", "Semantic status"], ["var(--chart-infinite, #ffe0e0)", "±Infinity"]]
    : [["var(--chart-zero, #eeeae2)", "Zero"], ["var(--chart-semantic, #fff2c9)", "Semantic status"], ["var(--chart-negative, #f3b7b7)", "Negative"], ["var(--chart-infinite, #ffe0e0)", "±Infinity"]];
  container.append(title);
  if (hasNumericScale) {
    container.append(colorbar, labels);
  } else {
    const empty = document.createElement("p");
    empty.className = "boundary-note";
    empty.textContent = scale === "base2"
      ? "No positive values for a log color scale."
      : "No finite values for a numeric color scale.";
    container.append(empty);
  }
  for (const [color, label] of notes) {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = color;
    const text = document.createElement("span");
    text.textContent = label;
    item.append(swatch, text);
    container.append(item);
  }
  const special = document.createElement("p");
  special.className = "boundary-note";
  special.textContent = "Amber fill = semantic status · Amber outline = mixed outcomes · Gray hatch = Missing · Red outline + ∞ = Infinity";
  container.append(special);
}

function isSemanticHeatState(state) {
  return [
    VALUE_STATES.NOT_EVALUATED,
    VALUE_STATES.CACHED_NOT_RESUMABLE,
    VALUE_STATES.NO_FINITE_BREAK_EVEN,
    VALUE_STATES.NO_OVERLAP_WINDOW,
  ].includes(state);
}

function colorLabel(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span;
}

function wireRovingCells(cells, columnCount) {
  if (!cells.length) return;
  if (!cells.some((cell) => cell.getAttribute("tabindex") === "0")) cells[0].setAttribute("tabindex", "0");
  const byCoordinate = new Map(cells.map((cell) => [
    `${cell.dataset.gridRow}|${cell.dataset.gridColumn}`, cell,
  ]));
  cells.forEach((cell) => {
    cell.addEventListener("keydown", (event) => {
      let target = null;
      const row = Number(cell.dataset.gridRow);
      const column = Number(cell.dataset.gridColumn);
      if (event.key === "ArrowRight") target = byCoordinate.get(`${row}|${Math.min(columnCount - 1, column + 1)}`);
      if (event.key === "ArrowLeft") target = byCoordinate.get(`${row}|${Math.max(0, column - 1)}`);
      if (event.key === "ArrowDown") target = byCoordinate.get(`${row + 1}|${column}`);
      if (event.key === "ArrowUp") target = byCoordinate.get(`${row - 1}|${column}`);
      if (event.key === "Home") target = cells[0];
      if (event.key === "End") target = cells[cells.length - 1];
      if (!target) return;
      event.preventDefault();
      cells.forEach((item) => item.setAttribute("tabindex", item === target ? "0" : "-1"));
      target.focus();
    });
  });
}
