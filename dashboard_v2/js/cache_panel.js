import { attachTooltip, svgElement } from "./chart_core.js";
import { escapeHtml, formatCompact, formatNumber } from "./metrics.js";

const WIDTH = 760;
const HEIGHT = 360;
const BOX = Object.freeze({ left: 72, right: 730, top: 34, bottom: 292 });
const GIB = 1024 ** 3;
const SERIES_COLORS = ["var(--chart-series-0, #2868c7)", "var(--chart-series-1, #087568)", "var(--chart-series-2, #9b5f00)", "var(--chart-series-3, #7155bd)", "var(--chart-series-4, #c33d63)", "var(--chart-series-5, #2c8ea6)"];
const CHURN = Object.freeze([
  ["cache_eviction_bytes", "Eviction", "var(--chart-series-0, #2868c7)"],
  ["cache_self_eviction_bytes", "Self-eviction", "var(--chart-series-2, #c16a00)"],
  ["cache_rejected_bytes", "Rejected", "var(--chart-series-4, #c33d63)"],
]);

export function renderCachePanel({
  panel, summary, hitChart, churnChart, caption, tooltip, rows, selectedId, onSelect,
}) {
  const cacheRows = rows.filter((row) => hasCache(row));
  panel.hidden = cacheRows.length === 0;
  summary.replaceChildren();
  hitChart.replaceChildren();
  churnChart.replaceChildren();
  if (!cacheRows.length) return;

  const selected = cacheRows.find((row) => row.id === selectedId) || null;
  caption.textContent = selected
    ? `${selected.id} selected · cards show this Case; curves retain the full capacity sweep.`
    : `${cacheRows.length.toLocaleString()} cache-enabled cases · cards show maxima across the current sweep.`;
  renderSummaryCards(summary, cacheRows, selected);
  renderHitChart(hitChart, tooltip, cacheRows, selectedId, onSelect);
  renderChurnChart(churnChart, tooltip, cacheRows, selectedId, onSelect);
}

function renderSummaryCards(container, rows, selected) {
  const source = selected ? [selected] : rows;
  const specs = [
    ["Inserted", "cache_inserted_bytes", "bytes"],
    ["Deduplicated", "cache_deduplicated_insert_bytes", "bytes"],
    ["Peak protected", "cache_peak_protected_bytes", "bytes"],
    ["Final residency", "cache_final_used_bytes", "bytes"],
  ];
  for (const [label, key] of specs) {
    const value = Math.max(0, ...source.map((row) => Number(row.metrics?.[key]) || 0));
    const card = document.createElement("article");
    card.className = "cache-summary-card";
    const title = document.createElement("span");
    title.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = formatBytes(value);
    const note = document.createElement("small");
    note.textContent = selected ? selected.id : "sweep maximum";
    card.append(title, strong, note);
    container.append(card);
  }
}

function renderHitChart(container, tooltip, rows, selectedId, onSelect) {
  const svg = baseSvg("Prefix boundaries by cache capacity");
  container.append(svg);
  const scale = capacityScale(rows);
  drawAxes(svg, scale, {
    yTicks: [0, .25, .5, .75, 1],
    yFor: (value) => BOX.bottom - clamp(Number(value), 0, 1) * (BOX.bottom - BOX.top),
    yLabel: (value) => `${Math.round(value * 100)}%`,
    yTitle: "Prefix boundary ratio",
  });

  const groups = groupBySeries(rows);
  const points = [];
  [...groups.entries()].forEach(([key, group], index) => {
    const color = SERIES_COLORS[index % SERIES_COLORS.length];
    const ordered = [...group].sort(capacityOrder);
    drawPath(svg, ordered, scale.xFor, (row) => hitY(row, "configured_prefix_ratio"), color, "cache-candidate-line configured");
    drawPath(svg, ordered, scale.xFor, (row) => hitY(row, "shareable_prefix_ratio"), color, "cache-candidate-line aligned");
    drawPath(svg, ordered, scale.xFor, (row) => hitY(row, "resident_prefix_ratio"), color, "cache-actual-line");
    drawPath(svg, ordered, scale.xFor, (row) => hitY(row, "recovery_boundary_ratio"), color, "cache-recovery-line");
    for (const row of ordered) {
      const point = selectablePoint({
        row,
        cx: scale.xFor(row),
        cy: hitY(row, "resident_prefix_ratio"),
        color,
        selectedId,
        tooltip,
        onSelect,
        html: hitTooltip(row),
        className: "cache-hit-point",
      });
      svg.append(point);
      points.push(point);
    }
    if (index < 6) drawSeriesLabel(svg, key, color, index);
  });
  drawHitLegend(svg);
  wirePoints(points);
}

function renderChurnChart(container, tooltip, rows, selectedId, onSelect) {
  const svg = baseSvg("Cache churn bytes by cache capacity");
  container.append(svg);
  const scale = capacityScale(rows);
  const maximum = Math.max(1, ...rows.flatMap((row) => CHURN.map(([key]) => Number(row.metrics?.[key]) || 0)));
  const yForValue = (value) => {
    const score = Math.log2(1 + Math.max(0, Number(value) || 0));
    return BOX.bottom - score / Math.log2(1 + maximum) * (BOX.bottom - BOX.top);
  };
  const yTickValues = byteTicks(maximum);
  drawAxes(svg, scale, {
    yTicks: yTickValues,
    yFor: yForValue,
    yLabel: formatBytesShort,
    yTitle: "Logical bytes",
  });

  const groups = groupBySeries(rows);
  const points = [];
  [...groups.values()].forEach((group, groupIndex) => {
    const ordered = [...group].sort(capacityOrder);
    CHURN.forEach(([key, label, color], metricIndex) => {
      drawPath(svg, ordered, scale.xFor, (row) => yForValue(row.metrics?.[key]), color, `cache-churn-line metric-${metricIndex}`, groupIndex ? .34 : .9);
      for (const row of ordered) {
        const point = selectablePoint({
          row,
          cx: scale.xFor(row),
          cy: yForValue(row.metrics?.[key]),
          color,
          selectedId,
          tooltip,
          onSelect,
          html: churnTooltip(row, label, key),
          className: `cache-churn-point metric-${metricIndex}`,
          radius: groupIndex ? 3.2 : 4.2,
        });
        svg.append(point);
        points.push(point);
      }
    });
  });
  drawChurnLegend(svg);
  wirePoints(points);
}

function baseSvg(label) {
  return svgElement("svg", {
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    role: "group",
    "aria-label": label,
  });
}

function drawAxes(svg, scale, { yTicks, yFor, yLabel, yTitle }) {
  for (const value of yTicks) {
    const y = yFor(value);
    svg.append(svgElement("line", { x1: BOX.left, x2: BOX.right, y1: y, y2: y, class: "grid-line" }));
    svg.append(svgElement("text", { x: BOX.left - 10, y: y + 4, class: "tick-label", "text-anchor": "end" }, yLabel(value)));
  }
  for (const tick of scale.ticks) {
    const x = scale.xForCapacity(tick.value);
    svg.append(svgElement("line", { x1: x, x2: x, y1: BOX.top, y2: BOX.bottom, class: "grid-line vertical" }));
    svg.append(svgElement("text", { x, y: BOX.bottom + 22, class: "tick-label", "text-anchor": "middle" }, tick.label));
  }
  if (scale.hasZero && scale.hasPositive) {
    svg.append(svgElement("line", { x1: scale.divider, x2: scale.divider, y1: BOX.top, y2: BOX.bottom, class: "semantic-divider" }));
    svg.append(svgElement("text", { x: scale.divider - 7, y: BOX.top + 15, class: "semantic-lane-label", "text-anchor": "end" }, "zero capacity"));
  }
  svg.append(svgElement("text", { x: (BOX.left + BOX.right) / 2, y: HEIGHT - 12, class: "axis-label", "text-anchor": "middle" }, "Total cache capacity · positive values use log₂ GiB"));
  svg.append(svgElement("text", { x: 17, y: (BOX.top + BOX.bottom) / 2, class: "axis-label", transform: `rotate(-90 17 ${(BOX.top + BOX.bottom) / 2})`, "text-anchor": "middle" }, yTitle));
}

function capacityScale(rows) {
  const capacities = [...new Set(rows.map(capacityBytes))].sort((a, b) => a - b);
  const positives = capacities.filter((value) => value > 0);
  const hasZero = capacities.includes(0);
  const hasPositive = positives.length > 0;
  const zeroX = BOX.left + 24;
  const positiveLeft = hasZero && hasPositive ? BOX.left + 105 : BOX.left + 16;
  const minScore = hasPositive ? Math.log2(positives[0] / GIB) : 0;
  const maxScore = hasPositive ? Math.log2(positives.at(-1) / GIB) : minScore;
  const xForCapacity = (bytes) => {
    if (Number(bytes) === 0) return zeroX;
    if (!hasPositive) return (BOX.left + BOX.right) / 2;
    if (minScore === maxScore) return (positiveLeft + BOX.right - 14) / 2;
    const ratio = (Math.log2(Number(bytes) / GIB) - minScore) / (maxScore - minScore);
    return positiveLeft + ratio * (BOX.right - 14 - positiveLeft);
  };
  const ticks = capacities.map((value) => ({
    value,
    label: value === 0 ? "0" : capacityLabel(value),
  }));
  return {
    hasZero,
    hasPositive,
    zeroX,
    divider: BOX.left + 72,
    ticks,
    xForCapacity,
    xFor: (row) => xForCapacity(capacityBytes(row)),
  };
}

function drawPath(svg, rows, xFor, yFor, color, className, opacity = 1) {
  if (!rows.length) return;
  const coordinates = rows.map((row) => `${xFor(row)},${yFor(row)}`).join(" ");
  svg.append(svgElement("polyline", {
    points: coordinates,
    fill: "none",
    stroke: color,
    opacity,
    class: className,
  }));
}

function selectablePoint({ row, cx, cy, color, selectedId, tooltip, onSelect, html, className, radius = 4.8 }) {
  const point = svgElement("circle", {
    cx,
    cy,
    r: row.id === selectedId ? radius + 2 : radius,
    fill: color,
    class: `${className}${row.id === selectedId ? " is-selected" : ""}`,
    role: "button",
    tabindex: row.id === selectedId ? "0" : "-1",
    "aria-label": `${row.id}, cache ${capacityLabel(capacityBytes(row))}, resident prefix ${formatNumber((Number(row.metrics?.resident_prefix_ratio) || 0) * 100)} percent`,
    "data-case-id": row.id,
  });
  attachTooltip(point, tooltip, html, (element) => onSelect(row, element));
  return point;
}

function wirePoints(points) {
  if (!points.length) return;
  if (!points.some((point) => point.getAttribute("tabindex") === "0")) points[0].setAttribute("tabindex", "0");
  points.forEach((point, index) => {
    point.addEventListener("keydown", (event) => {
      let target = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") target = points[(index + 1) % points.length];
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") target = points[(index - 1 + points.length) % points.length];
      if (event.key === "Home") target = points[0];
      if (event.key === "End") target = points.at(-1);
      if (!target) return;
      event.preventDefault();
      points.forEach((item) => item.setAttribute("tabindex", item === target ? "0" : "-1"));
      target.focus();
    });
  });
}

function drawHitLegend(svg) {
  const items = [
    ["Resident", "cache-actual-line"],
    ["Recovery", "cache-recovery-line"],
    ["Object-shareable prefix", "cache-candidate-line aligned"],
    ["Configured candidate", "cache-candidate-line configured"],
  ];
  items.forEach(([label, className], index) => {
    const x = BOX.left + index * 192;
    svg.append(svgElement("line", { x1: x, x2: x + 26, y1: 17, y2: 17, class: className, stroke: "var(--chart-axis, #405064)" }));
    svg.append(svgElement("text", { x: x + 32, y: 21, class: "cache-legend-label" }, label));
  });
}

function drawChurnLegend(svg) {
  CHURN.forEach(([, label, color], index) => {
    const x = BOX.left + index * 185;
    svg.append(svgElement("line", { x1: x, x2: x + 26, y1: 17, y2: 17, stroke: color, class: "cache-churn-line" }));
    svg.append(svgElement("text", { x: x + 32, y: 21, class: "cache-legend-label" }, label));
  });
}

function drawSeriesLabel(svg, key, color, index) {
  if (index > 2) return;
  const label = key.length > 42 ? `${key.slice(0, 39)}…` : key;
  svg.append(svgElement("text", { x: BOX.right - 4, y: BOX.top + 15 + index * 15, fill: color, class: "cache-series-label", "text-anchor": "end" }, label));
}

function hitTooltip(row) {
  return `<strong>${escapeHtml(row.id)} · Prefix Cache</strong><dl>
    <dt>Capacity</dt><dd>${escapeHtml(capacityLabel(capacityBytes(row)))}</dd>
    <dt>Configured candidate</dt><dd>${percentage(row.metrics?.configured_prefix_ratio)}</dd>
    <dt>Object-shareable</dt><dd>${percentage(row.metrics?.shareable_prefix_ratio)}</dd>
    <dt>Resident prefix</dt><dd>${percentage(row.metrics?.resident_prefix_ratio)}</dd>
    <dt>Recovery boundary</dt><dd>${percentage(row.metrics?.recovery_boundary_ratio)}</dd>
    <dt>Model / seq</dt><dd>${escapeHtml(row.model)} · ${formatCompact(row.seq_len)}</dd>
  </dl>`;
}

function churnTooltip(row, label, key) {
  return `<strong>${escapeHtml(row.id)} · ${escapeHtml(label)}</strong><dl>
    <dt>Capacity</dt><dd>${escapeHtml(capacityLabel(capacityBytes(row)))}</dd>
    <dt>${escapeHtml(label)}</dt><dd>${escapeHtml(formatBytes(Number(row.metrics?.[key]) || 0))}</dd>
    <dt>Final residency</dt><dd>${escapeHtml(formatBytes(Number(row.metrics?.cache_final_used_bytes) || 0))}</dd>
    <dt>Resident prefix</dt><dd>${percentage(row.metrics?.resident_prefix_ratio)}</dd>
  </dl>`;
}

function groupBySeries(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = [row.model, `L${row.seq_len}`, `P${formatNumber(row.prefix_ratio * 100)}%`, `B${row.block_size_tokens}`, row.arrival_model, `IO${row.max_io_bytes ?? "∞"}`, ...(row.encoder_state == null ? [] : [`Encoder ${row.encoder_state}`, `output ${row.encoder_tail_policy}`])].join(" · ");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
}

function hitY(row, key) {
  return BOX.bottom - clamp(Number(row.metrics?.[key]) || 0, 0, 1) * (BOX.bottom - BOX.top);
}

function capacityBytes(row) {
  return Number(row.metrics?.cache_total_capacity_bytes) || 0;
}

function capacityOrder(a, b) {
  return capacityBytes(a) - capacityBytes(b) || String(a.id).localeCompare(String(b.id));
}

function hasCache(row) {
  return row?.metrics && Object.prototype.hasOwnProperty.call(row.metrics, "resident_prefix_ratio");
}

function capacityLabel(bytes) {
  if (!bytes) return "0 GiB";
  return `${formatNumber(bytes / GIB)} GiB`;
}

function percentage(value) {
  return `${formatNumber((Number(value) || 0) * 100)}%`;
}

function formatBytes(value) {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes >= GIB) return `${formatNumber(bytes / GIB)} GiB`;
  if (bytes >= 1024 ** 2) return `${formatNumber(bytes / (1024 ** 2))} MiB`;
  if (bytes >= 1024) return `${formatNumber(bytes / 1024)} KiB`;
  return `${formatNumber(bytes)} B`;
}

function formatBytesShort(value) {
  return formatBytes(value).replace(" ", "");
}

function byteTicks(maximum) {
  if (maximum <= 1) return [0, maximum];
  const result = [0];
  const maxPower = Math.ceil(Math.log2(maximum));
  const step = Math.max(1, Math.ceil(maxPower / 4));
  for (let power = step; power < maxPower; power += step) result.push(2 ** power);
  result.push(maximum);
  return [...new Set(result)];
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}
