import {
  classifyValue, formatAxisMetric, formatMetric, formatNumber, numericValue, VALUE_STATES,
} from "./metrics.js";
import { RESULT_STATES, semanticLaneLabel } from "./result_semantics.js";

export const SVG_NS = "http://www.w3.org/2000/svg";
export const SERIES_COLORS = [
  "#2868c7", "#087568", "#bd6500", "#7155bd", "#c33d63", "#287f9c",
  "#58851e", "#9c4f16", "#4859b8", "#008a73", "#aa3d32", "#5d6875",
  "#85600c", "#744481", "#1f7cba", "#9b506c", "#5a7b34", "#b34f23",
];

let paletteTheme;
let paletteStyle;
const paletteColors = new Map();

// Read once per effective theme, not once per heatmap cell or series point.
// Hex fallbacks keep the light palette and numeric color interpolation intact.
export function chartColor(name, fallback) {
  const theme = document.documentElement.dataset.theme;
  if (!paletteStyle || paletteTheme !== theme) {
    paletteTheme = theme;
    paletteStyle = getComputedStyle(document.documentElement);
    paletteColors.clear();
  }
  const key = `${name}:${fallback}`;
  if (!paletteColors.has(key)) {
    paletteColors.set(key, paletteStyle.getPropertyValue(`--${name}`).trim() || fallback);
  }
  return paletteColors.get(key);
}

export function svgElement(name, attributes = {}, text = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  if (text !== null) node.textContent = text;
  return node;
}

export function colorForIndex(index) {
  const slot = index % SERIES_COLORS.length;
  return chartColor(`chart-series-${slot}`, SERIES_COLORS[slot]);
}

export function buildLinearScale(domainMin, domainMax, rangeMin, rangeMax) {
  let min = Number(domainMin);
  let max = Number(domainMax);
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = min + 1;
  if (max === min) { min -= .5; max += .5; }
  const scale = (value) => rangeMin + ((Number(value) - min) / (max - min)) * (rangeMax - rangeMin);
  scale.domain = [min, max];
  scale.ticks = linearTicks(min, max, 6);
  return scale;
}

export function buildXAxisLayout(hitRates, mode, left, right) {
  if (mode === "cache-capacity" || mode === "context-length") return resourceXAxisLayout(hitRates, mode, left, right);
  const hits = hitRates.map(numericValue).filter(Number.isFinite);
  const hasFullHit = mode === "miss-log" && hits.some((hit) => hit >= 1);
  const fullHitWidth = hasFullHit ? 96 : 0;
  const numericRight = right - fullHitWidth;
  const finiteValues = hits
    .map((hit) => mode === "hit-linear" ? hit : (hit >= 1 ? NaN : -Math.log10(Math.max(1 - hit, 1e-12))))
    .filter(Number.isFinite);
  const uniqueFiniteValues = [...new Set(finiteValues)];
  let min = Math.min(...finiteValues);
  let max = Math.max(...finiteValues);
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = min + 1;
  const isSingleHit = mode === "hit-linear" && uniqueFiniteValues.length === 1;
  const singleHit = isSingleHit ? uniqueFiniteValues[0] : null;
  if (isSingleHit) {
    min = singleHit - .5;
    max = singleHit + .5;
  }
  if (mode !== "hit-linear") {
    const padding = Math.max(.12, (max - min) * .05);
    min = Math.max(0, min - padding);
    max += padding;
  }
  const scale = buildLinearScale(min, max, left, numericRight);
  if (mode === "hit-linear") {
    scale.ticks = isSingleHit
      ? [singleHit]
      : buildHitRateTicks(finiteValues, Math.min(...finiteValues), Math.max(...finiteValues), left, numericRight);
  }
  scale.numericRight = numericRight;
  scale.fullHit = hasFullHit ? {
    dividerX: numericRight + 12,
    x: numericRight + 54,
    right,
  } : null;
  scale.xForHit = (hitRate) => {
    const hit = Number(hitRate);
    if (mode === "miss-log" && hit >= 1 && scale.fullHit) return scale.fullHit.x;
    if (mode === "hit-linear" && isSingleHit) return (left + numericRight) / 2;
    const value = mode === "hit-linear" ? hit : -Math.log10(Math.max(1 - hit, 1e-12));
    return scale(value);
  };
  scale.xForValue = scale.xForHit;
  return scale;
}

function resourceXAxisLayout(values, mode, left, right) {
  const numbers = values.map(numericValue).filter((value) => Number.isFinite(value) && value >= 0);
  const positive = [...new Set(numbers.filter((value) => value > 0))].sort((a, b) => a - b);
  const hasZero = mode === "cache-capacity" && numbers.includes(0);
  const numericLeft = hasZero && positive.length ? left + 96 : left;
  const min = Math.log2(positive[0] || 1);
  const max = Math.log2(positive.at(-1) || 2);
  const logarithm = buildLinearScale(min, max, numericLeft, right);
  const scale = (value) => {
    const number = numericValue(value);
    if (number === 0 && hasZero) return positive.length ? left + 32 : (left + right) / 2;
    if (!Number.isFinite(number) || number <= 0) return NaN;
    return positive.length === 1 ? (numericLeft + right) / 2 : logarithm(Math.log2(number));
  };
  const stride = Math.max(1, Math.ceil(positive.length / 9));
  scale.ticks = positive.filter((_, index) => index % stride === 0 || index === positive.length - 1);
  scale.xForValue = scale;
  scale.numericLeft = numericLeft;
  scale.numericRight = right;
  scale.zeroCapacity = hasZero ? { x: scale(0), dividerX: positive.length ? left + 72 : null } : null;
  return scale;
}

export function buildHitRateTicks(hitRates, min, max, left, right) {
  const observed = [...new Set(hitRates.map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  if (!observed.length || !Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];

  const highTail = min >= .4 && max > .9;
  const candidates = observed.length <= 12
    ? observed
    : highTail
      ? [min, .5, .7, .85, .9, .95, .98, .99, .999, .9999, max]
      : [min, ...linearTicks(min, max, 6), max];
  const ticks = [...new Set(candidates
    .map((value) => Number(Number(value).toFixed(10)))
    .filter((value) => value >= min - 1e-12 && value <= max + 1e-12))]
    .sort((a, b) => a - b);
  if (ticks.length <= 2) return ticks;

  const position = (value) => left + ((value - min) / (max - min)) * (right - left);
  const minimumGap = 58;
  const selected = [ticks[0], ticks.at(-1)];
  for (const tick of ticks.slice(1, -1)) {
    if (selected.every((value) => Math.abs(position(value) - position(tick)) >= minimumGap)) {
      selected.push(tick);
    }
  }
  return selected.sort((a, b) => a - b);
}

export function buildYScale(values, type, top, bottom, semanticStates = [], options = {}) {
  if (type === "linear") return linearValueScale(values, top, bottom, semanticStates, options);
  return type === "symlog"
    ? symlogScale(values, top, bottom, semanticStates)
    : base2Scale(values, top, bottom, semanticStates);
}

function linearValueScale(values, top, bottom, semanticStates, options) {
  const domain = valueDomain(values, "linear", options);
  const special = [
    ...(values.includes(Infinity) ? [{ key: "positive-infinity", label: "+∞" }] : []),
    ...semanticLaneDefinitions(semanticStates),
    ...(values.includes(-Infinity) ? [{ key: "negative-infinity", label: "−∞" }] : []),
    ...(values.some((value) => classifyValue(value) === VALUE_STATES.MISSING) ? [{ key: "missing", label: "Missing" }] : []),
  ];
  const hasTopLane = special[0]?.key === "positive-infinity";
  const plotTop = top + (hasTopLane ? 26 : 0);
  const bottomLanes = special.slice(hasTopLane ? 1 : 0);
  const plotBottom = bottom - bottomLanes.length * 26;
  const lanes = [
    ...(hasTopLane ? [{ ...special[0], y: top + 13 }] : []),
    ...bottomLanes.map((lane, index) => ({ ...lane, y: plotBottom + index * 26 + 13 })),
  ];
  const numericScale = buildLinearScale(domain.min, domain.max, plotBottom, plotTop);
  const laneY = (key) => lanes.find((lane) => lane.key === key)?.y ?? plotBottom;
  const scale = (value) => {
    if (classifyValue(value) === VALUE_STATES.MISSING) return laneY("missing");
    if (value === Infinity) return laneY("positive-infinity");
    if (value === -Infinity) return laneY("negative-infinity");
    return numericScale(value);
  };
  scale.ticks = numericScale.ticks;
  if (domain.min < 0 && !scale.ticks.includes(0)) scale.ticks = [...scale.ticks, 0].sort((a, b) => a - b);
  scale.domain = [domain.min, domain.max];
  scale.zeroY = numericScale(0);
  scale.semanticY = laneY;
  scale.plotTop = plotTop;
  scale.plotBottom = plotBottom;
  scale.specialLanes = lanes;
  return scale;
}

function base2Scale(values, top, bottom, semanticStates = []) {
  const states = values.map((value) => ({ state: classifyValue(value), value: Number(value) }));
  const positive = values.filter((value) => Number.isFinite(value) && value > 0);
  const topLanes = states.some((item) => item.state === VALUE_STATES.INFINITE && item.value > 0)
    ? [{ key: "positive-infinity", label: "+∞" }] : [];
  const bottomLanes = semanticLaneDefinitions(semanticStates);
  if (states.some((item) => item.state === VALUE_STATES.INFINITE && item.value < 0)) bottomLanes.push({ key: "negative-infinity", label: "−∞" });
  if (states.some((item) => item.state === VALUE_STATES.NEGATIVE && Number.isFinite(item.value))) bottomLanes.push({ key: "negative", label: "Negative" });
  if (states.some((item) => item.state === VALUE_STATES.ZERO)) bottomLanes.push({ key: "zero", label: "Zero" });
  if (states.some((item) => item.state === VALUE_STATES.MISSING)) bottomLanes.push({ key: "missing", label: "Missing" });
  const laneHeight = 26;
  const plotTop = top + topLanes.length * laneHeight;
  const plotBottom = bottom - bottomLanes.length * laneHeight;
  const rawMin = positive.length ? Math.min(...positive) : 1;
  const rawMax = positive.length ? Math.max(...positive) : 2;
  let minPower = Math.floor(Math.log2(rawMin));
  let maxPower = Math.ceil(Math.log2(rawMax));
  if (minPower === maxPower) { minPower -= 1; maxPower += 1; }
  const lanes = [];
  topLanes.forEach((lane, index) => lanes.push({ ...lane, y: top + laneHeight * index + laneHeight / 2 }));
  bottomLanes.forEach((lane, index) => lanes.push({ ...lane, y: plotBottom + laneHeight * index + laneHeight / 2 }));
  const laneY = (key, fallback) => lanes.find((lane) => lane.key === key)?.y ?? fallback;
  const scale = (value) => {
    const state = classifyValue(value);
    if (state === VALUE_STATES.INFINITE) return Number(value) < 0
      ? laneY("negative-infinity", plotBottom)
      : laneY("positive-infinity", plotTop);
    if (state === VALUE_STATES.NEGATIVE) return laneY("negative", plotBottom);
    if (state === VALUE_STATES.ZERO) return laneY("zero", plotBottom);
    if (state === VALUE_STATES.MISSING) return laneY("missing", plotBottom);
    const position = (Math.log2(Number(value)) - minPower) / (maxPower - minPower);
    return plotBottom - Math.max(0, Math.min(1, position)) * (plotBottom - plotTop);
  };
  const span = maxPower - minPower;
  const maxTickCount = Math.max(2, Math.floor((plotBottom - plotTop) / 22) + 1);
  const step = Math.max(1, Math.ceil((span + 1) / maxTickCount));
  scale.ticks = [];
  for (let power = minPower; power <= maxPower; power += step) scale.ticks.push(2 ** power);
  if (scale.ticks.at(-1) !== 2 ** maxPower) scale.ticks.push(2 ** maxPower);
  if (!positive.length) scale.ticks = [];
  scale.zeroY = laneY("zero", plotBottom);
  scale.semanticY = (state) => laneY(state, plotBottom);
  scale.domain = [2 ** minPower, 2 ** maxPower];
  scale.plotTop = plotTop;
  scale.plotBottom = plotBottom;
  scale.specialLanes = lanes;
  return scale;
}

function symlogScale(values, top, bottom, semanticStates = []) {
  const finite = values.filter(Number.isFinite);
  const hasPositiveInfinity = values.some((value) => value === Infinity);
  const hasNegativeInfinity = values.some((value) => value === -Infinity);
  const hasMissing = values.some((value) => classifyValue(value) === VALUE_STATES.MISSING);
  const laneHeight = 26;
  const plotTop = top + (hasPositiveInfinity ? laneHeight : 0);
  const bottomLanes = [
    ...semanticLaneDefinitions(semanticStates),
    ...(hasNegativeInfinity ? [{ key: "negative-infinity", label: "−∞" }] : []),
    ...(hasMissing ? [{ key: "missing", label: "Missing" }] : []),
  ];
  const plotBottom = bottom - bottomLanes.length * laneHeight;
  const lanes = [
    ...(hasPositiveInfinity ? [{ key: "positive-infinity", label: "+∞", y: top + laneHeight / 2 }] : []),
    ...bottomLanes.map((lane, index) => ({ ...lane, y: plotBottom + laneHeight * index + laneHeight / 2 })),
  ];
  const laneY = (key, fallback) => lanes.find((lane) => lane.key === key)?.y ?? fallback;
  const maxAbs = Math.max(1, ...finite.map(Math.abs));
  const constant = Math.max(1, 10 ** Math.floor(Math.log10(maxAbs) - 2));
  const transform = (value) => Math.sign(value) * Math.log2(1 + Math.abs(value) / constant);
  const extent = Math.max(1, transform(maxAbs));
  const middle = (plotTop + plotBottom) / 2;
  const scale = (value) => {
    const state = classifyValue(value);
    if (state === VALUE_STATES.INFINITE) return Number(value) < 0
      ? laneY("negative-infinity", plotBottom)
      : laneY("positive-infinity", plotTop);
    if (state === VALUE_STATES.MISSING) return laneY("missing", plotBottom);
    return middle - (transform(Number(value)) / extent) * ((plotBottom - plotTop) / 2);
  };
  const magnitudes = [maxAbs, maxAbs / 10, constant].filter((value, index, array) => value >= constant && array.indexOf(value) === index);
  scale.ticks = [...magnitudes.map((value) => -value), 0, ...magnitudes.slice().reverse()].sort((a, b) => a - b);
  scale.zeroY = middle;
  scale.semanticY = (state) => laneY(state, plotBottom);
  scale.domain = [-maxAbs, maxAbs];
  scale.constant = constant;
  scale.plotTop = plotTop;
  scale.plotBottom = plotBottom;
  scale.specialLanes = lanes;
  return scale;
}

function semanticLaneDefinitions(states) {
  const ordered = [
    RESULT_STATES.NOT_EVALUATED,
    RESULT_STATES.CACHED_NOT_RESUMABLE,
    RESULT_STATES.NO_FINITE_BREAK_EVEN,
    RESULT_STATES.NO_OVERLAP_WINDOW,
  ];
  const present = new Set(states || []);
  return ordered
    .filter((state) => present.has(state))
    .map((state) => ({ key: state, label: semanticLaneLabel(state) }));
}

export function drawAxes(svg, { xScale, yScale, xFormatter, yMetric, box, xLabel, yLabel, threshold = NaN }) {
  const { left, right, top, bottom } = box;
  const numericRight = xScale.numericRight ?? right;
  const plotTop = yScale.plotTop ?? top;
  const plotBottom = yScale.plotBottom ?? bottom;
  const grid = svgElement("g", { "aria-hidden": "true" });
  for (const tick of yScale.ticks) {
    const y = yScale(tick);
    grid.append(svgElement("line", { x1: left, x2: numericRight, y1: y, y2: y, class: tick === 0 ? "zero-line" : "grid-line" }));
    grid.append(svgElement("text", { x: left - 10, y: y + 4, class: "tick-label y-tick-label", "text-anchor": "end" }, formatAxisMetric(tick, yMetric)));
  }
  for (const tick of xScale.ticks) {
    const x = xScale(tick);
    grid.append(svgElement("line", { x1: x, x2: x, y1: plotTop, y2: plotBottom, class: "grid-line" }));
    grid.append(svgElement("text", { x, y: bottom + 20, class: "tick-label x-tick-label", "text-anchor": "middle" }, xFormatter(tick)));
  }
  grid.append(svgElement("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "axis-line" }));
  grid.append(svgElement("line", { x1: left, x2: left, y1: plotTop, y2: plotBottom, class: "axis-line" }));

  for (const lane of yScale.specialLanes || []) {
    grid.append(svgElement("line", { x1: left, x2: numericRight, y1: lane.y, y2: lane.y, class: "semantic-lane-line" }));
    grid.append(svgElement("text", { x: left - 10, y: lane.y + 4, class: "semantic-lane-label", "text-anchor": "end" }, lane.label));
  }
  if (xScale.fullHit) {
    grid.append(svgElement("line", {
      x1: xScale.fullHit.dividerX, x2: xScale.fullHit.dividerX,
      y1: plotTop, y2: plotBottom, class: "semantic-divider",
    }));
    grid.append(svgElement("text", {
      x: xScale.fullHit.x, y: bottom + 18, class: "semantic-lane-label", "text-anchor": "middle",
    }, "Full hit"));
    grid.append(svgElement("text", {
      x: xScale.fullHit.x, y: bottom + 32, class: "tick-label", "text-anchor": "middle",
    }, "miss=0"));
  }

  if (xScale.zeroCapacity) {
    const { x, dividerX } = xScale.zeroCapacity;
    if (dividerX !== null) grid.append(svgElement("line", { x1: dividerX, x2: dividerX, y1: plotTop, y2: plotBottom, class: "semantic-divider" }));
    grid.append(svgElement("text", { x, y: bottom + 18, class: "semantic-lane-label", "text-anchor": "middle" }, "Zero capacity"));
    grid.append(svgElement("text", { x, y: bottom + 32, class: "tick-label", "text-anchor": "middle" }, "0 GiB"));
  }

  if (Number.isFinite(threshold) && threshold > 0) {
    const y = yScale(threshold);
    grid.append(svgElement("line", { x1: left, x2: numericRight, y1: y, y2: y, class: "threshold-line" }));
    grid.append(svgElement("text", { x: numericRight - 4, y: y - 6, class: "threshold-label", "text-anchor": "end" }, `nominal proxy ${formatMetric(threshold, yMetric)}`));
  }
  grid.append(svgElement("text", { x: (left + right) / 2, y: bottom + 52, class: "axis-label", "text-anchor": "middle" }, xLabel));
  grid.append(svgElement("text", { x: 18, y: (plotTop + plotBottom) / 2, class: "axis-label", transform: `rotate(-90 18 ${(plotTop + plotBottom) / 2})`, "text-anchor": "middle" }, yLabel));
  svg.append(grid);
}

export function addMissingPattern(svg) {
  const defs = svgElement("defs");
  const pattern = svgElement("pattern", { id: "missingPattern", width: 8, height: 8, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
  pattern.append(svgElement("rect", { width: 8, height: 8, fill: chartColor("chart-missing-bg", "#e2e6e9") }));
  pattern.append(svgElement("line", { x1: 0, x2: 0, y1: 0, y2: 8, stroke: chartColor("chart-missing-line", "#aab3bb"), "stroke-width": 2 }));
  defs.append(pattern);
  svg.append(defs);
}

export function attachTooltip(element, tooltip, html, onActivate) {
  const show = (event) => {
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    positionTooltip(tooltip, event.clientX, event.clientY);
  };
  element.addEventListener("pointerenter", show);
  element.addEventListener("pointermove", show);
  element.addEventListener("pointerleave", () => { tooltip.hidden = true; });
  element.addEventListener("focus", () => {
    const box = element.getBoundingClientRect();
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    positionTooltip(tooltip, box.left + box.width / 2, box.top);
  });
  element.addEventListener("blur", () => { tooltip.hidden = true; });
  const activate = () => {
    tooltip.hidden = true;
    onActivate?.(element);
  };
  element.addEventListener("click", activate);
  element.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activate();
    }
  });
}

export function positionTooltip(tooltip, x, y) {
  const pad = 14;
  const maxLeft = window.innerWidth - tooltip.offsetWidth - pad;
  const maxTop = window.innerHeight - tooltip.offsetHeight - pad;
  tooltip.style.left = `${Math.max(pad, Math.min(maxLeft, x + 14))}px`;
  tooltip.style.top = `${Math.max(pad, Math.min(maxTop, y + 14))}px`;
}

export function valueDomain(values, type, options = {}) {
  const finite = values.map(numericValue).filter(Number.isFinite);
  if (type === "linear") {
    const min = Math.min(0, options.domain?.[0] ?? 0, ...finite);
    const maximum = Math.max(0, options.domain?.[1] ?? 0, ...finite);
    const max = maximum === min ? min + 1 : maximum;
    return { type, min, max, diverging: Boolean(options.diverging), maxAbs: Math.max(Math.abs(min), Math.abs(max)) };
  }
  if (type === "symlog") {
    return { type, maxAbs: Math.max(1, ...finite.map(Math.abs)) };
  }
  const positive = finite.filter((value) => value > 0);
  return {
    type,
    minPositive: positive.length ? Math.min(...positive) : 1,
    maxPositive: positive.length ? Math.max(...positive) : 2,
  };
}

export function valueColor(value, type, domain) {
  const state = classifyValue(value);
  if (state === VALUE_STATES.MISSING) return "url(#missingPattern)";
  if (state === VALUE_STATES.INFINITE) return chartColor("chart-infinite", "#ffe0e0");
  const number = Number(value);
  if (type === "linear") {
    if (domain.diverging) {
      if (number === 0) return chartColor("chart-zero", "#eeeae2");
      const extent = number < 0 ? Math.abs(domain.min) : domain.max;
      const ratio = Math.min(1, Math.abs(number) / Math.max(Number.MIN_VALUE, extent));
      return number < 0 ? mixColor(chartColor("heat-negative-low", "#f8dfdf"), chartColor("heat-negative-high", "#c93636"), ratio) : mixColor(chartColor("heat-positive-low", "#deeee8"), chartColor("heat-positive-high", "#087568"), ratio);
    }
    return mixColor(chartColor("heat-low", "#e8f0fb"), chartColor("heat-high", "#2868c7"), Math.max(0, Math.min(1, (number - domain.min) / (domain.max - domain.min))));
  }
  if (type === "symlog") {
    if (number === 0) return chartColor("chart-zero", "#eeeae2");
    const power = Math.min(1, Math.log1p(Math.abs(number)) / Math.log1p(domain.maxAbs));
    return number < 0 ? mixColor(chartColor("heat-negative-low", "#f8dfdf"), chartColor("heat-negative-high", "#c93636"), power) : mixColor(chartColor("heat-positive-low", "#deeee8"), chartColor("heat-positive-high", "#087568"), power);
  }
  if (number < 0) return chartColor("chart-negative", "#f3b7b7");
  if (number === 0) return chartColor("chart-zero", "#eeeae2");
  const minLog = Math.log2(Math.max(Number.MIN_VALUE, domain.minPositive));
  const maxLog = Math.log2(Math.max(domain.minPositive, domain.maxPositive));
  const ratio = maxLog === minLog ? .5 : Math.max(0, Math.min(1, (Math.log2(number) - minLog) / (maxLog - minLog)));
  return mixColor(chartColor("heat-low", "#e8f0fb"), chartColor("heat-high", "#2868c7"), ratio);
}

function mixColor(from, to, ratio) {
  const parse = (hex) => [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));
  const a = parse(from);
  const b = parse(to);
  const rgb = a.map((value, index) => Math.round(value + (b[index] - value) * ratio));
  return `rgb(${rgb.join(",")})`;
}

function linearTicks(min, max, count) {
  const ticks = [];
  const step = (max - min) / Math.max(1, count - 1);
  for (let index = 0; index < count; index += 1) ticks.push(min + index * step);
  return ticks;
}
