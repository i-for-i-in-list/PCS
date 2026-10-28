import {
  classifyValue, formatCompact, formatMetric, formatMetricForRow, formatNumber,
  riskForRow, riskReasonsForRow, VALUE_STATES,
} from "./metrics.js";
import { recoveryOutcome } from "./result_semantics.js";

export const PAGE_SIZE = 100;

const COLUMNS = [
  { key: "id", label: "Case", width: 82, value: (row) => row.id },
  { key: "model", label: "Model", width: 170, value: (row) => row.model },
  { key: "seq_len", label: "seq_len", width: 85, value: (row) => row.seq_len, format: formatCompact },
  { key: "prefix_ratio", label: "Prefix", width: 74, value: (row) => row.prefix_ratio, format: (value) => `${formatNumber(value * 100)}%` },
  { key: "block_size_tokens", label: "Block", width: 63, value: (row) => row.block_size_tokens },
  { key: "arrival_model", label: "Arrival", width: 67, value: (row) => row.arrival_model },
  { key: "max_io_bytes", label: "Max IO", width: 78, value: (row) => row.max_io_bytes, format: (value) => value ?? "None" },
  { key: "optimistic_overlap_GBps_p99", label: "Optimistic BW", width: 148, value: (row) => row.metrics?.optimistic_overlap_GBps_p99, format: (_value, row) => formatMetricForRow(row, "optimistic_overlap_GBps_p99") },
  { key: "dependency_aware_break_even_GBps_p99", label: "Break-even", width: 168, value: (row) => row.metrics?.dependency_aware_break_even_GBps_p99, format: (_value, row) => formatMetricForRow(row, "dependency_aware_break_even_GBps_p99") },
  { key: "read_avg_GBps", label: "Read", width: 88, value: (row) => row.metrics?.read_avg_GBps, format: (value) => formatMetric(value, "GB/s") },
  { key: "write_avg_GBps", label: "Write", width: 88, value: (row) => row.metrics?.write_avg_GBps, format: (value) => formatMetric(value, "GB/s") },
  { key: "aggregate_dependency_chunked_iops", label: "Dependency IOPS", width: 112, value: (row) => row.metrics?.aggregate_dependency_chunked_iops, format: (value) => formatMetric(value, "IOPS") },
  { key: "dependency_ttft_savings_us_p50", label: "TTFT upper bound", width: 126, value: (row) => row.metrics?.dependency_ttft_savings_us_p50, format: (_value, row) => formatMetricForRow(row, "dependency_ttft_savings_us_p50") },
  { key: "status", label: "Status / Risk", width: 210, value: (row) => riskReasonsForRow(row).join(", ") || "Normal" },
];

const CACHE_COLUMNS = [
  { key: "shareable_prefix_ratio", label: "Shareable", width: 92, value: (row) => row.metrics?.shareable_prefix_ratio, format: (value) => `${formatNumber(Number(value) * 100)}%` },
  { key: "resident_prefix_ratio", label: "Resident", width: 92, value: (row) => row.metrics?.resident_prefix_ratio, format: (value) => `${formatNumber(Number(value) * 100)}%` },
  { key: "recovery_boundary_ratio", label: "Recovery", width: 118, value: (row) => row.metrics?.recovery_boundary_ratio, format: (value, row) => recoveryOutcome(row) === "resident_not_recoverable" ? "None (0%)" : `${formatNumber(Number(value) * 100)}%` },
  { key: "cache_total_capacity_GiB", label: "Cache GiB", width: 90, value: (row) => row.metrics?.cache_total_capacity_GiB, format: (value) => formatMetric(value, "GiB") },
  { key: "cache_eviction_GiB", label: "Evicted GiB", width: 98, value: (row) => row.metrics?.cache_eviction_GiB, format: (value) => formatMetric(value, "GiB") },
];

export function createTableState() {
  return { filter: "", sortKey: "id", sortDirection: "asc", page: 1 };
}

export function ensureSelectedPage(rows, state, selectedId) {
  const ordered = orderRows(filterRows(rows, state.filter), state, columnsForRows(rows));
  const index = ordered.findIndex((row) => row.id === selectedId);
  if (index >= 0) state.page = Math.floor(index / PAGE_SIZE) + 1;
}

export function renderResultsTable({
  container, caption, rows, selectedId, state, onSelect, onStateChange,
  thresholdMetric = null, threshold = NaN,
}) {
  container.replaceChildren();
  const columns = columnsForRows(rows);
  const filtered = filterRows(rows, state.filter);
  const ordered = orderRows(filtered, state, columns);
  const pageCount = Math.max(1, Math.ceil(ordered.length / PAGE_SIZE));
  state.page = Math.max(1, Math.min(pageCount, state.page));
  const start = (state.page - 1) * PAGE_SIZE;
  const pageRows = ordered.slice(start, start + PAGE_SIZE);
  caption.textContent = `${filtered.length.toLocaleString()} / ${rows.length.toLocaleString()} cases · page ${state.page} of ${pageCount}`;

  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  const table = document.createElement("table");
  table.className = "results-table";
  table.setAttribute("aria-label", "Sweep cases and derived risk status");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.style.width = `${column.width}px`;
    if (state.sortKey === column.key) th.setAttribute("aria-sort", state.sortDirection === "asc" ? "ascending" : "descending");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sort-button";
    button.textContent = `${column.label}${state.sortKey === column.key ? (state.sortDirection === "asc" ? " ↑" : " ↓") : ""}`;
    button.addEventListener("click", () => {
      if (state.sortKey === column.key) state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
      else { state.sortKey = column.key; state.sortDirection = "asc"; }
      state.page = 1;
      onStateChange();
    });
    th.append(button);
    headRow.append(th);
  }
  head.append(headRow);
  table.append(head);

  const body = document.createElement("tbody");
  if (!pageRows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = columns.length;
    cell.className = "empty-cell";
    cell.textContent = rows.length ? "No cases match this filter." : "Run a sweep to populate results.";
    row.append(cell);
    body.append(row);
  }
  const interactiveRows = [];
  for (const row of pageRows) {
    const tr = document.createElement("tr");
    const risk = riskForRow(row);
    tr.className = risk === "normal" ? "" : `risk-${risk}`;
    tr.tabIndex = row.id === selectedId ? 0 : -1;
    tr.dataset.caseId = row.id;
    tr.setAttribute("aria-selected", String(row.id === selectedId));
    tr.setAttribute("aria-label", `${row.id}, ${row.model}, prefix ${formatNumber(row.prefix_ratio * 100)} percent, ${risk}`);
    tr.addEventListener("click", () => { tr.focus(); onSelect(row, tr); });
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(row, tr); }
    });
    for (const column of columns) {
      const td = document.createElement("td");
      const value = column.key === "status"
        ? riskReasonsForRow(row, thresholdMetric, threshold).join(", ") || "Normal"
        : column.value(row);
      td.textContent = column.key === "status"
        ? value
        : column.format ? column.format(value, row) : String(value ?? "Missing");
      td.title = td.textContent;
      if (column.key === "dependency_ttft_savings_us_p50") {
        const stateValue = classifyValue(value);
        if ([VALUE_STATES.NEGATIVE, VALUE_STATES.INFINITE].includes(stateValue)) {
          const marker = document.createElement("span");
          marker.className = "status-marker";
          marker.textContent = stateValue === VALUE_STATES.INFINITE ? "∞" : "!";
          td.append(marker);
        }
      }
      tr.append(td);
    }
    body.append(tr);
    interactiveRows.push(tr);
  }
  wireRovingRows(interactiveRows);
  table.append(body);
  scroll.append(table);
  container.append(scroll);

  const pagination = document.createElement("nav");
  pagination.className = "table-pagination";
  pagination.setAttribute("aria-label", "Case table pagination");
  const status = document.createElement("span");
  status.textContent = pageRows.length ? `Rows ${start + 1}–${start + pageRows.length}` : "No rows";
  pagination.append(status);
  pagination.append(pageButton("Previous", state.page === 1, () => { state.page -= 1; onStateChange(); }));
  for (const page of pageWindow(state.page, pageCount)) {
    const button = pageButton(String(page), false, () => { state.page = page; onStateChange(); });
    if (page === state.page) { button.setAttribute("aria-current", "page"); button.classList.add("active"); }
    pagination.append(button);
  }
  pagination.append(pageButton("Next", state.page === pageCount, () => { state.page += 1; onStateChange(); }));
  container.append(pagination);
}

function pageButton(label, disabled, handler) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", handler);
  return button;
}

function columnsForRows(rows) {
  const applicable = [...COLUMNS];
  if (rows.some((row) => row.encoder_state != null)) applicable.splice(2, 0,
    { key: "encoder_state", label: "Encoder state", width: 115, value: (row) => row.encoder_state ?? "—" },
    { key: "encoder_tail_policy", label: "Tail output", width: 112, value: (row) => row.encoder_tail_policy ?? "—" });
  if (!rows.some((row) => row.analysis_mode === "cache_replay")) return applicable;
  const columns = applicable.map((column) => (
    column.key === "prefix_ratio" ? { ...column, label: "Configured", width: 108 } : column
  ));
  const hitIndex = columns.findIndex((column) => column.key === "prefix_ratio");
  columns.splice(hitIndex + 1, 0, ...CACHE_COLUMNS);
  return columns;
}

function filterRows(rows, filter) {
  const needle = String(filter || "").trim().toLowerCase();
  if (!needle) return [...rows];
  return rows.filter((row) => [
    row.id, row.model, row.attention, row.seq_len, row.prefix_ratio,
    row.metrics?.resident_prefix_ratio, row.metrics?.cache_total_capacity_GiB,
    row.block_size_tokens, row.arrival_model, row.max_io_bytes, row.encoder_state, row.encoder_tail_policy,
  ].some((value) => String(value ?? "").toLowerCase().includes(needle)));
}

function orderRows(rows, state, columns = COLUMNS) {
  const column = columns.find((item) => item.key === state.sortKey) || columns[0];
  const direction = state.sortDirection === "desc" ? -1 : 1;
  return [...rows].sort((a, b) => compare(column.value(a), column.value(b)) * direction || String(a.id).localeCompare(String(b.id)));
}

function compare(a, b) {
  if (a === b) return 0;
  if (a === null || a === undefined || Number.isNaN(a)) return 1;
  if (b === null || b === undefined || Number.isNaN(b)) return -1;
  return typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
}

function pageWindow(current, total) {
  const pages = new Set([1, total]);
  for (let page = current - 1; page <= current + 1; page += 1) if (page >= 1 && page <= total) pages.add(page);
  return [...pages].sort((a, b) => a - b);
}

function wireRovingRows(rows) {
  if (!rows.length) return;
  if (!rows.some((row) => row.tabIndex === 0)) rows[0].tabIndex = 0;
  rows.forEach((row, index) => {
    row.addEventListener("keydown", (event) => {
      let target = null;
      if (event.key === "ArrowDown") target = rows[(index + 1) % rows.length];
      if (event.key === "ArrowUp") target = rows[(index - 1 + rows.length) % rows.length];
      if (event.key === "Home") target = rows[0];
      if (event.key === "End") target = rows[rows.length - 1];
      if (!target) return;
      event.preventDefault();
      rows.forEach((item) => { item.tabIndex = item === target ? 0 : -1; });
      target.focus();
    });
  });
}
