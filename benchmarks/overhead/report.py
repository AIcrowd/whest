"""Terminal and browser reporting for overhead benchmark runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.overhead import SCHEMA_VERSION
from benchmarks.overhead.artifacts import _json_text


def _count_by(cases: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "passed": 0, "failed": 0}
    )
    for case in cases:
        value = str(case.get(key, "unknown"))
        bucket = buckets[value]
        bucket["count"] += 1
        if case.get("passed", False):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    return dict(sorted(buckets.items()))


def _comparison_maps(
    comparison: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not comparison:
        return {}, {}

    base_cases = {
        row["case_id"]: row for row in comparison.get("base", {}).get("cases", [])
    }
    deltas = {
        row["case_id"]: row
        for row in [
            *comparison.get("regressions", []),
            *comparison.get("improvements", []),
        ]
    }
    return base_cases, deltas


def build_browser_report_payload(
    run: dict[str, Any], comparison: dict[str, Any] | None = None
) -> dict[str, Any]:
    base_cases, delta_rows = _comparison_maps(comparison)
    flattened_cases: list[dict[str, Any]] = []

    for case in run.get("cases", []):
        numpy_row = case.get("numpy", {})
        whest_row = case.get("whest", {})
        startup = case.get("startup", {})
        steady_state_delta_ns = float(whest_row.get("median_ns", 0.0)) - float(
            numpy_row.get("median_ns", 0.0)
        )
        startup_delta_ns = float(
            startup.get("whest", {}).get("elapsed_ns", 0.0)
        ) - float(startup.get("numpy", {}).get("elapsed_ns", 0.0))
        details = case.get("whest_details", {})
        operations = details.get("operations", {})
        baseline_case = base_cases.get(case["case_id"], {})
        comparison_row = delta_rows.get(case["case_id"], {})

        flattened_cases.append(
            {
                "case_id": case["case_id"],
                "op_name": case.get("op_name"),
                "family": case.get("family"),
                "surface": case.get("surface"),
                "size_name": case.get("size_name"),
                "dtype": case.get("dtype"),
                "source_file": case.get("source_file"),
                "ratio": case.get("ratio"),
                "threshold": case.get("threshold"),
                "passed": case.get("passed"),
                "policy_source": case.get("policy_source"),
                "numpy_median_ns": numpy_row.get("median_ns"),
                "whest_median_ns": whest_row.get("median_ns"),
                "steady_state_delta_ns": steady_state_delta_ns,
                "startup_ratio": startup.get("ratio"),
                "startup_numpy_ns": startup.get("numpy", {}).get("elapsed_ns"),
                "startup_whest_ns": startup.get("whest", {}).get("elapsed_ns"),
                "startup_delta_ns": startup_delta_ns,
                "flops_used": details.get("flops_used"),
                "op_count": details.get("op_count"),
                "tracked_time_s": details.get("tracked_time_s"),
                "operation_names": sorted(operations.keys()),
                "operations": operations,
                "baseline_ratio": baseline_case.get("ratio"),
                "ratio_delta": comparison_row.get("ratio_delta"),
            }
        )

    flattened_cases.sort(
        key=lambda row: (float(row.get("ratio", 0.0)), row["case_id"]),
        reverse=True,
    )

    comparison_summary = None
    if comparison is not None:
        comparison_summary = {
            "shared_case_count": comparison.get("shared_case_count", 0),
            "regression_count": len(comparison.get("regressions", [])),
            "improvement_count": len(comparison.get("improvements", [])),
            "top_regressions": comparison.get("regressions", [])[:5],
            "top_improvements": comparison.get("improvements", [])[:5],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": run.get("manifest", {}),
        "environment": run.get("environment", {}),
        "summary": run.get("summary", {}),
        "accountability": run.get("accountability", {}),
        "operations": run.get("operations", []),
        "cases": flattened_cases,
        "top_cases_by_ratio": flattened_cases[:10],
        "aggregates": {
            "families": _count_by(flattened_cases, "family"),
            "surfaces": _count_by(flattened_cases, "surface"),
            "sizes": _count_by(flattened_cases, "size_name"),
        },
        "comparison": comparison_summary,
    }


def render_browser_report(payload: dict[str, Any]) -> str:
    embedded_json = _json_text(payload).strip().replace("</", "<\\/")
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whest Overhead Benchmark Report</title>
  <style>
    :root {
      color-scheme: light;
      --coral: #F0524D;
      --coral-hover: #D23934;
      --coral-light: #FEF2F1;
      --gray-900: #292C2D;
      --gray-600: #5D5F60;
      --gray-400: #AAACAD;
      --gray-200: #D9DCDC;
      --gray-100: #F1F3F5;
      --gray-50: #F8F9F9;
      --white: #FFFFFF;
      --success: #23B761;
      --warning: #FA9E33;
      --info: #4A7CFF;
      --error: #F0524D;
      --font-display-serif: "Newsreader", "Source Serif 4", Georgia, serif;
      --font-app-sans: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      --font-mono: "JetBrains Mono", "IBM Plex Mono", "SFMono-Regular", monospace;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --radius: 0.5rem;
      --radius-pill: 999px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--gray-50);
      color: var(--gray-900);
      font-family: var(--font-app-sans);
      font-size: 14px;
      line-height: 1.5;
    }
    .report-shell {
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px;
    }
    h1, h2, h3, p {
      margin: 0;
    }
    .eyebrow {
      color: var(--gray-400);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }
    .report-section-card {
      background: var(--white);
      border: 1px solid var(--gray-200);
      border-radius: 8px;
      box-shadow: var(--shadow-sm);
    }
    .report-hero {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
    }
    .brand-line {
      display: flex;
      align-items: baseline;
      gap: 10px;
    }
    .brand-line h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand-line .dot {
      color: var(--coral);
    }
    .hero-subtitle {
      margin-top: 8px;
      color: var(--gray-600);
      max-width: 70ch;
    }
    .report-meta {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .hero-mode-pill,
    .summary-pill,
    .case-status-pill,
    .policy-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.18rem 0.6rem;
      border-radius: var(--radius-pill);
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      white-space: nowrap;
      border: 1px solid var(--gray-200);
    }
    .hero-mode-pill {
      background: var(--coral-light);
      color: var(--coral-hover);
      border-color: rgba(240, 82, 77, 0.25);
    }
    .summary-pill {
      background: var(--gray-100);
      color: var(--gray-600);
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 16px;
    }
    .metric-tile {
      padding: 16px 18px;
      background: var(--white);
      border: 1px solid var(--gray-200);
      border-radius: 8px;
      box-shadow: var(--shadow-sm);
    }
    .metric-label {
      color: var(--gray-400);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .metric-value {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .metric-note {
      margin-top: 6px;
      color: var(--gray-600);
      font-size: 12px;
    }
    .overview-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .section-card-header {
      padding: 16px 18px 12px;
      border-bottom: 1px solid var(--gray-200);
    }
    .section-card-header h2 {
      margin-top: 6px;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .section-card-body {
      padding: 16px 18px 18px;
    }
    .environment-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }
    .environment-row {
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--gray-200);
      background: var(--gray-50);
    }
    .environment-row-label {
      color: var(--gray-400);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .environment-row-value {
      display: block;
      margin-top: 8px;
      font-family: var(--font-mono);
      font-size: 12px;
      line-height: 1.45;
      color: var(--gray-900);
      word-break: break-word;
    }
    .top-list {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 10px;
    }
    .top-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px;
      border: 1px solid var(--gray-200);
      border-radius: 8px;
      background: var(--gray-50);
    }
    .code-token,
    .operation-token,
    .category-chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.18rem 0.6rem;
      border-radius: var(--radius-pill);
      border: 1px solid var(--gray-200);
      background: var(--gray-100);
      color: var(--gray-600);
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      white-space: nowrap;
    }
    .code-token,
    .operation-token {
      font-family: var(--font-mono);
    }
    .section-stack {
      display: grid;
      gap: 16px;
      margin-top: 16px;
    }
    .controls-card {
      padding: 16px 18px;
    }
    .controls-header h2 {
      margin-top: 6px;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .controls-subtitle {
      margin-top: 8px;
      color: var(--gray-600);
      max-width: 80ch;
    }
    .filter-grid {
      display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(5, minmax(120px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .operation-filter-grid {
      display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(2, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .control-field label {
      display: block;
      margin-bottom: 8px;
      color: var(--gray-400);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 600;
    }
    input, select {
      width: 100%;
      min-height: 40px;
      border-radius: 6px;
      border: 1px solid var(--gray-200);
      background: var(--white);
      color: var(--gray-900);
      padding: 8px 12px;
      font: inherit;
    }
    input:focus,
    select:focus {
      outline: none;
      border-color: var(--coral);
      box-shadow: 0 0 0 2px rgba(240, 82, 77, 0.15);
    }
    .operation-inventory,
    .technical-surface {
      overflow: hidden;
    }
    .table-wrap {
      overflow: auto;
      max-height: 66vh;
      border-top: 1px solid var(--gray-200);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    thead th {
      text-align: left;
      color: var(--gray-600);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 10px 16px;
      border-bottom: 2px solid var(--gray-200);
      position: sticky;
      top: 0;
      background: var(--gray-50);
      z-index: 1;
    }
    .sort-button {
      display: inline-flex;
      align-items: center;
      justify-content: flex-start;
      gap: 8px;
      width: 100%;
      padding: 0;
      border: 0;
      background: none;
      color: inherit;
      font: inherit;
      letter-spacing: inherit;
      text-transform: inherit;
      cursor: pointer;
    }
    .sort-button:hover,
    .sort-button.is-active {
      color: var(--coral-hover);
    }
    .sort-indicator {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 12px;
      color: var(--gray-400);
      font-family: var(--font-mono);
      font-size: 11px;
      line-height: 1;
    }
    .sort-button.is-active .sort-indicator {
      color: var(--coral-hover);
    }
    tbody td {
      padding: 10px 16px;
      border-bottom: 1px solid var(--gray-100);
      vertical-align: top;
    }
    tbody tr:hover {
      background: var(--gray-50);
    }
    code {
      font-family: var(--font-mono);
      font-size: 12px;
    }
    .case-cell-stack,
    .operation-cell-stack {
      display: grid;
      gap: 8px;
    }
    .case-status-pill {
      background: var(--gray-100);
      color: var(--gray-600);
    }
    .case-status-pass {
      background: rgba(35, 183, 97, 0.13);
      color: #126E39;
      border-color: rgba(35, 183, 97, 0.25);
    }
    .case-status-fail {
      background: var(--coral-light);
      color: var(--coral-hover);
      border-color: rgba(240, 82, 77, 0.25);
    }
    .policy-pill {
      background: var(--gray-100);
      color: var(--gray-600);
      max-width: 100%;
    }
    .value-block {
      display: grid;
      gap: 6px;
    }
    .value-detail {
      color: var(--gray-600);
      font-size: 12px;
      line-height: 1.45;
    }
    .empty-state {
      padding: 28px 16px;
      text-align: center;
      color: var(--gray-600);
      font-size: 14px;
    }
    .op-link {
      color: var(--coral-hover);
      font-weight: 600;
      text-decoration: none;
    }
    .op-link:hover {
      text-decoration: underline;
    }
    .category-chip {
      background: rgba(74, 124, 255, 0.12);
      color: #1E4ACC;
      border-color: rgba(74, 124, 255, 0.3);
    }
    .coverage-missing {
      background: rgba(250, 158, 51, 0.15);
      color: #A8650B;
      border-color: rgba(250, 158, 51, 0.3);
    }
    .coverage-idle {
      background: var(--gray-100);
      color: var(--gray-600);
    }
    .section-spacer {
      margin-top: 16px;
    }
    @media (max-width: 1080px) {
      .overview-grid {
        grid-template-columns: 1fr;
      }
      .filter-grid {
        grid-template-columns: repeat(2, minmax(180px, 1fr));
      }
      .operation-filter-grid {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 760px) {
      .report-shell {
        padding: 14px;
      }
      .filter-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="report-shell">
    <header class="report-hero report-section-card">
      <div>
        <div class="eyebrow">Whest overhead harness</div>
        <div class="brand-line">
          <h1>whest<span class="dot">.</span>bench overhead report</h1>
        </div>
        <p class="hero-subtitle">
          Scan measured wrapper overhead across the documented API inventory, then drill into the concrete benchmark cases behind each operation.
        </p>
        <div class="report-meta">
          <span id="hero-mode-pill" class="hero-mode-pill">mode</span>
          <span id="comparison-badge"></span>
        </div>
      </div>
    </header>

    <section id="summary-cards" class="summary-grid"></section>

    <div class="overview-grid">
      <section class="report-section-card">
        <div class="section-card-header">
          <div>
            <div class="eyebrow">Runtime fingerprint</div>
            <h2>Environment</h2>
          </div>
        </div>
        <div class="section-card-body">
          <div id="environment-panel" class="environment-grid"></div>
        </div>
      </section>
      <section class="report-section-card">
        <div class="section-card-header">
          <div>
            <div class="eyebrow">Coverage & accountability</div>
            <h2>Inventory Notes</h2>
          </div>
        </div>
        <div class="section-card-body">
          <ul id="accountability-panel" class="top-list"></ul>
        </div>
      </section>
    </div>

    <section id="case-drilldown" class="controls-card report-section-card section-spacer">
      <div class="controls-header">
        <div>
          <div class="eyebrow">Operation inventory</div>
          <h2>All Documented Operations</h2>
          <p class="controls-subtitle">
            The main scan surface mirrors the docs API list: one row per operation, with measured overhead where available and explicit coverage states everywhere else. Click any column header to sort the visible rows.
          </p>
        </div>
      </div>
      <div class="operation-filter-grid">
        <div class="control-field">
          <label for="operation-search">Search operations</label>
          <input id="operation-search" type="search" placeholder="Operation name, module, notes, or refs">
        </div>
        <div class="control-field">
          <label for="coverage-filter">Coverage</label>
          <select id="coverage-filter"></select>
        </div>
        <div class="control-field">
          <label for="category-filter">Category</label>
          <select id="category-filter"></select>
        </div>
      </div>
    </section>

    <section class="operation-inventory report-section-card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th><button type="button" class="sort-button" data-sort-table="operations" data-sort-key="name">Operation<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="operations" data-sort-key="module">Module<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="operations" data-sort-key="category">Type<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="operations" data-sort-key="coverage_status">Coverage<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="operations" data-sort-key="representative_ratio">Representative<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="operations" data-sort-key="worst_ratio">Worst<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="operations" data-sort-key="measured_case_count">Cases<span class="sort-indicator"></span></button></th>
            </tr>
          </thead>
          <tbody id="operation-table-body"></tbody>
        </table>
      </div>
    </section>

    <section class="controls-card report-section-card section-spacer">
      <div class="controls-header">
        <div>
          <div class="eyebrow">Case drill-down</div>
          <h2>Measured Cases</h2>
          <p class="controls-subtitle">
            Use the lower table to inspect concrete benchmark cases, policy outcomes, startup ratios, and the `whest` operation log behind a measured row. Click any column header to sort the visible rows.
          </p>
        </div>
      </div>
      <div class="filter-grid">
        <div class="control-field">
          <label for="case-search">Search</label>
          <input id="case-search" type="search" placeholder="Case id, family, source file, or operation">
        </div>
        <div class="control-field">
          <label for="family-filter">Family</label>
          <select id="family-filter"></select>
        </div>
        <div class="control-field">
          <label for="surface-filter">Surface</label>
          <select id="surface-filter"></select>
        </div>
        <div class="control-field">
          <label for="size-filter">Size</label>
          <select id="size-filter"></select>
        </div>
        <div class="control-field">
          <label for="status-filter">Status</label>
          <select id="status-filter"></select>
        </div>
        <div class="control-field">
          <label for="sort-filter">Sort</label>
          <select id="sort-filter"></select>
        </div>
      </div>
    </section>

    <section class="technical-surface report-section-card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="case_id">Case<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="family">Family<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="surface">Surface<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="size_name">Size<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="passed">Status<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="ratio">Ratio<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="threshold">Threshold<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="steady_state_delta_ns">Delta<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="startup_ratio">Startup<span class="sort-indicator"></span></button></th>
              <th><button type="button" class="sort-button" data-sort-table="cases" data-sort-key="op_count">Ops<span class="sort-indicator"></span></button></th>
            </tr>
          </thead>
          <tbody id="case-table-body"></tbody>
        </table>
      </div>
    </section>
  </div>

  <script id="report-data" type="application/json">__REPORT_JSON__</script>
  <script>
    function reviveFloat(_key, value) {
      if (value && typeof value === "object" && Object.keys(value).length === 1 && value.__whest_float__) {
        if (value.__whest_float__ === "inf") return Infinity;
        if (value.__whest_float__ === "-inf") return -Infinity;
        if (value.__whest_float__ === "nan") return NaN;
      }
      return value;
    }

    const data = JSON.parse(document.getElementById("report-data").textContent, reviveFloat);

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatNumber(value, digits = 2) {
      if (value === null || value === undefined) return "—";
      if (Number.isNaN(value)) return "NaN";
      if (!Number.isFinite(value)) return value > 0 ? "∞" : "-∞";
      return Number(value).toFixed(digits);
    }

    function formatInt(value) {
      if (value === null || value === undefined) return "—";
      if (Number.isNaN(value)) return "NaN";
      if (!Number.isFinite(value)) return value > 0 ? "∞" : "-∞";
      return Math.round(Number(value)).toLocaleString();
    }

    function formatRatio(value) {
      return `${formatNumber(value)}x`;
    }

    function optionMarkup(value, label) {
      return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
    }

    function buildSummaryCards() {
      const summary = data.summary || {};
      const measuredOps = (data.operations || []).filter((row) => (row.measured_case_count || 0) > 0).length;
      const cards = [
        ["Mode", data.manifest?.mode || "unknown", "Selected harness mode for this artifact."],
        ["Measured ops", formatInt(measuredOps), "Documented operations with measured benchmark coverage in this run."],
        ["Measured cases", formatInt(summary.case_count || 0), "Concrete benchmark cases recorded in this run."],
        ["Worst ratio", formatRatio(summary.worst_ratio || 0), "Largest measured steady-state slowdown versus NumPy."],
      ];
      if (data.comparison) {
        cards.push([
          "Shared vs Baseline",
          formatInt(data.comparison.shared_case_count || 0),
          "Cases available in both the current run and the comparison baseline.",
        ]);
      }
      document.getElementById("summary-cards").innerHTML = cards.map(([label, value, note]) => `
        <article class="metric-tile">
          <div class="metric-label">${escapeHtml(label)}</div>
          <div class="metric-value">${escapeHtml(value)}</div>
          <div class="metric-note">${escapeHtml(note)}</div>
        </article>
      `).join("");

      document.getElementById("hero-mode-pill").textContent = `mode · ${data.manifest?.mode || "unknown"}`;
    }

    function buildEnvironmentPanel() {
      const software = data.environment?.software || {};
      const hardware = data.environment?.hardware || {};
      const rows = [
        ["Python", software.python],
        ["NumPy", software.numpy],
        ["OS", software.os],
        ["BLAS", software.blas],
        ["Arch", hardware.arch],
        ["CPU cores", hardware.cpu_cores],
        ["Timestamp", data.environment?.timestamp],
      ].filter(([, value]) => value !== undefined && value !== null);

      document.getElementById("environment-panel").innerHTML = rows.map(([label, value]) => `
        <div class="environment-row">
          <div class="environment-row-label">${escapeHtml(label)}</div>
          <code class="environment-row-value">${escapeHtml(value)}</code>
        </div>
      `).join("");
    }

    function buildAccountabilityPanel() {
      const accountability = data.accountability || {};
      const items = [
        ["documented ops", formatInt((data.operations || []).length)],
        ["discovered callables", formatInt(accountability.discovered_total || 0)],
        ["discovery-only callables", formatInt((accountability.unclassified_operations || []).length)],
        ["missing docs/discovery matches", formatInt((accountability.discovered_missing_in_docs || []).length + (accountability.documented_missing_in_discovery || []).length)],
      ];
      document.getElementById("accountability-panel").innerHTML = items.map(([label, value]) => `
        <li class="top-item">
          <span>${escapeHtml(label)}</span>
          <span class="summary-pill">${escapeHtml(value)}</span>
        </li>
      `).join("");
    }

    function setSelectOptions(id, values, label) {
      const select = document.getElementById(id);
      select.innerHTML = optionMarkup("", `All ${label}`) + values.map((value) => optionMarkup(value, value)).join("");
    }

    function initializeOperationFilters() {
      setSelectOptions(
        "coverage-filter",
        Array.from(new Set((data.operations || []).map((row) => row.coverage_status).filter(Boolean))).sort(),
        "coverage states"
      );
      setSelectOptions(
        "category-filter",
        Array.from(new Set((data.operations || []).map((row) => row.category).filter(Boolean))).sort(),
        "categories"
      );
    }

    function initializeCaseFilters() {
      setSelectOptions("family-filter", Object.keys(data.aggregates?.families || {}), "families");
      setSelectOptions("surface-filter", Object.keys(data.aggregates?.surfaces || {}), "surfaces");
      setSelectOptions("size-filter", Object.keys(data.aggregates?.sizes || {}), "sizes");
      document.getElementById("status-filter").innerHTML = [
        optionMarkup("", "All statuses"),
        optionMarkup("pass", "Passing"),
        optionMarkup("fail", "Failing"),
      ].join("");
      document.getElementById("sort-filter").innerHTML = [
        optionMarkup("ratio:desc", "Sort: ratio desc"),
        optionMarkup("steady_state_delta_ns:desc", "Sort: delta desc"),
        optionMarkup("startup_ratio:desc", "Sort: startup ratio desc"),
        optionMarkup("case_id:asc", "Sort: case id asc"),
        optionMarkup("family:asc", "Sort: family asc"),
        optionMarkup("surface:asc", "Sort: surface asc"),
        optionMarkup("size_name:asc", "Sort: size asc"),
        optionMarkup("passed:asc", "Sort: fail first"),
        optionMarkup("threshold:desc", "Sort: threshold desc"),
        optionMarkup("op_count:desc", "Sort: ops desc"),
      ].join("");
      document.getElementById("sort-filter").value = "ratio:desc";

      const badge = document.getElementById("comparison-badge");
      if (data.comparison) {
        badge.innerHTML = `<span class="summary-pill">baseline diff · ${formatInt(
          data.comparison.shared_case_count
        )} shared cases</span>`;
      }
    }

    const SIZE_RANK = {
      tiny: 0,
      small: 1,
      medium: 2,
      large: 3,
      huge: 4,
    };
    const COVERAGE_RANK = {
      partial_error: 0,
      benchmark_error: 1,
      measured: 2,
      profile_missing: 3,
      unsupported: 4,
      excluded: 5,
      not_in_run: 6,
    };
    const OPERATION_SORT_DEFAULTS = {
      name: "asc",
      module: "asc",
      category: "asc",
      coverage_status: "asc",
      representative_ratio: "desc",
      worst_ratio: "desc",
      measured_case_count: "desc",
    };
    const CASE_SORT_DEFAULTS = {
      case_id: "asc",
      family: "asc",
      surface: "asc",
      size_name: "asc",
      passed: "asc",
      ratio: "desc",
      threshold: "desc",
      steady_state_delta_ns: "desc",
      startup_ratio: "desc",
      op_count: "desc",
    };
    const operationSortState = { key: "worst_ratio", direction: "desc" };
    const caseSortState = { key: "ratio", direction: "desc" };

    function defaultSortDirection(tableName, key) {
      const defaults = tableName === "operations" ? OPERATION_SORT_DEFAULTS : CASE_SORT_DEFAULTS;
      return defaults[key] || "asc";
    }

    function toggleSort(state, tableName, key) {
      if (state.key === key) {
        state.direction = state.direction === "asc" ? "desc" : "asc";
        return;
      }
      state.key = key;
      state.direction = defaultSortDirection(tableName, key);
    }

    function syncCaseSortControl() {
      const control = document.getElementById("sort-filter");
      const value = `${caseSortState.key}:${caseSortState.direction}`;
      if (control instanceof HTMLSelectElement) control.value = value;
    }

    function applyCaseSortSelection(value) {
      const [key, direction] = String(value || "").split(":");
      caseSortState.key = CASE_SORT_DEFAULTS[key] ? key : "ratio";
      caseSortState.direction = direction === "asc" ? "asc" : "desc";
      syncCaseSortControl();
    }

    function normalizeNumber(value) {
      if (value === null || value === undefined || value === "") return null;
      const number = Number(value);
      return Number.isNaN(number) ? null : number;
    }

    function compareNumbers(a, b, direction = "asc") {
      const left = normalizeNumber(a);
      const right = normalizeNumber(b);
      if (left === null && right === null) return 0;
      if (left === null) return 1;
      if (right === null) return -1;
      if (left === right) return 0;
      return direction === "asc" ? left - right : right - left;
    }

    function compareStrings(a, b, direction = "asc") {
      const left = a === null || a === undefined || a === "" ? null : String(a);
      const right = b === null || b === undefined || b === "" ? null : String(b);
      if (left === null && right === null) return 0;
      if (left === null) return 1;
      if (right === null) return -1;
      const result = left.localeCompare(right, undefined, { sensitivity: "base" });
      return direction === "asc" ? result : -result;
    }

    function compareValues(a, b, direction = "asc") {
      const leftNumber = normalizeNumber(a);
      const rightNumber = normalizeNumber(b);
      if (leftNumber !== null || rightNumber !== null) {
        return compareNumbers(leftNumber, rightNumber, direction);
      }
      return compareStrings(a, b, direction);
    }

    function operationSortValue(row, key) {
      switch (key) {
        case "name":
          return row.name || row.slug;
        case "module":
          return row.module;
        case "category":
          return row.category;
        case "coverage_status":
          return COVERAGE_RANK[row.coverage_status] ?? 999;
        case "representative_ratio":
          return row.representative_ratio;
        case "worst_ratio":
          return row.worst_ratio;
        case "measured_case_count":
          return row.measured_case_count;
        default:
          return row[key];
      }
    }

    function caseSortValue(row, key) {
      switch (key) {
        case "case_id":
          return row.case_id;
        case "family":
          return row.family;
        case "surface":
          return row.surface;
        case "size_name":
          return SIZE_RANK[row.size_name] ?? 999;
        case "passed":
          return row.passed ? 1 : 0;
        case "ratio":
          return row.ratio;
        case "threshold":
          return row.threshold;
        case "steady_state_delta_ns":
          return row.steady_state_delta_ns;
        case "startup_ratio":
          return row.startup_ratio;
        case "op_count":
          return row.op_count ?? (row.operation_names || []).length;
        default:
          return row[key];
      }
    }

    function compareOperationRows(a, b) {
      const primary = compareValues(
        operationSortValue(a, operationSortState.key),
        operationSortValue(b, operationSortState.key),
        operationSortState.direction
      );
      if (primary !== 0) return primary;
      return compareStrings(a.name || a.slug, b.name || b.slug, "asc");
    }

    function compareCaseRows(a, b) {
      const primary = compareValues(
        caseSortValue(a, caseSortState.key),
        caseSortValue(b, caseSortState.key),
        caseSortState.direction
      );
      if (primary !== 0) return primary;
      return compareStrings(a.case_id, b.case_id, "asc");
    }

    function updateSortButtons() {
      document.querySelectorAll(".sort-button[data-sort-table]").forEach((button) => {
        const tableName = button.getAttribute("data-sort-table");
        const key = button.getAttribute("data-sort-key");
        const state = tableName === "operations" ? operationSortState : caseSortState;
        const active = state.key === key;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
        const indicator = button.querySelector(".sort-indicator");
        if (indicator) indicator.textContent = active ? (state.direction === "asc" ? "^" : "v") : "";
        const header = button.closest("th");
        if (header) header.setAttribute("aria-sort", active ? (state.direction === "asc" ? "ascending" : "descending") : "none");
      });
    }

    function activeOperationRows() {
      const search = document.getElementById("operation-search").value.trim().toLowerCase();
      const coverage = document.getElementById("coverage-filter").value;
      const category = document.getElementById("category-filter").value;
      const rows = (data.operations || []).filter((row) => {
        if (coverage && row.coverage_status !== coverage) return false;
        if (category && row.category !== category) return false;
        if (!search) return true;
        const haystack = [
          row.name,
          row.slug,
          row.module,
          row.category,
          row.summary,
          row.notes,
          row.numpy_ref,
          row.whest_ref,
        ].join(" ").toLowerCase();
        return haystack.includes(search);
      });
      rows.sort(compareOperationRows);
      return rows;
    }

    function renderOperationTable() {
      const rows = activeOperationRows();
      if (!rows.length) {
        document.getElementById("operation-table-body").innerHTML = `
          <tr><td class="empty-state" colspan="7">No operations match the current filter set.</td></tr>
        `;
        return;
      }
      document.getElementById("operation-table-body").innerHTML = rows.map((row, index) => `
        <tr>
          <td>
            <div class="operation-cell-stack">
              <a class="op-link" href="#case-drilldown" data-case-filter="${escapeHtml(row.slug || row.name || "")}">${escapeHtml(row.name || row.slug || "unknown")}</a>
              <div class="value-detail">${escapeHtml(row.summary || "")}</div>
              <div class="value-detail"><code>${escapeHtml(row.whest_ref || "—")}</code></div>
              ${row.error_messages && row.error_messages.length ? `<div id="operation-row-note-${index}" class="value-detail">${escapeHtml(row.error_messages[0])}</div>` : ""}
            </div>
          </td>
          <td><span class="code-token">${escapeHtml(row.module || "—")}</span></td>
          <td><span class="category-chip">${escapeHtml(row.category || "—")}</span></td>
          <td>
            <span class="case-status-pill ${row.coverage_status === "measured" ? "case-status-pass" : row.coverage_status === "partial_error" || row.coverage_status === "benchmark_error" ? "case-status-fail" : row.coverage_status === "profile_missing" ? "coverage-missing" : "coverage-idle"}">
              ${escapeHtml(row.coverage_status || "unknown")}
            </span>
          </td>
          <td>${row.representative_ratio == null ? "—" : escapeHtml(formatRatio(row.representative_ratio))}</td>
          <td>${row.worst_ratio == null ? "—" : escapeHtml(formatRatio(row.worst_ratio))}</td>
          <td>
            <div class="value-block">
              <div>${escapeHtml(formatInt(row.measured_case_count || 0))}${row.expected_case_count ? ` / ${escapeHtml(formatInt(row.expected_case_count))}` : ""}</div>
              ${row.benchmark_error_count ? `<div class="value-detail">${escapeHtml(formatInt(row.benchmark_error_count))} benchmark errors</div>` : ""}
            </div>
          </td>
        </tr>
      `).join("");
      document.querySelectorAll(".op-link[data-case-filter]").forEach((link) => {
        link.addEventListener("click", (event) => {
          const filterValue = event.currentTarget?.getAttribute("data-case-filter") || "";
          const caseSearch = document.getElementById("case-search");
          if (!(caseSearch instanceof HTMLInputElement)) return;
          caseSearch.value = filterValue;
          renderCaseTable();
          caseSearch.focus();
        });
      });
      rows.forEach((row, index) => {
        if (!row.error_messages || !row.error_messages.length) return;
        const rowElement = document.getElementById(`operation-row-note-${index}`);
        if (rowElement) rowElement.textContent = row.error_messages[0];
      });
    }

    function activeCaseRows() {
      const search = document.getElementById("case-search").value.trim().toLowerCase();
      const family = document.getElementById("family-filter").value;
      const surface = document.getElementById("surface-filter").value;
      const size = document.getElementById("size-filter").value;
      const status = document.getElementById("status-filter").value;

      const rows = (data.cases || []).filter((row) => {
        if (family && row.family !== family) return false;
        if (surface && row.surface !== surface) return false;
        if (size && row.size_name !== size) return false;
        if (status === "pass" && !row.passed) return false;
        if (status === "fail" && row.passed) return false;
        if (!search) return true;
        const haystack = [
          row.case_id,
          row.family,
          row.surface,
          row.size_name,
          row.dtype,
          row.source_file,
          ...(row.operation_names || []),
        ].join(" ").toLowerCase();
        return haystack.includes(search);
      });

      rows.sort(compareCaseRows);
      return rows;
    }

    function renderCaseTable() {
      const rows = activeCaseRows();
      if (!rows.length) {
        document.getElementById("case-table-body").innerHTML = `
          <tr>
            <td class="empty-state" colspan="10">No cases match the current filter set.</td>
          </tr>
        `;
        return;
      }

      document.getElementById("case-table-body").innerHTML = rows.map((row) => `
        <tr>
          <td>
            <div class="case-cell-stack">
              <code class="code-token">${escapeHtml(row.case_id || "unknown")}</code>
              <div class="value-detail">${escapeHtml(row.source_file || "—")}</div>
              <div class="value-detail">
                ${row.op_name ? `<span class="operation-token">${escapeHtml(row.op_name)}</span>` : ""}
              </div>
            </div>
          </td>
          <td><span class="summary-pill">${escapeHtml(row.family || "—")}</span></td>
          <td><span class="summary-pill">${escapeHtml(row.surface || "—")}</span></td>
          <td><span class="summary-pill">${escapeHtml(row.size_name || "—")}</span></td>
          <td>
            <div class="value-block">
              <span class="case-status-pill ${row.passed ? "case-status-pass" : "case-status-fail"}">
                ${row.passed ? "Pass" : "Fail"}
              </span>
              <span class="policy-pill">${escapeHtml(row.policy_source || "default")}</span>
            </div>
          </td>
          <td>
            <div class="value-block">
              <div>${escapeHtml(formatRatio(row.ratio))}</div>
              <div class="value-detail">${escapeHtml(formatInt(row.whest_median_ns))} ns <code>whest</code></div>
            </div>
          </td>
          <td>
            <div class="value-block">
              <div>${escapeHtml(formatRatio(row.threshold))}</div>
              <div class="value-detail">${escapeHtml(formatInt(row.numpy_median_ns))} ns <code>numpy</code></div>
            </div>
          </td>
          <td>
            <div class="value-block">
              <div>${escapeHtml(formatInt(row.steady_state_delta_ns))} ns</div>
              <div class="value-detail">
                ${row.ratio_delta === undefined || row.ratio_delta === null ? "—" : `${escapeHtml(formatNumber(row.ratio_delta))}x vs baseline`}
              </div>
            </div>
          </td>
          <td>
            <div class="value-block">
              <div>${escapeHtml(formatRatio(row.startup_ratio))}</div>
              <div class="value-detail">${escapeHtml(formatInt(row.startup_delta_ns))} ns</div>
            </div>
          </td>
          <td>
            <div class="operation-cell-stack">
              ${(row.operation_names || []).map((name) => `<span class="operation-token">${escapeHtml(name)}</span>`).join("") || '<span class="summary-pill">—</span>'}
            </div>
          </td>
        </tr>
      `).join("");
    }

    buildSummaryCards();
    buildEnvironmentPanel();
    buildAccountabilityPanel();
    initializeOperationFilters();
    initializeCaseFilters();
    applyCaseSortSelection("ratio:desc");
    document.querySelectorAll(".sort-button[data-sort-table]").forEach((button) => {
      button.addEventListener("click", (event) => {
        const element = event.currentTarget;
        const tableName = element.getAttribute("data-sort-table");
        const key = element.getAttribute("data-sort-key");
        if (!tableName || !key) return;
        if (tableName === "operations") {
          toggleSort(operationSortState, tableName, key);
          renderOperationTable();
        } else {
          toggleSort(caseSortState, tableName, key);
          syncCaseSortControl();
          renderCaseTable();
        }
        updateSortButtons();
      });
    });
    renderOperationTable();
    renderCaseTable();
    updateSortButtons();
    ["operation-search", "coverage-filter", "category-filter"]
      .forEach((id) => {
        const element = document.getElementById(id);
        element.addEventListener("input", renderOperationTable);
        element.addEventListener("change", renderOperationTable);
      });
    ["case-search", "family-filter", "surface-filter", "size-filter", "status-filter"]
      .forEach((id) => {
        const element = document.getElementById(id);
        element.addEventListener("input", renderCaseTable);
        element.addEventListener("change", renderCaseTable);
      });
    const caseSortFilter = document.getElementById("sort-filter");
    caseSortFilter.addEventListener("change", (event) => {
      applyCaseSortSelection(event.currentTarget.value);
      renderCaseTable();
      updateSortButtons();
    });
  </script>
</body>
</html>
""".replace("__REPORT_JSON__", embedded_json)


def write_browser_report(
    root: Path, *, run: dict[str, Any], comparison: dict[str, Any] | None = None
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = build_browser_report_payload(run, comparison=comparison)
    (root / "report_data.json").write_text(_json_text(payload), encoding="utf-8")
    report_path = root / "report.html"
    report_path.write_text(render_browser_report(payload), encoding="utf-8")
    return report_path


def render_terminal_summary(
    manifest: dict[str, object], cases: list[dict[str, object]]
) -> str:
    """Render a compact terminal summary."""
    total = len(cases)
    failed = [case for case in cases if not case.get("passed", False)]
    passed = total - len(failed)
    unclassified = manifest.get("unclassified_operations", [])

    lines = [
        f"Mode: {manifest.get('mode', 'unknown')}",
        f"Cases: {total} total, {passed} passed, {len(failed)} failed",
        f"Discovery-only operations: {len(unclassified)}",
    ]

    if not failed:
        lines.append("All cases passed.")
        return "\n".join(lines)

    lines.append("Worst regressions:")
    ranked = sorted(
        failed,
        key=lambda case: (
            float(case.get("ratio", 0.0)) - float(case.get("threshold", 0.0)),
            float(case.get("ratio", 0.0)),
        ),
        reverse=True,
    )
    for case in ranked[:3]:
        lines.append(
            f"- {case['case_id']}: {float(case['ratio']):.2f}x > "
            f"{float(case['threshold']):.2f}x ({case['policy_source']})"
        )

    return "\n".join(lines)
