#!/usr/bin/env python
"""Generate enriched empirical-weights documentation from weights.json.

Reads ``src/mechestim/data/weights.json`` (which must contain
``meta.per_op_details``) and produces:

1. ``docs/reference/empirical-weights.csv`` -- 25-column review spreadsheet.
2. ``docs/reference/empirical-weights.md``  -- human-readable markdown page.

Usage::

    uv run python scripts/generate_empirical_weights_docs.py

Requires the enriched weights.json (with ``meta.per_op_details``).  If that
section is missing, the script prints an error and exits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "src" / "mechestim" / "data" / "weights.json"
CSV_OUT = REPO_ROOT / "docs" / "reference" / "empirical-weights.csv"
MD_OUT = REPO_ROOT / "docs" / "reference" / "empirical-weights.md"

# ---------------------------------------------------------------------------
# Category display names and ordering
# ---------------------------------------------------------------------------

#: Maps op-name prefixes / registry categories to display categories.
#: Order here defines the table order in the markdown output.
DISPLAY_CATEGORIES = [
    "Pointwise Unary",
    "Pointwise Binary",
    "Reductions",
    "Sorting",
    "FFT",
    "Linalg",
    "Contractions",
    "Polynomial",
    "Random",
    "Misc",
    "Window",
]

#: Map from per_op_details "category" field to display category.
CATEGORY_MAP: dict[str, str] = {
    "counted_unary": "Pointwise Unary",
    "counted_binary": "Pointwise Binary",
    "counted_reduction": "Reductions",
    "counted_custom": "Misc",  # default for custom, overridden below
}


def _classify_op(op_name: str, detail_category: str) -> str:
    """Determine display category for an operation."""
    if op_name.startswith("fft."):
        return "FFT"
    if op_name.startswith("linalg."):
        return "Linalg"
    if op_name.startswith("random."):
        return "Random"
    if op_name in _CONTRACTION_OPS:
        return "Contractions"
    if op_name in _SORTING_OPS:
        return "Sorting"
    if op_name in _WINDOW_OPS:
        return "Window"
    if op_name in _POLYNOMIAL_OPS:
        return "Polynomial"
    return CATEGORY_MAP.get(detail_category, "Misc")


_CONTRACTION_OPS = {
    "dot",
    "matmul",
    "inner",
    "vdot",
    "vecdot",
    "outer",
    "tensordot",
    "kron",
    "einsum",
}

_SORTING_OPS = {
    "sort",
    "argsort",
    "lexsort",
    "partition",
    "argpartition",
    "searchsorted",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "in1d",
    "isin",
    "intersect1d",
    "setdiff1d",
    "setxor1d",
    "union1d",
}

_WINDOW_OPS = {"bartlett", "blackman", "hamming", "hanning", "kaiser"}

_POLYNOMIAL_OPS = {
    "polyval",
    "polyfit",
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyder",
    "polyint",
    "poly",
    "roots",
}

# ---------------------------------------------------------------------------
# Weight tier classification
# ---------------------------------------------------------------------------


def _weight_tier(w: float) -> str:
    if w < 0.5:
        return "negligible"
    if w < 2.0:
        return "baseline"
    if w < 20.0:
        return "moderate"
    if w < 100.0:
        return "heavy"
    return "extreme"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "Operation",
    "Category",
    "Perf Weight",
    "Timing Weight",
    "Absolute Alpha",
    "Perf/Timing Ratio",
    "Weight Tier",
    "Zero FP",
    "Alpha Min",
    "Alpha Max",
    "Alpha CV",
    "Analytical Formula",
    "Analytical FLOPs",
    "Benchmark Code",
    "Benchmark Size",
    "Repeats",
    "Perf Instructions (total)",
    "Timing (ns, total)",
    "Time per Analytical FLOP (ns)",
    "Baseline Alpha",
    "Baseline Command",
    "Baseline Perf Instructions",
    "Baseline Timing (ns)",
    "Implementation URL",
    "Notes",
]


def _safe_div(a: float, b: float) -> str:
    """Divide a/b, returning 'N/A' if b is zero-ish."""
    if b is None or abs(b) < 1e-15:
        return "N/A"
    return f"{a / b:.4f}"


def _alpha_cv(alphas: list[float]) -> str:
    """Coefficient of variation of alpha values."""
    if len(alphas) < 2:
        return "N/A"
    m = statistics.mean(alphas)
    if abs(m) < 1e-15:
        return "N/A"
    return f"{statistics.stdev(alphas) / m:.4f}"


def _time_per_flop(timing_ns: float | None, analytical_flops: int | None,
                   repeats: int | None) -> str:
    """Compute timing_ns / (analytical_flops * repeats)."""
    if timing_ns is None or analytical_flops is None or repeats is None:
        return "N/A"
    denom = analytical_flops * repeats
    if denom == 0:
        return "N/A"
    return f"{timing_ns / denom:.4f}"


def build_rows(data: dict) -> list[dict]:
    """Build one row dict per operation from enriched weights.json."""
    weights = data["weights"]
    details = data["meta"]["per_op_details"]
    timing_weights = data["meta"]["validation"].get("timing_weights", {})

    rows = []
    for op_name, perf_weight in weights.items():
        d = details.get(op_name, {})
        tw = timing_weights.get(op_name, 0.0)
        category = d.get("category", "counted_custom")
        display_cat = _classify_op(op_name, category)
        alphas = d.get("distribution_alphas", [])
        analytical_flops = d.get("analytical_flops")
        repeats = d.get("repeats")
        timing_ns = d.get("timing_ns_total")

        row = {
            "Operation": op_name,
            "Category": display_cat,
            "Perf Weight": f"{perf_weight:.4f}",
            "Timing Weight": f"{tw:.4f}" if tw else "0.0000",
            "Absolute Alpha": f"{d.get('absolute_alpha', 0.0):.4f}",
            "Perf/Timing Ratio": _safe_div(perf_weight, tw),
            "Weight Tier": _weight_tier(perf_weight),
            "Zero FP": "TRUE" if perf_weight < 0.001 else "FALSE",
            "Alpha Min": f"{min(alphas):.4f}" if alphas else "N/A",
            "Alpha Max": f"{max(alphas):.4f}" if alphas else "N/A",
            "Alpha CV": _alpha_cv(alphas),
            "Analytical Formula": d.get("analytical_formula", ""),
            "Analytical FLOPs": str(analytical_flops) if analytical_flops is not None else "",
            "Benchmark Code": d.get("bench_code", ""),
            "Benchmark Size": d.get("benchmark_size", ""),
            "Repeats": str(repeats) if repeats is not None else "",
            "Perf Instructions (total)": str(d.get("perf_instructions_total", "")),
            "Timing (ns, total)": str(timing_ns) if timing_ns is not None else "",
            "Time per Analytical FLOP (ns)": _time_per_flop(timing_ns, analytical_flops, repeats),
            "Baseline Alpha": f"{d.get('baseline_alpha', 0.0):.4f}",
            "Baseline Command": d.get("baseline_bench_code", ""),
            "Baseline Perf Instructions": str(d.get("baseline_perf_instructions_total", "")),
            "Baseline Timing (ns)": str(d.get("baseline_timing_ns_total", "")),
            "Implementation URL": d.get("cost_impl_url", ""),
            "Notes": d.get("notes", ""),
        }
        rows.append(row)

    # Sort by category order, then by perf weight descending
    cat_order = {c: i for i, c in enumerate(DISPLAY_CATEGORIES)}
    rows.sort(key=lambda r: (
        cat_order.get(r["Category"], 999),
        -float(r["Perf Weight"]),
    ))
    return rows


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _impl_link(url: str) -> str:
    """Format an implementation URL as a short markdown link."""
    if not url:
        return ""
    # Extract file:line from URL
    # https://github.com/.../src/mechestim/_pointwise.py#L253
    parts = url.split("/blob/main/")
    if len(parts) < 2:
        return f"[link]({url})"
    file_part = parts[1]
    # src/mechestim/_pointwise.py#L253 -> _pointwise.py:253
    path_and_anchor = file_part.split("#L")
    filename = Path(path_and_anchor[0]).name
    if len(path_and_anchor) == 2:
        return f"[{filename}:{path_and_anchor[1]}]({url})"
    return f"[{filename}]({url})"


def _format_weight(w_str: str) -> str:
    """Format a weight value for display."""
    try:
        v = float(w_str)
        return f"{v:.4f}"
    except (ValueError, TypeError):
        return w_str


def generate_markdown(rows: list[dict], data: dict) -> str:
    """Generate the full markdown document."""
    meta = data["meta"]
    hw = meta["hardware"]
    sw = meta["software"]
    bc = meta["benchmark_config"]
    meth = meta["methodology"]
    val = meta["validation"]
    pv = val["perf_vs_timing"]

    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    # ------------------------------------------------------------------
    # Title + introduction
    # ------------------------------------------------------------------
    w("# Empirical FLOP Weights")
    w()
    w("## Introduction")
    w()
    w("Per-operation FLOP weights are multiplicative correction factors that bridge")
    w("the gap between mechestim's analytical cost formulas and the actual")
    w("floating-point instruction cost observed on hardware. When weights are")
    w("loaded, the effective cost of an operation becomes:")
    w()
    w("$$")
    w("\\text{cost}(\\text{op}) = \\texttt{analytical\\_formula}(\\text{shapes}) "
      "\\times \\text{weight}(\\text{op})")
    w("$$")
    w()
    w("A weight of 25.9 for `sin` means that each analytical FLOP of sine costs")
    w("approximately 26 times more in actual floating-point instructions than a")
    w("FLOP of addition.")
    w()

    # ------------------------------------------------------------------
    # Methodology
    # ------------------------------------------------------------------
    w("## Methodology")
    w()
    w("### The unified correction-factor formula")
    w()
    w("Every weight is computed from the same two-step formula:")
    w()
    w("$$")
    w("$\\alpha$(\\text{op}) = \\text{median}_{D} "
      "\\left[ \\frac{\\sum \\texttt{fp\\_arith\\_inst\\_retired.*} "
      "\\times \\texttt{simd\\_width}}"
      "{C(\\text{op}, \\text{params}) \\times R} \\right]")
    w("$$")
    w()
    w("$$")
    w("\\text{weight}(\\text{op}) = "
      "\\frac{$\\alpha$(\\text{op})}{$\\alpha$(\\text{add})}")
    w("$$")
    w()
    w("Where:")
    w()
    w("- $\\alpha$(op) is the **raw correction factor** -- the ratio of "
      "hardware-observed FP instructions to the analytical FLOP count.")
    w("- `fp_arith_inst_retired.*` are Intel PMU hardware counters "
      "that count retired floating-point arithmetic instructions, "
      "weighted by SIMD lane count.")
    w("- $C$(op, params) is the analytical FLOP count from mechestim's "
      "cost formula (e.g., `numel(output)` for pointwise ops).")
    w("- $R$ is the number of repeats per distribution.")
    w("- The **median** across 3 input distributions is reported.")
    w()

    # ------------------------------------------------------------------
    # Measurement environment
    # ------------------------------------------------------------------
    w("## Measurement environment")
    w()
    w('!!! info "Calibration platform"')
    w()
    cache = hw.get("cache", {})
    l3_mb = cache.get("L3", 0) / 1024 if cache.get("L3", 0) > 1024 else cache.get("L3", 0)
    w(f"    - **CPU:** {hw['cpu_model']}")
    w(f"    - **Cores:** {hw['cpu_cores']} physical / {hw['cpu_threads']} threads")
    w(f"    - **RAM:** {hw['ram_gb']} GB")
    w(f"    - **Arch:** {hw['arch']} (AVX-512 capable)")
    w(f"    - **Cache:** L1d {cache.get('L1d', '?')} KB, "
      f"L1i {cache.get('L1i', '?')} KB, "
      f"L2 {cache.get('L2', '?')} KB, "
      f"L3 {l3_mb:.0f} MB")
    w("    - **Instance:** AWS EC2 c6i.metal (bare metal -- full PMU access)")
    w(f"    - **OS:** {sw['os']}")
    w(f"    - **Python:** {sw['python'].split('(')[0].strip()}")
    w(f"    - **NumPy:** {sw['numpy']}")
    w(f"    - **BLAS:** {sw['blas']}")
    w(f"    - **Measurement mode:** {bc['measurement_mode']} "
      f"(hardware counters: `fp_arith_inst_retired.*`)")
    w(f"    - **dtype:** {bc['dtype']}")
    w(f"    - **Repeats:** {bc['repeats']} per distribution")
    w(f"    - **Distributions:** {bc['distributions']} per operation")
    w(f"    - **Methodology version:** {meth['version']}")
    w(f"    - **Baseline $\\alpha$(add):** {meth['baseline_alpha']}")
    ts = meta.get("timestamp", "")
    if ts:
        w(f"    - **Date:** {ts[:10]}")
    dur = meta.get("duration_seconds")
    if dur:
        w(f"    - **Total calibration time:** {dur} seconds")
    w()

    # ------------------------------------------------------------------
    # Baseline details
    # ------------------------------------------------------------------
    w("## Baseline details")
    w()
    w("All weights are normalized against element-wise addition (`np.add`):")
    w()

    # Find the first op that has baseline details
    details = meta.get("per_op_details", {})
    sample_detail = None
    for op_d in details.values():
        if op_d.get("baseline_bench_code"):
            sample_detail = op_d
            break

    if sample_detail:
        w(f"- **Benchmark command:** `{sample_detail.get('baseline_bench_code', 'np.add(x, y, out=_out)')}`")
        w(f"- **Array size:** {sample_detail.get('benchmark_size', 'n=10_000_000')}, dtype=float64")
        w(f"- **Measured perf instructions:** {sample_detail.get('baseline_perf_instructions_total', 'N/A')}")
        w(f"- **Measured timing:** {sample_detail.get('baseline_timing_ns_total', 'N/A')} ns")
        w(f"- **$\\alpha$(add):** {meth['baseline_alpha']}")
    else:
        w(f"- **Benchmark command:** `np.add(x, y, out=_out)`")
        w(f"- **$\\alpha$(add):** {meth['baseline_alpha']}")
    w()

    # ------------------------------------------------------------------
    # Download link
    # ------------------------------------------------------------------
    w("**[Download full review spreadsheet (CSV)](empirical-weights.csv)**")
    w()

    # ------------------------------------------------------------------
    # Per-category tables
    # ------------------------------------------------------------------
    w("## Weight tables")
    w()

    # Group rows by category
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["Category"]].append(r)

    for cat in DISPLAY_CATEGORIES:
        cat_rows = by_cat.get(cat, [])
        if not cat_rows:
            continue
        w(f"### {cat} ({len(cat_rows)} operations)")
        w()
        w("| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |")
        w("|:---|--------:|----------:|---------------:|:--------|:-----|:------|")
        for r in cat_rows:
            op = f"`{r['Operation']}`"
            pw = _format_weight(r["Perf Weight"])
            tw = _format_weight(r["Timing Weight"])
            tpf = r["Time per Analytical FLOP (ns)"]
            formula = r["Analytical Formula"]
            impl = _impl_link(r["Implementation URL"])
            notes = r["Notes"]
            w(f"| {op} | {pw} | {tw} | {tpf} | {formula} | {impl} | {notes} |")
        w()

    # ------------------------------------------------------------------
    # Category summary
    # ------------------------------------------------------------------
    w("## Summary by category")
    w()
    w("| Category | Count | Avg Weight | Min | Max |")
    w("|:---------|------:|-----------:|----:|----:|")
    for cat in DISPLAY_CATEGORIES:
        cat_rows = by_cat.get(cat, [])
        if not cat_rows:
            continue
        wts = [float(r["Perf Weight"]) for r in cat_rows]
        avg = statistics.mean(wts)
        mn = min(wts)
        mx = max(wts)
        w(f"| {cat} | {len(cat_rows)} | {avg:.4f} | {mn:.4f} | {mx:.4f} |")
    w()
    total = sum(len(by_cat.get(c, [])) for c in DISPLAY_CATEGORIES)
    w(f"**Total benchmarked operations:** {total}")
    w()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    w("## Validation")
    w()
    w("Every operation is measured in both **perf mode** (hardware counters) and")
    w("**timing mode** (wall-clock nanoseconds).")
    w()
    w("### Correlation statistics")
    w()
    w("| Metric | Value | Interpretation |")
    w("|--------|------:|:---------------|")
    w(f"| Pearson $r$ | {pv['pearson_r']:.4f} | "
      f"Linear correlation between perf and timing weight vectors. |")
    w(f"| Spearman $\\rho$ | {pv['spearman_rho']:.4f} | "
      f"Rank correlation -- are the orderings consistent? |")
    w()

    max_div = pv.get("max_divergence", {})
    if max_div:
        w("### Maximum divergence")
        w()
        w("| Field | Value |")
        w("|:------|:------|")
        w(f"| Operation | `{max_div.get('op', 'N/A')}` |")
        w(f"| Perf weight | {max_div.get('perf_weight', 'N/A')} |")
        w(f"| Timing weight | {max_div.get('timing_weight', 'N/A')} |")
        w(f"| Ratio | {max_div.get('ratio', 'N/A'):.1f} |")
        w()

    w("### Interpreting divergence")
    w()
    w("The moderate correlation values and large max divergence for BLAS operations are")
    w("**expected**. Perf mode counts FP instructions regardless of execution time,")
    w("while timing mode measures wall-clock time including memory bandwidth and cache")
    w("effects. BLAS operations achieve near-peak FLOP throughput, so their per-instruction")
    w("timing is much lower than for scalar pointwise operations. For pointwise ops")
    w("(which dominate the count), the two modes agree well in relative ordering.")
    w()
    w('!!! warning "Correlation caveats"')
    w()
    w("    The Pearson and Spearman values span all operations, including BLAS/linalg")
    w("    ops where timing and perf divergence is structurally expected. For the")
    w("    subset of pointwise operations, both correlations are substantially higher.")
    w()

    # ------------------------------------------------------------------
    # Known limitations
    # ------------------------------------------------------------------
    w("## Known limitations")
    w()
    w("### Trace anomaly (subprocess overhead)")
    w()
    w("The `trace` operation shows an anomalously high weight because its analytical")
    w("formula is $n$ (the matrix dimension), which is small, while the subprocess")
    w("measurement captures fixed per-process overhead that dominates at small input")
    w("sizes. The weight for `trace` should be interpreted with caution.")
    w()
    w("### BLAS vectorization effects")
    w()
    w("Operations backed by optimized BLAS routines (`matmul`, `dot`, contraction ops)")
    w("show weights below 1.0 because FMA instructions fuse two analytical FLOPs into")
    w("one hardware instruction. The sub-unity weights are correct -- they reflect")
    w("real hardware instruction counts.")
    w()
    w("### Random number generators")
    w()
    w("RNG weights vary dramatically (0.0001 to 367) because the analytical formula")
    w("(`numel(output)`) captures only the output size, not the internal algorithmic")
    w("complexity. Complex distributions like `hypergeometric` involve rejection")
    w("sampling loops that execute many FP instructions per output element.")
    w()

    # ------------------------------------------------------------------
    # Related pages
    # ------------------------------------------------------------------
    w("## Related pages")
    w()
    w("- [How to calibrate weights](../how-to/calibrate-weights.md)")
    w("- [FLOP counting model](../concepts/flop-counting-model.md)")
    w("- [Operation audit](operation-audit.md)")
    w("- [Agent cheat sheet](for-agents.md)")
    w()

    return "\n".join(lines)


def write_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote markdown to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate enriched empirical-weights docs from weights.json."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_PATH,
        help="Path to weights.json (default: %(default)s)",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=CSV_OUT,
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=MD_OUT,
        help="Output markdown path (default: %(default)s)",
    )
    args = parser.parse_args()

    data = load_data(args.weights)

    # Check for per_op_details
    if "per_op_details" not in data.get("meta", {}):
        print(
            "ERROR: weights.json does not contain meta.per_op_details.\n"
            "Run the enriched benchmark suite first (Phase 1 + 2) to produce\n"
            "a weights.json with per-op raw measurement details.\n"
            f"File: {args.weights}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = build_rows(data)
    write_csv(rows, args.csv_out)
    md_content = generate_markdown(rows, data)
    write_markdown(md_content, args.md_out)

    print(f"\nDone. {len(rows)} operations processed.")
    print(f"  CSV: {args.csv_out}")
    print(f"  MD:  {args.md_out}")


if __name__ == "__main__":
    main()
