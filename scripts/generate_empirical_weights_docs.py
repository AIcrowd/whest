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
CSV_OUT = REPO_ROOT / "src" / "mechestim" / "data" / "weights.csv"
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

# ---------------------------------------------------------------------------
# Section A: Review columns — what the reviewer reads and tunes
# Section B: Evidence columns — raw measurements for anyone who needs to dig
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    # --- Section A: Review ---
    "Operation",
    "Status",
    "Category",
    "Cost Formula",
    "Weight",
    "Reviewer Weight",
    "Effective Cost Example",
    "vs Add",
    "Confidence",
    "Notes",
    # --- Section B: Evidence ---
    "Exclusion Reason",
    "Hardware FP Instructions (per analytical FLOP)",
    "Timing Weight",
    "Perf/Timing Agreement",
    "Measurement Spread (CV)",
    "Benchmark Command",
    "Benchmark Size",
    "Total Perf Instructions",
    "Total Timing (ns)",
    "Implementation",
    "Weight Tier",
    "Repeats",
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


def _empty_row(op_name: str, status: str, reason: str,
               category: str, notes: str) -> dict:
    """Build a row with only status/reason fields populated."""
    row = {col: "" for col in CSV_COLUMNS}
    row["Operation"] = op_name
    row["Status"] = status
    row["Exclusion Reason"] = reason
    row["Category"] = category
    row["Notes"] = notes
    row["Reviewer Weight"] = ""
    return row


def _confidence(alphas: list[float]) -> str:
    """Classify measurement confidence from distribution spread."""
    if len(alphas) < 2:
        return ""
    m = statistics.mean(alphas)
    if abs(m) < 1e-15:
        return "low"
    cv = statistics.stdev(alphas) / m
    if cv < 0.05:
        return "high"
    if cv < 0.20:
        return "medium"
    return "low"


def _effective_cost_example(formula: str, weight: float,
                            op_name: str = "") -> str:
    """Compute a concrete cost example for the reviewer.

    Shows: ``weight * C(example_input)`` = effective weighted FLOPs.
    Uses a small, concrete input size so the reviewer can sanity-check.
    """
    if weight == 0:
        return ""
    lower = formula.lower() if formula else ""

    # Per-element formulas (pointwise, reductions, random, etc.)
    if "numel" in lower or formula in ("n", ""):
        effective = 1000 * weight
        desc = "1000 elements"
    # Matrix multiply
    elif "m*n*k" in lower:
        c = 2 * 32 * 32 * 32  # 32x32 @ 32x32
        effective = c * weight
        desc = f"32x32 matmul, C={c}"
    # Sort / search with log factor
    elif "log" in lower and "n" in lower and "ceil" in lower:
        c = 1000 * math.ceil(math.log2(1000))
        effective = c * weight
        desc = f"1000 elements, C={c}"
    # Polynomial with degree
    elif "degree" in lower or "deg" in lower:
        c = 2 * 100 * 10
        effective = c * weight
        desc = f"n=100 deg=10, C={c}"
    # Linalg: n^3 formulas
    elif "n^3" in lower or "n^2" in lower:
        # Use n=32 as example (small matrix)
        if "n^3 / 3" in lower:
            c = 32 ** 3 // 3
        elif "n^3" in lower:
            c = 32 ** 3
        else:
            c = 32 ** 2
        effective = c * weight
        desc = f"n=32, C={c}"
    # Inner/vdot: 2*N
    elif formula.startswith("2*N") or formula == "2*N":
        c = 2 * 1000
        effective = c * weight
        desc = f"1000-element dot, C={c}"
    # Cross, trace, simple n formulas
    elif formula.startswith(("6 *", "min(")):
        c = 1000
        effective = c * weight
        desc = f"n=1000, C={c}"
    # Convolution: n * k
    elif "n * k" in lower:
        c = 1000 * 100  # n=1000, k=100
        effective = c * weight
        desc = f"n=1000 k=100, C={c}"
    # Statistical: 2 * f^2 * s
    elif "f^2" in lower:
        c = 2 * 10 * 10 * 100  # f=10, s=100
        effective = c * weight
        desc = f"10 features x 100 samples, C={c}"
    # Fallback: just show weight * 1000
    else:
        c = 1000
        effective = c * weight
        desc = f"C={c}"

    effective = c * weight
    return f"{effective:,.0f} FLOPs ({desc})"


# Alias map: excluded op -> canonical benchmarked op whose weight it inherits
_ALIAS_MAP = {
    "acos": "arccos", "acosh": "arccosh", "asin": "arcsin",
    "asinh": "arcsinh", "atan": "arctan", "atan2": "arctan2",
    "atanh": "arctanh", "pow": "power", "absolute": "abs",
    "amax": "max", "amin": "min", "around": "rint", "fix": "trunc",
    "round": "rint", "nanargmax": "argmax", "nanargmin": "argmin",
    "nancumprod": "cumprod", "nancumsum": "cumsum",
    "cumulative_prod": "cumprod", "cumulative_sum": "cumsum",
    "ptp": "max", "divmod": "floor_divide", "trapz": "trapezoid",
}

# Exclusion sets with reasons
_EXCLUSIONS = {
    "bitwise": (
        frozenset({"bitwise_and", "bitwise_count", "bitwise_invert",
                    "bitwise_left_shift", "bitwise_not", "bitwise_or",
                    "bitwise_right_shift", "bitwise_xor", "invert",
                    "left_shift", "right_shift", "gcd", "lcm"}),
        "Integer/bitwise operation — no floating-point instructions retired",
    ),
    "complex": (
        frozenset({"angle", "conj", "conjugate", "imag", "real",
                    "real_if_close", "iscomplex", "iscomplexobj",
                    "isreal", "isrealobj", "sort_complex"}),
        "Complex-number operation — weight depends on dtype (real vs complex)",
    ),
    "linalg_delegate": (
        frozenset({"linalg.cond", "linalg.cross", "linalg.matmul",
                    "linalg.matrix_norm", "linalg.matrix_power",
                    "linalg.matrix_rank", "linalg.multi_dot", "linalg.norm",
                    "linalg.outer", "linalg.tensordot", "linalg.tensorinv",
                    "linalg.tensorsolve", "linalg.trace", "linalg.vecdot",
                    "linalg.vector_norm"}),
        "Delegates to a primary op (e.g., linalg.matmul -> matmul)",
    ),
    "random_alias": (
        frozenset({"random.bytes", "random.random_integers",
                    "random.ranf", "random.sample"}),
        "Alias or removed in NumPy 2.x",
    ),
    "no_fp_work": (
        frozenset({"einsum_path"}),
        "Planning operation — no floating-point computation",
    ),
    "version_dependent": (
        frozenset({"isnat"}),
        "datetime64-only operation — no floating-point instructions",
    ),
}


def build_rows(data: dict) -> list[dict]:
    """Build one row dict per operation from enriched weights.json.

    Includes benchmarked ops, aliases, and excluded ops with reasons.
    """
    from mechestim._registry import REGISTRY

    weights = data["weights"]
    details = data["meta"]["per_op_details"]
    timing_weights = data["meta"]["validation"].get("timing_weights", {})
    abs_alphas = data["meta"]["validation"].get("absolute_correction_factors", {})
    add_weight = weights.get("add", 1.0)

    rows = []

    # --- Benchmarked ops ---
    for op_name, perf_weight in weights.items():
        d = details.get(op_name, {})
        tw = timing_weights.get(op_name, 0.0)
        category = d.get("category", "counted_custom")
        display_cat = _classify_op(op_name, category)
        alphas = d.get("distribution_alphas", [])
        analytical_flops = d.get("analytical_flops")
        repeats = d.get("repeats")
        timing_ns = d.get("timing_ns_total")
        absolute_alpha = abs_alphas.get(op_name, d.get("absolute_alpha", 0.0))
        vs_add = perf_weight / add_weight if add_weight else 0.0
        formula = d.get("analytical_formula", "")

        row = {
            # Section A: Review
            "Operation": op_name,
            "Status": "benchmarked",
            "Category": display_cat,
            "Cost Formula": formula,
            "Weight": f"{perf_weight:.4f}",
            "Reviewer Weight": "",
            "Effective Cost Example": _effective_cost_example(formula, perf_weight, analytical_flops),
            "vs Add": f"{vs_add:.1f}x",
            "Confidence": _confidence(alphas),
            "Notes": d.get("notes", ""),
            # Section B: Evidence
            "Exclusion Reason": "",
            "Hardware FP Instructions (per analytical FLOP)": f"{absolute_alpha:.2f}",
            "Timing Weight": f"{tw:.4f}" if tw else "0.0000",
            "Perf/Timing Agreement": _safe_div(perf_weight, tw),
            "Measurement Spread (CV)": _alpha_cv(alphas),
            "Benchmark Command": d.get("bench_code", ""),
            "Benchmark Size": d.get("benchmark_size", ""),
            "Total Perf Instructions": str(d.get("perf_instructions_total", "")),
            "Total Timing (ns)": str(timing_ns) if timing_ns is not None else "",
            "Implementation": d.get("cost_impl_url", ""),
            "Weight Tier": _weight_tier(perf_weight),
            "Repeats": str(repeats) if repeats is not None else "",
        }
        rows.append(row)

    # --- Alias ops (inherit weight from canonical op) ---
    for alias, canonical in sorted(_ALIAS_MAP.items()):
        if alias in weights:
            continue
        entry = REGISTRY.get(alias, {})
        canon_weight = weights.get(canonical, 0.0)
        canon_alpha = abs_alphas.get(canonical, 0.0)
        vs_add = canon_weight / add_weight if add_weight and canon_weight else 0.0
        reason = f"Alias of {canonical}"
        if canon_weight:
            reason += f" (weight: {canon_weight:.2f})"
        row = _empty_row(alias, "alias", reason, entry.get("category", ""), entry.get("notes", ""))
        row["Weight"] = f"{canon_weight:.4f}" if canon_weight else ""
        row["vs Add"] = f"{vs_add:.1f}x" if vs_add else ""
        row["Hardware FP Instructions (per analytical FLOP)"] = f"{canon_alpha:.2f}" if canon_alpha else ""
        rows.append(row)

    # --- Excluded ops ---
    for group_name, (op_set, reason) in _EXCLUSIONS.items():
        for op_name in sorted(op_set):
            if op_name in weights:
                continue
            entry = REGISTRY.get(op_name, {})
            rows.append(_empty_row(op_name, "excluded", reason, entry.get("category", ""), entry.get("notes", "")))

    # --- Free ops (0 FLOPs) ---
    for name, entry in sorted(REGISTRY.items()):
        if entry["category"] == "free" and name not in weights:
            row = _empty_row(name, "free", "Zero FLOP cost — not benchmarked", "free", entry.get("notes", ""))
            row["Weight"] = "0"
            row["Cost Formula"] = "0"
            rows.append(row)

    # --- Blacklisted ops ---
    for name, entry in sorted(REGISTRY.items()):
        if entry["category"] == "blacklisted" and name not in weights:
            rows.append(_empty_row(name, "blacklisted", "Intentionally unsupported in mechestim", "blacklisted", entry.get("notes", "")))

    # Sort: benchmarked first (by category then weight), then alias, excluded, free, blacklisted
    status_order = {"benchmarked": 0, "alias": 1, "excluded": 2, "free": 3, "blacklisted": 4}
    cat_order = {c: i for i, c in enumerate(DISPLAY_CATEGORIES)}
    rows.sort(key=lambda r: (
        status_order.get(r["Status"], 99),
        cat_order.get(r["Category"], 999),
        -float(r["Weight"]) if r["Weight"] else 0,
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
    """Format an implementation URL as a short markdown link.

    Uses backtick code spans for the link text to prevent markdown from
    interpreting underscores in filenames (e.g. __init__.py) as emphasis.
    """
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
    # Escape underscores in filename to prevent markdown emphasis
    safe_name = filename.replace("_", r"\_")
    if len(path_and_anchor) == 2:
        return f"[{safe_name}:{path_and_anchor[1]}]({url})"
    return f"[{safe_name}]({url})"


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
    w("# FLOP Weight Calibration Results")
    w()
    w("## Introduction")
    w()
    w("Per-operation FLOP weights are multiplicative correction factors that bridge")
    w("the gap between mechestim's analytical cost formulas and the actual")
    w("floating-point instruction cost observed on hardware. When weights are")
    w("loaded, the effective cost of an operation becomes:")
    w()
    w("$$")
    w("\\text{cost}(\\text{op}) = C(\\text{op}, \\text{shapes}) "
      "\\times w(\\text{op})")
    w("$$")
    w()
    w("where $C$ is the analytical FLOP formula and $w$ is the weight.")
    w("A weight of 25.9 for `sin` means that each analytical FLOP of sine costs")
    w("approximately 26 times more in actual floating-point instructions than a")
    w("FLOP of addition.")
    w()

    # ------------------------------------------------------------------
    # Methodology
    # ------------------------------------------------------------------
    w("## Methodology")
    w()
    w("### The correction formula")
    w()
    w("Every weight is computed from the same two-step formula:")
    w()
    w("$$")
    w("\\alpha(\\text{op}) = \\mathrm{median}_{D} "
      "\\left[ \\frac{F(\\text{op})}"
      "{C(\\text{op}, \\text{params}) \\times R} \\right]")
    w("$$")
    w()
    w("$$")
    w("w(\\text{op}) = "
      "\\frac{\\alpha(\\text{op})}{\\alpha(\\text{add})}")
    w("$$")
    w()
    w("Where:")
    w()
    w("- $\\alpha(\\text{op})$ is the **raw correction factor** -- the ratio of "
      "hardware-observed FP instructions to the analytical FLOP count.")
    w("- $F(\\text{op})$ is the total SIMD-width-weighted count of retired "
      "floating-point instructions, measured via the Intel PMU counters "
      "`fp_arith_inst_retired.*` (scalar x1, 128-bit x2, 256-bit x4, 512-bit x8).")
    w("- $C(\\text{op}, \\text{params})$ is the analytical FLOP count from mechestim's "
      "cost formula (e.g., `numel(output)` for pointwise ops).")
    w("- $R$ is the number of repeats per distribution.")
    w("- The **median** across 3 input distributions is reported.")
    w()

    # ------------------------------------------------------------------
    # Measurement environment
    # ------------------------------------------------------------------
    w("## Measurement environment")
    w()
    cache = hw.get("cache", {})
    l3_mb = cache.get("L3", 0) / 1024 if cache.get("L3", 0) > 1024 else cache.get("L3", 0)
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w(f"| CPU | {hw['cpu_model']} |")
    w(f"| Cores | {hw['cpu_cores']} physical / {hw['cpu_threads']} threads |")
    w(f"| RAM | {hw['ram_gb']} GB |")
    w(f"| Arch | {hw['arch']} (AVX-512 capable) |")
    w(f"| Cache | L1d {cache.get('L1d', '?')} KB, "
      f"L1i {cache.get('L1i', '?')} KB, "
      f"L2 {cache.get('L2', '?')} KB, "
      f"L3 {l3_mb:.0f} MB |")
    w("| Instance | AWS EC2 c6i.metal (bare metal, full PMU access) |")
    w(f"| OS | {sw['os']} |")
    w(f"| Python | {sw['python'].split('(')[0].strip()} |")
    w(f"| NumPy | {sw['numpy']} |")
    w(f"| BLAS | {sw['blas']} |")
    w(f"| Measurement mode | {bc['measurement_mode']} "
      f"(hardware counters: `fp_arith_inst_retired.*`) |")
    w(f"| dtype | {bc['dtype']} |")
    w(f"| Repeats | {bc['repeats']} per distribution |")
    w(f"| Distributions | {bc['distributions']} per operation |")
    w(f"| Methodology version | {meth['version']} |")
    w(f"| Baseline alpha(add) | {meth['baseline_alpha']} |")
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
        w(f"- **$\\alpha(\\text{{add}})$:** {meth['baseline_alpha']}")
    else:
        w(f"- **Benchmark command:** `np.add(x, y, out=_out)`")
        w(f"- **$\\alpha(\\text{{add}})$:** {meth['baseline_alpha']}")
    w()

    # ------------------------------------------------------------------
    # Download link
    # ------------------------------------------------------------------
    w("**[Download full review spreadsheet (CSV)](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/data/weights.csv)**")
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
        w("| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |")
        w("|:---|-------:|-------:|:-----------|:--------|:-----|:------|")
        for r in cat_rows:
            op = f"`{r['Operation']}`"
            wt = _format_weight(r["Weight"])
            vs = r["vs Add"]
            conf = r["Confidence"]
            formula = r["Cost Formula"]
            impl = _impl_link(r["Implementation"])
            notes = r["Notes"]
            w(f"| {op} | {wt} | {vs} | {conf} | {formula} | {impl} | {notes} |")
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
        wts = [float(r["Weight"]) for r in cat_rows
               if r["Weight"] and r["Weight"] != "N/A"]
        if not wts:
            continue
        avg = statistics.mean(wts)
        mn = min(wts)
        mx = max(wts)
        w(f"| {cat} | {len(cat_rows)} | {avg:.2f} | {mn:.4f} | {mx:.4f} |")
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
    w("**Correlation caveats:**")
    w("The Pearson and Spearman values span all operations, including BLAS/linalg")
    w("ops where timing and perf divergence is structurally expected. For the")
    w("subset of pointwise operations, both correlations are substantially higher.")
    w()

    # ------------------------------------------------------------------
    # Known limitations
    # ------------------------------------------------------------------
    w("## Known limitations")
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
