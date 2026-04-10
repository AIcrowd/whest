"""Benchmark runner — CLI orchestrator for mechestim benchmarks.

Methodology v2.0
-----------------
Every benchmark category returns ``alpha(op)`` values: the ratio of measured
hardware cost (FP instructions or elapsed time) to the analytical FLOP count.
The runner normalises all alphas by ``alpha(add)`` (the baseline) to produce
the final weights::

    weight(op) = alpha(op) / alpha(add)

This unification means *all* categories — including linalg and FFT — go
through the same normalisation path.  Raw alpha values are preserved in
``meta.validation.absolute_correction_factors`` for scientific analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from typing import Any

from benchmarks._baseline import measure_baseline
from benchmarks._fft import benchmark_fft
from benchmarks._linalg import benchmark_linalg
from benchmarks._metadata import collect_metadata
from benchmarks._pointwise import benchmark_pointwise
from benchmarks._polynomial import benchmark_polynomial
from benchmarks._random import benchmark_random
from benchmarks._reductions import benchmark_reductions
from benchmarks._sorting import benchmark_sorting
from benchmarks.dashboard import render_html, render_terminal

# -- optional new-category imports (modules may not exist yet) -------------
try:
    from benchmarks._contractions import benchmark_contractions
except ImportError:
    benchmark_contractions = None  # type: ignore[assignment]

try:
    from benchmarks._misc import benchmark_misc
except ImportError:
    benchmark_misc = None  # type: ignore[assignment]

try:
    from benchmarks._window import benchmark_window
except ImportError:
    benchmark_window = None  # type: ignore[assignment]


ALL_CATEGORIES = sorted(
    {
        "pointwise",
        "reductions",
        "sorting",
        "random",
        "polynomial",
        "linalg",
        "fft",
        "contractions",
        "misc",
        "window",
    }
)

_BENCHMARK_FUNCS: dict[str, Any] = {
    "pointwise": benchmark_pointwise,
    "reductions": benchmark_reductions,
    "linalg": benchmark_linalg,
    "fft": benchmark_fft,
    "sorting": benchmark_sorting,
    "random": benchmark_random,
    "polynomial": benchmark_polynomial,
}
# Register new categories only if their modules are available.
if benchmark_contractions is not None:
    _BENCHMARK_FUNCS["contractions"] = benchmark_contractions
if benchmark_misc is not None:
    _BENCHMARK_FUNCS["misc"] = benchmark_misc
if benchmark_window is not None:
    _BENCHMARK_FUNCS["window"] = benchmark_window

# Approximate op counts per category (for progress bar total)
_APPROX_OP_COUNTS: dict[str, int] = {
    "pointwise": 70,
    "reductions": 28,
    "linalg": 14,
    "fft": 14,
    "sorting": 7,
    "random": 10,
    "polynomial": 10,
    "contractions": 10,
    "misc": 25,
    "window": 5,
}


def normalize_weights(
    raw_alpha: dict[str, float], alpha_add: float
) -> dict[str, float]:
    """Divide each raw alpha(op) value by *alpha_add*.

    If *alpha_add* is zero (or very close), fall back to 1.0 so the raw
    values pass through unchanged.

    Parameters
    ----------
    raw_alpha:
        Mapping of op names to raw alpha values (measured / analytical).
    alpha_add:
        The baseline alpha(add) value used as normalisation constant.
    """
    if alpha_add == 0:
        alpha_add = 1.0
    return {k: v / alpha_add for k, v in raw_alpha.items()}


def _compute_validation_stats(
    perf_weights: dict[str, float],
    timing_weights: dict[str, float],
) -> dict[str, Any]:
    """Compute Pearson r, Spearman rho, and max divergence between two weight sets.

    Returns a dict suitable for ``meta.validation.perf_vs_timing``.
    If scipy is unavailable or there are fewer than 3 common ops, returns
    an empty dict with a ``note`` explaining why.
    """
    common_ops = sorted(set(perf_weights) & set(timing_weights))
    if len(common_ops) < 3:
        return {"note": f"too few common ops ({len(common_ops)}) for validation"}

    perf_vals = [perf_weights[op] for op in common_ops]
    timing_vals = [timing_weights[op] for op in common_ops]

    stats: dict[str, Any] = {}

    # Pearson & Spearman (optional scipy dependency)
    try:
        from scipy.stats import pearsonr, spearmanr

        r, _ = pearsonr(perf_vals, timing_vals)
        rho, _ = spearmanr(perf_vals, timing_vals)
        stats["pearson_r"] = round(r, 6)
        stats["spearman_rho"] = round(rho, 6)
    except ImportError:
        stats["note"] = "scipy not available — correlation stats skipped"

    # Max divergence
    max_div_op = ""
    max_div_ratio = 0.0
    for op in common_ops:
        pw = perf_weights[op]
        tw = timing_weights[op]
        if tw > 0:
            ratio = abs(pw / tw - 1.0)
        elif pw > 0:
            ratio = float("inf")
        else:
            ratio = 0.0
        if ratio > max_div_ratio:
            max_div_ratio = ratio
            max_div_op = op

    if max_div_op:
        stats["max_divergence"] = {
            "op": max_div_op,
            "perf_weight": round(perf_weights[max_div_op], 4),
            "timing_weight": round(timing_weights[max_div_op], 4),
            "ratio": round(
                perf_weights[max_div_op] / timing_weights[max_div_op]
                if timing_weights[max_div_op]
                else float("inf"),
                4,
            ),
        }

    return stats


def _unpack_benchmark_result(
    result: dict[str, float] | tuple[dict[str, float], dict[str, dict]],
) -> tuple[dict[str, float], dict[str, dict]]:
    """Unpack a benchmark function return value.

    Handles both the legacy ``dict`` return and the new ``(alphas, details)``
    tuple return for backward compatibility.
    """
    if isinstance(result, tuple):
        return result
    return result, {}


def _run_category_loop(
    cats: list[str],
    dtype: str,
    repeats: int,
    log_fn: Any,
    use_rich: bool = False,
    console: Any = None,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Run all benchmark categories and collect raw alpha(op) values.

    Returns a tuple of ``(alphas, details)`` where *alphas* maps op names to
    raw alpha values and *details* maps op names to per-op benchmark metadata.
    """
    alphas: dict[str, float] = {}
    all_details: dict[str, dict] = {}

    if use_rich:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        total_ops = sum(_APPROX_OP_COUNTS.get(c, 10) for c in cats)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[current_op]}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        with progress:
            task_id = progress.add_task(
                "Benchmarking", total=total_ops, current_op="starting..."
            )
            for cat in cats:
                if cat not in _BENCHMARK_FUNCS:
                    continue
                progress.update(task_id, current_op=f"benchmarking {cat}")
                func = _BENCHMARK_FUNCS[cat]
                raw_alphas, raw_details = _unpack_benchmark_result(
                    func(dtype=dtype, repeats=repeats)
                )
                alphas.update(raw_alphas)
                all_details.update(raw_details)
                progress.advance(
                    task_id, advance=_APPROX_OP_COUNTS.get(cat, len(raw_alphas))
                )
            progress.update(task_id, current_op="done")
    else:
        for cat in cats:
            if cat not in _BENCHMARK_FUNCS:
                log_fn(f"Unknown category: {cat!r}, skipping.")
                continue
            log_fn(f"Benchmarking {cat} ...")
            func = _BENCHMARK_FUNCS[cat]
            raw_alphas, raw_details = _unpack_benchmark_result(
                func(dtype=dtype, repeats=repeats)
            )
            alphas.update(raw_alphas)
            all_details.update(raw_details)

    return alphas, all_details


def _enrich_details(
    all_details: dict[str, dict],
    *,
    weights: dict[str, float],
    alpha_add: float,
    baseline_n: int,
    baseline_bench_code: str,
    repeats: int,
    timing_baseline: float,
) -> None:
    """Enrich per-op detail dicts with computed fields, URLs, and notes.

    Mutates *all_details* in place.
    """
    # -- registry notes ----------------------------------------------------
    try:
        from mechestim._registry import REGISTRY
    except ImportError:
        REGISTRY = {}  # type: ignore[assignment]

    # -- implementation URLs -----------------------------------------------
    try:
        from benchmarks._impl_urls import build_url_map

        url_map = build_url_map(list(all_details.keys()))
    except Exception:
        url_map = {}

    for op, detail in all_details.items():
        # weight-derived fields
        detail["perf_weight"] = weights.get(op, 0.0)
        detail["absolute_alpha"] = round(weights.get(op, 0.0) * alpha_add, 6)

        # baseline reference fields
        detail["baseline_alpha"] = round(alpha_add, 6)
        detail["baseline_analytical_flops"] = baseline_n
        detail["baseline_bench_code"] = baseline_bench_code
        detail["baseline_perf_instructions_total"] = round(
            alpha_add * baseline_n * repeats, 2
        )

        # baseline timing (from validation loop, if available)
        if timing_baseline > 0:
            detail["baseline_timing_ns_total"] = round(
                timing_baseline * baseline_n * repeats, 2
            )

        # registry notes
        detail["notes"] = REGISTRY.get(op, {}).get("notes", "")

        # implementation URL
        url = url_map.get(op, "")
        if url:
            detail["cost_impl_url"] = url


def run_benchmarks(
    dtype: str = "float64",
    output: str | None = None,
    html: str | None = None,
    categories: list[str] | None = None,
    repeats: int = 10,
) -> dict[str, Any]:
    """Run the full benchmark suite and return the results dict."""

    if categories is None or "all" in categories:
        cats = list(ALL_CATEGORIES)
    else:
        cats = list(categories)

    # -- measurement mode --------------------------------------------------
    from benchmarks._perf import measurement_mode

    mode = measurement_mode()

    # -- try rich ----------------------------------------------------------
    try:
        from rich.console import Console

        console = Console(stderr=True)
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    def _log(msg: str) -> None:
        if use_rich:
            console.print(msg)  # type: ignore[union-attr]
        else:
            print(msg, file=sys.stderr)

    _log(
        f"[bold]Measurement mode:[/bold] {mode}"
        if use_rich
        else f"Measurement mode: {mode}"
    )
    if mode == "timing":
        _log(
            "  [dim](perf not available — using wall-clock time as proxy)[/dim]"
            if use_rich
            else "  (perf not available — using wall-clock time as proxy)"
        )

    t_start = _time.monotonic()

    _log("Collecting metadata ...")
    meta = collect_metadata(dtype=dtype, repeats=repeats, distributions=3)
    meta["benchmark_config"]["measurement_mode"] = mode

    # -- baseline ----------------------------------------------------------
    _log("Measuring baseline (np.add) ...")
    baseline_fpe = measure_baseline(dtype=dtype, repeats=repeats)

    # -- baseline constants ---------------------------------------------------
    _BASELINE_N = 10_000_000  # default n in measure_baseline()
    _BASELINE_BENCH_CODE = "np.add(x, y, out=_out)"

    # -- primary measurement loop ------------------------------------------
    _log("Running primary measurement loop ...")
    raw_alphas, all_details = _run_category_loop(
        cats, dtype, repeats, _log, use_rich=use_rich, console=console
    )

    # -- normalise ALL categories by alpha(add) ----------------------------
    weights = normalize_weights(raw_alphas, baseline_fpe)

    # -- dual-mode validation (Step 1.2) -----------------------------------
    # If primary mode is perf, re-run in timing mode for validation.
    # If primary mode is timing, we have no perf to compare against.
    validation: dict[str, Any] = {
        "absolute_correction_factors": {k: round(v, 6) for k, v in raw_alphas.items()},
    }

    timing_weights: dict[str, float] = {}
    timing_alphas: dict[str, float] = {}
    timing_baseline: float = 0.0

    if mode == "perf" and not os.environ.get("MECHESTIM_SKIP_VALIDATION"):
        _log("Running timing-mode validation loop ...")
        # Force timing mode for the re-run
        _orig_env = os.environ.get("MECHESTIM_FORCE_TIMING")
        os.environ["MECHESTIM_FORCE_TIMING"] = "1"
        try:
            timing_baseline = measure_baseline(dtype=dtype, repeats=repeats)
            timing_alphas, _timing_details = _run_category_loop(
                cats, dtype, repeats, _log, use_rich=False, console=None
            )
        finally:
            if _orig_env is None:
                os.environ.pop("MECHESTIM_FORCE_TIMING", None)
            else:
                os.environ["MECHESTIM_FORCE_TIMING"] = _orig_env

        timing_weights = normalize_weights(timing_alphas, timing_baseline)
        timing_weights = {k: round(v, 4) for k, v in timing_weights.items()}
        validation["timing_weights"] = timing_weights
        validation["perf_vs_timing"] = _compute_validation_stats(
            weights, timing_weights
        )

        # -- inject timing data into per-op details -------------------------
        for op, t_alpha in timing_alphas.items():
            if op in all_details:
                all_details[op]["timing_ns_total"] = round(
                    t_alpha * _BASELINE_N * repeats, 2
                )
            if op in timing_weights and op in all_details:
                all_details[op]["timing_weight"] = timing_weights[op]

    # -- methodology metadata (Step 1.3 + 1.4) ----------------------------
    meta["methodology"] = {
        "version": "2.0",
        "formula": (
            "weight(op) = alpha(op) / alpha(add), "
            "where alpha(op) = median(perf_instructions / analytical_FLOPs)"
        ),
        "baseline_alpha": round(baseline_fpe, 6),
        "note": (
            "analytical_FLOPs from mechestim registry; "
            "perf_instructions are SIMD-width-weighted "
            "fp_arith_inst_retired counts"
        ),
    }
    meta["validation"] = validation

    # -- round weights -----------------------------------------------------
    weights = {k: round(v, 4) for k, v in weights.items()}

    # -- enrich per-op details ---------------------------------------------
    _enrich_details(
        all_details,
        weights=weights,
        alpha_add=baseline_fpe,
        baseline_n=_BASELINE_N,
        baseline_bench_code=_BASELINE_BENCH_CODE,
        repeats=repeats,
        timing_baseline=timing_baseline,
    )

    # -- store per-op details in meta --------------------------------------
    meta["per_op_details"] = all_details

    duration = round(_time.monotonic() - t_start, 1)
    meta["duration_seconds"] = duration

    result = {"meta": meta, "weights": weights}

    # -- write JSON --------------------------------------------------------
    if output:
        _log(f"Writing JSON to {output} ...")
        with open(output, "w") as f:
            json.dump(result, f, indent=2)

    # -- terminal dashboard ------------------------------------------------
    summary = render_terminal(meta, weights, baseline_fpe, len(weights), duration)
    print(summary, file=sys.stderr)

    # -- optional HTML report ----------------------------------------------
    if html:
        _log(f"Writing HTML report to {html} ...")
        html_content = render_html(meta, weights, baseline_fpe, len(weights), duration)
        with open(html, "w") as f:
            f.write(html_content)

    return result


def main() -> None:
    """CLI entry-point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run mechestim benchmarks and produce FPE weight tables.",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        help="NumPy dtype to benchmark (default: float64)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--html",
        default=None,
        help="Path to write HTML dashboard report",
    )
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        default=None,
        help="Category to benchmark (may be repeated; default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeat measurements (default: 10)",
    )
    args = parser.parse_args()

    categories = args.categories if args.categories else ["all"]

    run_benchmarks(
        dtype=args.dtype,
        output=args.output,
        html=args.html,
        categories=categories,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
