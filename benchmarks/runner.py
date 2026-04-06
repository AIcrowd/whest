"""Benchmark runner — CLI orchestrator for mechestim benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
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

# Categories that get normalized by baseline_fpe
_NORMALIZED_CATEGORIES = {"pointwise", "reductions", "sorting", "random", "polynomial"}

# Categories that use correction factors directly (no normalization)
_DIRECT_CATEGORIES = {"linalg", "fft"}

ALL_CATEGORIES = sorted(_NORMALIZED_CATEGORIES | _DIRECT_CATEGORIES)

_BENCHMARK_FUNC_NAMES: dict[str, str] = {
    "pointwise": "benchmark_pointwise",
    "reductions": "benchmark_reductions",
    "linalg": "benchmark_linalg",
    "fft": "benchmark_fft",
    "sorting": "benchmark_sorting",
    "random": "benchmark_random",
    "polynomial": "benchmark_polynomial",
}


def normalize_weights(raw_fpe: dict[str, float], baseline_fpe: float) -> dict[str, float]:
    """Divide each raw FPE value by *baseline_fpe*.

    If *baseline_fpe* is zero (or very close), fall back to 1.0 so the raw
    values pass through unchanged.
    """
    if baseline_fpe == 0:
        baseline_fpe = 1.0
    return {k: v / baseline_fpe for k, v in raw_fpe.items()}


def run_benchmarks(
    dtype: str = "float64",
    output: str | None = None,
    html: str | None = None,
    categories: list[str] | None = None,
    repeats: int = 10,
) -> dict[str, Any]:
    """Run the full benchmark suite and return the results dict."""

    if categories is None or "all" in categories:
        cats = ALL_CATEGORIES
    else:
        cats = categories

    # -- metadata ----------------------------------------------------------
    print("Collecting metadata ...", file=sys.stderr)
    meta = collect_metadata(dtype=dtype, repeats=repeats)

    # -- baseline ----------------------------------------------------------
    print("Measuring baseline (np.add) ...", file=sys.stderr)
    baseline_fpe = measure_baseline(dtype=dtype, repeats=repeats)

    # -- run each category -------------------------------------------------
    weights: dict[str, float] = {}

    import benchmarks.runner as _self_module

    for cat in cats:
        if cat not in _BENCHMARK_FUNC_NAMES:
            print(f"Unknown category: {cat!r}, skipping.", file=sys.stderr)
            continue

        print(f"Benchmarking {cat} ...", file=sys.stderr)
        func = getattr(_self_module, _BENCHMARK_FUNC_NAMES[cat])
        raw = func(dtype=dtype, repeats=repeats)

        if cat in _NORMALIZED_CATEGORIES:
            normalized = normalize_weights(raw, baseline_fpe)
        else:
            # linalg / fft — use correction factors directly
            normalized = raw

        weights.update(normalized)

    # -- round weights -----------------------------------------------------
    weights = {k: round(v, 4) for k, v in weights.items()}

    result = {"meta": meta, "weights": weights}

    # -- write JSON --------------------------------------------------------
    if output:
        print(f"Writing JSON to {output} ...", file=sys.stderr)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)

    # -- terminal dashboard ------------------------------------------------
    summary = render_terminal(weights)
    print(summary, file=sys.stderr)

    # -- optional HTML report ----------------------------------------------
    if html:
        print(f"Writing HTML report to {html} ...", file=sys.stderr)
        html_content = render_html(weights, meta)
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
