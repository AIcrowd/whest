"""Benchmark runner — CLI orchestrator for mechestim benchmarks."""

from __future__ import annotations

import argparse
import json
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

# In perf mode: linalg/fft return correction factors (already normalized).
# In timing mode: ALL categories return raw timing, need baseline normalization.
_CORRECTION_CATEGORIES = {"linalg", "fft"}

ALL_CATEGORIES = sorted(
    {"pointwise", "reductions", "sorting", "random", "polynomial"}
    | _CORRECTION_CATEGORIES
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

# Approximate op counts per category (for progress bar total)
_APPROX_OP_COUNTS: dict[str, int] = {
    "pointwise": 70,
    "reductions": 28,
    "linalg": 14,
    "fft": 14,
    "sorting": 7,
    "random": 10,
    "polynomial": 10,
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

    # -- measurement mode --------------------------------------------------
    from benchmarks._perf import measurement_mode

    mode = measurement_mode()

    # -- try rich ----------------------------------------------------------
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        console = Console(stderr=True)
        use_rich = True
    except ImportError:
        use_rich = False

    def _log(msg: str) -> None:
        if use_rich:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    _log(f"[bold]Measurement mode:[/bold] {mode}" if use_rich else f"Measurement mode: {mode}")
    if mode == "timing":
        _log(
            "  [dim](perf not available — using wall-clock time as proxy)[/dim]"
            if use_rich else
            "  (perf not available — using wall-clock time as proxy)"
        )

    t_start = _time.monotonic()

    _log("Collecting metadata ...")
    meta = collect_metadata(dtype=dtype, repeats=repeats, distributions=3)
    meta["benchmark_config"]["measurement_mode"] = mode

    # -- baseline ----------------------------------------------------------
    _log("Measuring baseline (np.add) ...")
    baseline_fpe = measure_baseline(dtype=dtype, repeats=repeats)

    # -- helper: should we normalize? --------------------------------------
    def _should_normalize(cat: str) -> bool:
        """In timing mode, normalize everything. In perf mode, skip linalg/fft."""
        if mode == "perf" and cat in _CORRECTION_CATEGORIES:
            return False
        return True

    # -- run each category -------------------------------------------------
    weights: dict[str, float] = {}

    if use_rich:
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
                raw = func(dtype=dtype, repeats=repeats)

                if _should_normalize(cat):
                    weights.update(normalize_weights(raw, baseline_fpe))
                else:
                    weights.update(raw)
                progress.advance(task_id, advance=_APPROX_OP_COUNTS.get(cat, len(raw)))
            progress.update(task_id, current_op="done")
    else:
        for cat in cats:
            if cat not in _BENCHMARK_FUNCS:
                print(f"Unknown category: {cat!r}, skipping.", file=sys.stderr)
                continue
            print(f"Benchmarking {cat} ...", file=sys.stderr)
            func = _BENCHMARK_FUNCS[cat]
            raw = func(dtype=dtype, repeats=repeats)

            if _should_normalize(cat):
                weights.update(normalize_weights(raw, baseline_fpe))
            else:
                weights.update(raw)

    # -- round weights -----------------------------------------------------
    weights = {k: round(v, 4) for k, v in weights.items()}

    duration = round(_time.monotonic() - t_start, 1)
    meta["duration_seconds"] = duration

    result = {"meta": meta, "weights": weights}

    # -- write JSON --------------------------------------------------------
    if output:
        _log(f"Writing JSON to {output} ...")
        with open(output, "w") as f:
            json.dump(result, f, indent=2)

    # -- terminal dashboard ------------------------------------------------
    summary = render_terminal(
        meta, weights, baseline_fpe, len(weights), duration
    )
    print(summary, file=sys.stderr)

    # -- optional HTML report ----------------------------------------------
    if html:
        _log(f"Writing HTML report to {html} ...")
        html_content = render_html(
            meta, weights, baseline_fpe, len(weights), duration
        )
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
