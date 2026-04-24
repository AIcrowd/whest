"""Benchmark dashboard with rich terminal and HTML output."""

from __future__ import annotations

import html as _html
from io import StringIO
from typing import Any

_REDUCTION_NAMES = frozenset(
    {
        "sum",
        "mean",
        "std",
        "var",
        "max",
        "min",
        "prod",
        "any",
        "all",
        "cumsum",
        "cumprod",
        "nansum",
        "nanmean",
        "nanstd",
        "nanvar",
        "nanmax",
        "nanmin",
        "nanprod",
        "median",
        "nanmedian",
        "percentile",
        "nanpercentile",
        "quantile",
        "nanquantile",
        "count_nonzero",
        "average",
        "argmax",
        "argmin",
        "nancumprod",
        "nancumsum",
    }
)
_SORTING_NAMES = frozenset(
    {
        "sort",
        "argsort",
        "partition",
        "argpartition",
        "searchsorted",
        "unique",
        "lexsort",
    }
)
_POLY_NAMES = frozenset(
    {
        "polyval",
        "polyfit",
        "polyder",
        "polyint",
        "polyadd",
        "polysub",
        "polymul",
        "polydiv",
        "poly",
        "roots",
    }
)


def _categorize_weights(weights: dict[str, float]) -> dict[str, dict[str, float]]:
    """Group weights by category based on op_name prefix."""
    categories: dict[str, dict[str, float]] = {}
    for op, w in sorted(weights.items()):
        if op.startswith("linalg."):
            cat = "Linalg"
        elif op.startswith("fft."):
            cat = "FFT"
        elif op.startswith("random."):
            cat = "Random"
        elif op in _REDUCTION_NAMES:
            cat = "Reductions"
        elif op in _SORTING_NAMES:
            cat = "Sorting"
        elif op in _POLY_NAMES:
            cat = "Polynomial"
        else:
            cat = "Pointwise"
        categories.setdefault(cat, {})[op] = w
    return {k: v for k, v in categories.items() if v}


def _bar_text(value: float, max_value: float, width: int = 20) -> str:
    """Create a unicode bar string scaled to max_value."""
    if max_value <= 0:
        return ""
    ratio = min(value / max_value, 1.0)
    full = int(ratio * width)
    return "\u2588" * max(full, 0)


def render_terminal(
    meta: dict[str, Any],
    weights: dict[str, float],
    baseline_fpe: float,
    total_ops: int,
    duration_seconds: float,
) -> str:
    """Render benchmark results to a terminal string.

    Uses ``rich`` for styled output, falling back to plain text.
    """
    try:
        return _render_terminal_rich(
            meta, weights, baseline_fpe, total_ops, duration_seconds
        )
    except Exception:
        return _render_terminal_plain(
            meta, weights, baseline_fpe, total_ops, duration_seconds
        )


def _render_terminal_rich(
    meta: dict[str, Any],
    weights: dict[str, float],
    baseline_fpe: float,
    total_ops: int,
    duration_seconds: float,
) -> str:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100)

    hw = meta.get("hardware", {})
    sw = meta.get("software", {})
    bc = meta.get("benchmark_config", {})
    mode = bc.get("measurement_mode", "unknown")

    header_lines = [
        f"Timestamp : {meta.get('timestamp', 'N/A')[:19]}",
        f"CPU       : {hw.get('cpu_model', 'N/A')} ({hw.get('cpu_cores', '?')} cores, {hw.get('arch', '?')})",
        f"dtype     : {bc.get('dtype', 'N/A')}",
        f"NumPy     : {sw.get('numpy', 'N/A')} ({sw.get('blas', 'N/A')})",
        f"Mode      : {mode}"
        + (" (wall-clock proxy)" if mode == "timing" else " (hardware counters)"),
    ]
    console.print(Panel("\n".join(header_lines), title="flopscope FLOP Weight Benchmark"))

    categories = _categorize_weights(weights)

    for cat_name, ops in categories.items():
        # Scale bars per-category so each category uses its full width
        cat_max = max(ops.values()) if ops else 1.0

        table = Table(title=cat_name, show_lines=False)
        table.add_column("Operation", style="cyan", min_width=25)
        table.add_column("Weight", justify="right", style="green", min_width=10)
        table.add_column("", min_width=22)  # bar column

        for op, w in sorted(ops.items(), key=lambda x: -x[1]):
            bar = _bar_text(w, cat_max, width=20)
            table.add_row(op, f"{w:.2f}", f"[yellow]{bar}[/yellow]")
        console.print(table)
        console.print()

    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    footer_lines = [
        f"Baseline: np.add = 1.0 ({baseline_fpe:.4f} raw units/elem)",
        f"Total operations benchmarked: {total_ops}",
        f"Duration: {mins}m {secs:02d}s",
    ]
    console.print(Panel("\n".join(footer_lines), title="Summary"))

    return buf.getvalue()


def _render_terminal_plain(
    meta: dict[str, Any],
    weights: dict[str, float],
    baseline_fpe: float,
    total_ops: int,
    duration_seconds: float,
) -> str:
    lines: list[str] = []
    hw = meta.get("hardware", {})
    sw = meta.get("software", {})
    bc = meta.get("benchmark_config", {})

    lines.append("=" * 70)
    lines.append("  flopscope FLOP Weight Benchmark")
    lines.append("=" * 70)
    lines.append(f"  Timestamp : {meta.get('timestamp', 'N/A')[:19]}")
    lines.append(f"  CPU       : {hw.get('cpu_model', 'N/A')}")
    lines.append(f"  dtype     : {bc.get('dtype', 'N/A')}")
    lines.append(f"  NumPy     : {sw.get('numpy', 'N/A')}")
    lines.append("")

    categories = _categorize_weights(weights)

    for cat_name, ops in categories.items():
        cat_max = max(ops.values()) if ops else 1.0
        lines.append(f"  --- {cat_name} ---")
        for op, w in sorted(ops.items(), key=lambda x: -x[1]):
            bar = _bar_text(w, cat_max, width=20)
            lines.append(f"    {op:<28s} {w:>10.2f}  {bar}")
        lines.append("")

    lines.append("-" * 70)
    lines.append(f"  Baseline: np.add = {baseline_fpe:.4f} raw units/elem")
    lines.append(f"  Total operations: {total_ops}")
    lines.append(f"  Duration: {duration_seconds:.1f}s")
    lines.append("=" * 70)

    return "\n".join(lines)


def render_html(
    meta: dict[str, Any],
    weights: dict[str, float],
    baseline_fpe: float,
    total_ops: int,
    duration_seconds: float,
) -> str:
    """Render benchmark results as a standalone HTML document."""
    hw = meta.get("hardware", {})
    sw = meta.get("software", {})
    bc = meta.get("benchmark_config", {})
    categories = _categorize_weights(weights)

    esc = _html.escape

    parts: list[str] = []
    parts.append("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>flopscope FLOP Weight Benchmark</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #333; }
  h1 { border-bottom: 2px solid #2563eb; padding-bottom: .5rem; }
  .meta { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1rem 1.5rem; margin-bottom: 1.5rem; line-height: 1.8; }
  .meta b { min-width: 80px; display: inline-block; }
  h2 { color: #1e40af; margin-top: 2rem; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
  th { text-align: left; border-bottom: 2px solid #cbd5e1; padding: .5rem; }
  td { padding: .4rem .5rem; border-bottom: 1px solid #e2e8f0; }
  .bar-cell { width: 40%; }
  .bar { height: 1.2rem; background: linear-gradient(90deg, #22c55e, #16a34a);
         border-radius: 3px; min-width: 2px; }
  .summary { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
             padding: 1rem 1.5rem; margin-top: 2rem; }
</style>
</head>
<body>
<h1>flopscope FLOP Weight Benchmark</h1>
""")

    parts.append('<div class="meta">')
    parts.append(f"<b>Timestamp:</b> {esc(str(meta.get('timestamp', 'N/A'))[:19])}<br>")
    parts.append(
        f"<b>CPU:</b> {esc(str(hw.get('cpu_model', 'N/A')))} "
        f"({hw.get('cpu_cores', '?')} cores, {esc(str(hw.get('arch', '?')))})<br>"
    )
    parts.append(f"<b>dtype:</b> {esc(str(bc.get('dtype', 'N/A')))} | ")
    parts.append(
        f"<b>NumPy:</b> {esc(str(sw.get('numpy', 'N/A')))} ({esc(str(sw.get('blas', 'N/A')))})<br>"
    )
    mode = bc.get("measurement_mode", "unknown")
    mode_label = (
        "hardware counters (perf)" if mode == "perf" else "wall-clock time (proxy)"
    )
    parts.append(f"<b>Mode:</b> {esc(mode_label)}")
    parts.append("</div>")

    for cat_name, ops in categories.items():
        cat_max = max(ops.values()) if ops else 1.0
        parts.append(f"<h2>{esc(cat_name)}</h2>")
        parts.append(
            "<table><tr><th>Operation</th><th>Weight</th>"
            "<th class='bar-cell'></th></tr>"
        )
        for op, w in sorted(ops.items(), key=lambda x: -x[1]):
            pct = (w / cat_max * 100) if cat_max > 0 else 0
            parts.append(
                f"<tr><td><code>{esc(op)}</code></td>"
                f"<td style='text-align:right'>{w:.2f}</td>"
                f'<td class="bar-cell">'
                f'<div class="bar" style="width:{pct:.1f}%"></div>'
                f"</td></tr>"
            )
        parts.append("</table>")

    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    parts.append('<div class="summary">')
    parts.append(
        f"<b>Baseline:</b> np.add = 1.0 ({baseline_fpe:.4f} raw units/elem)<br>"
    )
    parts.append(f"<b>Total operations:</b> {total_ops}<br>")
    parts.append(f"<b>Duration:</b> {mins}m {secs:02d}s")
    parts.append("</div>")
    parts.append("</body></html>")

    return "\n".join(parts)
