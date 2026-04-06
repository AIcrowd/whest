"""Benchmark dashboard with rich terminal and HTML output."""

from __future__ import annotations

import html as _html
from io import StringIO
from typing import Any

_REDUCTION_NAMES = frozenset(
    {"sum", "mean", "std", "var", "max", "min", "prod", "any", "all", "cumsum", "cumprod"}
)
_SORTING_NAMES = frozenset({"sort", "argsort", "partition", "argpartition"})
_POLY_NAMES = frozenset({"polyval", "polyfit", "polyder", "polyint", "polyadd", "polysub", "polymul", "polydiv"})


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
    """Create a unicode bar string."""
    if max_value <= 0:
        return ""
    ratio = min(value / max_value, 1.0)
    full = int(ratio * width)
    blocks = "\u2588" * full
    return blocks


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
        return _render_terminal_rich(meta, weights, baseline_fpe, total_ops, duration_seconds)
    except Exception:
        return _render_terminal_plain(meta, weights, baseline_fpe, total_ops, duration_seconds)


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

    header_lines = [
        f"Timestamp : {meta.get('timestamp', 'N/A')}",
        f"CPU       : {hw.get('cpu_model', 'N/A')} ({hw.get('cpu_cores', '?')} cores, {hw.get('arch', '?')})",
        f"dtype     : {bc.get('dtype', 'N/A')}",
        f"NumPy     : {sw.get('numpy', 'N/A')}",
        f"BLAS      : {sw.get('blas', 'N/A')}",
    ]
    console.print(Panel("\n".join(header_lines), title="Benchmark Environment"))

    categories = _categorize_weights(weights)
    max_w = max(weights.values()) if weights else 1.0

    for cat_name, ops in categories.items():
        table = Table(title=cat_name)
        table.add_column("Operation", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Bar")
        for op, w in sorted(ops.items(), key=lambda x: -x[1]):
            table.add_row(op, f"{w:.2f}", _bar_text(w, max_w))
        console.print(table)

    footer_lines = [
        f"Baseline FPE rate : {baseline_fpe:.4f}",
        f"Total operations  : {total_ops}",
        f"Duration          : {duration_seconds:.1f}s",
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

    lines.append("=" * 60)
    lines.append("Benchmark Environment")
    lines.append("=" * 60)
    lines.append(f"  Timestamp : {meta.get('timestamp', 'N/A')}")
    lines.append(f"  CPU       : {hw.get('cpu_model', 'N/A')}")
    lines.append(f"  dtype     : {bc.get('dtype', 'N/A')}")
    lines.append(f"  NumPy     : {sw.get('numpy', 'N/A')}")
    lines.append(f"  BLAS      : {sw.get('blas', 'N/A')}")
    lines.append("")

    categories = _categorize_weights(weights)
    max_w = max(weights.values()) if weights else 1.0

    for cat_name, ops in categories.items():
        lines.append(f"--- {cat_name} ---")
        for op, w in sorted(ops.items(), key=lambda x: -x[1]):
            lines.append(f"  {op:<30s} {w:>8.2f}  {_bar_text(w, max_w)}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"  Baseline FPE rate : {baseline_fpe:.4f}")
    lines.append(f"  Total operations  : {total_ops}")
    lines.append(f"  Duration          : {duration_seconds:.1f}s")
    lines.append("=" * 60)

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
    max_w = max(weights.values()) if weights else 1.0

    esc = _html.escape

    parts: list[str] = []
    parts.append(
        """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark Dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #333; }
  h1 { border-bottom: 2px solid #2563eb; padding-bottom: .5rem; }
  .meta { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1rem 1.5rem; margin-bottom: 1.5rem; }
  .meta span { display: inline-block; margin-right: 2rem; }
  h2 { color: #1e40af; margin-top: 2rem; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
  th { text-align: left; border-bottom: 2px solid #cbd5e1; padding: .5rem; }
  td { padding: .5rem; border-bottom: 1px solid #e2e8f0; }
  .bar-cell { width: 40%; }
  .bar { height: 1.2rem; background: #22c55e; border-radius: 3px; }
  .summary { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
             padding: 1rem 1.5rem; margin-top: 2rem; }
</style>
</head>
<body>
<h1>Benchmark Dashboard</h1>
"""
    )

    parts.append('<div class="meta">')
    parts.append(f"<span><b>Timestamp:</b> {esc(str(meta.get('timestamp', 'N/A')))}</span>")
    parts.append(
        f"<span><b>CPU:</b> {esc(str(hw.get('cpu_model', 'N/A')))} "
        f"({hw.get('cpu_cores', '?')} cores)</span>"
    )
    parts.append(f"<span><b>dtype:</b> {esc(str(bc.get('dtype', 'N/A')))}</span>")
    parts.append(f"<span><b>NumPy:</b> {esc(str(sw.get('numpy', 'N/A')))}</span>")
    parts.append(f"<span><b>BLAS:</b> {esc(str(sw.get('blas', 'N/A')))}</span>")
    parts.append("</div>")

    for cat_name, ops in categories.items():
        parts.append(f"<h2>{esc(cat_name)}</h2>")
        parts.append("<table><tr><th>Operation</th><th>Weight</th><th class='bar-cell'>Bar</th></tr>")
        for op, w in sorted(ops.items(), key=lambda x: -x[1]):
            pct = (w / max_w * 100) if max_w > 0 else 0
            parts.append(
                f"<tr><td>{esc(op)}</td><td>{w:.2f}</td>"
                f'<td class="bar-cell"><div class="bar" style="width:{pct:.1f}%"></div></td></tr>'
            )
        parts.append("</table>")

    parts.append('<div class="summary">')
    parts.append(f"<b>Baseline FPE rate:</b> {baseline_fpe:.4f}<br>")
    parts.append(f"<b>Total operations:</b> {total_ops}<br>")
    parts.append(f"<b>Duration:</b> {duration_seconds:.1f}s")
    parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)
