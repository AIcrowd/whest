"""Budget display rendering with Rich (optional) and plain-text fallback."""

from __future__ import annotations

from mechestim._budget import budget_data


def _format_flops(n: int) -> str:
    """Format a FLOP count with thousands separators."""
    return f"{n:,}"


def _pct(used: int, total: int) -> str:
    """Return a percentage string."""
    if total == 0:
        return "0.0%"
    return f"{100 * used / total:.1f}%"


def _usage_color(used: int, total: int) -> str:
    """Return a Rich color name based on usage percentage."""
    if total == 0:
        return "white"
    ratio = used / total
    if ratio < 0.5:
        return "green"
    if ratio < 0.8:
        return "yellow"
    return "red"


def _plain_text_summary() -> str:
    """Render a plain-text budget summary across all namespaces."""
    data = budget_data(by_namespace=True)
    if data["flops_used"] == 0 and not data.get("by_namespace"):
        return "No budget data recorded yet."

    lines = [
        "mechestim FLOP Budget Summary",
        "=" * 50,
        f"  Total budget:    {_format_flops(data['flop_budget']):>20}",
        f"  Used:            {_format_flops(data['flops_used']):>20}  ({_pct(data['flops_used'], data['flop_budget'])})",
        f"  Remaining:       {_format_flops(data['flops_remaining']):>20}  ({_pct(data['flops_remaining'], data['flop_budget'])})",
    ]

    by_ns = data.get("by_namespace", {})
    if by_ns:
        lines.append("")
        for ns, ns_data in by_ns.items():
            label = ns if ns is not None else "(default)"
            lines.append(f"  [{label}]")
            lines.append(f"    Budget:  {_format_flops(ns_data['flop_budget']):>16}")
            lines.append(f"    Used:    {_format_flops(ns_data['flops_used']):>16}  ({_pct(ns_data['flops_used'], ns_data['flop_budget'])})")
            ops = ns_data.get("operations", {})
            if ops:
                lines.append("    Operations:")
                for op_name, op_info in sorted(ops.items(), key=lambda x: -x[1]["flop_cost"]):
                    call_word = "call" if op_info["calls"] == 1 else "calls"
                    op_pct = _pct(op_info["flop_cost"], ns_data["flops_used"])
                    lines.append(
                        f"      {op_name:<20} {_format_flops(op_info['flop_cost']):>12}  ({op_pct:>6})  [{op_info['calls']} {call_word}]"
                    )
            lines.append("")

    ops = data.get("operations", {})
    if ops:
        lines.append("  All operations (session total):")
        for op_name, op_info in sorted(ops.items(), key=lambda x: -x[1]["flop_cost"]):
            call_word = "call" if op_info["calls"] == 1 else "calls"
            op_pct = _pct(op_info["flop_cost"], data["flops_used"])
            lines.append(
                f"    {op_name:<20} {_format_flops(op_info['flop_cost']):>12}  ({op_pct:>6})  [{op_info['calls']} {call_word}]"
            )

    return "\n".join(lines)


def _rich_summary():
    """Render a Rich-formatted budget summary."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    data = budget_data(by_namespace=True)
    if data["flops_used"] == 0 and not data.get("by_namespace"):
        return Panel("No budget data recorded yet.", title="mechestim Budget")

    color = _usage_color(data["flops_used"], data["flop_budget"])

    table = Table(title="mechestim FLOP Budget Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("%", justify="right")

    table.add_row("Total budget", _format_flops(data["flop_budget"]), "")
    table.add_row(
        "Used",
        Text(_format_flops(data["flops_used"]), style=color),
        Text(_pct(data["flops_used"], data["flop_budget"]), style=color),
    )
    table.add_row(
        "Remaining",
        _format_flops(data["flops_remaining"]),
        _pct(data["flops_remaining"], data["flop_budget"]),
    )

    by_ns = data.get("by_namespace", {})
    if by_ns:
        table.add_section()
        table.add_row(Text("By Namespace", style="bold magenta"), "", "")
        for ns, ns_data in by_ns.items():
            label = ns if ns is not None else "(default)"
            ns_color = _usage_color(ns_data["flops_used"], ns_data["flop_budget"])
            table.add_row(
                Text(f"  {label}", style="bold"),
                Text(_format_flops(ns_data["flops_used"]), style=ns_color),
                Text(_pct(ns_data["flops_used"], ns_data["flop_budget"]), style=ns_color),
            )
            ops = ns_data.get("operations", {})
            for op_name, op_info in sorted(ops.items(), key=lambda x: -x[1]["flop_cost"]):
                call_word = "call" if op_info["calls"] == 1 else "calls"
                table.add_row(
                    f"    {op_name}",
                    _format_flops(op_info["flop_cost"]),
                    f"[{op_info['calls']} {call_word}]",
                )

    return table


def render_budget_summary():
    """Return a Rich renderable if Rich is installed, otherwise plain text."""
    try:
        import rich  # noqa: F401
        return _rich_summary()
    except ImportError:
        return _plain_text_summary()


class _PlainTextLive:
    """Fallback live display that prints summary on exit."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print(_plain_text_summary())
        return None


def budget_live():
    """Return a live-updating budget display context manager."""
    try:
        from rich.live import Live

        class _RichBudgetLive:
            def __init__(self):
                self._live = None

            def __enter__(self):
                self._live = Live(_rich_summary(), refresh_per_second=2)
                self._live.__enter__()
                return self

            def __exit__(self, *args):
                if self._live is not None:
                    self._live.update(_rich_summary())
                    self._live.__exit__(*args)
                return None

        return _RichBudgetLive()
    except ImportError:
        return _PlainTextLive()


def budget_summary():
    """Print or return the session-wide budget summary."""
    result = render_budget_summary()
    try:
        get_ipython  # noqa: F821
        return result
    except NameError:
        if isinstance(result, str):
            print(result)
        else:
            from rich.console import Console
            Console().print(result)
        return None
