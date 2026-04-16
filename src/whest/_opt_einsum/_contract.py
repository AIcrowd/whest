"""Contains contract_path and supporting types (stripped from opt_einsum contract.py).

Excluded: contract, _core_contract, ContractExpression, _einsum, _tensordot,
_transpose, backends/sharing imports, _filter_einsum_defaults,
format_const_einsum_str, shape_only.
"""

from collections.abc import Collection
from dataclasses import dataclass, field
from decimal import Decimal
from functools import cached_property
from hashlib import sha1
from typing import Any, Literal, overload

from . import _blas as blas
from . import _helpers as helpers
from . import _parser as parser
from . import _paths as paths
from ._hsluv import rgb_distance_hex, rich_label_palette
from ._subgraph_symmetry import SubgraphSymmetryOracle
from ._symmetry import PermutationGroup, symmetric_flop_count
from ._typing import (
    ArrayType,
    ContractionListType,
    OptimizeKind,
    PathType,
)

__all__ = [
    "contract_path",
    "PathInfo",
    "StepInfo",
]

_RICH_SYMMETRY_STYLES = {
    "S": "bold bright_cyan",
    "C": "bold bright_magenta",
    "D": "bold bright_yellow",
    "W": "bold bright_green",
}

## Common types

_MemoryLimit = None | int | Decimal | Literal["max_input"]


@dataclass
class StepInfo:
    """Per-step diagnostics for a contraction path."""

    subscript: str
    """Einsum subscript for this step, e.g. ``"ijk,ai->ajk"``."""

    flop_cost: int
    """Symmetry-aware FLOP cost (FMA = 1 op)."""

    input_shapes: list[tuple[int, ...]]
    """Shapes of the input operands for this step."""

    output_shape: tuple[int, ...]
    """Shape of the output operand for this step."""

    input_groups: list[PermutationGroup | None]
    """PermutationGroup for each input in this step."""

    output_group: PermutationGroup | None
    """PermutationGroup of the output, or None."""

    dense_flop_cost: int
    """FLOP cost without symmetry (FMA = 1 op)."""

    symmetry_savings: float
    """Fraction saved: ``1 - (flop_cost / dense_flop_cost)``. Zero when no symmetry."""

    blas_type: str | bool = False

    inner_group: PermutationGroup | None = None
    """PermutationGroup among the contracted (summed) labels, or None.
    Describes inner-summation redundancy from the W-side of the
    subgraph symmetry oracle."""
    """BLAS classification for this step (e.g. 'GEMM', 'SYMM', False)."""

    path_indices: tuple[int, ...] = ()
    """The SSA-id contraction tuple for this step (the entry from
    ``PathInfo.path[i]``). Useful for cross-referencing the table with
    the raw path field."""

    inner_applied: bool = False
    """Whether inner (W-side) symmetry was actually applied to reduce
    the FLOP cost at this step.  True only when ``use_inner_symmetry``
    is enabled and all W-group labels are contracted at this step."""

    merged_subset: frozenset[int] | None = None
    """Subset of *original* operand positions that this step's output
    intermediate covers. For step 0 contracting two original operands i
    and j, this is ``frozenset({i, j})``. For later steps it's the union
    of the subsets of all SSA inputs being contracted. This is the exact
    key the symmetry oracle uses for ``oracle.sym(...)`` lookups, so it
    makes the symmetry column directly attributable. ``None`` when no
    oracle was provided."""


@dataclass
class PathInfo:
    """Information about a contraction path with per-step symmetry diagnostics."""

    path: list[tuple[int, ...]]
    """The optimized contraction path (list of index-tuples)."""

    steps: list[StepInfo]
    """Per-step diagnostics."""

    naive_cost: int
    """Naive (single-step) FLOP cost (FMA = 1 op)."""

    optimized_cost: int
    """Sum of per-step costs (FMA = 1 op)."""

    largest_intermediate: int
    """Number of elements in the largest intermediate tensor."""

    speedup: float
    """``naive_cost / optimized_cost``."""

    input_subscripts: str = ""
    """Comma-separated input subscripts, e.g. ``"ij,jk,kl"``."""

    output_subscript: str = ""
    """Output subscript, e.g. ``"il"``."""

    size_dict: dict[str, int] = field(default_factory=dict)
    """Mapping from index label to dimension size."""

    optimizer_used: str = ""
    """Name of the path-finding function actually invoked. For ``optimize='auto'``
    or ``'auto-hq'`` this resolves to the underlying inner choice
    (e.g. ``'optimal'``, ``'branch_2'``, ``'dynamic_programming'``,
    ``'random_greedy_128'``) so users can tell which algorithm produced
    the path. For explicit choices it matches the requested name. Empty
    string for the trivial num_ops <= 2 case where no optimizer runs."""

    # Legacy fields for backward-compat with opt_einsum tests
    contraction_list: ContractionListType = field(default_factory=list)
    scale_list: list[int] = field(default_factory=list)
    size_list: list[int] = field(default_factory=list)
    _oe_naive_cost: int = 0
    _oe_opt_cost: int = 0

    @property
    def opt_cost(self) -> Decimal:
        """Legacy: opt_einsum-style cost (FMA = 1 op)."""
        return Decimal(self._oe_opt_cost)

    @property
    def eq(self) -> str:
        return f"{self.input_subscripts}->{self.output_subscript}"

    @staticmethod
    def _preferred_label_style_index(label: str, total_slots: int) -> int | None:
        """Return the preferred stable palette slot for a label."""
        if not label or not label[0].isalpha():
            return None
        return int.from_bytes(sha1(label.encode("utf-8")).digest()[:2], "big") % (
            total_slots
        )

    @cached_property
    def _label_styles(self) -> dict[str, str]:
        """Assign non-colliding styles for the active labels in this expression."""
        labels = list(dict.fromkeys(ch for ch in self.eq if ch.isalpha()))
        slot_count = max(64, len(labels))
        label_styles = tuple(f"bold {color}" for color in rich_label_palette(slot_count))
        used_slots: set[int] = set()
        styles: dict[str, str] = {}
        total_slots = len(label_styles)

        for label in labels:
            preferred = self._preferred_label_style_index(label, total_slots)
            if preferred is None:
                styles[label] = "bold"
                continue

            if not used_slots:
                used_slots.add(preferred)
                styles[label] = label_styles[preferred]
                continue

            best_slot: int | None = None
            best_score: tuple[float, int] | None = None
            for slot, style in enumerate(label_styles):
                if slot in used_slots:
                    continue

                color = style.rsplit(" ", 1)[-1]
                min_distance = min(
                    rgb_distance_hex(
                        color, label_styles[used_slot].rsplit(" ", 1)[-1]
                    )
                    for used_slot in used_slots
                )
                circular_preference_distance = min(
                    (slot - preferred) % total_slots,
                    (preferred - slot) % total_slots,
                )
                score = (min_distance, -circular_preference_distance)
                if best_score is None or score > best_score:
                    best_score = score
                    best_slot = slot

            assert best_slot is not None
            used_slots.add(best_slot)
            styles[label] = label_styles[best_slot]

        return styles

    def _label_style(self, label: str) -> str:
        """Return the resolved style for a label within this expression."""
        return self._label_styles.get(label, "bold")

    def _style_text_charwise(self, text: str):
        from rich.text import Text

        result = Text()
        for ch in text:
            if ch.isalpha():
                result.append(ch, style=self._label_style(ch))
            elif ch in ",->[]{}()<>✓×:":
                result.append(ch, style="dim")
            else:
                result.append(ch)
        return result

    def _rich_eq_text(self):
        """Render the full einsum expression with global label styling."""
        from rich.text import Text

        result = Text()
        prefix = "Complete contraction: "
        result.append(prefix, style="bold")
        result.append_text(self._style_text_charwise(self.eq))
        return result

    def _rich_subscript_text(self, subscript: str):
        """Render a subscript or step expression with global label styling."""
        return self._style_text_charwise(subscript)

    def _rich_index_sizes_text(self):
        """Render the index-size summary with label styling."""
        return self._style_text_charwise(self._fmt_index_sizes())

    def _rich_symmetry_token_text(self, token: str):
        from rich.text import Text

        if token == "-":
            return Text("-", style="dim")
        if token in {"×", "→"}:
            return Text(token, style="dim")
        if token.startswith("PermGroup⟨"):
            return self._style_text_charwise(token)

        result = Text()
        if token.startswith("W"):
            sym_style = _RICH_SYMMETRY_STYLES["W"]
            result.append("W", style=sym_style)
            if token.startswith("W✓"):
                result.append("✓", style=sym_style)
            if ":" in token:
                result.append(":", style=sym_style)
            remainder = token.split(":", 1)[1].lstrip() if ":" in token else token[1:]
            if remainder:
                result.append(" ", style="dim")
                result.append_text(self._rich_symmetry_token_text(remainder))
            return result

        if token[0] in _RICH_SYMMETRY_STYLES and token[1:].split("{", 1)[0].isdigit():
            prefix = token[0]
            digits = []
            i = 1
            while i < len(token) and token[i].isdigit():
                digits.append(token[i])
                i += 1
            result.append(prefix, style=_RICH_SYMMETRY_STYLES[prefix])
            result.append("".join(digits), style=_RICH_SYMMETRY_STYLES[prefix])
            if i < len(token) and token[i] == "{":
                result.append("{", style="dim")
                i += 1
                while i < len(token) and token[i] != "}":
                    ch = token[i]
                    if ch.isalpha():
                        result.append(ch, style=self._label_style(ch))
                    elif ch == ",":
                        result.append(ch, style="dim")
                    else:
                        result.append(ch)
                    i += 1
                if i < len(token) and token[i] == "}":
                    result.append("}", style="dim")
                return result

        return self._style_text_charwise(token)

    def _rich_step_sym_text(self, step: StepInfo):
        from rich.text import Text

        in_parts = [self._fmt_sym(s) for s in step.input_groups]
        out_part = self._fmt_sym(step.output_group)
        w_part = self._fmt_sym(step.inner_group)
        if all(p == "-" for p in in_parts) and out_part == "-" and w_part == "-":
            return Text("-", style="dim")

        result = Text()
        for idx, part in enumerate(in_parts):
            if idx:
                result.append(" × ", style="dim")
            result.append_text(self._rich_symmetry_token_text(part))
        result.append(" → ", style="dim")
        result.append_text(self._rich_symmetry_token_text(out_part))
        if w_part != "-":
            result.append("  [", style="dim")
            result.append(
                "W✓" if step.inner_applied else "W", style=_RICH_SYMMETRY_STYLES["W"]
            )
            result.append(": ", style="dim")
            result.append_text(self._rich_symmetry_token_text(w_part))
            result.append("]", style="dim")
        return result

    def _fmt_overall_savings(self) -> str:
        """Format total optimized-vs-dense savings for the whole contraction."""
        if self.naive_cost <= 0:
            return "0.0%"
        return f"{1 - (self.optimized_cost / self.naive_cost):.1%}"

    def _rich_metric_pill(
        self,
        label: str,
        value: str | Any,
        *,
        highlight: bool = False,
        value_style: str | None = None,
        border_style: str | None = None,
    ):
        from rich import box
        from rich.panel import Panel
        from rich.text import Text

        resolved_value_style = value_style or ("bold cyan" if highlight else "bold")
        resolved_border_style = border_style or ("cyan" if highlight else "dim")
        body = Text()
        body.append(label, style="bold")
        body.append(": ", style="dim")
        if not value:
            body.append("-", style=resolved_value_style)
        elif isinstance(value, Text):
            if highlight:
                value = value.copy()
                value.stylize(resolved_value_style)
            body.append_text(value)
        else:
            body.append(str(value), style=resolved_value_style)
        return Panel.fit(
            body,
            box=box.ROUNDED,
            padding=(0, 1),
            border_style=resolved_border_style,
        )

    def _rich_summary_strip(self):
        from rich.columns import Columns

        pills = []
        pills.append(self._rich_metric_pill("Naive cost", f"{self.naive_cost:,}"))
        pills.append(
            self._rich_metric_pill(
                "Optimized cost", f"{self.optimized_cost:,}", highlight=True
            )
        )
        speedup_style = "bold green" if self.speedup > 1 else "bold"
        speedup_border = "green" if self.speedup > 1 else "dim"
        pills.append(
            self._rich_metric_pill(
                "Speedup",
                f"{self.speedup:.3f}x",
                value_style=speedup_style,
                border_style=speedup_border,
            )
        )
        savings_style = (
            "bold green" if self.optimized_cost < self.naive_cost else "bold"
        )
        savings_border = "green" if self.optimized_cost < self.naive_cost else "dim"
        pills.append(
            self._rich_metric_pill(
                "Savings",
                self._fmt_overall_savings(),
                value_style=savings_style,
                border_style=savings_border,
            )
        )
        pills.append(
            self._rich_metric_pill(
                "Largest intermediate", f"{self.largest_intermediate:,} elements"
            )
        )
        if self.size_dict:
            pills.append(
                self._rich_metric_pill("Index sizes", self._rich_index_sizes_text())
            )
        if self.optimizer_used:
            pills.append(self._rich_metric_pill("Optimizer", self.optimizer_used))
        return Columns(pills, expand=True, equal=False, padding=(0, 1))

    def _rich_verbose_detail_text(
        self, step: StepInfo, cumulative: int, *, step_index: int | None = None
    ):
        from rich.text import Text

        shape = (
            "(" + ",".join(str(d) for d in step.output_shape) + ")"
            if step.output_shape
            else "()"
        )
        result = Text()
        if step_index is not None:
            result.append(f"step {step_index}: ", style="dim")
        result.append("subset=", style="dim")
        result.append(self._fmt_subset(step.merged_subset), style="bold")
        result.append("\n")
        result.append("out_shape=", style="dim")
        result.append(shape, style="bold")
        result.append("\n")
        result.append("cumulative=", style="dim")
        result.append(f"{cumulative:,}", style="bold cyan")
        return result

    @staticmethod
    def _try_named_group(k: int, order: int) -> str | None:
        """Return the named prefix (e.g. 'S3') if recognised, else None."""
        if order == 1:
            return None
        from math import factorial

        if order == factorial(k):
            return f"S{k}"
        if order == k:
            return f"C{k}"
        if order == 2 * k and k >= 3:
            return f"D{k}"
        return None

    @staticmethod
    def _fmt_generators(group: PermutationGroup, labels: tuple) -> str:
        """Format generators in cycle notation with labels."""
        parts = []
        for gen in group.generators:
            if gen.is_identity:
                continue
            cycles = gen.cyclic_form
            if not cycles:
                continue
            perm_str = "".join(
                "(" + " ".join(labels[i] for i in cycle) + ")" for cycle in cycles
            )
            parts.append(perm_str)
        return ", ".join(parts) if parts else "e"

    def _fmt_sym(self, group: PermutationGroup | None) -> str:
        """Format a PermutationGroup for display."""
        if group is None:
            return "-"
        labels = group._labels or tuple(str(i) for i in range(group.degree))
        k = group.degree
        order = group.order()

        name = self._try_named_group(k, order)
        if name is not None:
            return f"{name}{{{','.join(labels)}}}"

        orbits = [orb for orb in group.orbits() if len(orb) >= 2]
        if not orbits:
            return "-"

        if len(orbits) == 1:
            orbit = orbits[0]
            moved_labels = tuple(labels[i] for i in sorted(orbit))
            mk = len(moved_labels)
            name = self._try_named_group(mk, order)
            if name is not None:
                return f"{name}{{{','.join(moved_labels)}}}"

        gen_str = self._fmt_generators(group, labels)
        return f"PermGroup⟨{gen_str}⟩"

    def _fmt_step_sym(self, step: StepInfo) -> str:
        """Format inputs→output symmetry transformation for one step."""
        in_parts = [self._fmt_sym(s) for s in step.input_groups]
        out_part = self._fmt_sym(step.output_group)
        w_part = self._fmt_sym(step.inner_group)
        if all(p == "-" for p in in_parts) and out_part == "-" and w_part == "-":
            return ""
        result = f"{' × '.join(in_parts)} → {out_part}"
        if w_part != "-":
            w_prefix = "W✓" if step.inner_applied else "W"
            result += f"  [{w_prefix}: {w_part}]"
        return result

    def _fmt_index_sizes(self) -> str:
        """Format index sizes compactly. Groups indices with the same size."""
        if not self.size_dict:
            return ""
        from collections import defaultdict

        by_size: dict[int, list[str]] = defaultdict(list)
        for idx, sz in self.size_dict.items():
            by_size[sz].append(idx)
        parts = []
        for sz, idxs in sorted(by_size.items(), key=lambda kv: (-len(kv[1]), -kv[0])):
            idxs_sorted = sorted(idxs)
            parts.append(f"{'='.join(idxs_sorted)}={sz}")
        return ", ".join(parts)

    @staticmethod
    def _fmt_contract(step: StepInfo) -> str:
        """Format the path-supplied contraction tuple, e.g. '(0, 1)'."""
        if not step.path_indices:
            return "-"
        if len(step.path_indices) == 2:
            return f"({step.path_indices[0]}, {step.path_indices[1]})"
        return "(" + ",".join(str(p) for p in step.path_indices) + ")"

    def _fmt_unique_dense(self, step: StepInfo) -> str:
        """Show output and inner unique/dense element counts."""
        from math import prod

        from whest._opt_einsum._symmetry import unique_elements

        if step.flop_cost == step.dense_flop_cost:
            return "-"

        parts: list[str] = []

        if step.output_group is not None and step.output_shape:
            out_str = step.subscript.split("->")[1] if "->" in step.subscript else ""
            out_total = prod(step.output_shape)
            out_unique = unique_elements(
                frozenset(out_str), self.size_dict, perm_group=step.output_group
            )
            if out_unique != out_total:
                parts.append(f"V:{out_unique:,}/{out_total:,}")

        if step.inner_applied and step.inner_group is not None:
            lhs = (
                step.subscript.split("->")[0]
                if "->" in step.subscript
                else step.subscript
            )
            out_str = step.subscript.split("->")[1] if "->" in step.subscript else ""
            contracted = frozenset(lhs.replace(",", "")) - frozenset(out_str)
            if contracted:
                inner_total = prod(self.size_dict[c] for c in contracted)
                inner_unique = unique_elements(
                    contracted, self.size_dict, perm_group=step.inner_group
                )
                if inner_unique != inner_total:
                    parts.append(f"W:{inner_unique:,}/{inner_total:,}")

        return " ".join(parts) if parts else "-"

    @staticmethod
    def _fmt_subset(s: frozenset[int] | None) -> str:
        if s is None:
            return "-"
        if not s:
            return "{}"
        return "{" + ",".join(str(i) for i in sorted(s)) + "}"

    def _header_lines(self) -> list[str]:
        sizes_line = self._fmt_index_sizes()
        header_lines = [
            f"  Complete contraction:  {self.eq}",
            f"      Naive cost (whest):  {self.naive_cost:,}",
            f"  Optimized cost (whest):  {self.optimized_cost:,}",
            f"                     Speedup:  {self.speedup:.3f}x",
            f"                     Savings:  {self._fmt_overall_savings()}",
            f"       Largest intermediate:  {self.largest_intermediate:,} elements",
        ]
        if sizes_line:
            header_lines.append(f"                Index sizes:  {sizes_line}")
        if self.optimizer_used:
            header_lines.append(f"                  Optimizer:  {self.optimizer_used}")
        return header_lines

    def _rich_step_table(self, verbose: bool = False):
        from rich import box
        from rich.table import Table

        any_unique = any(
            s.dense_flop_cost > 0 and s.flop_cost != s.dense_flop_cost
            for s in self.steps
        )

        contract_width = max(
            len("contract"),
            max((len(self._fmt_contract(step)) for step in self.steps), default=0),
        )
        subscript_width = min(
            24,
            max(
                len("subscript"),
                max((len(step.subscript) for step in self.steps), default=0),
            ),
        )
        flops_width = max(
            len("flops"),
            max((len(f"{step.flop_cost:,}") for step in self.steps), default=0),
        )
        dense_width = max(
            len("dense_flops"),
            max((len(f"{step.dense_flop_cost:,}") for step in self.steps), default=0),
        )
        savings_width = max(
            len("savings"),
            max(
                (len(f"{step.symmetry_savings:0.1%}") for step in self.steps), default=0
            ),
        )
        blas_width = max(
            len("blas"),
            max(
                (
                    len(str(step.blas_type) if step.blas_type else "-")
                    for step in self.steps
                ),
                default=0,
            ),
        )
        unique_width = None
        if any_unique:
            unique_width = max(
                len("unique/total"),
                max(
                    (len(self._fmt_unique_dense(step)) for step in self.steps),
                    default=0,
                ),
            )

        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            box=box.HEAVY,
            pad_edge=False,
            padding=(0, 1),
            collapse_padding=True,
        )
        table.add_column("step", justify="right", no_wrap=True, width=len("step"))
        table.add_column("contract", justify="left", no_wrap=True, width=contract_width)
        table.add_column("subscript", overflow="fold", width=subscript_width)
        table.add_column("flops", justify="right", no_wrap=True, width=flops_width)
        table.add_column(
            "dense_flops", justify="right", no_wrap=True, width=dense_width
        )
        table.add_column("savings", justify="right", no_wrap=True, width=savings_width)
        table.add_column("blas", no_wrap=True, width=blas_width)
        if any_unique:
            table.add_column("unique/total", no_wrap=True, width=unique_width)
        table.add_column(
            "symmetry (inputs → output)",
            overflow="fold",
            min_width=len("symmetry (inputs → output)"),
            ratio=1,
        )

        cumulative = 0
        for i, step in enumerate(self.steps):
            row = [
                str(i),
                self._fmt_contract(step),
                self._rich_subscript_text(step.subscript),
                f"{step.flop_cost:,}",
                f"{step.dense_flop_cost:,}",
                f"{step.symmetry_savings:>7.1%}",
                str(step.blas_type) if step.blas_type else "-",
            ]
            if any_unique:
                row.append(self._fmt_unique_dense(step))
            row.append(self._rich_step_sym_text(step) or "-")
            table.add_row(*row)
            if verbose:
                cumulative += step.flop_cost
                detail_row = [""] * len(table.columns)
                detail_row[2] = self._rich_verbose_detail_text(step, cumulative)
                detail_row[-1] = self._rich_verbose_detail_text(
                    step, cumulative, step_index=i
                )
                table.add_row(*detail_row)

        return table

    def _rich_renderable(self, verbose: bool = False):
        from rich.console import Group
        from rich.panel import Panel

        expr = self._rich_eq_text()
        summary = self._rich_summary_strip()
        table = self._rich_step_table(verbose=verbose)
        return Panel(
            Group(expr, summary, table),
            title="[bold cyan]einsum_path[/bold cyan]",
            border_style="cyan",
        )

    def format_table(self, verbose: bool = False) -> str:
        """Render the path info as a printable table.

        Parameters
        ----------
        verbose : bool, optional
            When True, emit an additional indented details row under each
            step showing the operand subset covered by the intermediate,
            its output shape, the unique-vs-dense element counts that the
            symmetry savings derive from, and the cumulative cost so far.
            Useful for debugging why a particular step's savings are what
            they are. Default False.
        """
        sym_strs = [self._fmt_step_sym(s) for s in self.steps]
        max_sym_width = max((len(s) for s in sym_strs), default=0)
        header_lines = self._header_lines()

        # Common columns: step, contract, subscript, flops, dense_flops, savings, blas
        # Plus: symmetry (when any step has symmetry) and unique/dense (when any
        # step has reduced cost).
        any_unique = any(
            s.dense_flop_cost > 0 and s.flop_cost != s.dense_flop_cost
            for s in self.steps
        )

        contract_strs = [self._fmt_contract(s) for s in self.steps]
        contract_col_width = max(
            len("contract"), max((len(c) for c in contract_strs), default=0)
        )
        unique_col_width = max(
            len("unique/total"),
            max((len(self._fmt_unique_dense(s)) for s in self.steps), default=0),
        )

        # Build the header line
        cols = [
            f"{'step':>4}",
            f"{'contract':<{contract_col_width}}",
            f"{'subscript':<30}",
            f"{'flops':>14}",
            f"{'dense_flops':>14}",
            f"{'savings':>8}",
            f"{'blas':<8}",
        ]
        if any_unique:
            cols.append(f"{'unique/total':<{unique_col_width}}")
        sym_col_width = min(max(max_sym_width, len("symmetry (inputs → output)")), 60)
        cols.append(f"{'symmetry (inputs → output)':<{sym_col_width}}")

        header_row = "  ".join(cols)
        width = max(len(header_row), 84)
        lines = header_lines + ["-" * width, header_row, "-" * width]

        cumulative = 0
        for i, step in enumerate(self.steps):
            blas_label = str(step.blas_type) if step.blas_type else "-"
            row_parts = [
                f"{i:>4}",
                f"{contract_strs[i]:<{contract_col_width}}",
                f"{step.subscript:<30}",
                f"{step.flop_cost:>14,}",
                f"{step.dense_flop_cost:>14,}",
                f"{step.symmetry_savings:>7.1%}",
                f"{blas_label:<8}",
            ]
            if any_unique:
                row_parts.append(f"{self._fmt_unique_dense(step):<{unique_col_width}}")
            sym_str = sym_strs[i] or "-"
            if len(sym_str) > sym_col_width:
                sym_str = sym_str[: sym_col_width - 1] + "…"
            row_parts.append(f"{sym_str:<{sym_col_width}}")
            lines.append("  ".join(row_parts))

            cumulative += step.flop_cost
            if verbose:
                # Indented details row: subset, out_shape, cumulative cost.
                # Aligned under the subscript column for visual clarity.
                subset_str = self._fmt_subset(step.merged_subset)
                shape_str = (
                    "(" + ",".join(str(d) for d in step.output_shape) + ")"
                    if step.output_shape
                    else "()"
                )
                detail_parts = [
                    f"subset={subset_str}",
                    f"out_shape={shape_str}",
                    f"cumulative={cumulative:,}",
                ]
                lines.append("        " + "  ".join(detail_parts))

        return "\n".join(lines)

    def __rich__(self):
        return self._rich_renderable()

    def print(self, verbose: bool = False) -> None:
        """Print using Rich when available, otherwise plain text.

        Notes
        -----
        Builtin ``print(info)`` still goes through ``__str__`` and remains
        the plain-text fallback. This convenience method chooses the Rich
        renderer whenever Rich is importable, including the Rich verbose
        layout when ``verbose=True``.
        """
        import builtins

        try:
            from rich.console import Console
        except ImportError:
            builtins.print(self.format_table(verbose=verbose))
            return None

        if verbose:
            Console().print(self._rich_renderable(verbose=True))
        else:
            Console().print(self)
        return None

    def __str__(self) -> str:
        return self.format_table(verbose=False)

    def __repr__(self) -> str:
        return self.__str__()


def _choose_memory_arg(memory_limit: _MemoryLimit, size_list: list[int]) -> int | None:
    if memory_limit == "max_input":
        return max(size_list)

    if isinstance(memory_limit, str):
        raise ValueError(
            "memory_limit must be None, int, or the string Literal['max_input']."
        )

    if memory_limit is None:
        return None

    if memory_limit < 1:
        if memory_limit == -1:
            return None
        else:
            raise ValueError("Memory limit must be larger than 0, or -1")

    return int(memory_limit)


# Overload for contract_path(einsum_string, *operands)
@overload
def contract_path(
    subscripts: str,
    *operands: ArrayType,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
    symmetry_oracle: SubgraphSymmetryOracle | None = None,
) -> tuple[PathType, PathInfo]: ...


# Overload for contract_path(operand, indices, operand, indices, ....)
@overload
def contract_path(
    subscripts: ArrayType,
    *operands: ArrayType | Collection[int],
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
    symmetry_oracle: SubgraphSymmetryOracle | None = None,
) -> tuple[PathType, PathInfo]: ...


def contract_path(
    subscripts: Any,
    *operands: Any,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
    symmetry_oracle: SubgraphSymmetryOracle | None = None,
) -> tuple[PathType, PathInfo]:
    """Find a contraction order `path`, without performing the contraction.

    Parameters:
          subscripts: Specifies the subscripts for summation.
          *operands: These are the arrays for the operation.
          use_blas: Do you use BLAS for valid operations, may use extra memory for more intermediates.
          optimize: Choose the type of path the contraction will be optimized with.
                - if a list is given uses this as the path.
                - `'optimal'` An algorithm that explores all possible ways of
                contracting the listed tensors. Scales factorially with the number of
                terms in the contraction.
                - `'dp'` A faster (but essentially optimal) algorithm that uses
                dynamic programming to exhaustively search all contraction paths
                without outer-products.
                - `'greedy'` An cheap algorithm that heuristically chooses the best
                pairwise contraction at each step. Scales linearly in the number of
                terms in the contraction.
                - `'random-greedy'` Run a randomized version of the greedy algorithm
                32 times and pick the best path.
                - `'random-greedy-128'` Run a randomized version of the greedy
                algorithm 128 times and pick the best path.
                - `'branch-all'` An algorithm like optimal but that restricts itself
                to searching 'likely' paths. Still scales factorially.
                - `'branch-2'` An even more restricted version of 'branch-all' that
                only searches the best two options at each step. Scales exponentially
                with the number of terms in the contraction.
                - `'auto'` Choose the best of the above algorithms whilst aiming to
                keep the path finding time below 1ms.
                - `'auto-hq'` Aim for a high quality contraction, choosing the best
                of the above algorithms whilst aiming to keep the path finding time
                below 1sec.

          memory_limit: Give the upper bound of the largest intermediate tensor contract will build.
                - None or -1 means there is no limit
                - `max_input` means the limit is set as largest input tensor
                - a positive integer is taken as an explicit limit on the number of elements

                The default is None. Note that imposing a limit can make contractions
                exponentially slower to perform.

          shapes: Whether ``contract_path`` should assume arrays (the default) or array shapes have been supplied.

    Returns:
          path: The optimized einsum contraction path
          PathInfo: A printable object containing various information about the path found.

    Notes:
          The resulting path indicates which terms of the input contraction should be
          contracted first, the result of this contraction is then appended to the end of
          the contraction list.

    Examples:
          We can begin with a chain dot example. In this case, it is optimal to
          contract the b and c tensors represented by the first element of the path (1,
          2). The resulting tensor is added to the end of the contraction and the
          remaining contraction, `(0, 1)`, is then executed.

      ```python
      path_info = contract_path('ij,jk,kl->il', (2,3), (3,4), (4,5), shapes=True)
      print(path_info[0])
      #> [(1, 2), (0, 1)]
      ```
    """
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    # Track which optimizer is actually invoked, for display in PathInfo.
    # Resolved below once we know num_ops (for auto/auto-hq's inner choice).
    optimizer_used: str = ""

    # Python side parsing
    operands_ = [subscripts] + list(operands)
    input_subscripts, output_subscript, operands_prepped = parser.parse_einsum_input(
        operands_, shapes=shapes
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    input_sets = [frozenset(x) for x in input_list]
    if shapes:
        input_shapes = list(operands_prepped)
    else:
        input_shapes = [parser.get_shape(x) for x in operands_prepped]
    output_set = frozenset(output_subscript)
    indices = frozenset(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict: dict[str, int] = {}
    for tnum, term in enumerate(input_list):
        sh = input_shapes[tnum]

        if len(sh) != len(term):
            raise ValueError(
                f"Einstein sum subscript '{input_list[tnum]}' does not contain the "
                f"correct number of indices for operand {tnum}."
            )
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])

            if char in size_dict:
                # For broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        f"Size of label '{char}' for operand {tnum} ({size_dict[char]}) does not match previous "
                        f"terms ({dim})."
                    )
            else:
                size_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = [
        helpers.compute_size_by_dict(term, size_dict)
        for term in input_list + [output_subscript]
    ]
    memory_arg = _choose_memory_arg(memory_limit, size_list)

    num_ops = len(input_list)

    # Compute naive cost
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = helpers.flop_count(indices, inner_product, num_ops, size_dict)

    # Compute the path
    if optimize is False:
        path_tuple: PathType = [tuple(range(num_ops))]
        optimizer_used = "none"
    elif not isinstance(optimize, (str, paths.PathOptimizer)):
        # Custom path supplied (a list of tuples)
        path_tuple = optimize  # type: ignore
        optimizer_used = "explicit_path"
    elif num_ops <= 2:
        # Nothing to be optimized
        path_tuple = [tuple(range(num_ops))]
        optimizer_used = "trivial"
    elif isinstance(optimize, paths.PathOptimizer):
        # Custom path optimizer instance supplied
        if symmetry_oracle is not None:
            try:
                path_tuple = optimize(
                    input_sets,
                    output_set,
                    size_dict,
                    memory_arg,
                    symmetry_oracle=symmetry_oracle,
                )
            except TypeError:
                path_tuple = optimize(input_sets, output_set, size_dict, memory_arg)
        else:
            path_tuple = optimize(input_sets, output_set, size_dict, memory_arg)
        optimizer_used = type(optimize).__name__
    else:
        path_optimizer = paths.get_path_fn(optimize)
        if symmetry_oracle is not None:
            path_tuple = path_optimizer(
                input_sets,
                output_set,
                size_dict,
                memory_arg,
                symmetry_oracle=symmetry_oracle,
            )
        else:
            path_tuple = path_optimizer(input_sets, output_set, size_dict, memory_arg)
        # Resolve auto/auto-hq to the inner choice the routing made.
        if optimize == "auto":
            inner_fn = paths._AUTO_CHOICES.get(num_ops, paths.greedy)
            optimizer_used = getattr(inner_fn, "__name__", str(inner_fn))
        elif optimize == "auto-hq":
            from ._path_random import random_greedy_128

            inner_fn = paths._AUTO_HQ_CHOICES.get(num_ops, random_greedy_128)
            optimizer_used = getattr(inner_fn, "__name__", str(inner_fn))
        else:
            optimizer_used = optimize

    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []
    step_infos: list[StepInfo] = []

    # Track symmetries through contractions using the oracle
    # ssa_ids[position] gives the SSA id for that operand position in input_list
    ssa_ids: list[int] = list(range(num_ops))
    next_ssa = num_ops

    # ssa_to_subset: maps SSA id -> frozenset of original operand indices.
    # Always populated (even without an oracle) so that StepInfo.merged_subset
    # is available for display, since the subset reconstruction is cheap and
    # purely a function of the path.
    ssa_to_subset: dict[int, frozenset[int]] = {
        k: frozenset({k}) for k in range(num_ops)
    }

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path_tuple):
        # Preserve the original (path-supplied) tuple for display before
        # we sort it for the popping convention.
        original_path_tuple = tuple(contract_inds)
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        # Snapshot per-operand index sets before find_contraction mutates
        # input_sets (needed for Φ cost model's per-operand free counts).
        _pre_input_sets = [input_sets[ci] for ci in contract_inds]

        contract_tuple = helpers.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost using oracle if available
        if symmetry_oracle is not None:
            # Look up each input's symmetry from the oracle before merging.
            # This is used both for cost computation (via merged_subset →
            # result_sym) and for BLAS classification (input_groups
            # enables SYMM/SYMV/SYDT labelling in can_blas).
            step_syms: list = [None] * len(contract_inds)
            merged_subset: frozenset[int] = frozenset()
            for pos_in_step, ci in enumerate(contract_inds):
                ssa_id = ssa_ids[ci]
                subset_i = ssa_to_subset[ssa_id]
                step_syms[pos_in_step] = symmetry_oracle.sym(subset_i).output
                merged_subset = merged_subset | subset_i

            subset_sym = symmetry_oracle.sym(merged_subset)
            result_sym = subset_sym.output

            # Per-operand free index counts for Φ cost model.
            _free_counts = tuple(len(s - idx_removed) for s in _pre_input_sets)

            from whest._config import get_setting

            _use_inner = bool(get_setting("use_inner_symmetry"))

            cost = symmetric_flop_count(
                idx_contract,
                bool(idx_removed),
                len(contract_inds),
                size_dict,
                output_group=subset_sym.output,
                output_indices=out_inds,
                inner_group=subset_sym.inner,
                inner_indices=idx_removed if idx_removed else None,
                use_inner_symmetry=_use_inner,
                per_operand_free_counts=_free_counts,
            )

            # Determine whether inner symmetry was actually applied at
            # this step (for display: W✓ vs W).
            _step_inner_applied = False
            if _use_inner and subset_sym.inner is not None and idx_removed:
                _gl = (
                    set(subset_sym.inner._labels) if subset_sym.inner._labels else set()
                )
                _step_inner_applied = bool(_gl and _gl <= set(idx_removed))
        else:
            step_syms = [None] * len(contract_inds)
            result_sym = None
            cost = helpers.flop_count(
                idx_contract, bool(idx_removed), len(contract_inds), size_dict
            )

        # Dense cost is always the opt_einsum flop_count (no symmetry)
        dense_cost = helpers.flop_count(
            idx_contract, bool(idx_removed), len(contract_inds), size_dict
        )

        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shapes.pop(x) for x in contract_inds]

        # Update SSA id tracking: compute merged subset and assign new SSA id.
        # Always tracked (even without an oracle) so StepInfo.merged_subset
        # is available for display.
        new_merged_subset: frozenset[int] = frozenset()
        for ci in contract_inds:
            ssa_id = ssa_ids[ci]
            new_merged_subset = new_merged_subset | ssa_to_subset[ssa_id]
        for ci in contract_inds:
            ssa_ids.pop(ci)
        ssa_to_subset[next_ssa] = new_merged_subset
        ssa_ids.append(next_ssa)
        next_ssa += 1

        if use_blas:
            do_blas = blas.can_blas(
                tmp_inputs,
                "".join(out_inds),
                idx_removed,
                tmp_shapes,
                input_groups=step_syms if symmetry_oracle is not None else None,
            )
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path_tuple)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = parser.find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        input_shapes.append(shp_result)

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # Build StepInfo
        step_flop = cost
        step_dense = dense_cost
        savings = 1.0 - (step_flop / step_dense) if step_dense > 0 else 0.0

        step_infos.append(
            StepInfo(
                subscript=einsum_str,
                flop_cost=step_flop,
                input_shapes=list(tmp_shapes),
                output_shape=shp_result,
                input_groups=list(step_syms),
                output_group=result_sym,
                inner_group=(subset_sym.inner if symmetry_oracle is not None else None),
                dense_flop_cost=step_dense,
                symmetry_savings=savings,
                blas_type=do_blas,
                inner_applied=(
                    _step_inner_applied if symmetry_oracle is not None else False
                ),
                path_indices=original_path_tuple,
                merged_subset=new_merged_subset,
            )
        )

        # for large expressions saving the remaining terms at each step can
        # incur a large memory footprint - and also be messy to print
        if len(input_list) <= 20:
            remaining: tuple[str, ...] | None = tuple(input_list)
        else:
            remaining = None

        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    # naive_cost already computed with flop_count
    optimized_cost = sum(s.flop_cost for s in step_infos)

    path_print = PathInfo(
        path=list(path_tuple),
        steps=step_infos,
        naive_cost=naive_cost,
        optimized_cost=optimized_cost,
        largest_intermediate=max(size_list, default=1),
        speedup=naive_cost / max(optimized_cost, 1),
        input_subscripts=input_subscripts,
        output_subscript=output_subscript,
        size_dict=dict(size_dict),
        optimizer_used=optimizer_used,
        contraction_list=contraction_list,
        scale_list=scale_list,
        size_list=size_list,
        _oe_naive_cost=naive_cost,
        _oe_opt_cost=opt_cost,
    )

    return path_tuple, path_print
