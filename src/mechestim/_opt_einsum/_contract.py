"""Contains contract_path and supporting types (stripped from opt_einsum contract.py).

Excluded: contract, _core_contract, ContractExpression, _einsum, _tensordot,
_transpose, backends/sharing imports, _filter_einsum_defaults,
format_const_einsum_str, shape_only.
"""

from collections.abc import Collection
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, overload

from . import _blas as blas
from . import _helpers as helpers
from . import _parser as parser
from . import _paths as paths
from ._symmetry import IndexSymmetry, propagate_symmetry, symmetric_flop_count
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

## Common types

_MemoryLimit = None | int | Decimal | Literal["max_input"]


@dataclass
class StepInfo:
    """Per-step diagnostics for a contraction path."""

    subscript: str
    """Einsum subscript for this step, e.g. ``"ijk,ai->ajk"``."""

    flop_cost: int
    """Symmetry-aware FLOP cost (opt_einsum convention: includes op_factor)."""

    input_shapes: list[tuple[int, ...]]
    """Shapes of the input operands for this step."""

    output_shape: tuple[int, ...]
    """Shape of the output operand for this step."""

    input_symmetries: list[IndexSymmetry | None]
    """IndexSymmetry for each input in this step."""

    output_symmetry: IndexSymmetry | None
    """IndexSymmetry of the output, or None."""

    dense_flop_cost: int
    """FLOP cost without symmetry (opt_einsum convention: includes op_factor)."""

    symmetry_savings: float
    """Fraction saved: ``1 - (flop_cost / dense_flop_cost)``. Zero when no symmetry."""

    blas_type: str | bool = False
    """BLAS classification for this step (e.g. 'GEMM', 'SYMM', False)."""


@dataclass
class PathInfo:
    """Information about a contraction path with per-step symmetry diagnostics."""

    path: list[tuple[int, ...]]
    """The optimized contraction path (list of index-tuples)."""

    steps: list[StepInfo]
    """Per-step diagnostics."""

    naive_cost: int
    """Naive (single-step) FLOP cost (opt_einsum convention with op_factor)."""

    optimized_cost: int
    """Sum of per-step costs (opt_einsum convention with op_factor)."""

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

    # Legacy fields for backward-compat with opt_einsum tests
    contraction_list: ContractionListType = field(default_factory=list)
    scale_list: list[int] = field(default_factory=list)
    size_list: list[int] = field(default_factory=list)
    _oe_naive_cost: int = 0
    _oe_opt_cost: int = 0

    @property
    def opt_cost(self) -> Decimal:
        """Legacy: opt_einsum-style cost (using flop_count with op_factor)."""
        return Decimal(self._oe_opt_cost)

    @property
    def eq(self) -> str:
        return f"{self.input_subscripts}->{self.output_subscript}"

    def __str__(self) -> str:
        def fmt_sym(sym: IndexSymmetry | None) -> str:
            """Format an IndexSymmetry as e.g. 'S3{i,j,k}' or 'S2{i,j}·S2{k,l}'."""
            if not sym:
                return "-"

            def fmt_block(block: tuple) -> str:
                if len(block) == 1:
                    return block[0]
                return f"({''.join(block)})"

            def fmt_group(g: frozenset) -> str:
                blocks = sorted(g)
                return f"S{len(g)}{{{','.join(fmt_block(b) for b in blocks)}}}"

            return "·".join(fmt_group(g) for g in sym)

        def fmt_step_sym(step: StepInfo) -> str:
            """Format inputs→output symmetry transformation for one step."""
            in_parts = [fmt_sym(s) for s in step.input_symmetries]
            out_part = fmt_sym(step.output_symmetry)
            if all(p == "-" for p in in_parts) and out_part == "-":
                return ""
            return f"{' × '.join(in_parts)} → {out_part}"

        def fmt_index_sizes() -> str:
            """Format index sizes compactly. Groups indices with the same size."""
            if not self.size_dict:
                return ""
            # Group indices by size: {20: ['a','b','c'], 5: ['d']} -> "a=b=c=20, d=5"
            from collections import defaultdict

            by_size: dict[int, list[str]] = defaultdict(list)
            for idx, sz in self.size_dict.items():
                by_size[sz].append(idx)
            parts = []
            for sz, idxs in sorted(
                by_size.items(), key=lambda kv: (-len(kv[1]), -kv[0])
            ):
                idxs_sorted = sorted(idxs)
                parts.append(f"{'='.join(idxs_sorted)}={sz}")
            return ", ".join(parts)

        # Decide layout: single-line if symmetry column fits, otherwise two-line
        sym_strs = [fmt_step_sym(s) for s in self.steps]
        max_sym_width = max((len(s) for s in sym_strs), default=0)
        any_sym = any(s for s in sym_strs)
        sizes_line = fmt_index_sizes()

        header_lines = [
            f"  Complete contraction:  {self.eq}",
            f"      Naive cost (mechestim):  {self.naive_cost:,}",
            f"  Optimized cost (mechestim):  {self.optimized_cost:,}",
            f"                     Speedup:  {self.speedup:.3f}x",
            f"       Largest intermediate:  {self.largest_intermediate:,} elements",
        ]
        if sizes_line:
            header_lines.append(f"                Index sizes:  {sizes_line}")

        if not any_sym:
            # No symmetry anywhere — keep the original compact format
            width = 84
            lines = header_lines + [
                "-" * width,
                f"{'step':>4}  {'subscript':<30} {'flops':>14} {'dense_flops':>14} {'savings':>8}  {'blas':<8}",
                "-" * width,
            ]
            for i, step in enumerate(self.steps):
                blas_label = str(step.blas_type) if step.blas_type else "-"
                lines.append(
                    f"{i:>4}  {step.subscript:<30} {step.flop_cost:>14,} {step.dense_flop_cost:>14,} {step.symmetry_savings:>7.1%}  {blas_label:<8}"
                )
            return "\n".join(lines)

        # With symmetry: include a "symmetry" column
        sym_col_width = min(max(max_sym_width, len("symmetry (inputs → output)")), 60)
        width = 32 + 14 + 14 + 9 + 9 + sym_col_width + 6  # rough total
        lines = header_lines + [
            "-" * width,
            f"{'step':>4}  {'subscript':<30} {'flops':>14} {'dense_flops':>14} {'savings':>8}  {'blas':<8}  {'symmetry (inputs → output)':<{sym_col_width}}",
            "-" * width,
        ]
        for i, step in enumerate(self.steps):
            blas_label = str(step.blas_type) if step.blas_type else "-"
            sym_str = sym_strs[i] or "-"
            # Truncate if longer than column width
            if len(sym_str) > sym_col_width:
                sym_str = sym_str[: sym_col_width - 1] + "…"
            lines.append(
                f"{i:>4}  {step.subscript:<30} {step.flop_cost:>14,} {step.dense_flop_cost:>14,} {step.symmetry_savings:>7.1%}  {blas_label:<8}  {sym_str:<{sym_col_width}}"
            )
        return "\n".join(lines)

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
    input_symmetries: list[IndexSymmetry | None] | None = None,
    induced_output_symmetry: IndexSymmetry | None = None,
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
    input_symmetries: list[IndexSymmetry | None] | None = None,
    induced_output_symmetry: IndexSymmetry | None = None,
) -> tuple[PathType, PathInfo]: ...


def contract_path(
    subscripts: Any,
    *operands: Any,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
    input_symmetries: list[IndexSymmetry | None] | None = None,
    induced_output_symmetry: IndexSymmetry | None = None,
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
    elif not isinstance(optimize, (str, paths.PathOptimizer)):
        # Custom path supplied
        path_tuple = optimize  # type: ignore
    elif num_ops <= 2:
        # Nothing to be optimized
        path_tuple = [tuple(range(num_ops))]
    elif isinstance(optimize, paths.PathOptimizer):
        # Custom path optimizer supplied
        if input_symmetries is not None or induced_output_symmetry is not None:
            try:
                path_tuple = optimize(
                    input_sets,
                    output_set,
                    size_dict,
                    memory_arg,
                    input_symmetries=input_symmetries,
                    induced_output_symmetry=induced_output_symmetry,
                )
            except TypeError:
                path_tuple = optimize(input_sets, output_set, size_dict, memory_arg)
        else:
            path_tuple = optimize(input_sets, output_set, size_dict, memory_arg)
    else:
        path_optimizer = paths.get_path_fn(optimize)
        if input_symmetries is not None or induced_output_symmetry is not None:
            try:
                path_tuple = path_optimizer(
                    input_sets,
                    output_set,
                    size_dict,
                    memory_arg,
                    input_symmetries=input_symmetries,
                    induced_output_symmetry=induced_output_symmetry,
                )
            except TypeError:
                path_tuple = path_optimizer(
                    input_sets, output_set, size_dict, memory_arg
                )
        else:
            path_tuple = path_optimizer(input_sets, output_set, size_dict, memory_arg)

    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []
    step_infos: list[StepInfo] = []

    # Track symmetries through contractions if provided
    sym_list: list[IndexSymmetry | None] | None = None
    if input_symmetries is not None or induced_output_symmetry is not None:
        sym_list = (
            list(input_symmetries) if input_symmetries is not None else [None] * num_ops
        )

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path_tuple):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract_tuple = helpers.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost, scale, and size — symmetry-aware when available
        if sym_list is not None:
            # Gather symmetries for the contracted tensors (before popping)
            step_syms = [sym_list[i] for i in contract_inds]
            # Propagate symmetry pairwise through all contracted tensors
            if len(contract_inds) == 1:
                result_sym = step_syms[0]
            else:
                # Get the index sets for the contracted tensors
                step_input_sets = [frozenset(input_list[i]) for i in contract_inds]
                # Fold left: propagate pairwise, using out_inds as the surviving indices
                accum_sym = step_syms[0]
                accum_k = step_input_sets[0]
                for si in range(1, len(step_syms)):
                    k_other = step_input_sets[si]
                    accum_sym = propagate_symmetry(
                        accum_sym,
                        accum_k,
                        step_syms[si],
                        k_other,
                        out_inds,
                        induced_output_symmetry=induced_output_symmetry,
                    )
                    accum_k = accum_k | k_other
                result_sym = accum_sym

            cost = symmetric_flop_count(
                idx_contract,
                bool(idx_removed),
                len(contract_inds),
                size_dict,
                input_symmetries=step_syms,
                output_symmetry=result_sym,
                output_indices=out_inds,
            )
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
        if sym_list is not None:
            for x in contract_inds:
                sym_list.pop(x)

        if use_blas:
            do_blas = blas.can_blas(
                tmp_inputs,
                "".join(out_inds),
                idx_removed,
                tmp_shapes,
                input_symmetries=step_syms if sym_list is not None else None,
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
        if sym_list is not None:
            sym_list.append(result_sym)

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # Build StepInfo — use opt_einsum cost convention (includes op_factor)
        step_flop = cost  # already has op_factor
        step_dense = dense_cost  # already has op_factor
        savings = 1.0 - (step_flop / step_dense) if step_dense > 0 else 0.0

        step_infos.append(
            StepInfo(
                subscript=einsum_str,
                flop_cost=step_flop,
                input_shapes=list(tmp_shapes),
                output_shape=shp_result,
                input_symmetries=list(step_syms),
                output_symmetry=result_sym,
                dense_flop_cost=step_dense,
                symmetry_savings=savings,
                blas_type=do_blas,
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

    # naive_cost already computed with flop_count (includes op_factor)
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
        contraction_list=contraction_list,
        scale_list=scale_list,
        size_list=size_list,
        _oe_naive_cost=naive_cost,
        _oe_opt_cost=opt_cost,
    )

    return path_tuple, path_print
