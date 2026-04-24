"""FLOP cost calculators for whest operations."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whest._symmetric import SymmetryInfo


def parse_einsum_subscripts(subscripts: str) -> tuple[list[list[str]], list[str]]:
    """Parse an einsum subscript string into input and output index lists.

    Parameters
    ----------
    subscripts : str
        Einsum subscript string (e.g., ``'ij,jk->ik'``).

    Returns
    -------
    inputs : list of list of str
        Index labels for each input operand.
    output : list of str
        Index labels for the output.
    """
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        input_part, output_part = subscripts.split("->")
        output = list(output_part)
    else:
        input_part = subscripts
        all_labels: list[str] = []
        for part in input_part.split(","):
            all_labels.extend(list(part))
        counts = Counter(all_labels)
        output = sorted(lbl for lbl, c in counts.items() if c == 1)
    inputs = [list(part) for part in input_part.split(",")]
    return inputs, output


def einsum_cost(
    subscripts: str,
    shapes: list[tuple[int, ...]],
    operand_symmetries: list[SymmetryInfo | None] | None = None,
) -> int:
    """FLOP cost of an einsum operation.

    Delegates to ``contract_path`` from opt_einsum, which uses ``flop_count``
    with ``op_factor`` (FMA = 1 FLOP; see ``_cost_model.FMA_COST``).

    Parameters
    ----------
    subscripts : str
        Einsum subscript string.
    shapes : list of tuple of int
        Shapes of the input operands.
    operand_symmetries : list of SymmetryInfo or None, optional
        Symmetry information for each input operand.

    Returns
    -------
    int
        Estimated FLOP count.
    """
    from whest._opt_einsum import contract_path

    # Convert SymmetryInfo -> PermutationGroup for oracle
    oracle = None
    if operand_symmetries and any(s is not None for s in operand_symmetries):
        from whest._einsum import _symmetry_info_to_perm_groups

        input_parts = subscripts.replace(" ", "").split("->")[0].split(",")
        output_str = subscripts.split("->")[1] if "->" in subscripts else ""

        perm_groups = [
            _symmetry_info_to_perm_groups(sym, chars)
            for sym, chars in zip(operand_symmetries, input_parts, strict=False)
        ]

        from whest._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle

        # Use sentinel objects as operands (identity-based detection not needed here)
        sentinel_operands = [object() for _ in range(len(input_parts))]
        oracle = SubgraphSymmetryOracle(
            operands=sentinel_operands,
            subscript_parts=input_parts,
            per_op_groups=perm_groups,
            output_chars=output_str,
        )

    _, path_info = contract_path(
        subscripts, *shapes, shapes=True, symmetry_oracle=oracle
    )
    return path_info.optimized_cost


def analytical_pointwise_cost(
    shape: tuple[int, ...], symmetry_info: SymmetryInfo | None = None
) -> int:
    """FLOP cost of a pointwise (element-wise) operation.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array.
    symmetry_info : SymmetryInfo or None, optional
        If provided, only unique elements are counted.

    Returns
    -------
    int
        Estimated FLOP count (one per element, or one per unique element).
    """
    if symmetry_info is not None:
        return max(symmetry_info.unique_elements, 1)
    result = 1
    for dim in shape:
        result *= dim
    return max(result, 1)


def _normalize_axis(
    axis: int | tuple[int, ...] | None, ndim: int
) -> tuple[int, ...] | None:
    """Return normalized reduction axes as a sorted tuple, or None for full reduction."""
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    normalized = tuple(sorted((a % ndim) if ndim > 0 else a for a in axis))
    return normalized


def _compute_output_unique_count(
    groups: list | None,
    input_shape: tuple[int, ...],
    reduced_axes: tuple[int, ...] | None,
) -> int:
    """Number of unique outputs after reducing ``reduced_axes`` of ``input_shape``.

    Uses ``propagate_symmetry_reduce`` to obtain the surviving output symmetry
    groups, then multiplies Burnside counts (over symmetric axes) by free-axis
    sizes (kept non-symmetric axes).
    """
    # Runtime import to avoid circular dependency at module load.
    from whest._symmetric import propagate_symmetry_reduce

    ndim = len(input_shape)
    if reduced_axes is None:
        # Full reduction → scalar output.
        return 1
    kept_axes = [d for d in range(ndim) if d not in set(reduced_axes)]
    if not kept_axes:
        return 1

    # Propagate to get the surviving output symmetry groups.
    # (Operates on input-axis numbering internally; returns output-axis numbering.)
    out_groups = None
    if groups:
        out_groups = propagate_symmetry_reduce(
            groups, ndim, reduced_axes, keepdims=False
        )

    # Output shape in output-axis numbering.
    output_shape = tuple(input_shape[d] for d in kept_axes)

    # Sum Burnside unique counts over the output groups; multiply free-dim sizes.
    accounted: set[int] = set()
    total = 1
    if out_groups:
        for group in out_groups:
            axes = group.axes
            if axes is None:
                continue
            size_dict = {i: output_shape[axes[i]] for i in range(group.degree)}
            total *= group.burnside_unique_count(size_dict)
            accounted.update(axes)
    for i, size in enumerate(output_shape):
        if i not in accounted:
            total *= size
    return total


def _compute_R_unique_count(
    groups: list | None,
    input_shape: tuple[int, ...],
    reduced_axes: tuple[int, ...] | None,
) -> int:
    """Number of unique inputs feeding one output slice.

    Only **inner-clean** sym groups (``g.axes ⊆ R``) contribute input-side
    savings — they act entirely within the reduced axes and combine equivalent
    values. Split groups (spanning both R and K) and output-only groups
    (``g.axes ⊆ K``) contribute no savings here.
    """
    ndim = len(input_shape)
    if reduced_axes is None:
        reduced_axes = tuple(range(ndim))
    reduced_set = set(reduced_axes)

    accounted: set[int] = set()
    total = 1
    if groups:
        for group in groups:
            axes = group.axes
            if axes is None:
                continue
            axes_set = set(axes)
            if not axes_set.issubset(reduced_set):
                # Not inner-clean: either output-only (g.axes ⊆ K) — doesn't
                # touch reduced axes — or split. Neither contributes to u_R.
                continue
            size_dict = {i: input_shape[axes[i]] for i in range(group.degree)}
            total *= group.burnside_unique_count(size_dict)
            accounted.update(axes)
    for d in reduced_axes:
        if d not in accounted:
            total *= input_shape[d]
    return total


def analytical_reduction_cost(
    input_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
    symmetry_info: SymmetryInfo | None = None,
) -> int:
    """FLOP cost of a reduction operation.

    The cost of a reduction is the number of accumulations performed:
    for each output, the first input value is a free copy, and the remaining
    ``u_R − 1`` values are accumulated in. Total cost:

    .. math::

        \\text{cost} = \\text{unique\\_out} \\times (u_R - 1)

    where ``unique_out`` is the number of unique outputs (accounting for
    output-side symmetry) and ``u_R`` is the number of unique inputs feeding
    one output slice (accounting for inner-clean input symmetry).

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input array.
    axis : int, tuple of int, or None, optional
        Axis or axes along which to reduce. If None, reduce over all elements.
    symmetry_info : SymmetryInfo or None, optional
        If provided, symmetry is used to count unique outputs and unique
        per-output inputs. Only inner-clean groups (g.axes ⊆ reduced axes)
        contribute per-output input savings; split groups do not.

    Returns
    -------
    int
        Estimated FLOP count. Clamped to a minimum of 1 for degenerate shapes
        (scalar, size-1 axis, empty shape) so every reduction registers at
        least 1 flop for budget tracking purposes.
    """
    ndim = len(input_shape)
    reduced_axes = _normalize_axis(axis, ndim)
    groups = symmetry_info.groups if symmetry_info is not None else None

    unique_out = _compute_output_unique_count(groups, input_shape, reduced_axes)
    u_R = _compute_R_unique_count(groups, input_shape, reduced_axes)

    cost = unique_out * (u_R - 1)
    return max(cost, 1)


# Backward-compatible internal aliases. The public weighted API lives in
# ``whest.flops`` and wraps these analytical formulas.
pointwise_cost = analytical_pointwise_cost
reduction_cost = analytical_reduction_cost


def svd_cost(m: int, n: int, k: int | None = None) -> int:
    """FLOP cost of a (truncated) SVD.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    k : int or None, optional
        Number of singular values/vectors to compute. Defaults to min(m, n).

    Returns
    -------
    int
        Estimated FLOP count: m * n * k.

    Notes
    -----
    Based on Golub-Reinsch bidiagonalization.
    """
    if k is None:
        k = min(m, n)
    return m * n * k


def _ceil_log2(n: int) -> int:
    """Return ceil(log2(n)), minimum 1.

    Parameters
    ----------
    n : int
        Input value.

    Returns
    -------
    int
        ceil(log2(n)), with a floor of 1.
    """
    if n <= 1:
        return 1
    return max(math.ceil(math.log2(n)), 1)


def sort_cost(n: int) -> int:
    """FLOP cost of comparison-based sort.

    Parameters
    ----------
    n : int
        Number of elements to sort.

    Returns
    -------
    int
        Estimated FLOP count: n * ceil(log2(n)).
    """
    if n <= 0:
        return 1
    return max(n * _ceil_log2(n), 1)


def search_cost(queries: int, sorted_size: int) -> int:
    """FLOP cost of binary search.

    Parameters
    ----------
    queries : int
        Number of search queries.
    sorted_size : int
        Size of the sorted array being searched.

    Returns
    -------
    int
        Estimated FLOP count: queries * ceil(log2(sorted_size)).
    """
    if queries <= 0:
        return 1
    return max(queries * _ceil_log2(sorted_size), 1)
