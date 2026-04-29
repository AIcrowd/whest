"""FLOP cost calculators for whest operations."""

from __future__ import annotations

import math
from collections import Counter

from whest._perm_group import SymmetryGroup
from whest._symmetry_utils import unique_elements_for_shape, validate_symmetry_group


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
    operand_symmetries: list[SymmetryGroup | None] | None = None,
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
    operand_symmetries : list of SymmetryGroup or None, optional
        Exact symmetry group for each input operand.

    Returns
    -------
    int
        Estimated FLOP count.
    """
    from whest._opt_einsum import contract_path

    oracle = None
    if operand_symmetries and any(s is not None for s in operand_symmetries):
        input_parts = subscripts.replace(" ", "").split("->")[0].split(",")
        output_str = subscripts.split("->")[1] if "->" in subscripts else ""
        perm_groups = []
        for symmetry, chars, shape in zip(
            operand_symmetries, input_parts, shapes, strict=False
        ):
            if symmetry is None:
                perm_groups.append(None)
                continue
            validate_symmetry_group(symmetry, ndim=len(shape), shape=shape)
            axes = (
                symmetry.axes
                if symmetry.axes is not None
                else tuple(range(symmetry.degree))
            )
            if len(axes) < 2 or symmetry.order() <= 1:
                perm_groups.append(None)
                continue
            labeled_group = SymmetryGroup(*symmetry.generators, axes=axes)
            labeled_group._labels = tuple(chars[axis] for axis in axes)
            perm_groups.append([labeled_group])

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
    shape: tuple[int, ...], symmetry: SymmetryGroup | None = None
) -> int:
    """FLOP cost of a pointwise (element-wise) operation.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array.
    symmetry : SymmetryGroup or None, optional
        If provided, only unique elements are counted.

    Returns
    -------
    int
        Estimated FLOP count (one per element, or one per unique element).
    """
    if symmetry is not None:
        return max(unique_elements_for_shape(symmetry, shape), 1)
    result = 1
    for dim in shape:
        result *= dim
    return max(result, 1)


def analytical_reduction_cost(
    input_shape: tuple[int, ...],
    axis: int | None = None,
    symmetry: SymmetryGroup | None = None,
) -> int:
    """FLOP cost of a reduction operation.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input array.
    axis : int or None, optional
        Axis along which to reduce. If None, reduce over all elements.
    symmetry : SymmetryGroup or None, optional
        If provided, only unique elements are counted.

    Returns
    -------
    int
        Estimated FLOP count (one per element).

    Notes
    -----
    The ``axis`` parameter is accepted for API consistency but does not
    affect the result: a reduction always touches every element regardless
    of which axis is reduced, so the cost is always ``prod(input_shape)``.
    """
    if symmetry is not None:
        return max(unique_elements_for_shape(symmetry, input_shape), 1)
    result = 1
    for dim in input_shape:
        result *= dim
    return max(result, 1)


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
