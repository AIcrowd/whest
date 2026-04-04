"""Einsum with analytical FLOP counting, symmetry detection, and path optimization."""

from __future__ import annotations

import numpy as _np

from mechestim._symmetric import SymmetricTensor, validate_symmetry
from mechestim._validation import check_nan_inf, require_budget


def _symmetry_info_to_index_symmetry(sym_info, subscript_chars: str):
    """Convert SymmetryInfo (positional axes) to IndexSymmetry (char labels).

    sym_info.symmetric_axes is like [(0, 1, 2)] -- positional indices.
    subscript_chars is like "ijk" -- the einsum subscript for this operand.
    Returns IndexSymmetry like [frozenset("ijk")], or None.
    """
    if sym_info is None:
        return None
    groups = []
    for group in sym_info.symmetric_axes:
        char_group = frozenset(subscript_chars[d] for d in group)
        if len(char_group) >= 2:
            groups.append(char_group)
    return groups if groups else None


def _execute_pairwise(path_info, operands: list):
    """Execute pairwise contractions according to the optimized path."""
    ops = list(operands)
    for contract_inds, step in zip(path_info.path, path_info.steps):
        # Pop operands in reverse sorted order (same as opt_einsum convention)
        inds = sorted(contract_inds, reverse=True)
        tensors = [ops.pop(i) for i in inds]
        result = _np.einsum(step.subscript, *tensors)
        ops.append(result)
    return ops[0]


def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    optimize: str | bool = "auto",
    symmetric_axes: list[tuple[int, ...]] | None = None,
    **kwargs,
) -> _np.ndarray:
    """Evaluate Einstein summation with FLOP counting and optional path optimization.

    Wraps ``numpy.einsum`` with analytical FLOP cost computation and
    optional symmetry savings. If any input is a ``SymmetricTensor``,
    the cost is automatically reduced. If ``symmetric_axes`` is provided
    and the output passes validation, a ``SymmetricTensor`` is returned.

    All contractions go through opt_einsum's ``contract_path`` to find an
    optimal pairwise decomposition. The FLOP cost uses opt_einsum's cost
    model which includes ``op_factor`` (multiply-add = 2 FLOPs for inner
    products).

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g., ``'ij,jk->ik'``).
    *operands : numpy.ndarray
        Input arrays. ``SymmetricTensor`` inputs are detected automatically
        for cost savings.
    optimize : str or bool, optional
        Contraction path strategy. Default ``'auto'``. ``False`` is treated
        as ``'auto'`` (all einsums go through contract_path).
    symmetric_axes : list of tuple of int, optional
        Output dimension symmetry groups. For example, ``[(0, 1)]`` declares
        the output is symmetric in its first two dimensions.

    Returns
    -------
    numpy.ndarray or SymmetricTensor
        The result of the einsum.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the FLOP budget.
    NoBudgetContextError
        If called outside a ``BudgetContext``.
    SymmetryError
        If ``symmetric_axes`` is provided but the result is not symmetric.
    """
    budget = require_budget()
    shapes = [op.shape for op in operands]

    # Extract symmetry info from SymmetricTensor inputs
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
    ]

    # Convert SymmetryInfo -> IndexSymmetry for the path optimizer
    input_parts = subscripts.split("->")[0].split(",")
    index_symmetries = [
        _symmetry_info_to_index_symmetry(s, chars)
        for s, chars in zip(operand_symmetries, input_parts)
    ]
    has_symmetry = any(s is not None for s in index_symmetries)

    from mechestim._opt_einsum import contract_path as _contract_path

    path, path_info = _contract_path(
        subscripts,
        *shapes,
        shapes=True,
        optimize=optimize if optimize is not False else "auto",
        input_symmetries=index_symmetries if has_symmetry else None,
    )

    budget.deduct(
        "einsum",
        flop_cost=path_info.optimized_cost,
        subscripts=subscripts,
        shapes=tuple(shapes),
    )

    # Execute pairwise steps
    result = _execute_pairwise(path_info, list(operands))

    # Handle output symmetry wrapping
    if symmetric_axes and isinstance(result, _np.ndarray) and result.ndim >= 2:
        validate_symmetry(result, symmetric_axes)
        result = SymmetricTensor(result, symmetric_axes=symmetric_axes)

    check_nan_inf(result, "einsum")
    return result


def einsum_path(subscripts: str, *operands, optimize: str | bool = "auto"):
    """Compute the optimal contraction path without executing.

    Returns ``(path, PathInfo)`` with zero budget cost.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string.
    *operands : numpy.ndarray
        Input arrays.
    optimize : str or bool, optional
        Path optimization strategy. Default ``'auto'``.

    Returns
    -------
    path : list of tuple of int
        The contraction path.
    info : PathInfo
        Diagnostics including per-step costs and symmetry savings.
    """
    shapes = [op.shape for op in operands]
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
    ]
    input_parts = subscripts.split("->")[0].split(",")
    index_symmetries = [
        _symmetry_info_to_index_symmetry(s, chars)
        for s, chars in zip(operand_symmetries, input_parts)
    ]
    has_symmetry = any(s is not None for s in index_symmetries)

    from mechestim._opt_einsum import contract_path as _contract_path

    path, path_info = _contract_path(
        subscripts,
        *shapes,
        shapes=True,
        optimize=optimize if optimize is not False else "auto",
        input_symmetries=index_symmetries if has_symmetry else None,
    )
    return list(path), path_info
