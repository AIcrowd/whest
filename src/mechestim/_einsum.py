"""Einsum with analytical FLOP counting and symmetry detection."""
from __future__ import annotations

import numpy as _np

from mechestim._flops import einsum_cost
from mechestim._symmetric import SymmetricTensor, validate_symmetry
from mechestim._validation import check_nan_inf, require_budget


def _detect_repeated_operands(operands: tuple) -> list[int] | None:
    """Detect operands that are the same Python object (via id()).
    Returns indices of the largest repeated group, or None."""
    seen: dict[int, list[int]] = {}
    for i, op in enumerate(operands):
        obj_id = id(op)
        if obj_id not in seen:
            seen[obj_id] = []
        seen[obj_id].append(i)
    for indices in seen.values():
        if len(indices) >= 2:
            return indices
    return None


def einsum(subscripts: str, *operands: _np.ndarray, symmetric_dims: list[tuple[int, ...]] | None = None) -> _np.ndarray:
    """Evaluate Einstein summation with FLOP counting.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string.
    *operands : numpy.ndarray
        Input arrays.
    symmetric_dims : list of tuple of int, optional
        Output dimension symmetry groups for FLOP savings. Validated at runtime.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    BudgetExhaustedError, NoBudgetContextError, SymmetryError
    """
    budget = require_budget()
    shapes = [op.shape for op in operands]
    repeated = _detect_repeated_operands(operands)

    # Extract symmetry info from SymmetricTensor inputs
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None
        for op in operands
    ]
    has_operand_symmetry = any(s is not None for s in operand_symmetries)

    cost = einsum_cost(
        subscripts,
        shapes=list(shapes),
        repeated_operand_indices=repeated,
        symmetric_dims=symmetric_dims,
        operand_symmetries=operand_symmetries if has_operand_symmetry else None,
    )
    budget.deduct("einsum", flop_cost=cost, subscripts=subscripts, shapes=tuple(shapes))

    result = _np.einsum(subscripts, *operands)

    if symmetric_dims and isinstance(result, _np.ndarray) and result.ndim >= 2:
        validate_symmetry(result, symmetric_dims)
        result = SymmetricTensor(result, symmetric_dims=symmetric_dims)

    check_nan_inf(result, "einsum")
    return result
