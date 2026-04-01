"""Einsum with analytical FLOP counting and symmetry detection."""
from __future__ import annotations

import numpy as _np

from mechestim._flops import einsum_cost
from mechestim._validation import check_nan_inf, require_budget
from mechestim.errors import SymmetryError


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


def _validate_symmetric_dims(result: _np.ndarray, symmetric_dims: list[tuple[int, ...]]) -> None:
    """Validate that the result actually has the claimed symmetry."""
    for group in symmetric_dims:
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                axes = list(range(result.ndim))
                axes[group[i]], axes[group[j]] = axes[group[j]], axes[group[i]]
                transposed = result.transpose(axes)
                if not _np.allclose(result, transposed, atol=1e-6, rtol=1e-5):
                    max_dev = float(_np.max(_np.abs(result - transposed)))
                    raise SymmetryError(dims=group, max_deviation=max_dev)


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

    cost = einsum_cost(subscripts, shapes=list(shapes), repeated_operand_indices=repeated, symmetric_dims=symmetric_dims)
    budget.deduct("einsum", flop_cost=cost, subscripts=subscripts, shapes=tuple(shapes))

    result = _np.einsum(subscripts, *operands)

    if symmetric_dims and isinstance(result, _np.ndarray) and result.ndim >= 2:
        _validate_symmetric_dims(result, symmetric_dims)

    check_nan_inf(result, "einsum")
    return result
