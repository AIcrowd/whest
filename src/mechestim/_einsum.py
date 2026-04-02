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


def einsum(
    subscripts: str,
    *operands: _np.ndarray,
    symmetric_dims: list[tuple[int, ...]] | None = None,
) -> _np.ndarray:
    """Evaluate Einstein summation with FLOP counting.

    Wraps ``numpy.einsum`` with analytical FLOP cost computation and
    optional symmetry savings. If any input is a ``SymmetricTensor``,
    the cost is automatically reduced. If ``symmetric_dims`` is provided
    and the output passes validation, a ``SymmetricTensor`` is returned.

    FLOP Cost
    ---------
    Product of all index dimensions, divided by symmetry factors
    (repeated operands and symmetric dim groups). ``SymmetricTensor``
    inputs further reduce cost based on unique element count.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript string (e.g., ``'ij,jk->ik'``).
    *operands : numpy.ndarray
        Input arrays. ``SymmetricTensor`` inputs are detected automatically
        for cost savings.
    symmetric_dims : list of tuple of int, optional
        Output dimension symmetry groups. For example, ``[(0, 1)]`` declares
        the output is symmetric in its first two dimensions. Validated at
        runtime with ``atol=1e-6, rtol=1e-5``.

    Returns
    -------
    numpy.ndarray or SymmetricTensor
        The result of the einsum. Returns ``SymmetricTensor`` when
        ``symmetric_dims`` is provided and validation passes.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the FLOP budget.
    NoBudgetContextError
        If called outside a ``BudgetContext``.
    SymmetryError
        If ``symmetric_dims`` is provided but the result is not symmetric.

    See Also
    --------
    numpy.einsum : NumPy's einsum documentation for subscript syntax.
    """
    budget = require_budget()
    shapes = [op.shape for op in operands]
    repeated = _detect_repeated_operands(operands)

    # Extract symmetry info from SymmetricTensor inputs
    operand_symmetries = [
        op.symmetry_info if isinstance(op, SymmetricTensor) else None for op in operands
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
