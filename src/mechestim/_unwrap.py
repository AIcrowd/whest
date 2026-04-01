# src/mechestim/_unwrap.py
"""Unwrap wrapper with FLOP counting."""
from __future__ import annotations
import numpy as _np
from mechestim._validation import require_budget, validate_ndarray

def unwrap_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of phase unwrapping. Formula: numel(input). Source: Diff + conditional adjustment."""
    numel = 1
    for d in shape: numel *= d
    return max(numel, 1)

def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):
    budget = require_budget()
    validate_ndarray(p)
    cost = unwrap_cost(p.shape)
    budget.deduct("unwrap", flop_cost=cost, subscripts=None, shapes=(p.shape,))
    kwargs = {"axis": axis, "period": period}
    if discont is not None:
        kwargs["discont"] = discont
    return _np.unwrap(p, **kwargs)
