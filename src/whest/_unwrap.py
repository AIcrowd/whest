# src/mechestim/_unwrap.py
"""Unwrap wrapper with FLOP counting."""

from __future__ import annotations

import numpy as _np

from mechestim._docstrings import attach_docstring
from mechestim._validation import require_budget


def unwrap_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of phase unwrapping.

    Parameters
    ----------
    shape : tuple of int
        Input array shape.

    Returns
    -------
    int
        Estimated FLOP count: numel(input).

    Notes
    -----
    Cost covers element-wise differencing and conditional adjustment.
    """
    numel = 1
    for d in shape:
        numel *= d
    return max(numel, 1)


def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):
    budget = require_budget()
    if not isinstance(p, _np.ndarray):
        p = _np.asarray(p)
    cost = unwrap_cost(p.shape)
    budget.deduct("unwrap", flop_cost=cost, subscripts=None, shapes=(p.shape,))
    kwargs = {"axis": axis, "period": period}
    if discont is not None:
        kwargs["discont"] = discont
    return _np.unwrap(p, **kwargs)


attach_docstring(unwrap, _np.unwrap, "counted_custom", "numel(input) FLOPs")

import sys as _sys  # noqa: E402

from mechestim._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
