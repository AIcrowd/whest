# src/mechestim/_unwrap.py
"""Unwrap wrapper with FLOP counting."""
from __future__ import annotations
import numpy as _np
from mechestim._validation import require_budget, validate_ndarray

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
    for d in shape: numel *= d
    return max(numel, 1)


def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):
    """Unwrap by taking the complement of large deltas w.r.t. the period.

    Wraps ``numpy.unwrap`` with FLOP counting.

    Parameters
    ----------
    p : numpy.ndarray
        Input array.
    discont : float, optional
        Maximum discontinuity between values. Default is ``period / 2``.
    axis : int, optional
        Axis along which unwrap will operate. Default is ``-1``.
    period : float, optional
        Size of the range over which the input wraps. Default is ``2 * pi``.

    Returns
    -------
    numpy.ndarray
        Unwrapped array.

    Notes
    -----
    **mechestim cost:** numel(input) FLOPs.

    See Also
    --------
    numpy.unwrap : Full NumPy documentation.
    """
    budget = require_budget()
    validate_ndarray(p)
    cost = unwrap_cost(p.shape)
    budget.deduct("unwrap", flop_cost=cost, subscripts=None, shapes=(p.shape,))
    kwargs = {"axis": axis, "period": period}
    if discont is not None:
        kwargs["discont"] = discont
    return _np.unwrap(p, **kwargs)
