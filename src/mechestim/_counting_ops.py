"""Counted wrappers for trace, histogram, and generation operations.

These are operations that look "free" but involve genuine computation.
Each function charges a FLOP cost to the active budget.
"""

from __future__ import annotations

import builtins as _builtins

import numpy as _np

from mechestim._docstrings import attach_docstring
from mechestim._flops import _ceil_log2
from mechestim._validation import require_budget

# ---------------------------------------------------------------------------
# Reductions disguised as free
# ---------------------------------------------------------------------------


def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    budget = require_budget()
    a = _np.asarray(a)
    cost = _builtins.max(_builtins.min(a.shape[axis1], a.shape[axis2]), 1)
    budget.deduct("trace", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


attach_docstring(
    trace, _np.trace, "counted_custom", "min(a.shape[axis1], a.shape[axis2]) FLOPs"
)


def allclose(a, b, **kwargs):
    budget = require_budget()
    a = _np.asarray(a)
    b = _np.asarray(b)
    out_shape = _np.broadcast_shapes(a.shape, b.shape)
    numel = 1
    for d in out_shape:
        numel *= d
    cost = _builtins.max(numel, 1)
    budget.deduct(
        "allclose", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    )
    return _np.allclose(a, b, **kwargs)


attach_docstring(allclose, _np.allclose, "counted_custom", "numel(a) FLOPs")


def array_equal(a, b, **kwargs):
    budget = require_budget()
    a = _np.asarray(a)
    b = _np.asarray(b)
    # array_equal does not broadcast; returns False on shape mismatch
    cost = _builtins.max(a.size, b.size, 1)
    budget.deduct(
        "array_equal", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    )
    return _np.array_equal(a, b, **kwargs)


attach_docstring(array_equal, _np.array_equal, "counted_custom", "numel(a) FLOPs")


def array_equiv(a, b):
    budget = require_budget()
    a = _np.asarray(a)
    b = _np.asarray(b)
    # array_equiv broadcasts; returns False if shapes are incompatible
    try:
        out_shape = _np.broadcast_shapes(a.shape, b.shape)
        numel = 1
        for d in out_shape:
            numel *= d
        cost = _builtins.max(numel, 1)
    except ValueError:
        cost = _builtins.max(a.size, b.size, 1)
    budget.deduct(
        "array_equiv", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)
    )
    return _np.array_equiv(a, b)


attach_docstring(array_equiv, _np.array_equiv, "counted_custom", "numel(a) FLOPs")


# ---------------------------------------------------------------------------
# Histogram & counting
# ---------------------------------------------------------------------------


def histogram(a, bins=10, **kwargs):
    budget = require_budget()
    a = _np.asarray(a)
    n = a.size
    if isinstance(bins, _builtins.int):
        cost = _builtins.max(n * _ceil_log2(bins), 1)
    elif isinstance(bins, _builtins.str):
        cost = _builtins.max(n, 1)
    else:
        bins_arr = _np.asarray(bins)
        cost = _builtins.max(n * _ceil_log2(_builtins.len(bins_arr)), 1)
    budget.deduct("histogram", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.histogram(a, bins=bins, **kwargs)


attach_docstring(
    histogram,
    _np.histogram,
    "counted_custom",
    "n * ceil(log2(bins)) FLOPs when bins is int; n FLOPs otherwise",
)


def histogram2d(x, y, bins=10, **kwargs):
    budget = require_budget()
    x = _np.asarray(x)
    y = _np.asarray(y)
    n = x.size

    # Determine bx, by
    if isinstance(bins, _builtins.int):
        cost = _builtins.max(n * (_ceil_log2(bins) + _ceil_log2(bins)), 1)
    elif (
        isinstance(bins, (_builtins.list, tuple))
        and _builtins.len(bins) == 2
        and isinstance(bins[0], _builtins.int)
        and isinstance(bins[1], _builtins.int)
    ):
        bx, by = bins[0], bins[1]
        cost = _builtins.max(n * (_ceil_log2(bx) + _ceil_log2(by)), 1)
    elif isinstance(bins, (_builtins.list, tuple)) and _builtins.len(bins) == 2:
        b0 = _np.asarray(bins[0])
        b1 = _np.asarray(bins[1])
        if b0.ndim >= 1 and b1.ndim >= 1:
            cost = _builtins.max(
                n * (_ceil_log2(_builtins.len(b0)) + _ceil_log2(_builtins.len(b1))), 1
            )
        else:
            cost = _builtins.max(n, 1)
    else:
        cost = _builtins.max(n, 1)

    budget.deduct(
        "histogram2d", flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
    )
    return _np.histogram2d(x, y, bins=bins, **kwargs)


attach_docstring(
    histogram2d,
    _np.histogram2d,
    "counted_custom",
    "n * (ceil(log2(bx)) + ceil(log2(by))) FLOPs when bins is int pair; n FLOPs otherwise",
)


def histogramdd(sample, bins=10, **kwargs):
    budget = require_budget()
    sample = _np.asarray(sample)
    # sample shape: (n, d) or (n,) for 1-d
    if sample.ndim == 1:
        n = sample.shape[0]
        d = 1
    else:
        n, d = sample.shape[0], sample.shape[1]

    if isinstance(bins, _builtins.int):
        cost = _builtins.max(n * d * _ceil_log2(bins), 1)
    elif isinstance(bins, (_builtins.list, tuple)):
        total_log = 0
        for b in bins:
            if isinstance(b, _builtins.int):
                total_log += _ceil_log2(b)
            else:
                b_arr = _np.asarray(b)
                if b_arr.ndim >= 1 and b_arr.size > 0:
                    total_log += _ceil_log2(_builtins.len(b_arr))
                else:
                    total_log += 1
        cost = _builtins.max(n * total_log, 1)
    else:
        cost = _builtins.max(n, 1)

    budget.deduct(
        "histogramdd", flop_cost=cost, subscripts=None, shapes=(sample.shape,)
    )
    return _np.histogramdd(sample, bins=bins, **kwargs)


attach_docstring(
    histogramdd,
    _np.histogramdd,
    "counted_custom",
    "n * d * ceil(log2(bins)) FLOPs when bins is int; n FLOPs otherwise",
)


def histogram_bin_edges(a, bins=10, **kwargs):
    budget = require_budget()
    a = _np.asarray(a)
    cost = _builtins.max(a.size, 1)
    budget.deduct(
        "histogram_bin_edges", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    )
    return _np.histogram_bin_edges(a, bins=bins, **kwargs)


attach_docstring(
    histogram_bin_edges, _np.histogram_bin_edges, "counted_custom", "numel(a) FLOPs"
)


def bincount(x, **kwargs):
    budget = require_budget()
    x = _np.asarray(x)
    cost = _builtins.max(x.size, 1)
    budget.deduct("bincount", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.bincount(x, **kwargs)


attach_docstring(bincount, _np.bincount, "counted_custom", "numel(x) FLOPs")


# ---------------------------------------------------------------------------
# Array generation with arithmetic
# ---------------------------------------------------------------------------


def logspace(start, stop, num=50, **kwargs):
    budget = require_budget()
    cost = _builtins.max(num, 1)
    budget.deduct("logspace", flop_cost=cost, subscripts=None, shapes=((num,),))
    return _np.logspace(start, stop, num=num, **kwargs)


attach_docstring(logspace, _np.logspace, "counted_custom", "num FLOPs")


def geomspace(start, stop, num=50, **kwargs):
    budget = require_budget()
    cost = _builtins.max(num, 1)
    budget.deduct("geomspace", flop_cost=cost, subscripts=None, shapes=((num,),))
    return _np.geomspace(start, stop, num=num, **kwargs)


attach_docstring(geomspace, _np.geomspace, "counted_custom", "num FLOPs")


def vander(x, N=None, **kwargs):
    budget = require_budget()
    x = _np.asarray(x)
    n = _builtins.len(x)
    if N is None:
        N = n
    cost = _builtins.max(n * (N - 1), 1)
    budget.deduct("vander", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.vander(x, N=N, **kwargs)


attach_docstring(vander, _np.vander, "counted_custom", "len(x) * (N-1) FLOPs")

# ---------------------------------------------------------------------------
# Apply & piecewise (formerly blacklisted)
# ---------------------------------------------------------------------------


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Counted version of np.apply_along_axis. Cost: numel(output)."""
    budget = require_budget()
    if not isinstance(arr, _np.ndarray):
        arr = _np.asarray(arr)
    result = _np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct(
        "apply_along_axis", flop_cost=cost, subscripts=None, shapes=(arr.shape,)
    )
    return result


attach_docstring(
    apply_along_axis,
    _np.apply_along_axis,
    "counted_custom",
    "numel(output) FLOPs",
)


def apply_over_axes(func, a, axes):
    """Counted version of np.apply_over_axes. Cost: numel(output)."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    result = _np.apply_over_axes(func, a, axes)
    cost = result.size if hasattr(result, "size") else 1
    budget.deduct("apply_over_axes", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return result


attach_docstring(
    apply_over_axes,
    _np.apply_over_axes,
    "counted_custom",
    "numel(output) FLOPs",
)


def piecewise(x, condlist, funclist, *args, **kw):
    """Counted version of np.piecewise. Cost: numel(input)."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    result = _np.piecewise(x, condlist, funclist, *args, **kw)
    cost = x.size
    budget.deduct("piecewise", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return result


attach_docstring(
    piecewise,
    _np.piecewise,
    "counted_custom",
    "numel(input) FLOPs",
)


import sys as _sys  # noqa: E402

from mechestim._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
