"""Counted wrappers for sorting, search, and set operations."""

from __future__ import annotations

import inspect as _inspect

import numpy as _np

from whest._docstrings import attach_docstring
from whest._flops import search_cost, sort_cost
from whest._validation import require_budget


def _sort_cost_nd(a: _np.ndarray, axis: int) -> int:
    """Total sort cost for an n-d array sorted along *axis*.

    Total cost = num_slices * sort_cost(n) where
    num_slices = numel(a) / a.shape[axis].
    """
    n = a.shape[axis]
    numel = 1
    for d in a.shape:
        numel *= d
    num_slices = numel // n if n > 0 else 1
    return max(num_slices * sort_cost(n), 1)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def sort(a, axis=-1, **kwargs):
    """Counted version of ``numpy.sort``. Cost: n*ceil(log2(n)) FLOPs per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim == 0:
        cost = 1
    elif axis is None:
        cost = sort_cost(a.size)
    else:
        ax = axis % a.ndim
        cost = _sort_cost_nd(a, ax)
    budget.deduct("sort", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.sort(a, axis=axis, **kwargs)


attach_docstring(sort, _np.sort, "counted_custom", "n*ceil(log2(n)) FLOPs per slice")
sort.__signature__ = _inspect.signature(_np.sort)


def argsort(a, axis=-1, **kwargs):
    """Counted version of ``numpy.argsort``. Cost: n*ceil(log2(n)) FLOPs per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim == 0:
        cost = 1
    elif axis is None:
        cost = sort_cost(a.size)
    else:
        ax = axis % a.ndim
        cost = _sort_cost_nd(a, ax)
    budget.deduct("argsort", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.argsort(a, axis=axis, **kwargs)


attach_docstring(
    argsort, _np.argsort, "counted_custom", "n*ceil(log2(n)) FLOPs per slice"
)
argsort.__signature__ = _inspect.signature(_np.argsort)


def lexsort(keys, axis=-1):
    """Counted version of ``numpy.lexsort``.

    Cost: k * sort_cost(n) where k = len(keys) and n = len(keys[0]).
    """
    budget = require_budget()
    # keys is a sequence of arrays; convert to list for inspection
    keys_list = list(keys)
    k = len(keys_list)
    if k == 0:
        cost = 1
    else:
        # numpy.lexsort uses the last key as primary; length is shape along axis
        first = _np.asarray(keys_list[0])
        n = first.shape[axis] if first.ndim > 0 else 1
        cost = max(k * sort_cost(n), 1)
    shapes = tuple(_np.asarray(key).shape for key in keys_list)
    budget.deduct("lexsort", flop_cost=cost, subscripts=None, shapes=shapes)
    return _np.lexsort(keys_list, axis=axis)


attach_docstring(lexsort, _np.lexsort, "counted_custom", "k*n*ceil(log2(n)) FLOPs")


def partition(a, kth, axis=-1, **kwargs):
    """Counted version of ``numpy.partition``. Cost: n * len(kth) per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    # kth can be int or sequence of ints
    kth_count = len(kth) if hasattr(kth, "__len__") else 1
    if a.ndim == 0:
        cost = 1
    else:
        ax = axis if axis is not None else -1
        ax = ax % a.ndim
        n = a.shape[ax]
        numel = 1
        for d in a.shape:
            numel *= d
        num_slices = numel // n if n > 0 else 1
        cost = max(num_slices * n * kth_count, 1)
    budget.deduct("partition", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.partition(a, kth, axis=axis, **kwargs)


attach_docstring(
    partition, _np.partition, "counted_custom", "n FLOPs per slice (linear quickselect)"
)
partition.__signature__ = _inspect.signature(_np.partition)


def argpartition(a, kth, axis=-1, **kwargs):
    """Counted version of ``numpy.argpartition``. Cost: n * len(kth) per slice."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    # kth can be int or sequence of ints
    kth_count = len(kth) if hasattr(kth, "__len__") else 1
    if a.ndim == 0:
        cost = 1
    else:
        ax = axis if axis is not None else -1
        ax = ax % a.ndim
        n = a.shape[ax]
        numel = 1
        for d in a.shape:
            numel *= d
        num_slices = numel // n if n > 0 else 1
        cost = max(num_slices * n * kth_count, 1)
    budget.deduct("argpartition", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.argpartition(a, kth, axis=axis, **kwargs)


attach_docstring(
    argpartition,
    _np.argpartition,
    "counted_custom",
    "n FLOPs per slice (linear quickselect)",
)
argpartition.__signature__ = _inspect.signature(_np.argpartition)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def searchsorted(a, v, **kwargs):
    """Counted version of ``numpy.searchsorted``.

    Cost: m * ceil(log2(n)) where m = numel(v) and n = len(a).
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    v_arr = _np.asarray(v)
    n = a.shape[0] if a.ndim > 0 else 1
    m = max(v_arr.size, 1)
    cost = search_cost(m, n)
    budget.deduct(
        "searchsorted",
        flop_cost=cost,
        subscripts=None,
        shapes=(a.shape, v_arr.shape),
    )
    return _np.searchsorted(a, v, **kwargs)


attach_docstring(
    searchsorted,
    _np.searchsorted,
    "counted_custom",
    "m*ceil(log2(n)) FLOPs (m queries into sorted array of size n)",
)
searchsorted.__signature__ = _inspect.signature(_np.searchsorted)


def digitize(x, bins, **kwargs):
    """Counted version of ``numpy.digitize``.

    Cost: n * ceil(log2(len(bins))) where n = numel(x).
    """
    budget = require_budget()
    x_arr = _np.asarray(x)
    bins_arr = _np.asarray(bins)
    n = max(x_arr.size, 1)
    cost = search_cost(n, max(len(bins_arr), 1))
    budget.deduct(
        "digitize",
        flop_cost=cost,
        subscripts=None,
        shapes=(x_arr.shape, bins_arr.shape),
    )
    return _np.digitize(x, bins, **kwargs)


attach_docstring(
    digitize,
    _np.digitize,
    "counted_custom",
    "n*ceil(log2(len(bins))) FLOPs",
)
digitize.__signature__ = _inspect.signature(_np.digitize)


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


def _unique_cost(ar):
    """Compute sort-based cost for uniqueness: n * ceil(log2(n))."""
    if not isinstance(ar, _np.ndarray):
        ar = _np.asarray(ar)
    n = max(ar.size, 1)
    return sort_cost(n)


def unique(ar, **kwargs):
    """Counted version of ``numpy.unique``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    ar_arr = _np.asarray(ar)
    cost = _unique_cost(ar_arr)
    budget.deduct("unique", flop_cost=cost, subscripts=None, shapes=(ar_arr.shape,))
    return _np.unique(ar_arr, **kwargs)


attach_docstring(unique, _np.unique, "counted_custom", "n*ceil(log2(n)) FLOPs")
unique.__signature__ = _inspect.signature(_np.unique)


def unique_all(x, /):
    """Counted version of ``numpy.unique_all``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    budget.deduct("unique_all", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,))
    return _np.unique_all(x_arr)


attach_docstring(unique_all, _np.unique_all, "counted_custom", "n*ceil(log2(n)) FLOPs")


def unique_counts(x, /):
    """Counted version of ``numpy.unique_counts``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    budget.deduct(
        "unique_counts", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)
    )
    return _np.unique_counts(x_arr)


attach_docstring(
    unique_counts, _np.unique_counts, "counted_custom", "n*ceil(log2(n)) FLOPs"
)


def unique_inverse(x, /):
    """Counted version of ``numpy.unique_inverse``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    budget.deduct(
        "unique_inverse", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)
    )
    return _np.unique_inverse(x_arr)


attach_docstring(
    unique_inverse, _np.unique_inverse, "counted_custom", "n*ceil(log2(n)) FLOPs"
)


def unique_values(x, /):
    """Counted version of ``numpy.unique_values``. Cost: n*ceil(log2(n)) FLOPs."""
    budget = require_budget()
    x_arr = _np.asarray(x)
    cost = _unique_cost(x_arr)
    budget.deduct(
        "unique_values", flop_cost=cost, subscripts=None, shapes=(x_arr.shape,)
    )
    return _np.unique_values(x_arr)


attach_docstring(
    unique_values, _np.unique_values, "counted_custom", "n*ceil(log2(n)) FLOPs"
)


# ---------------------------------------------------------------------------
# Set operations  (cost = (n+m) * ceil(log2(n+m)))
# ---------------------------------------------------------------------------


def _set_cost(ar1, ar2):
    """Compute cost for set operations on two arrays."""
    n = _np.asarray(ar1).size
    m = _np.asarray(ar2).size
    total = max(n + m, 1)
    return sort_cost(total)


def in1d(ar1, ar2, **kwargs):
    """Counted version of ``numpy.in1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    budget.deduct("in1d", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape))
    return _np.in1d(ar1, ar2, **kwargs)


attach_docstring(in1d, _np.in1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs")
in1d.__signature__ = _inspect.signature(_np.in1d)


def isin(element, test_elements, **kwargs):
    """Counted version of ``numpy.isin``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    el = _np.asarray(element)
    te = _np.asarray(test_elements)
    cost = _set_cost(el, te)
    budget.deduct("isin", flop_cost=cost, subscripts=None, shapes=(el.shape, te.shape))
    return _np.isin(element, test_elements, **kwargs)


attach_docstring(isin, _np.isin, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs")
isin.__signature__ = _inspect.signature(_np.isin)


def intersect1d(ar1, ar2, **kwargs):
    """Counted version of ``numpy.intersect1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    budget.deduct(
        "intersect1d", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    )
    return _np.intersect1d(ar1, ar2, **kwargs)


attach_docstring(
    intersect1d, _np.intersect1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs"
)
intersect1d.__signature__ = _inspect.signature(_np.intersect1d)


def union1d(ar1, ar2):
    """Counted version of ``numpy.union1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    budget.deduct(
        "union1d", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    )
    return _np.union1d(ar1, ar2)


attach_docstring(union1d, _np.union1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs")


def setdiff1d(ar1, ar2, **kwargs):
    """Counted version of ``numpy.setdiff1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    budget.deduct(
        "setdiff1d", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    )
    return _np.setdiff1d(ar1, ar2, **kwargs)


attach_docstring(
    setdiff1d, _np.setdiff1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs"
)
setdiff1d.__signature__ = _inspect.signature(_np.setdiff1d)


def setxor1d(ar1, ar2, **kwargs):
    """Counted version of ``numpy.setxor1d``. Cost: (n+m)*ceil(log2(n+m)) FLOPs."""
    budget = require_budget()
    a1 = _np.asarray(ar1)
    a2 = _np.asarray(ar2)
    cost = _set_cost(a1, a2)
    budget.deduct(
        "setxor1d", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    )
    return _np.setxor1d(ar1, ar2, **kwargs)


attach_docstring(
    setxor1d, _np.setxor1d, "counted_custom", "(n+m)*ceil(log2(n+m)) FLOPs"
)
setxor1d.__signature__ = _inspect.signature(_np.setxor1d)

import sys as _sys  # noqa: E402

from whest._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
