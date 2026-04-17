"""Tests for numpy 2.3+ behavioral shims preserving pre-2.3 semantics."""

from __future__ import annotations

import numpy as np

import whest as we

# -------------------------------------------------------------------------
# count_nonzero: axis=None must return Python int on all numpy versions
# -------------------------------------------------------------------------


def test_count_nonzero_axis_none_returns_python_int():
    arr = np.array([0, 1, 2, 0, 3])
    result = we.count_nonzero(arr)
    assert type(result) is int, f"expected exact `int`, got {type(result).__name__}"
    assert result == 3


def test_count_nonzero_explicit_axis_none_returns_python_int():
    arr = np.array([[0, 1], [2, 0]])
    result = we.count_nonzero(arr, axis=None)
    assert type(result) is int
    assert result == 2


def test_count_nonzero_with_axis_returns_ndarray():
    """Negative case: shim does not interfere when axis is not None."""
    arr = np.array([[0, 1, 2], [3, 0, 4]])
    result = we.count_nonzero(arr, axis=0)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 1, 2])


def test_count_nonzero_int_return_with_shim_forced(monkeypatch):
    """Force the shim's code path even on numpy <2.3 to verify the int coercion."""
    import whest._pointwise as _pointwise

    monkeypatch.setattr(_pointwise, "_NUMPY_GE_2_3", True)
    arr = np.array([0, 1, 2, 0, 3])
    result = _pointwise.count_nonzero(arr)
    assert type(result) is int
    assert result == 3
