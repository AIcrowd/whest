"""Tests for numpy 2.3+ behavioral shims preserving pre-2.3 semantics."""

from __future__ import annotations

import numpy as np

import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
# -------------------------------------------------------------------------
# count_nonzero: axis=None must return Python int on all numpy versions
# -------------------------------------------------------------------------


def test_count_nonzero_axis_none_returns_python_int():
    arr = np.array([0, 1, 2, 0, 3])
    result = fnp.count_nonzero(arr)
    assert type(result) is int, f"expected exact `int`, got {type(result).__name__}"
    assert result == 3


def test_count_nonzero_explicit_axis_none_returns_python_int():
    arr = np.array([[0, 1], [2, 0]])
    result = fnp.count_nonzero(arr, axis=None)
    assert type(result) is int
    assert result == 2


def test_count_nonzero_with_axis_returns_ndarray():
    """Negative case: shim does not interfere when axis is not None."""
    arr = np.array([[0, 1, 2], [3, 0, 4]])
    result = fnp.count_nonzero(arr, axis=0)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 1, 2])


def test_count_nonzero_int_return_is_unconditional():
    """Coercion to int for axis=None is unconditional regardless of numpy version."""
    import flopscope._pointwise as _pointwise

    arr = np.array([0, 1, 2, 0, 3])
    result = _pointwise.count_nonzero(arr)
    assert type(result) is int
    assert result == 3


# -------------------------------------------------------------------------
# unique: string / complex input must return sorted values on all versions
# -------------------------------------------------------------------------


def test_unique_strings_returns_sorted():
    arr = np.array(["banana", "apple", "cherry", "apple"])
    result = fnp.unique(arr)
    np.testing.assert_array_equal(result, np.array(["apple", "banana", "cherry"]))


def test_unique_complex_returns_sorted():
    arr = np.array([2 + 1j, 1 + 0j, 2 + 0j, 1 + 0j], dtype=np.complex128)
    result = fnp.unique(arr)
    expected = np.sort(np.array([1 + 0j, 2 + 0j, 2 + 1j]))
    np.testing.assert_array_equal(result, expected)


def test_unique_numeric_sort_unaffected():
    """Negative case: shim is a no-op for non-string / non-complex dtypes."""
    arr = np.array([3.0, 1.0, 2.0, 1.0])
    result = fnp.unique(arr)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_unique_with_return_index_delegates_to_numpy():
    """Shim does not attempt to re-sort when auxiliary arrays are requested."""
    arr = np.array(["b", "a", "c"])
    values, idx = fnp.unique(arr, return_index=True)
    # Shape / type consistency; actual order is whatever numpy returns.
    assert values.shape == (3,)
    assert idx.shape == (3,)


def test_unique_shim_forced_for_strings(monkeypatch):
    """Force the shim's code path to verify the re-sort on numpy <2.3 too."""
    import flopscope._sorting_ops as _sorting_ops

    monkeypatch.setattr(_sorting_ops, "_NUMPY_GE_2_3", True)
    # Input order matters — fnp need something numpy would return unsorted
    # if the 2.3+ behavior were active. Since fnp're on 2.2, the raw call
    # sorts; our re-sort is a no-op. So test that the RESULT is sorted
    # either way.
    arr = np.array(["banana", "apple", "cherry"])
    result = _sorting_ops.unique(arr)
    np.testing.assert_array_equal(result, np.array(["apple", "banana", "cherry"]))
