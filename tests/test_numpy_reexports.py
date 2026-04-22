"""Tests for numpy re-exports in whest.

Two categories:
1. Parametrized identity tests — proves each re-export IS the numpy counterpart
2. Functional smoke tests — proves the re-exports actually work in realistic use
"""

from __future__ import annotations

import numpy as np
import pytest

import whest as we

# ----- Parametrized identity tests -----

NEW_EXPORTS = [
    # Abstract dtype hierarchy
    ("floating", np.floating),
    ("integer", np.integer),
    ("number", np.number),
    ("dtype", np.dtype),
    # Concrete dtypes
    ("uint16", np.uint16),
    ("uint32", np.uint32),
    ("uint64", np.uint64),
    # Dtype info
    ("finfo", np.finfo),
    ("iinfo", np.iinfo),
    # Error state
    ("errstate", np.errstate),
    ("seterr", np.seterr),
    ("geterr", np.geterr),
    # Iteration
    ("ndindex", np.ndindex),
    ("ndenumerate", np.ndenumerate),
    ("broadcast", np.broadcast),
    ("nditer", np.nditer),
    # Print options
    ("set_printoptions", np.set_printoptions),
    ("get_printoptions", np.get_printoptions),
    ("printoptions", np.printoptions),
]


@pytest.mark.parametrize("name,expected", NEW_EXPORTS)
def test_reexport_identity(name, expected):
    """Every new whest export is identical to its numpy counterpart."""
    actual = getattr(we, name)
    assert actual is expected, f"we.{name} is {actual!r}, expected {expected!r}"


# ----- Functional: abstract dtype hierarchy -----


def test_floating_type_check():
    """we.floating correctly identifies float dtypes."""
    assert np.issubdtype(we.float32, we.floating)
    assert np.issubdtype(we.float64, we.floating)
    assert not np.issubdtype(we.int32, we.floating)


def test_integer_type_check():
    """we.integer correctly identifies integer dtypes."""
    assert np.issubdtype(we.int32, we.integer)
    assert np.issubdtype(we.uint32, we.integer)
    assert not np.issubdtype(we.float32, we.integer)


def test_number_type_check():
    """we.number is the parent of both floating and integer."""
    assert np.issubdtype(we.float32, we.number)
    assert np.issubdtype(we.int64, we.number)
    assert np.issubdtype(we.complex64, we.number)


def test_dtype_construction():
    """we.dtype constructs dtypes from strings or types."""
    assert we.dtype("float32") == we.float32
    assert we.dtype(we.int32).kind == "i"


# ----- Functional: concrete dtypes -----


def test_uint_dtypes_usable():
    """The new uint dtypes can be used to create arrays."""
    a16 = we.array([1, 2, 3], dtype=we.uint16)
    a32 = we.array([1, 2, 3], dtype=we.uint32)
    a64 = we.array([1, 2, 3], dtype=we.uint64)
    assert a16.dtype == np.uint16
    assert a32.dtype == np.uint32
    assert a64.dtype == np.uint64


# ----- Functional: dtype info -----


def test_finfo_float32():
    """we.finfo returns sensible float32 metadata."""
    info = we.finfo(we.float32)
    assert info.eps > 0
    assert info.max > 0
    assert info.tiny > 0
    assert info.bits == 32


def test_iinfo_int32():
    """we.iinfo returns sensible int32 metadata."""
    info = we.iinfo(we.int32)
    assert info.min == -(2**31)
    assert info.max == 2**31 - 1
    assert info.bits == 32


# ----- Functional: error state -----


def test_errstate_suppresses_warnings():
    """we.errstate can suppress numpy warnings in a block."""
    a = we.array([1.0, 2.0, 3.0])
    b = we.array([0.0, 0.0, 0.0])
    with we.errstate(divide="ignore", invalid="ignore"):
        with we.BudgetContext(flop_budget=int(1e9)):
            result = a / b
    result_np = np.asarray(result)
    assert np.isinf(result_np[0]) or np.isnan(result_np[0])


def test_geterr_seterr_roundtrip():
    """we.geterr returns numpy's current error state, we.seterr updates it."""
    original = we.geterr()
    try:
        we.seterr(divide="ignore")
        current = we.geterr()
        assert current["divide"] == "ignore"
    finally:
        we.seterr(**original)


# ----- Functional: iteration utilities -----


def test_ndindex_iterates_shape():
    """we.ndindex iterates over all indices of a shape."""
    indices = list(we.ndindex(2, 3))
    assert len(indices) == 6
    assert indices[0] == (0, 0)
    assert indices[-1] == (1, 2)


def test_ndenumerate_yields_index_and_value():
    """we.ndenumerate yields (index, value) pairs."""
    a = we.array([[1.0, 2.0], [3.0, 4.0]])
    pairs = list(we.ndenumerate(a))
    assert len(pairs) == 4
    assert pairs[0] == ((0, 0), 1.0)


# ----- Functional: print options -----


def test_printoptions_context_manager():
    """we.printoptions temporarily changes print formatting."""
    a = we.array([1.234567891234])
    with we.printoptions(precision=2):
        short = str(a)
    with we.printoptions(precision=10):
        long = str(a)
    assert len(long) > len(short)


def test_set_get_printoptions_roundtrip():
    """we.set_printoptions changes state, we.get_printoptions returns it."""
    original = we.get_printoptions()
    try:
        we.set_printoptions(precision=3)
        current = we.get_printoptions()
        assert current["precision"] == 3
    finally:
        we.set_printoptions(**{k: v for k, v in original.items() if k != "legacy"})


# ----- Functional: we.typing submodule -----


def test_typing_submodule_has_NDArray():
    """we.typing.NDArray is available."""
    from whest.typing import ArrayLike, DTypeLike, NDArray

    assert NDArray is not None
    assert ArrayLike is not None
    assert DTypeLike is not None


def test_typing_NDArray_is_numpy_NDArray():
    """we.typing.NDArray is identical to numpy.typing.NDArray."""
    import numpy.typing as npt

    assert we.typing.NDArray is npt.NDArray


def test_typing_NDArray_accepts_whest_array():
    """A function annotated with NDArray accepts WhestArray at runtime."""
    from whest.typing import NDArray

    def f(x: NDArray) -> NDArray:
        return x * 2

    m = we.array([1.0, 2.0, 3.0])
    with we.BudgetContext(flop_budget=int(1e9)):
        result = f(m)
    assert isinstance(result, we.ndarray)


def test_no_legacy_symmetry_exports():
    for legacy_name in ("PermutationGroup", "Permutation", "Cycle", "SymmetryInfo"):
        assert not hasattr(we, legacy_name)
