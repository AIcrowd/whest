"""Tests for numpy re-exports in flopscope.

Two categories:
1. Parametrized identity tests — proves each re-export IS the numpy counterpart
2. Functional smoke tests — proves the re-exports actually work in realistic use
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
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
    """Every new flopscope export is identical to its numpy counterpart."""
    actual = getattr(fnp, name)
    assert actual is expected, f"fnp.{name} is {actual!r}, expected {expected!r}"


# ----- Functional: abstract dtype hierarchy -----


def test_floating_type_check():
    """fnp.floating correctly identifies float dtypes."""
    assert np.issubdtype(fnp.float32, fnp.floating)
    assert np.issubdtype(fnp.float64, fnp.floating)
    assert not np.issubdtype(fnp.int32, fnp.floating)


def test_integer_type_check():
    """fnp.integer correctly identifies integer dtypes."""
    assert np.issubdtype(fnp.int32, fnp.integer)
    assert np.issubdtype(fnp.uint32, fnp.integer)
    assert not np.issubdtype(fnp.float32, fnp.integer)


def test_number_type_check():
    """fnp.number is the parent of both floating and integer."""
    assert np.issubdtype(fnp.float32, fnp.number)
    assert np.issubdtype(fnp.int64, fnp.number)
    assert np.issubdtype(fnp.complex64, fnp.number)


def test_dtype_construction():
    """fnp.dtype constructs dtypes from strings or types."""
    assert fnp.dtype("float32") == fnp.float32
    assert fnp.dtype(fnp.int32).kind == "i"


# ----- Functional: concrete dtypes -----


def test_uint_dtypes_usable():
    """The new uint dtypes can be used to create arrays."""
    a16 = fnp.array([1, 2, 3], dtype=fnp.uint16)
    a32 = fnp.array([1, 2, 3], dtype=fnp.uint32)
    a64 = fnp.array([1, 2, 3], dtype=fnp.uint64)
    assert a16.dtype == np.uint16
    assert a32.dtype == np.uint32
    assert a64.dtype == np.uint64


# ----- Functional: dtype info -----


def test_finfo_float32():
    """fnp.finfo returns sensible float32 metadata."""
    info = fnp.finfo(fnp.float32)
    assert info.eps > 0
    assert info.max > 0
    assert info.tiny > 0
    assert info.bits == 32


def test_iinfo_int32():
    """fnp.iinfo returns sensible int32 metadata."""
    info = fnp.iinfo(fnp.int32)
    assert info.min == -(2**31)
    assert info.max == 2**31 - 1
    assert info.bits == 32


# ----- Functional: error state -----


def test_errstate_suppresses_warnings():
    """fnp.errstate can suppress numpy warnings in a block."""
    a = fnp.array([1.0, 2.0, 3.0])
    b = fnp.array([0.0, 0.0, 0.0])
    with fnp.errstate(divide="ignore", invalid="ignore"):
        with flops.BudgetContext(flop_budget=int(1e9)):
            result = a / b
    result_np = np.asarray(result)
    assert np.isinf(result_np[0]) or np.isnan(result_np[0])


def test_geterr_seterr_roundtrip():
    """fnp.geterr returns numpy's current error state, fnp.seterr updates it."""
    original = fnp.geterr()
    try:
        fnp.seterr(divide="ignore")
        current = fnp.geterr()
        assert current["divide"] == "ignore"
    finally:
        fnp.seterr(**original)


# ----- Functional: iteration utilities -----


def test_ndindex_iterates_shape():
    """fnp.ndindex iterates over all indices of a shape."""
    indices = list(fnp.ndindex(2, 3))
    assert len(indices) == 6
    assert indices[0] == (0, 0)
    assert indices[-1] == (1, 2)


def test_ndenumerate_yields_index_and_value():
    """fnp.ndenumerate yields (index, value) pairs."""
    a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    pairs = list(fnp.ndenumerate(a))
    assert len(pairs) == 4
    assert pairs[0] == ((0, 0), 1.0)


# ----- Functional: print options -----


def test_printoptions_context_manager():
    """fnp.printoptions temporarily changes print formatting."""
    a = fnp.array([1.234567891234])
    with fnp.printoptions(precision=2):
        short = str(a)
    with fnp.printoptions(precision=10):
        long = str(a)
    assert len(long) > len(short)


def test_set_get_printoptions_roundtrip():
    """fnp.set_printoptions changes state, fnp.get_printoptions returns it."""
    original = fnp.get_printoptions()
    try:
        fnp.set_printoptions(precision=3)
        current = fnp.get_printoptions()
        assert current["precision"] == 3
    finally:
        fnp.set_printoptions(**{k: v for k, v in original.items() if k != "legacy"})


# ----- Functional: fnp.typing submodule -----


def test_typing_submodule_has_NDArray():
    """fnp.typing.NDArray is available."""
    from flopscope.numpy.typing import ArrayLike, DTypeLike, NDArray

    assert NDArray is not None
    assert ArrayLike is not None
    assert DTypeLike is not None


def test_typing_NDArray_is_numpy_NDArray():
    """fnp.typing.NDArray is identical to numpy.typing.NDArray."""
    import numpy.typing as npt

    assert fnp.typing.NDArray is npt.NDArray


def test_typing_NDArray_accepts_flopscope_array():
    """A function annotated with NDArray accepts FlopscopeArray at runtime."""
    from flopscope.numpy.typing import NDArray

    def f(x: NDArray) -> NDArray:
        return x * 2

    m = fnp.array([1.0, 2.0, 3.0])
    with flops.BudgetContext(flop_budget=int(1e9)):
        result = f(m)
    assert isinstance(result, fnp.ndarray)
