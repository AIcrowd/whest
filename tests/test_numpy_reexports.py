"""Tests for numpy re-exports in mechestim.

Two categories:
1. Parametrized identity tests — proves each re-export IS the numpy counterpart
2. Functional smoke tests — proves the re-exports actually work in realistic use
"""

from __future__ import annotations

import mechestim as me
import numpy as np
import pytest

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
    """Every new mechestim export is identical to its numpy counterpart."""
    actual = getattr(me, name)
    assert actual is expected, (
        f"me.{name} is {actual!r}, expected {expected!r}"
    )


# ----- Functional: abstract dtype hierarchy -----


def test_floating_type_check():
    """me.floating correctly identifies float dtypes."""
    assert np.issubdtype(me.float32, me.floating)
    assert np.issubdtype(me.float64, me.floating)
    assert not np.issubdtype(me.int32, me.floating)


def test_integer_type_check():
    """me.integer correctly identifies integer dtypes."""
    assert np.issubdtype(me.int32, me.integer)
    assert np.issubdtype(me.uint32, me.integer)
    assert not np.issubdtype(me.float32, me.integer)


def test_number_type_check():
    """me.number is the parent of both floating and integer."""
    assert np.issubdtype(me.float32, me.number)
    assert np.issubdtype(me.int64, me.number)
    assert np.issubdtype(me.complex64, me.number)


def test_dtype_construction():
    """me.dtype constructs dtypes from strings or types."""
    assert me.dtype("float32") == me.float32
    assert me.dtype(me.int32).kind == "i"


# ----- Functional: concrete dtypes -----


def test_uint_dtypes_usable():
    """The new uint dtypes can be used to create arrays."""
    a16 = me.array([1, 2, 3], dtype=me.uint16)
    a32 = me.array([1, 2, 3], dtype=me.uint32)
    a64 = me.array([1, 2, 3], dtype=me.uint64)
    assert a16.dtype == np.uint16
    assert a32.dtype == np.uint32
    assert a64.dtype == np.uint64


# ----- Functional: dtype info -----


def test_finfo_float32():
    """me.finfo returns sensible float32 metadata."""
    info = me.finfo(me.float32)
    assert info.eps > 0
    assert info.max > 0
    assert info.tiny > 0
    assert info.bits == 32


def test_iinfo_int32():
    """me.iinfo returns sensible int32 metadata."""
    info = me.iinfo(me.int32)
    assert info.min == -(2**31)
    assert info.max == 2**31 - 1
    assert info.bits == 32
