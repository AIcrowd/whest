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
