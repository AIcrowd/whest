# src/whest/linalg/_aliases.py
"""Linalg namespace aliases that delegate to top-level whest operations.

These functions exist in numpy.linalg as convenience aliases.
FLOP costs are handled by the delegated top-level implementations.
"""

from __future__ import annotations

import numpy as _np

import whest as _me
from whest._docstrings import attach_docstring


def matmul(x1, x2, /):
    """Matrix multiply (linalg namespace). Delegates to whest.matmul."""
    return _me.matmul(x1, x2)


attach_docstring(
    matmul, _np.linalg.matmul, "linalg", "0 FLOPs (delegates to whest.matmul)"
)


def cross(x1, x2, /, *, axis=-1):
    """Cross product (linalg namespace). Delegates to whest.cross."""
    return _me.cross(x1, x2, axis=axis)


attach_docstring(
    cross, _np.linalg.cross, "linalg", "0 FLOPs (delegates to whest.cross)"
)


def outer(x1, x2, /):
    """Outer product (linalg namespace). Delegates to whest.outer."""
    return _me.outer(x1, x2)


attach_docstring(
    outer, _np.linalg.outer, "linalg", "0 FLOPs (delegates to whest.outer)"
)


def tensordot(x1, x2, /, *, axes=2):
    """Tensor dot product (linalg namespace). Delegates to whest.tensordot."""
    return _me.tensordot(x1, x2, axes=axes)


attach_docstring(
    tensordot,
    _np.linalg.tensordot,
    "linalg",
    "0 FLOPs (delegates to whest.tensordot)",
)


if hasattr(_np.linalg, "vecdot"):

    def vecdot(x1, x2, /, *, axis=-1):
        """Vector dot product (linalg namespace). Delegates to whest.vecdot."""
        return _me.vecdot(x1, x2, axis=axis)

    attach_docstring(
        vecdot,
        _np.linalg.vecdot,
        "linalg",
        "0 FLOPs (delegates to whest.vecdot)",
    )

else:
    from whest.errors import UnsupportedFunctionError

    def vecdot(*args, **kwargs):
        raise UnsupportedFunctionError("linalg.vecdot", min_version="2.1")


def diagonal(x, /, *, offset=0):
    """Diagonal (linalg namespace). Delegates to whest.diagonal. Cost: 0 FLOPs."""
    return _me.diagonal(x, offset=offset)


attach_docstring(diagonal, _np.linalg.diagonal, "linalg", "0 FLOPs (free)")


def matrix_transpose(x, /):
    """Transpose (linalg namespace). Delegates to whest.matrix_transpose. Cost: 0 FLOPs."""
    return _me.matrix_transpose(x)


attach_docstring(
    matrix_transpose, _np.linalg.matrix_transpose, "linalg", "0 FLOPs (free)"
)
