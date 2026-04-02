# src/mechestim/linalg/_aliases.py
"""Linalg namespace aliases that delegate to top-level mechestim operations.

These functions exist in numpy.linalg as convenience aliases.
FLOP costs are handled by the delegated top-level implementations.
"""
from __future__ import annotations

import numpy as _np

import mechestim as _me
from mechestim._docstrings import attach_docstring


def matmul(a, b):
    """Matrix multiply (linalg namespace). Delegates to mechestim.matmul."""
    return _me.matmul(a, b)

attach_docstring(matmul, _np.linalg.matmul, "linalg", "0 FLOPs (delegates to mechestim.matmul)")


def cross(a, b, **kwargs):
    """Cross product (linalg namespace). Delegates to mechestim.cross."""
    return _me.cross(a, b, **kwargs)

attach_docstring(cross, _np.linalg.cross, "linalg", "0 FLOPs (delegates to mechestim.cross)")


def outer(a, b):
    """Outer product (linalg namespace). Delegates to mechestim.outer."""
    return _me.outer(a, b)

attach_docstring(outer, _np.linalg.outer, "linalg", "0 FLOPs (delegates to mechestim.outer)")


def tensordot(a, b, axes=2):
    """Tensor dot product (linalg namespace). Delegates to mechestim.tensordot."""
    return _me.tensordot(a, b, axes=axes)

attach_docstring(tensordot, _np.linalg.tensordot, "linalg", "0 FLOPs (delegates to mechestim.tensordot)")


def vecdot(a, b, **kwargs):
    """Vector dot product (linalg namespace). Delegates to mechestim.vecdot."""
    return _me.vecdot(a, b, **kwargs)

attach_docstring(vecdot, _np.linalg.vecdot, "linalg", "0 FLOPs (delegates to mechestim.vecdot)")


def diagonal(a, **kwargs):
    """Diagonal (linalg namespace). Delegates to mechestim.diagonal. Cost: 0 FLOPs."""
    return _me.diagonal(a, **kwargs)

attach_docstring(diagonal, _np.linalg.diagonal, "linalg", "0 FLOPs (free)")


def matrix_transpose(a):
    """Transpose (linalg namespace). Delegates to mechestim.matrix_transpose. Cost: 0 FLOPs."""
    return _me.matrix_transpose(a)

attach_docstring(matrix_transpose, _np.linalg.matrix_transpose, "linalg", "0 FLOPs (free)")
