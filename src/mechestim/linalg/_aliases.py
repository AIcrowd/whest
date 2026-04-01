# src/mechestim/linalg/_aliases.py
"""Linalg namespace aliases that delegate to top-level mechestim operations.

These functions exist in numpy.linalg as convenience aliases.
FLOP costs are handled by the delegated top-level implementations.
"""
from __future__ import annotations

import mechestim as _me


def matmul(a, b):
    """Matrix multiply (linalg namespace). Delegates to mechestim.matmul."""
    return _me.matmul(a, b)


def cross(a, b, **kwargs):
    """Cross product (linalg namespace). Delegates to mechestim.cross."""
    return _me.cross(a, b, **kwargs)


def outer(a, b):
    """Outer product (linalg namespace). Delegates to mechestim.outer."""
    return _me.outer(a, b)


def tensordot(a, b, axes=2):
    """Tensor dot product (linalg namespace). Delegates to mechestim.tensordot."""
    return _me.tensordot(a, b, axes=axes)


def vecdot(a, b, **kwargs):
    """Vector dot product (linalg namespace). Delegates to mechestim.vecdot."""
    return _me.vecdot(a, b, **kwargs)


def diagonal(a, **kwargs):
    """Diagonal (linalg namespace). Delegates to mechestim.diagonal. Cost: 0 FLOPs."""
    return _me.diagonal(a, **kwargs)


def matrix_transpose(a):
    """Transpose (linalg namespace). Delegates to mechestim.matrix_transpose. Cost: 0 FLOPs."""
    return _me.matrix_transpose(a)
