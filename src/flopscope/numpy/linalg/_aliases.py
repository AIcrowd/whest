# src/flopscope/linalg/_aliases.py
"""Linalg namespace aliases that delegate to top-level flopscope operations.

These functions exist in numpy.linalg as convenience aliases.
FLOP costs are handled by the delegated top-level implementations.
"""

from __future__ import annotations

import numpy as _np

import flopscope.numpy as _me
from flopscope._docstrings import attach_docstring


def matmul(x1, x2, /):
    """Matrix multiply (linalg namespace). Delegates to flopscope.matmul."""
    return _me.matmul(x1, x2)


attach_docstring(
    matmul, _np.linalg.matmul, "linalg", "0 FLOPs (delegates to flopscope.matmul)"
)


def cross(x1, x2, /, *, axis=-1):
    """Cross product (linalg namespace). Uses np.linalg.cross for strict validation."""
    import builtins as _builtins

    from flopscope._ndarray import _asflopscope
    from flopscope._validation import require_budget

    budget = require_budget()
    x1_arr = _np.asarray(x1)
    x2_arr = _np.asarray(x2)
    out_shape = _np.broadcast_shapes(x1_arr.shape, x2_arr.shape)
    out_size = 1
    for d in out_shape:
        out_size *= d
    with budget.deduct(
        "linalg.cross",
        flop_cost=_builtins.max(out_size * 3, 1),
        subscripts=None,
        shapes=(x1_arr.shape, x2_arr.shape),
    ):
        result = _np.linalg.cross(x1, x2, axis=axis)
    if isinstance(result, _np.ndarray):
        return _asflopscope(result)
    return result


attach_docstring(
    cross, _np.linalg.cross, "linalg", "0 FLOPs (delegates to flopscope.cross)"
)


def outer(x1, x2, /):
    """Outer product (linalg namespace). Delegates to flopscope.outer."""
    return _me.outer(x1, x2)


attach_docstring(
    outer, _np.linalg.outer, "linalg", "0 FLOPs (delegates to flopscope.outer)"
)


def tensordot(x1, x2, /, *, axes=2):
    """Tensor dot product (linalg namespace). Delegates to flopscope.tensordot."""
    return _me.tensordot(x1, x2, axes=axes)


attach_docstring(
    tensordot,
    _np.linalg.tensordot,
    "linalg",
    "0 FLOPs (delegates to flopscope.tensordot)",
)


if hasattr(_np.linalg, "vecdot"):

    def vecdot(x1, x2, /, *, axis=-1):
        """Vector dot product (linalg namespace). Delegates to flopscope.vecdot."""
        return _me.vecdot(x1, x2, axis=axis)

    attach_docstring(
        vecdot,
        _np.linalg.vecdot,
        "linalg",
        "0 FLOPs (delegates to flopscope.vecdot)",
    )

else:
    from flopscope.errors import UnsupportedFunctionError

    def vecdot(*args, **kwargs):
        raise UnsupportedFunctionError("linalg.vecdot", min_version="2.1")


def diagonal(x, /, *, offset=0):
    """Diagonal (linalg namespace). Delegates to flopscope.diagonal. Cost: 0 FLOPs."""
    return _me.diagonal(x, offset=offset, axis1=-2, axis2=-1)


attach_docstring(diagonal, _np.linalg.diagonal, "linalg", "0 FLOPs (free)")


def matrix_transpose(x, /):
    """Transpose (linalg namespace). Delegates to flopscope.matrix_transpose. Cost: 0 FLOPs."""
    return _me.matrix_transpose(x)


attach_docstring(
    matrix_transpose, _np.linalg.matrix_transpose, "linalg", "0 FLOPs (free)"
)
