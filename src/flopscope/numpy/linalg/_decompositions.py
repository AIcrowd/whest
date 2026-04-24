# src/flopscope/linalg/_decompositions.py
"""Matrix decomposition wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""

from __future__ import annotations

import numpy as _np
from numpy.linalg._linalg import EighResult, EigResult, QRResult

from flopscope._docstrings import attach_docstring
from flopscope._validation import require_budget
from flopscope.numpy.linalg._solvers import _batch_size, _has_zero_dim


def cholesky_cost(n: int) -> int:
    """FLOP cost of Cholesky decomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for Cholesky decomposition.
    """
    return max(n**3, 1)


def cholesky(a, /, *, upper=False):
    """Cholesky decomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = cholesky_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.cholesky", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.cholesky(a, upper=upper)
    return result


attach_docstring(cholesky, _np.linalg.cholesky, "linalg", r"$n^3$ FLOPs")


def qr_cost(m: int, n: int) -> int:
    r"""FLOP cost of QR decomposition.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    int
        Estimated FLOP count: $m \cdot n \cdot \min(m, n)$.

    Notes
    -----
    Simplified cubic cost model for QR decomposition.
    """
    return max(m * n * min(m, n), 1)


def qr(a, mode="reduced"):
    """QR decomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    m, n = a.shape[-2], a.shape[-1]
    batch = _batch_size(a.shape)
    cost = qr_cost(m, n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct("linalg.qr", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.linalg.qr(a, mode=mode)
    if mode in ("reduced", "complete"):
        return QRResult(*result)
    return result


attach_docstring(qr, _np.linalg.qr, "linalg", r"$m \cdot n \cdot \min(m,n)$ FLOPs")


def eig_cost(n: int) -> int:
    """FLOP cost of eigendecomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for eigendecomposition.
    """
    return max(n**3, 1)


def eig(a):
    """Eigendecomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eig_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eig", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.eig(a)
    return EigResult(*result)


attach_docstring(eig, _np.linalg.eig, "linalg", r"$n^3$ FLOPs")


def eigh_cost(n: int) -> int:
    """FLOP cost of symmetric eigendecomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for symmetric eigendecomposition.
    """
    return max(n**3, 1)


def eigh(a, UPLO="L"):
    """Symmetric eigendecomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eigh_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eigh", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.eigh(a, UPLO=UPLO)
    return EighResult(_np.asarray(result.eigenvalues), _np.asarray(result.eigenvectors))


attach_docstring(eigh, _np.linalg.eigh, "linalg", r"$n^3$ FLOPs")


def eigvals_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for eigenvalue computation.
    """
    return max(n**3, 1)


def eigvals(a):
    """Eigenvalues (nonsymmetric) with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eigvals_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eigvals", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.eigvals(a)
    return result


attach_docstring(eigvals, _np.linalg.eigvals, "linalg", r"$n^3$ FLOPs")


def eigvalsh_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues of a symmetric matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for symmetric eigenvalue computation.
    """
    return max(n**3, 1)


def eigvalsh(a, UPLO="L"):
    """Eigenvalues (symmetric) with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = eigvalsh_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.eigvalsh", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.eigvalsh(a, UPLO=UPLO)
    return result


attach_docstring(eigvalsh, _np.linalg.eigvalsh, "linalg", r"$n^3$ FLOPs")


def svdvals_cost(m: int, n: int, k: int | None = None) -> int:
    """FLOP cost of computing singular values.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    k : int or None, optional
        Number of singular values to compute. Defaults to min(m, n).

    Returns
    -------
    int
        Estimated FLOP count: m * n * k.

    Notes
    -----
    Source: Golub-Reinsch bidiagonalization. Same cost model as SVD.
    """
    if k is None:
        k = min(m, n)
    return max(m * n * k, 1)


def svdvals(x, /, *, k: int | None = None):
    """Singular values with FLOP counting."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    m, n = x.shape[-2], x.shape[-1]
    batch = _batch_size(x.shape)
    if k is None:
        k = min(m, n)
    if min(m, n) > 0 and not (1 <= k <= min(m, n)):
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got k={k}")
    cost = svdvals_cost(m, n, k) * batch if not _has_zero_dim(x.shape) else 0
    with budget.deduct(
        "linalg.svdvals", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    ):
        result = _np.linalg.svdvals(x)
    if k < min(m, n):
        result = result[..., :k]
    return result


attach_docstring(svdvals, _np.linalg.svdvals, "linalg", r"$m \cdot n \cdot k$ FLOPs")
