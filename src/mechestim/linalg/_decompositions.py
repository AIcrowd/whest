# src/mechestim/linalg/_decompositions.py
"""Matrix decomposition wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""

from __future__ import annotations

import numpy as _np

from mechestim._cost_model import FMA_COST
from mechestim._docstrings import attach_docstring
from mechestim._validation import require_budget


def cholesky_cost(n: int) -> int:
    """FLOP cost of Cholesky decomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $n^3 / 3$.

    Notes
    -----
    Source: Golub & Van Loan, *Matrix Computations*, 4th ed., §4.2.
    Assumes standard column-outer-product Cholesky algorithm.
    """
    return max(n**3 // 3, 1)


def cholesky(a):
    """Cholesky decomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = cholesky_cost(n)
    budget.deduct("linalg.cholesky", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.cholesky(a)


attach_docstring(cholesky, _np.linalg.cholesky, "linalg", r"$n^3/3$ FLOPs")


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
        Estimated FLOP count: $mn^2 - n^3/3$ (FMA = 1 op) (for $m \geq n$).

    Notes
    -----
    Source: Golub & Van Loan, *Matrix Computations*, 4th ed., §5.2.
    Assumes Householder QR. For m < n, roles are swapped.
    """
    if m < n:
        m, n = n, m
    return max(FMA_COST * (m * n**2 - n**3 // 3), 1)


def qr(a, mode="reduced"):
    """QR decomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    cost = qr_cost(m, n)
    budget.deduct("linalg.qr", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.qr(a, mode=mode)


attach_docstring(qr, _np.linalg.qr, "linalg", r"$mn^2 - n^3/3$ FLOPs (Householder, FMA=1)")


def eig_cost(n: int) -> int:
    """FLOP cost of eigendecomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $10n^3$.

    Notes
    -----
    Source: Golub & Van Loan, *Matrix Computations*, 4th ed., §7.5.
    Assumes Francis double-shift QR algorithm. The constant ~10 accounts
    for Hessenberg reduction (~$10n^3/3$) plus ~2 QR iterations per
    eigenvalue. This is an accepted asymptotic estimate; actual count
    is data-dependent.
    """
    return max(10 * n**3, 1)


def eig(a):
    """Eigendecomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eig_cost(n)
    budget.deduct("linalg.eig", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.eig(a)


attach_docstring(eig, _np.linalg.eig, "linalg", r"$10n^3$ FLOPs (Francis QR)")


def eigh_cost(n: int) -> int:
    """FLOP cost of symmetric eigendecomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $(4/3)n^3$.

    Notes
    -----
    Source: Golub & Van Loan, *Matrix Computations*, 4th ed., §8.3.
    Assumes tridiagonalization via Householder followed by implicit
    QR sweeps.
    """
    return max((4 * n**3) // 3, 1)


def eigh(a, UPLO="L"):
    """Symmetric eigendecomposition with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eigh_cost(n)
    budget.deduct("linalg.eigh", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    vals, vecs = _np.linalg.eigh(a, UPLO=UPLO)
    return _np.asarray(vals), _np.asarray(vecs)


attach_docstring(
    eigh, _np.linalg.eigh, "linalg", r"$4n^3/3$ FLOPs (tridiagonal + QR sweeps)"
)


def eigvals_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $7n^3$.

    Notes
    -----
    Same algorithm as ``eig`` (Francis QR), but eigenvectors are not
    accumulated. Without back-accumulation of eigenvectors the cost
    is dominated by Hessenberg reduction (~(10/3)n^3) plus QR iterations
    (~4n^3), totalling ~7n^3.
    """
    return max(7 * n**3, 1)


def eigvals(a):
    """Eigenvalues (nonsymmetric) with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eigvals_cost(n)
    budget.deduct("linalg.eigvals", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.eigvals(a)


attach_docstring(eigvals, _np.linalg.eigvals, "linalg", r"$7n^3$ FLOPs")


def eigvalsh_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues of a symmetric matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Estimated FLOP count: $(4/3)n^3$.

    Notes
    -----
    Same algorithm as ``eigh``, but eigenvectors are not accumulated.
    """
    return max((4 * n**3) // 3, 1)


def eigvalsh(a, UPLO="L"):
    """Eigenvalues (symmetric) with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eigvalsh_cost(n)
    budget.deduct("linalg.eigvalsh", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.eigvalsh(a, UPLO=UPLO)


attach_docstring(eigvalsh, _np.linalg.eigvalsh, "linalg", r"$4n^3/3$ FLOPs")


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


def svdvals(a, k: int | None = None):
    """Singular values with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    if k is None:
        k = min(m, n)
    if not (1 <= k <= min(m, n)):
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got k={k}")
    cost = svdvals_cost(m, n, k)
    budget.deduct("linalg.svdvals", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.svdvals(a)[:k]


attach_docstring(svdvals, _np.linalg.svdvals, "linalg", r"$m \cdot n \cdot k$ FLOPs")
