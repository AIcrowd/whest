# src/mechestim/linalg/_decompositions.py
"""Matrix decomposition wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""
from __future__ import annotations

import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


def cholesky_cost(n: int) -> int:
    """FLOP cost of Cholesky decomposition of an (n, n) matrix.

    Formula: n^3 / 3
    Source: Golub & Van Loan, "Matrix Computations", 4th ed., §4.2
    Assumes: Standard column-outer-product Cholesky algorithm.
    """
    return max(n ** 3 // 3, 1)


def cholesky(a):
    """Cholesky decomposition with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = cholesky_cost(n)
    budget.deduct("linalg.cholesky", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.cholesky(a)


def qr_cost(m: int, n: int) -> int:
    """FLOP cost of QR decomposition of an (m, n) matrix.

    Formula: 2*m*n^2 - (2/3)*n^3 (for m >= n)
    Source: Golub & Van Loan, "Matrix Computations", 4th ed., §5.2
    Assumes: Householder QR. For m < n, swap roles.
    """
    if m < n:
        m, n = n, m
    return max(2 * m * n ** 2 - (2 * n ** 3) // 3, 1)


def qr(a, mode="reduced"):
    """QR decomposition with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    cost = qr_cost(m, n)
    budget.deduct("linalg.qr", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.qr(a, mode=mode)


def eig_cost(n: int) -> int:
    """FLOP cost of eigendecomposition of an (n, n) matrix.

    Formula: 10 * n^3
    Source: Golub & Van Loan, "Matrix Computations", 4th ed., §7.5
    Assumes: Francis double-shift QR algorithm. The constant ~10 accounts
    for Hessenberg reduction (~10n^3/3) plus ~2 QR iterations per eigenvalue.
    This is an accepted asymptotic estimate; actual count is data-dependent.
    """
    return max(10 * n ** 3, 1)


def eig(a):
    """Eigendecomposition with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eig_cost(n)
    budget.deduct("linalg.eig", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.eig(a)


def eigh_cost(n: int) -> int:
    """FLOP cost of symmetric eigendecomposition of an (n, n) matrix.

    Formula: (4/3) * n^3
    Source: Golub & Van Loan, "Matrix Computations", 4th ed., §8.3
    Assumes: Tridiagonalization via Householder + implicit QR sweeps.
    """
    return max((4 * n ** 3) // 3, 1)


def eigh(a, UPLO="L"):
    """Symmetric eigendecomposition with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eigh_cost(n)
    budget.deduct("linalg.eigh", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    vals, vecs = _np.linalg.eigh(a, UPLO=UPLO)
    return _np.asarray(vals), _np.asarray(vecs)


def eigvals_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues of an (n, n) matrix.

    Formula: 10 * n^3
    Source: Same algorithm as eig (Francis QR).
    """
    return max(10 * n ** 3, 1)


def eigvals(a):
    """Eigenvalues (nonsymmetric) with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eigvals_cost(n)
    budget.deduct("linalg.eigvals", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.eigvals(a)


def eigvalsh_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues of a symmetric (n, n) matrix.

    Formula: (4/3) * n^3
    Source: Same algorithm as eigh.
    """
    return max((4 * n ** 3) // 3, 1)


def eigvalsh(a, UPLO="L"):
    """Eigenvalues (symmetric) with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = eigvalsh_cost(n)
    budget.deduct("linalg.eigvalsh", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.eigvalsh(a, UPLO=UPLO)


def svdvals_cost(m: int, n: int) -> int:
    """FLOP cost of computing singular values of an (m, n) matrix.

    Formula: m * n * min(m, n)
    Source: Golub-Reinsch bidiagonalization. Same as full SVD cost model.
    """
    return max(m * n * min(m, n), 1)


def svdvals(a):
    """Singular values with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    cost = svdvals_cost(m, n)
    budget.deduct("linalg.svdvals", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.svdvals(a)
