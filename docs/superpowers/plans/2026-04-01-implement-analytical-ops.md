# Unblacklist Analytically-Calculable Operations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move 64 blacklisted numpy functions with known analytical FLOP costs from `blacklisted` to `counted_custom` or `free`, implementing wrappers with documented cost formulas.

**Architecture:** Each module (`linalg/`, `fft/`, top-level) gets domain-grouped files containing co-located cost functions and wrappers. Cost functions are pure `(shape params) -> int` functions with docstrings citing their formula and source. Wrappers follow the pattern: `require_budget()` → `validate_ndarray()` → compute cost → `budget.deduct()` → delegate to numpy → return. No `check_nan_inf` — we rely on native numpy behavior.

**Tech Stack:** Python 3.10+, NumPy 2.1.3 (pinned), pytest

**Spec:** `docs/superpowers/specs/2026-04-01-unblacklist-analytical-ops-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `src/mechestim/linalg/_aliases.py` | 7 free/delegation wrappers for linalg namespace aliases |
| `src/mechestim/linalg/_decompositions.py` | 7 counted wrappers + cost functions for matrix decompositions |
| `src/mechestim/linalg/_solvers.py` | 6 counted wrappers + cost functions for linear solvers |
| `src/mechestim/linalg/_properties.py` | 8 counted wrappers + cost functions for matrix properties |
| `src/mechestim/linalg/_compound.py` | 2 counted wrappers + cost functions for multi_dot, matrix_power |
| `src/mechestim/fft/_transforms.py` | 14 counted wrappers + cost functions for FFT operations |
| `src/mechestim/fft/_free.py` | 4 free wrappers for fftfreq, fftshift, etc. |
| `src/mechestim/_polynomial.py` | 10 counted wrappers + cost functions for legacy polynomial ops |
| `src/mechestim/_window.py` | 5 counted wrappers + cost functions for window functions |
| `src/mechestim/_unwrap.py` | 1 counted wrapper + cost function for unwrap |
| `tests/test_linalg_aliases.py` | Tests for linalg alias delegation |
| `tests/test_linalg_decompositions.py` | Tests for linalg decompositions |
| `tests/test_linalg_solvers.py` | Tests for linalg solvers |
| `tests/test_linalg_properties.py` | Tests for linalg properties |
| `tests/test_linalg_compound.py` | Tests for multi_dot, matrix_power |
| `tests/test_fft_transforms.py` | Tests for FFT transforms |
| `tests/test_fft_free.py` | Tests for FFT free ops |
| `tests/test_polynomial.py` | Tests for polynomial ops |
| `tests/test_window.py` | Tests for window functions |
| `tests/test_unwrap.py` | Tests for unwrap |

### Modified files

| File | Changes |
|---|---|
| `src/mechestim/linalg/__init__.py` | Add imports for all new linalg ops, update `__all__` |
| `src/mechestim/fft/__init__.py` | Add imports for all FFT ops, add `__all__` |
| `src/mechestim/__init__.py` | Add imports for polynomial, window, unwrap ops |
| `src/mechestim/flops.py` | Re-export all new cost functions |
| `src/mechestim/_registry.py` | Update 64 entries: category + notes with formulas |

---

## Task 1: Linalg Aliases

**Files:**
- Create: `src/mechestim/linalg/_aliases.py`
- Create: `tests/test_linalg_aliases.py`
- Modify: `src/mechestim/linalg/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_linalg_aliases.py
"""Tests for linalg namespace aliases that delegate to top-level mechestim ops."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestLinalgMatmul:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(4, 3)
        B = numpy.random.randn(3, 5)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import matmul
            result = matmul(A, B)
            assert numpy.allclose(result, numpy.matmul(A, B))

    def test_cost_deducted(self):
        A = numpy.random.randn(4, 3)
        B = numpy.random.randn(3, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import matmul
            matmul(A, B)
            assert budget.flops_used > 0

    def test_outside_context_raises(self):
        from mechestim.linalg import matmul
        with pytest.raises(NoBudgetContextError):
            matmul(numpy.ones((2, 2)), numpy.ones((2, 2)))


class TestLinalgCross:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import cross
            result = cross(a, b)
            assert numpy.allclose(result, numpy.cross(a, b))


class TestLinalgOuter:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import outer
            result = outer(a, b)
            assert numpy.allclose(result, numpy.outer(a, b))


class TestLinalgTensordot:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import tensordot
            result = tensordot(A, B, axes=1)
            assert numpy.allclose(result, numpy.tensordot(A, B, axes=1))


class TestLinalgVecdot:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import vecdot
            result = vecdot(a, b)
            assert numpy.allclose(result, numpy.vecdot(a, b))


class TestLinalgDiagonal:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import diagonal
            result = diagonal(A)
            assert numpy.allclose(result, numpy.diagonal(A))

    def test_zero_cost(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import diagonal
            diagonal(A)
            assert budget.flops_used == 0


class TestLinalgMatrixTranspose:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import matrix_transpose
            result = matrix_transpose(A)
            assert numpy.allclose(result, numpy.matrix_transpose(A))

    def test_zero_cost(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import matrix_transpose
            matrix_transpose(A)
            assert budget.flops_used == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_linalg_aliases.py -v`
Expected: FAIL — `ImportError: cannot import name 'matmul' from 'mechestim.linalg'`

- [ ] **Step 3: Write implementation**

```python
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
```

- [ ] **Step 4: Update linalg `__init__.py`**

Replace the full contents of `src/mechestim/linalg/__init__.py` with:

```python
"""Linear algebra submodule for mechestim."""
from mechestim.linalg._svd import svd  # noqa: F401
from mechestim.linalg._aliases import (  # noqa: F401
    matmul, cross, outer, tensordot, vecdot, diagonal, matrix_transpose,
)
from mechestim._registry import make_module_getattr as _make_module_getattr

__all__ = [
    "svd",
    "matmul", "cross", "outer", "tensordot", "vecdot",
    "diagonal", "matrix_transpose",
]

__getattr__ = _make_module_getattr(module_prefix="linalg.", module_label="mechestim.linalg")
```

- [ ] **Step 5: Update registry entries**

In `src/mechestim/_registry.py`, update these 7 entries:

- `"linalg.matmul"`: category `"free"`, notes `"Alias for numpy.matmul — delegates to mechestim.matmul."`
- `"linalg.cross"`: category `"free"`, notes `"Alias for numpy.cross — delegates to mechestim.cross."`
- `"linalg.outer"`: category `"free"`, notes `"Alias for numpy.outer — delegates to mechestim.outer."`
- `"linalg.tensordot"`: category `"free"`, notes `"Alias for numpy.tensordot — delegates to mechestim.tensordot."`
- `"linalg.vecdot"`: category `"free"`, notes `"Alias for numpy.vecdot — delegates to mechestim.vecdot."`
- `"linalg.diagonal"`: category `"free"`, notes `"View of diagonal — delegates to mechestim.diagonal. Cost: 0 FLOPs."`
- `"linalg.matrix_transpose"`: category `"free"`, notes `"Transpose view — delegates to mechestim.matrix_transpose. Cost: 0 FLOPs."`

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_linalg_aliases.py -v`
Expected: All PASS

- [ ] **Step 7: Run existing tests to verify no regressions**

Run: `pytest tests/test_linalg.py tests/test_registry.py tests/test_getattr.py -v`
Expected: All PASS (the `test_linalg_unsupported` test in `test_linalg.py` that tried to access `linalg.cholesky` should still raise `AttributeError` since cholesky is not yet implemented)

- [ ] **Step 8: Commit**

```bash
git add src/mechestim/linalg/_aliases.py src/mechestim/linalg/__init__.py src/mechestim/_registry.py tests/test_linalg_aliases.py
git commit -m "feat: add linalg namespace aliases (matmul, cross, outer, tensordot, vecdot, diagonal, matrix_transpose)"
```

---

## Task 2: Linalg Decompositions

**Files:**
- Create: `src/mechestim/linalg/_decompositions.py`
- Create: `tests/test_linalg_decompositions.py`
- Modify: `src/mechestim/linalg/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_linalg_decompositions.py
"""Tests for linalg decomposition wrappers with FLOP counting."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestCholesky:
    def test_result_matches_numpy(self):
        A = numpy.array([[4.0, 2.0], [2.0, 3.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import cholesky
            result = cholesky(A)
            assert numpy.allclose(result, numpy.linalg.cholesky(A))

    def test_cost(self):
        n = 10
        A = numpy.eye(n) * 10 + numpy.random.randn(n, n)
        A = A @ A.T  # make positive definite
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import cholesky
            cholesky(A)
            assert budget.flops_used == n**3 // 3

    def test_op_log(self):
        A = numpy.eye(3) * 10
        A = A @ A.T
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import cholesky
            cholesky(A)
            assert budget.op_log[-1].op_name == "linalg.cholesky"

    def test_outside_context_raises(self):
        from mechestim.linalg import cholesky
        with pytest.raises(NoBudgetContextError):
            cholesky(numpy.eye(3))


class TestQR:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(6, 4)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import qr
            Q, R = qr(A)
            Q_np, R_np = numpy.linalg.qr(A)
            assert numpy.allclose(numpy.abs(Q), numpy.abs(Q_np))
            assert numpy.allclose(numpy.abs(R), numpy.abs(R_np))

    def test_cost(self):
        m, n = 6, 4
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import qr
            qr(A)
            expected = 2 * m * n**2 - (2 * n**3) // 3
            assert budget.flops_used == expected

    def test_op_log(self):
        A = numpy.random.randn(4, 3)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import qr
            qr(A)
            assert budget.op_log[-1].op_name == "linalg.qr"


class TestEig:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import eig
            w, v = eig(A)
            w_np, v_np = numpy.linalg.eig(A)
            assert numpy.allclose(sorted(numpy.abs(w)), sorted(numpy.abs(w_np)))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import eig
            eig(A)
            assert budget.flops_used == 10 * n**3


class TestEigh:
    def test_result_matches_numpy(self):
        A = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import eigh
            w, v = eigh(A)
            w_np, v_np = numpy.linalg.eigh(A)
            assert numpy.allclose(w, w_np)

    def test_cost(self):
        n = 6
        A = numpy.random.randn(n, n)
        A = A + A.T
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import eigh
            eigh(A)
            assert budget.flops_used == (4 * n**3) // 3


class TestEigvals:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import eigvals
            w = eigvals(A)
            w_np = numpy.linalg.eigvals(A)
            assert numpy.allclose(sorted(numpy.abs(w)), sorted(numpy.abs(w_np)))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import eigvals
            eigvals(A)
            assert budget.flops_used == 10 * n**3


class TestEigvalsh:
    def test_cost(self):
        n = 6
        A = numpy.random.randn(n, n)
        A = A + A.T
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import eigvalsh
            eigvalsh(A)
            assert budget.flops_used == (4 * n**3) // 3


class TestSvdvals:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(6, 4)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import svdvals
            s = svdvals(A)
            s_np = numpy.linalg.svdvals(A)
            assert numpy.allclose(s, s_np)

    def test_cost(self):
        m, n = 6, 4
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import svdvals
            svdvals(A)
            assert budget.flops_used == m * n * min(m, n)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_linalg_decompositions.py -v`
Expected: FAIL — `ImportError: cannot import name 'cholesky' from 'mechestim.linalg'`

- [ ] **Step 3: Write implementation**

```python
# src/mechestim/linalg/_decompositions.py
"""Matrix decomposition wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""
from __future__ import annotations

import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eigendecomposition (nonsymmetric)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eigendecomposition (symmetric)
# ---------------------------------------------------------------------------

def eigh_cost(n: int) -> int:
    """FLOP cost of symmetric eigendecomposition of an (n, n) matrix.

    Formula: (4/3) * n^3
    Source: Golub & Van Loan, "Matrix Computations", 4th ed., §8.3
    Assumes: Tridiagonalization via Householder + implicit QR sweeps.
    Cheaper than nonsymmetric due to tridiagonal structure.
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
    return _np.linalg.eigh(a, UPLO=UPLO)


# ---------------------------------------------------------------------------
# Eigenvalues only (nonsymmetric)
# ---------------------------------------------------------------------------

def eigvals_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues of an (n, n) matrix.

    Formula: 10 * n^3
    Source: Same algorithm as eig (Francis QR), no eigenvector back-substitution
    but the dominant cost is the same.
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


# ---------------------------------------------------------------------------
# Eigenvalues only (symmetric)
# ---------------------------------------------------------------------------

def eigvalsh_cost(n: int) -> int:
    """FLOP cost of computing eigenvalues of a symmetric (n, n) matrix.

    Formula: (4/3) * n^3
    Source: Same algorithm as eigh (tridiag + QR sweeps).
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


# ---------------------------------------------------------------------------
# Singular values only
# ---------------------------------------------------------------------------

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
```

- [ ] **Step 4: Update linalg `__init__.py` — add decomposition imports**

Add to `src/mechestim/linalg/__init__.py`:

```python
from mechestim.linalg._decompositions import (  # noqa: F401
    cholesky, qr, eig, eigh, eigvals, eigvalsh, svdvals,
    cholesky_cost, qr_cost, eig_cost, eigh_cost, eigvals_cost, eigvalsh_cost, svdvals_cost,
)
```

Update `__all__` to include: `"cholesky", "qr", "eig", "eigh", "eigvals", "eigvalsh", "svdvals"`

- [ ] **Step 5: Update registry entries**

In `src/mechestim/_registry.py`, update these 7 entries from `"blacklisted"` to `"counted_custom"`:

- `"linalg.cholesky"`: notes `"Cholesky decomposition. Cost: n^3/3 (Golub & Van Loan §4.2)."`
- `"linalg.qr"`: notes `"QR decomposition. Cost: 2*m*n^2 - (2/3)*n^3 (Golub & Van Loan §5.2)."`
- `"linalg.eig"`: notes `"Eigendecomposition. Cost: 10*n^3 (Francis QR, Golub & Van Loan §7.5)."`
- `"linalg.eigh"`: notes `"Symmetric eigendecomposition. Cost: (4/3)*n^3 (Golub & Van Loan §8.3)."`
- `"linalg.eigvals"`: notes `"Eigenvalues only. Cost: 10*n^3 (same as eig)."`
- `"linalg.eigvalsh"`: notes `"Symmetric eigenvalues. Cost: (4/3)*n^3 (same as eigh)."`
- `"linalg.svdvals"`: notes `"Singular values only. Cost: m*n*min(m,n) (Golub-Reinsch)."`

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_linalg_decompositions.py tests/test_linalg.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/linalg/_decompositions.py src/mechestim/linalg/__init__.py src/mechestim/_registry.py tests/test_linalg_decompositions.py
git commit -m "feat: add linalg decompositions (cholesky, qr, eig, eigh, eigvals, eigvalsh, svdvals)"
```

---

## Task 3: Linalg Solvers

**Files:**
- Create: `src/mechestim/linalg/_solvers.py`
- Create: `tests/test_linalg_solvers.py`
- Modify: `src/mechestim/linalg/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_linalg_solvers.py
"""Tests for linalg solver wrappers with FLOP counting."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestSolve:
    def test_result_matches_numpy(self):
        A = numpy.array([[3.0, 1.0], [1.0, 2.0]])
        b = numpy.array([9.0, 8.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import solve
            result = solve(A, b)
            assert numpy.allclose(result, numpy.linalg.solve(A, b))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n) + numpy.eye(n) * 10
        b = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import solve
            solve(A, b)
            assert budget.flops_used == n**3

    def test_op_log(self):
        A = numpy.eye(3)
        b = numpy.ones(3)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import solve
            solve(A, b)
            assert budget.op_log[-1].op_name == "linalg.solve"

    def test_outside_context_raises(self):
        from mechestim.linalg import solve
        with pytest.raises(NoBudgetContextError):
            solve(numpy.eye(3), numpy.ones(3))


class TestInv:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import inv
            result = inv(A)
            assert numpy.allclose(result, numpy.linalg.inv(A))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n) + numpy.eye(n) * 10
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import inv
            inv(A)
            assert budget.flops_used == n**3


class TestLstsq:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(6, 4)
        b = numpy.random.randn(6)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import lstsq
            x, residuals, rank, sv = lstsq(A, b, rcond=None)
            x_np, _, _, _ = numpy.linalg.lstsq(A, b, rcond=None)
            assert numpy.allclose(x, x_np)

    def test_cost(self):
        m, n = 6, 4
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import lstsq
            lstsq(A, b, rcond=None)
            assert budget.flops_used == m * n * min(m, n)


class TestPinv:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(4, 3)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import pinv
            result = pinv(A)
            assert numpy.allclose(result, numpy.linalg.pinv(A))

    def test_cost(self):
        m, n = 4, 3
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import pinv
            pinv(A)
            assert budget.flops_used == m * n * min(m, n)


class TestTensorsolve:
    def test_cost(self):
        # tensorsolve reshapes to square then solves
        a = numpy.eye(24).reshape(6, 4, 2, 3, 4)
        b = numpy.ones(6 * 4)
        # After reshape: n = product of "ind" dims
        with BudgetContext(flop_budget=10**9):
            from mechestim.linalg import tensorsolve
            tensorsolve(a, b)


class TestTensorinv:
    def test_cost(self):
        a = numpy.eye(24).reshape(6, 4, 2, 3, 4)
        with BudgetContext(flop_budget=10**9):
            from mechestim.linalg import tensorinv
            tensorinv(a, ind=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_linalg_solvers.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write implementation**

```python
# src/mechestim/linalg/_solvers.py
"""Linear solver wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""
from __future__ import annotations

import math
import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve_cost(n: int) -> int:
    """FLOP cost of solving Ax = b for (n, n) matrix A.

    Formula: n^3
    Source: LU factorization (2/3 n^3) + forward/back substitution (2 n^2).
    Leading term: n^3.
    """
    return max(n ** 3, 1)


def solve(a, b):
    """Solve linear system with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"First argument must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = solve_cost(n)
    budget.deduct("linalg.solve", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.solve(a, b)


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------

def inv_cost(n: int) -> int:
    """FLOP cost of matrix inverse of an (n, n) matrix.

    Formula: n^3
    Source: LU factorization + solve for n right-hand sides.
    """
    return max(n ** 3, 1)


def inv(a):
    """Matrix inverse with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = inv_cost(n)
    budget.deduct("linalg.inv", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.inv(a)


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------

def lstsq_cost(m: int, n: int) -> int:
    """FLOP cost of least-squares solution of an (m, n) system.

    Formula: m * n * min(m, n)
    Source: NumPy uses LAPACK gelsd (SVD-based) by default.
    """
    return max(m * n * min(m, n), 1)


def lstsq(a, b, rcond=None):
    """Least-squares solution with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2:
        raise ValueError(f"First argument must be 2D, got {a.ndim}D")
    m, n = a.shape
    cost = lstsq_cost(m, n)
    budget.deduct("linalg.lstsq", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.lstsq(a, b, rcond=rcond)


# ---------------------------------------------------------------------------
# Pseudoinverse
# ---------------------------------------------------------------------------

def pinv_cost(m: int, n: int) -> int:
    """FLOP cost of pseudoinverse of an (m, n) matrix.

    Formula: m * n * min(m, n)
    Source: Computed via SVD.
    """
    return max(m * n * min(m, n), 1)


def pinv(a, rcond=None, hermitian=False):
    """Pseudoinverse with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    cost = pinv_cost(m, n)
    budget.deduct("linalg.pinv", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    kwargs = {"hermitian": hermitian}
    if rcond is not None:
        kwargs["rcond"] = rcond
    return _np.linalg.pinv(a, **kwargs)


# ---------------------------------------------------------------------------
# Tensor solve
# ---------------------------------------------------------------------------

def tensorsolve_cost(a_shape: tuple, ind: int | None = None) -> int:
    """FLOP cost of tensor solve.

    Reduces to linalg.solve after reshaping. Cost: n^3 where
    n = product of trailing dimensions of a.
    Source: Delegates to solve internally.
    """
    ndim = len(a_shape)
    if ind is None:
        ind = 2
    # After reshape, the effective square dimension is the product
    # of dimensions from ind onward
    n = 1
    for d in a_shape[ind:]:
        n *= d
    return max(n ** 3, 1)


def tensorsolve(a, b, axes=None):
    """Tensor solve with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    cost = tensorsolve_cost(a.shape)
    budget.deduct("linalg.tensorsolve", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.tensorsolve(a, b, axes=axes)


# ---------------------------------------------------------------------------
# Tensor inverse
# ---------------------------------------------------------------------------

def tensorinv_cost(a_shape: tuple, ind: int = 2) -> int:
    """FLOP cost of tensor inverse.

    Reduces to linalg.inv after reshaping. Cost: n^3 where
    n = product of first `ind` dimensions.
    Source: Delegates to inv internally.
    """
    n = 1
    for d in a_shape[:ind]:
        n *= d
    return max(n ** 3, 1)


def tensorinv(a, ind=2):
    """Tensor inverse with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    cost = tensorinv_cost(a.shape, ind=ind)
    budget.deduct("linalg.tensorinv", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.tensorinv(a, ind=ind)
```

- [ ] **Step 4: Update linalg `__init__.py` — add solver imports**

Add to `src/mechestim/linalg/__init__.py`:

```python
from mechestim.linalg._solvers import (  # noqa: F401
    solve, inv, lstsq, pinv, tensorsolve, tensorinv,
    solve_cost, inv_cost, lstsq_cost, pinv_cost, tensorsolve_cost, tensorinv_cost,
)
```

Update `__all__` to include: `"solve", "inv", "lstsq", "pinv", "tensorsolve", "tensorinv"`

- [ ] **Step 5: Update registry entries**

Update these 6 entries from `"blacklisted"` to `"counted_custom"`:

- `"linalg.solve"`: notes `"Solve Ax=b. Cost: n^3 (LU factorization)."`
- `"linalg.inv"`: notes `"Matrix inverse. Cost: n^3 (LU + solve)."`
- `"linalg.lstsq"`: notes `"Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD)."`
- `"linalg.pinv"`: notes `"Pseudoinverse. Cost: m*n*min(m,n) (via SVD)."`
- `"linalg.tensorsolve"`: notes `"Tensor solve. Cost: n^3 after reshape (delegates to solve)."`
- `"linalg.tensorinv"`: notes `"Tensor inverse. Cost: n^3 after reshape (delegates to inv)."`

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_linalg_solvers.py tests/test_linalg.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/linalg/_solvers.py src/mechestim/linalg/__init__.py src/mechestim/_registry.py tests/test_linalg_solvers.py
git commit -m "feat: add linalg solvers (solve, inv, lstsq, pinv, tensorsolve, tensorinv)"
```

---

## Task 4: Linalg Properties

**Files:**
- Create: `src/mechestim/linalg/_properties.py`
- Create: `tests/test_linalg_properties.py`
- Modify: `src/mechestim/linalg/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_linalg_properties.py
"""Tests for linalg property wrappers with FLOP counting."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestTrace:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import trace
            result = trace(A)
            assert result == numpy.trace(A)

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import trace
            trace(A)
            assert budget.flops_used == n


class TestDet:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import det
            result = det(A)
            assert numpy.isclose(result, numpy.linalg.det(A))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import det
            det(A)
            assert budget.flops_used == n**3


class TestSlogdet:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import slogdet
            sign, logdet = slogdet(A)
            sign_np, logdet_np = numpy.linalg.slogdet(A)
            assert numpy.isclose(sign, sign_np)
            assert numpy.isclose(logdet, logdet_np)

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import slogdet
            slogdet(A)
            assert budget.flops_used == n**3


class TestNorm:
    def test_vector_default(self):
        x = numpy.array([3.0, 4.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import norm
            result = norm(x)
            assert numpy.isclose(result, 5.0)

    def test_vector_default_cost(self):
        x = numpy.random.randn(10)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import norm
            norm(x)
            assert budget.flops_used == 10  # numel for L2

    def test_matrix_fro_cost(self):
        A = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import norm
            norm(A)  # default for matrix is Frobenius
            assert budget.flops_used == 2 * 20  # 2 * numel

    def test_matrix_ord2_cost(self):
        A = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import norm
            norm(A, ord=2)
            assert budget.flops_used == 4 * 5 * 4  # m*n*min(m,n) SVD cost

    def test_matrix_ord1_cost(self):
        A = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import norm
            norm(A, ord=1)
            assert budget.flops_used == 20  # numel

    def test_vector_p_norm_cost(self):
        x = numpy.random.randn(10)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import norm
            norm(x, ord=3)
            assert budget.flops_used == 2 * 10  # power + sum


class TestVectorNorm:
    def test_result_matches_numpy(self):
        x = numpy.array([3.0, 4.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import vector_norm
            result = vector_norm(x)
            assert numpy.isclose(result, 5.0)

    def test_cost(self):
        x = numpy.random.randn(10)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import vector_norm
            vector_norm(x)
            assert budget.flops_used == 10


class TestMatrixNorm:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(3, 4)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import matrix_norm
            result = matrix_norm(A)
            expected = numpy.linalg.matrix_norm(A)
            assert numpy.isclose(result, expected)

    def test_fro_cost(self):
        A = numpy.random.randn(3, 4)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import matrix_norm
            matrix_norm(A)  # default is Frobenius
            assert budget.flops_used == 2 * 12


class TestCond:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 0.0], [0.0, 2.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import cond
            result = cond(A)
            assert numpy.isclose(result, numpy.linalg.cond(A))

    def test_cost(self):
        m, n = 4, 3
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import cond
            cond(A)
            assert budget.flops_used == m * n * min(m, n)


class TestMatrixRank:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 0.0], [0.0, 0.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import matrix_rank
            result = matrix_rank(A)
            assert result == numpy.linalg.matrix_rank(A)

    def test_cost(self):
        m, n = 5, 3
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import matrix_rank
            matrix_rank(A)
            assert budget.flops_used == m * n * min(m, n)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_linalg_properties.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write implementation**

```python
# src/mechestim/linalg/_properties.py
"""Matrix property wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""
from __future__ import annotations

import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def trace_cost(n: int) -> int:
    """FLOP cost of matrix trace for an (n, n) matrix.

    Formula: n
    Source: Sum of n diagonal elements.
    """
    return max(n, 1)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Matrix trace with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    # Trace sums along the diagonal; the number of elements is min of the
    # two axis dimensions (adjusted by offset)
    n = min(a.shape[axis1], a.shape[axis2])
    if offset > 0:
        n = min(n, a.shape[axis2] - offset)
    elif offset < 0:
        n = min(n, a.shape[axis1] + offset)
    n = max(n, 0)
    cost = trace_cost(n)
    budget.deduct("linalg.trace", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)


# ---------------------------------------------------------------------------
# Determinant
# ---------------------------------------------------------------------------

def det_cost(n: int) -> int:
    """FLOP cost of determinant of an (n, n) matrix.

    Formula: n^3
    Source: LU factorization. Determinant = product of diagonal of U.
    """
    return max(n ** 3, 1)


def det(a):
    """Determinant with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = det_cost(n)
    budget.deduct("linalg.det", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.det(a)


# ---------------------------------------------------------------------------
# Sign and log-determinant
# ---------------------------------------------------------------------------

def slogdet_cost(n: int) -> int:
    """FLOP cost of sign and log-determinant of an (n, n) matrix.

    Formula: n^3
    Source: Same LU factorization as det.
    """
    return max(n ** 3, 1)


def slogdet(a):
    """Sign and log-determinant with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = slogdet_cost(n)
    budget.deduct("linalg.slogdet", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.slogdet(a)


# ---------------------------------------------------------------------------
# Norm (unified)
# ---------------------------------------------------------------------------

def norm_cost(shape: tuple[int, ...], ord=None) -> int:
    """FLOP cost of matrix or vector norm.

    Formula depends on ord parameter and array dimensionality:
      Vector (1D):
        ord=None/2:      numel (sqrt of sum of squares)
        ord=1/inf/-inf:  numel (abs + reduction)
        ord=0:           numel (count nonzero)
        numeric p:       2 * numel (power + sum)
      Matrix (2D):
        ord=None/'fro':  2 * numel (square + sum)
        ord=1/-1/inf/-inf: numel (abs + reduction)
        ord=2/-2:        m * n * min(m, n) (requires SVD)
    Source: Direct analysis of norm definitions.
    """
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)

    if len(shape) == 1:
        # Vector norms
        if ord is None or ord == 2 or ord == -2:
            return numel
        elif ord in (1, -1, _np.inf, -_np.inf, 0):
            return numel
        else:
            return 2 * numel  # |x|^p for general p
    else:
        # Matrix norms
        m, n = shape[-2], shape[-1]
        if ord is None or ord == "fro":
            return 2 * numel  # square + sum
        elif ord in (1, -1, _np.inf, -_np.inf):
            return numel  # abs + reduction
        elif ord == 2 or ord == -2:
            return m * n * min(m, n)  # SVD
        elif ord == "nuc":
            return m * n * min(m, n)  # SVD for nuclear norm
        else:
            return numel  # fallback


def norm(x, ord=None, axis=None, keepdims=False):
    """Matrix or vector norm with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    # Determine effective shape for cost calculation
    if axis is None:
        if x.ndim == 1:
            effective_shape = x.shape
        else:
            effective_shape = x.shape
    elif isinstance(axis, int):
        effective_shape = (x.shape[axis],)
    else:
        effective_shape = tuple(x.shape[ax] for ax in axis)
    cost = norm_cost(effective_shape, ord=ord)
    budget.deduct("linalg.norm", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# Vector norm
# ---------------------------------------------------------------------------

def vector_norm_cost(shape: tuple[int, ...], ord=None) -> int:
    """FLOP cost of vector norm.

    Formula: same as norm for vectors.
    Source: Direct analysis.
    """
    numel = max(1, 1)
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    if ord is None or ord == 2 or ord == -2 or ord in (1, -1, _np.inf, -_np.inf, 0):
        return numel
    return 2 * numel


def vector_norm(x, ord=2, axis=None, keepdims=False):
    """Vector norm with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    if axis is not None:
        if isinstance(axis, int):
            effective_shape = (x.shape[axis],)
        else:
            effective_shape = tuple(x.shape[ax] for ax in axis)
    else:
        effective_shape = x.shape
    cost = vector_norm_cost(effective_shape, ord=ord)
    budget.deduct("linalg.vector_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.vector_norm(x, ord=ord, axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# Matrix norm
# ---------------------------------------------------------------------------

def matrix_norm_cost(shape: tuple[int, ...], ord=None) -> int:
    """FLOP cost of matrix norm.

    Formula: same as norm for matrices (dispatches on ord).
    Source: Direct analysis.
    """
    m, n = shape[-2], shape[-1]
    numel = m * n
    if ord is None or ord == "fro":
        return 2 * numel
    elif ord in (1, -1, _np.inf, -_np.inf):
        return numel
    elif ord == 2 or ord == -2:
        return m * n * min(m, n)
    elif ord == "nuc":
        return m * n * min(m, n)
    return numel


def matrix_norm(x, ord="fro", keepdims=False):
    """Matrix norm with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    cost = matrix_norm_cost(x.shape, ord=ord)
    budget.deduct("linalg.matrix_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.matrix_norm(x, ord=ord, keepdims=keepdims)


# ---------------------------------------------------------------------------
# Condition number
# ---------------------------------------------------------------------------

def cond_cost(m: int, n: int) -> int:
    """FLOP cost of condition number of an (m, n) matrix.

    Formula: m * n * min(m, n)
    Source: Computed via SVD, then ratio of extremal singular values.
    """
    return max(m * n * min(m, n), 1)


def cond(x, p=None):
    """Condition number with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    if x.ndim != 2:
        raise ValueError(f"Input must be 2D, got {x.ndim}D")
    m, n = x.shape
    cost = cond_cost(m, n)
    budget.deduct("linalg.cond", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.cond(x, p=p)


# ---------------------------------------------------------------------------
# Matrix rank
# ---------------------------------------------------------------------------

def matrix_rank_cost(m: int, n: int) -> int:
    """FLOP cost of matrix rank of an (m, n) matrix.

    Formula: m * n * min(m, n)
    Source: Computed via SVD, then threshold count.
    """
    return max(m * n * min(m, n), 1)


def matrix_rank(A, tol=None, hermitian=False):
    """Matrix rank with FLOP counting."""
    budget = require_budget()
    validate_ndarray(A)
    if A.ndim != 2:
        raise ValueError(f"Input must be 2D, got {A.ndim}D")
    m, n = A.shape
    cost = matrix_rank_cost(m, n)
    budget.deduct("linalg.matrix_rank", flop_cost=cost, subscripts=None, shapes=(A.shape,))
    return _np.linalg.matrix_rank(A, tol=tol, hermitian=hermitian)
```

- [ ] **Step 4: Update linalg `__init__.py` — add property imports**

Add to `src/mechestim/linalg/__init__.py`:

```python
from mechestim.linalg._properties import (  # noqa: F401
    trace, det, slogdet, norm, vector_norm, matrix_norm, cond, matrix_rank,
    trace_cost, det_cost, slogdet_cost, norm_cost, vector_norm_cost,
    matrix_norm_cost, cond_cost, matrix_rank_cost,
)
```

Update `__all__` to include: `"trace", "det", "slogdet", "norm", "vector_norm", "matrix_norm", "cond", "matrix_rank"`

- [ ] **Step 5: Update registry entries**

Update these 8 entries from `"blacklisted"` to `"counted_custom"`:

- `"linalg.trace"`: notes `"Matrix trace. Cost: n (sum of diagonal elements)."`
- `"linalg.det"`: notes `"Determinant. Cost: n^3 (LU factorization)."`
- `"linalg.slogdet"`: notes `"Sign + log determinant. Cost: n^3 (LU factorization)."`
- `"linalg.norm"`: notes `"Norm. Cost depends on ord: numel for L1/inf, 2*numel for Frobenius, m*n*min(m,n) for ord=2."`
- `"linalg.vector_norm"`: notes `"Vector norm. Cost: numel (or 2*numel for general p-norm)."`
- `"linalg.matrix_norm"`: notes `"Matrix norm. Cost depends on ord: 2*numel for Frobenius, m*n*min(m,n) for ord=2."`
- `"linalg.cond"`: notes `"Condition number. Cost: m*n*min(m,n) (via SVD)."`
- `"linalg.matrix_rank"`: notes `"Matrix rank. Cost: m*n*min(m,n) (via SVD)."`

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_linalg_properties.py tests/test_linalg.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/linalg/_properties.py src/mechestim/linalg/__init__.py src/mechestim/_registry.py tests/test_linalg_properties.py
git commit -m "feat: add linalg properties (trace, det, slogdet, norm, vector_norm, matrix_norm, cond, matrix_rank)"
```

---

## Task 5: Linalg Compound Ops

**Files:**
- Create: `src/mechestim/linalg/_compound.py`
- Create: `tests/test_linalg_compound.py`
- Modify: `src/mechestim/linalg/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_linalg_compound.py
"""Tests for linalg compound operation wrappers with FLOP counting."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestMultiDot:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(10, 5)
        B = numpy.random.randn(5, 8)
        C = numpy.random.randn(8, 3)
        with BudgetContext(flop_budget=10**8):
            from mechestim.linalg import multi_dot
            result = multi_dot([A, B, C])
            expected = numpy.linalg.multi_dot([A, B, C])
            assert numpy.allclose(result, expected)

    def test_cost_positive(self):
        A = numpy.random.randn(10, 5)
        B = numpy.random.randn(5, 8)
        C = numpy.random.randn(8, 3)
        with BudgetContext(flop_budget=10**8) as budget:
            from mechestim.linalg import multi_dot
            multi_dot([A, B, C])
            assert budget.flops_used > 0

    def test_op_log(self):
        A = numpy.random.randn(4, 3)
        B = numpy.random.randn(3, 5)
        with BudgetContext(flop_budget=10**8) as budget:
            from mechestim.linalg import multi_dot
            multi_dot([A, B])
            assert budget.op_log[-1].op_name == "linalg.multi_dot"


class TestMatrixPower:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import matrix_power
            result = matrix_power(A, 3)
            expected = numpy.linalg.matrix_power(A, 3)
            assert numpy.allclose(result, expected)

    def test_cost_power_3(self):
        n = 4
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**8) as budget:
            from mechestim.linalg import matrix_power
            matrix_power(A, 3)
            # k=3, binary: 11 -> floor(log2(3))=1, popcount(3)=2
            # cost = (1 + 2 - 1) * n^3 = 2 * n^3
            assert budget.flops_used == 2 * n**3

    def test_cost_power_0(self):
        n = 4
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**8) as budget:
            from mechestim.linalg import matrix_power
            matrix_power(A, 0)
            assert budget.flops_used == 0  # identity, free

    def test_cost_power_1(self):
        n = 4
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**8) as budget:
            from mechestim.linalg import matrix_power
            matrix_power(A, 1)
            assert budget.flops_used == 0  # copy, free

    def test_cost_negative_power(self):
        n = 3
        A = numpy.random.randn(n, n) + numpy.eye(n) * 5
        with BudgetContext(flop_budget=10**8) as budget:
            from mechestim.linalg import matrix_power
            matrix_power(A, -2)
            # Inversion: n^3 + power cost for k=2
            # k=2, binary: 10 -> floor(log2(2))=1, popcount(2)=1
            # power_cost = (1 + 1 - 1) * n^3 = n^3
            # total = n^3 (inv) + n^3 (power) = 2 * n^3
            assert budget.flops_used == 2 * n**3

    def test_outside_context_raises(self):
        from mechestim.linalg import matrix_power
        with pytest.raises(NoBudgetContextError):
            matrix_power(numpy.eye(3), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_linalg_compound.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write implementation**

```python
# src/mechestim/linalg/_compound.py
"""Compound linalg operations with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""
from __future__ import annotations

import math
import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


# ---------------------------------------------------------------------------
# Multi-dot
# ---------------------------------------------------------------------------

def multi_dot_cost(shapes: list[tuple[int, ...]]) -> int:
    """FLOP cost of optimal matrix chain multiplication.

    Formula: Sum of individual matmul costs using optimal parenthesization.
    For two matrices (m, k) @ (k, n), matmul cost = m * k * n.
    The optimal chain order is determined by dynamic programming.
    Source: Standard matrix chain multiplication (Cormen et al., CLRS §15.2).
    """
    n = len(shapes)
    if n < 2:
        return 0

    # Extract dimensions: for chain A0 @ A1 @ ... @ A(n-1)
    # dims[i] = rows of matrix i, dims[n] = cols of last matrix
    dims = [s[0] for s in shapes] + [shapes[-1][-1]]

    if n == 2:
        return dims[0] * dims[1] * dims[2]

    # DP for optimal parenthesization
    # cost_table[i][j] = min cost to multiply matrices i..j
    cost_table = [[0] * n for _ in range(n)]
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            cost_table[i][j] = float("inf")
            for k in range(i, j):
                cost = (
                    cost_table[i][k]
                    + cost_table[k + 1][j]
                    + dims[i] * dims[k + 1] * dims[j + 1]
                )
                if cost < cost_table[i][j]:
                    cost_table[i][j] = cost

    return max(int(cost_table[0][n - 1]), 1)


def multi_dot(arrays, *, out=None):
    """Efficient multi-matrix dot product with FLOP counting."""
    budget = require_budget()
    for arr in arrays:
        validate_ndarray(arr)
    shapes = [arr.shape for arr in arrays]
    cost = multi_dot_cost(shapes)
    budget.deduct("linalg.multi_dot", flop_cost=cost, subscripts=None,
                  shapes=tuple(shapes))
    return _np.linalg.multi_dot(arrays, out=out)


# ---------------------------------------------------------------------------
# Matrix power
# ---------------------------------------------------------------------------

def _popcount(n: int) -> int:
    """Count the number of set bits in a non-negative integer."""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def matrix_power_cost(n: int, k: int) -> int:
    """FLOP cost of matrix power A^k for an (n, n) matrix.

    Formula:
      k = 0 or k = 1: 0 (identity or copy)
      k >= 2: (floor(log2(k)) + popcount(k) - 1) * n^3
      k < 0: n^3 (inversion) + matrix_power_cost(n, |k|)
    Source: Exponentiation by squaring. Each squaring or multiply is n^3.
    floor(log2(k)) squarings + (popcount(k) - 1) extra multiplies.
    """
    if k == 0 or k == 1:
        return 0
    if k < 0:
        # Inversion cost + power cost for |k|
        return n ** 3 + matrix_power_cost(n, abs(k))
    num_ops = math.floor(math.log2(k)) + _popcount(k) - 1
    return max(num_ops * n ** 3, 1)


def matrix_power(a, n):
    """Matrix power with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    size = a.shape[0]
    cost = matrix_power_cost(size, n)
    if cost > 0:
        budget.deduct("linalg.matrix_power", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    else:
        # Still log it for traceability, even if free
        budget.deduct("linalg.matrix_power", flop_cost=0, subscripts=None, shapes=(a.shape,))
    return _np.linalg.matrix_power(a, n)
```

- [ ] **Step 4: Update linalg `__init__.py` — add compound imports**

Add to `src/mechestim/linalg/__init__.py`:

```python
from mechestim.linalg._compound import (  # noqa: F401
    multi_dot, matrix_power,
    multi_dot_cost, matrix_power_cost,
)
```

Update `__all__` to include: `"multi_dot", "matrix_power"`

- [ ] **Step 5: Update registry entries**

Update these 2 entries from `"blacklisted"` to `"counted_custom"`:

- `"linalg.multi_dot"`: notes `"Chain matmul. Cost: sum of optimal chain matmul costs (CLRS §15.2)."`
- `"linalg.matrix_power"`: notes `"Matrix power. Cost: (floor(log2(k)) + popcount(k) - 1) * n^3 (exponentiation by squaring)."`

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_linalg_compound.py tests/test_linalg.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/linalg/_compound.py src/mechestim/linalg/__init__.py src/mechestim/_registry.py tests/test_linalg_compound.py
git commit -m "feat: add linalg compound ops (multi_dot, matrix_power)"
```

---

## Task 6: FFT Transforms

**Files:**
- Create: `src/mechestim/fft/_transforms.py`
- Create: `src/mechestim/fft/_free.py`
- Create: `tests/test_fft_transforms.py`
- Create: `tests/test_fft_free.py`
- Modify: `src/mechestim/fft/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fft_transforms.py
"""Tests for FFT transform wrappers with FLOP counting."""
import math
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestFft:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6):
            from mechestim.fft import fft
            result = fft(x)
            assert numpy.allclose(result, numpy.fft.fft(x))

    def test_cost(self):
        n = 16
        x = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import fft
            fft(x)
            expected = 5 * n * math.ceil(math.log2(n))
            assert budget.flops_used == expected

    def test_cost_with_n_param(self):
        x = numpy.random.randn(10)
        n = 32
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import fft
            fft(x, n=n)
            expected = 5 * n * math.ceil(math.log2(n))
            assert budget.flops_used == expected

    def test_op_log(self):
        x = numpy.random.randn(8)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import fft
            fft(x)
            assert budget.op_log[-1].op_name == "fft.fft"

    def test_outside_context_raises(self):
        from mechestim.fft import fft
        with pytest.raises(NoBudgetContextError):
            fft(numpy.ones(8))


class TestIfft:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(16) + 1j * numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6):
            from mechestim.fft import ifft
            result = ifft(x)
            assert numpy.allclose(result, numpy.fft.ifft(x))

    def test_cost(self):
        n = 16
        x = numpy.random.randn(n) + 1j * numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import ifft
            ifft(x)
            expected = 5 * n * math.ceil(math.log2(n))
            assert budget.flops_used == expected


class TestRfft:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6):
            from mechestim.fft import rfft
            result = rfft(x)
            assert numpy.allclose(result, numpy.fft.rfft(x))

    def test_cost(self):
        n = 16
        x = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import rfft
            rfft(x)
            expected = 5 * (n // 2) * math.ceil(math.log2(n))
            assert budget.flops_used == expected


class TestIrfft:
    def test_cost(self):
        n = 16
        x = numpy.fft.rfft(numpy.random.randn(n))
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import irfft
            irfft(x, n=n)
            expected = 5 * (n // 2) * math.ceil(math.log2(n))
            assert budget.flops_used == expected


class TestFft2:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6):
            from mechestim.fft import fft2
            result = fft2(x)
            assert numpy.allclose(result, numpy.fft.fft2(x))

    def test_cost(self):
        x = numpy.random.randn(8, 8)
        N = 64
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import fft2
            fft2(x)
            expected = 5 * N * math.ceil(math.log2(N))
            assert budget.flops_used == expected


class TestFftn:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(4, 4, 4)
        with BudgetContext(flop_budget=10**8):
            from mechestim.fft import fftn
            result = fftn(x)
            assert numpy.allclose(result, numpy.fft.fftn(x))


class TestHfft:
    def test_cost(self):
        n = 16
        x = numpy.random.randn(n) + 1j * numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import hfft
            hfft(x)
            out_n = 2 * (n - 1)  # hfft output length
            expected = 5 * out_n * math.ceil(math.log2(out_n))
            assert budget.flops_used == expected
```

```python
# tests/test_fft_free.py
"""Tests for FFT free operations."""
import numpy
from mechestim._budget import BudgetContext


class TestFftfreq:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import fftfreq
            result = fftfreq(8, d=1.0)
            assert numpy.allclose(result, numpy.fft.fftfreq(8, d=1.0))
            assert budget.flops_used == 0


class TestRfftfreq:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import rfftfreq
            result = rfftfreq(8, d=1.0)
            assert numpy.allclose(result, numpy.fft.rfftfreq(8, d=1.0))
            assert budget.flops_used == 0


class TestFftshift:
    def test_result_matches_numpy(self):
        x = numpy.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import fftshift
            result = fftshift(x)
            assert numpy.allclose(result, numpy.fft.fftshift(x))
            assert budget.flops_used == 0


class TestIfftshift:
    def test_result_matches_numpy(self):
        x = numpy.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.fft import ifftshift
            result = ifftshift(x)
            assert numpy.allclose(result, numpy.fft.ifftshift(x))
            assert budget.flops_used == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fft_transforms.py tests/test_fft_free.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write FFT transforms implementation**

```python
# src/mechestim/fft/_transforms.py
"""FFT transform wrappers with FLOP counting.

All FFT cost estimates use the Cooley-Tukey radix-2 butterfly model:
5 * N * log2(N) real floating-point operations for a length-N complex DFT.
The constant 5 accounts for 1 complex multiply + 1 complex add per butterfly
(= 4 real muls + 6 real adds = 10 ops per butterfly, N/2 butterflies per
stage, giving 5*N per stage * log2(N) stages).

Source: Cooley & Tukey (1965); Van Loan, "Computational Frameworks for the
Fast Fourier Transform" (1992), §1.4.
"""
from __future__ import annotations

import math
import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def fft_cost(n: int) -> int:
    """FLOP cost of 1-D complex FFT of length n.

    Formula: 5 * n * ceil(log2(n))
    Source: Cooley-Tukey radix-2.
    """
    if n <= 1:
        return 0
    return 5 * n * math.ceil(math.log2(n))


def rfft_cost(n: int) -> int:
    """FLOP cost of 1-D real FFT of length n.

    Formula: 5 * (n // 2) * ceil(log2(n))
    Source: Real-input optimization exploits conjugate symmetry.
    """
    if n <= 1:
        return 0
    return 5 * (n // 2) * math.ceil(math.log2(n))


def fftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of N-D complex FFT over given shape.

    Formula: 5 * N * ceil(log2(N)) where N = product of shape
    Source: Row-column decomposition of multi-dimensional FFT.
    """
    N = 1
    for d in shape:
        N *= d
    if N <= 1:
        return 0
    return 5 * N * math.ceil(math.log2(N))


def rfftn_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of N-D real FFT over given shape.

    Formula: 5 * (N // 2) * ceil(log2(N)) where N = product of shape
    Source: Real-input optimization for multi-dimensional FFT.
    """
    N = 1
    for d in shape:
        N *= d
    if N <= 1:
        return 0
    return 5 * (N // 2) * math.ceil(math.log2(N))


def hfft_cost(n_out: int) -> int:
    """FLOP cost of Hermitian FFT with output length n_out.

    Formula: 5 * n_out * ceil(log2(n_out))
    Source: Same as complex FFT on the output length.
    """
    if n_out <= 1:
        return 0
    return 5 * n_out * math.ceil(math.log2(n_out))


# ---------------------------------------------------------------------------
# 1-D transforms
# ---------------------------------------------------------------------------

def fft(a, n=None, axis=-1, norm=None):
    """1-D FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if n is None:
        n = a.shape[axis]
    cost = fft_cost(n)
    budget.deduct("fft.fft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.fft(a, n=n, axis=axis, norm=norm)


def ifft(a, n=None, axis=-1, norm=None):
    """Inverse 1-D FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if n is None:
        n = a.shape[axis]
    cost = fft_cost(n)
    budget.deduct("fft.ifft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ifft(a, n=n, axis=axis, norm=norm)


def rfft(a, n=None, axis=-1, norm=None):
    """1-D real FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if n is None:
        n = a.shape[axis]
    cost = rfft_cost(n)
    budget.deduct("fft.rfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.rfft(a, n=n, axis=axis, norm=norm)


def irfft(a, n=None, axis=-1, norm=None):
    """Inverse 1-D real FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if n is None:
        n = 2 * (a.shape[axis] - 1)
    cost = rfft_cost(n)
    budget.deduct("fft.irfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.irfft(a, n=n, axis=axis, norm=norm)


# ---------------------------------------------------------------------------
# 2-D transforms
# ---------------------------------------------------------------------------

def fft2(a, s=None, axes=(-2, -1), norm=None):
    """2-D FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        s = tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.fft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.fft2(a, s=s, axes=axes, norm=norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Inverse 2-D FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        s = tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.ifft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ifft2(a, s=s, axes=axes, norm=norm)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """2-D real FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        s = tuple(a.shape[ax] for ax in axes)
    cost = rfftn_cost(s)
    budget.deduct("fft.rfft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.rfft2(a, s=s, axes=axes, norm=norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """Inverse 2-D real FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        s = (a.shape[axes[0]], 2 * (a.shape[axes[1]] - 1))
    cost = rfftn_cost(s)
    budget.deduct("fft.irfft2", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.irfft2(a, s=s, axes=axes, norm=norm)


# ---------------------------------------------------------------------------
# N-D transforms
# ---------------------------------------------------------------------------

def fftn(a, s=None, axes=None, norm=None):
    """N-D FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        if axes is None:
            s = a.shape
        else:
            s = tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.fftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.fftn(a, s=s, axes=axes, norm=norm)


def ifftn(a, s=None, axes=None, norm=None):
    """Inverse N-D FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        if axes is None:
            s = a.shape
        else:
            s = tuple(a.shape[ax] for ax in axes)
    cost = fftn_cost(s)
    budget.deduct("fft.ifftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ifftn(a, s=s, axes=axes, norm=norm)


def rfftn(a, s=None, axes=None, norm=None):
    """N-D real FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        if axes is None:
            s = a.shape
        else:
            s = tuple(a.shape[ax] for ax in axes)
    cost = rfftn_cost(s)
    budget.deduct("fft.rfftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.rfftn(a, s=s, axes=axes, norm=norm)


def irfftn(a, s=None, axes=None, norm=None):
    """Inverse N-D real FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if s is None:
        if axes is None:
            s = tuple(d if i < len(a.shape) - 1 else 2 * (d - 1)
                      for i, d in enumerate(a.shape))
        else:
            s = tuple(a.shape[ax] if i < len(axes) - 1 else 2 * (a.shape[ax] - 1)
                      for i, ax in enumerate(axes))
    cost = rfftn_cost(s)
    budget.deduct("fft.irfftn", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.irfftn(a, s=s, axes=axes, norm=norm)


# ---------------------------------------------------------------------------
# Hermitian transforms
# ---------------------------------------------------------------------------

def hfft(a, n=None, axis=-1, norm=None):
    """Hermitian FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if n is None:
        n = 2 * (a.shape[axis] - 1)
    cost = hfft_cost(n)
    budget.deduct("fft.hfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.hfft(a, n=n, axis=axis, norm=norm)


def ihfft(a, n=None, axis=-1, norm=None):
    """Inverse Hermitian FFT with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if n is None:
        n = a.shape[axis]
    cost = hfft_cost(n)
    budget.deduct("fft.ihfft", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.fft.ihfft(a, n=n, axis=axis, norm=norm)
```

- [ ] **Step 4: Write FFT free ops implementation**

```python
# src/mechestim/fft/_free.py
"""Zero-FLOP FFT utility operations.

These functions generate indices or rearrange arrays — no arithmetic.
"""
from __future__ import annotations

import numpy as _np


def fftfreq(n, d=1.0):
    """FFT sample frequencies. Cost: 0 FLOPs."""
    return _np.fft.fftfreq(n, d=d)


def rfftfreq(n, d=1.0):
    """Real FFT sample frequencies. Cost: 0 FLOPs."""
    return _np.fft.rfftfreq(n, d=d)


def fftshift(x, axes=None):
    """Shift zero-frequency component to center. Cost: 0 FLOPs."""
    return _np.fft.fftshift(x, axes=axes)


def ifftshift(x, axes=None):
    """Inverse of fftshift. Cost: 0 FLOPs."""
    return _np.fft.ifftshift(x, axes=axes)
```

- [ ] **Step 5: Update fft `__init__.py`**

Replace the full contents of `src/mechestim/fft/__init__.py` with:

```python
"""FFT submodule for mechestim."""
from mechestim.fft._transforms import (  # noqa: F401
    fft, ifft, rfft, irfft,
    fft2, ifft2, rfft2, irfft2,
    fftn, ifftn, rfftn, irfftn,
    hfft, ihfft,
    fft_cost, rfft_cost, fftn_cost, rfftn_cost, hfft_cost,
)
from mechestim.fft._free import (  # noqa: F401
    fftfreq, rfftfreq, fftshift, ifftshift,
)
from mechestim._registry import make_module_getattr as _make_module_getattr

__all__ = [
    "fft", "ifft", "rfft", "irfft",
    "fft2", "ifft2", "rfft2", "irfft2",
    "fftn", "ifftn", "rfftn", "irfftn",
    "hfft", "ihfft",
    "fftfreq", "rfftfreq", "fftshift", "ifftshift",
]

__getattr__ = _make_module_getattr(module_prefix="fft.", module_label="mechestim.fft")
```

- [ ] **Step 6: Update registry — all 18 fft entries**

Update all FFT entries from `"blacklisted"`:

Transform ops → `"counted_custom"`:
- `"fft.fft"`: notes `"1-D FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2)."`
- `"fft.ifft"`: notes `"Inverse 1-D FFT. Cost: 5*n*ceil(log2(n))."`
- `"fft.rfft"`: notes `"1-D real FFT. Cost: 5*(n/2)*ceil(log2(n))."`
- `"fft.irfft"`: notes `"Inverse 1-D real FFT. Cost: 5*(n/2)*ceil(log2(n))."`
- `"fft.fft2"`: notes `"2-D FFT. Cost: 5*N*ceil(log2(N)), N=prod(shape)."`
- `"fft.ifft2"`: notes `"Inverse 2-D FFT. Cost: 5*N*ceil(log2(N))."`
- `"fft.rfft2"`: notes `"2-D real FFT. Cost: 5*(N/2)*ceil(log2(N))."`
- `"fft.irfft2"`: notes `"Inverse 2-D real FFT. Cost: 5*(N/2)*ceil(log2(N))."`
- `"fft.fftn"`: notes `"N-D FFT. Cost: 5*N*ceil(log2(N)), N=prod(shape)."`
- `"fft.ifftn"`: notes `"Inverse N-D FFT. Cost: 5*N*ceil(log2(N))."`
- `"fft.rfftn"`: notes `"N-D real FFT. Cost: 5*(N/2)*ceil(log2(N))."`
- `"fft.irfftn"`: notes `"Inverse N-D real FFT. Cost: 5*(N/2)*ceil(log2(N))."`
- `"fft.hfft"`: notes `"Hermitian FFT. Cost: 5*n*ceil(log2(n))."`
- `"fft.ihfft"`: notes `"Inverse Hermitian FFT. Cost: 5*n*ceil(log2(n))."`

Free ops → `"free"`:
- `"fft.fftfreq"`: notes `"FFT sample frequencies. Cost: 0 FLOPs (index generation)."`
- `"fft.rfftfreq"`: notes `"Real FFT sample frequencies. Cost: 0 FLOPs (index generation)."`
- `"fft.fftshift"`: notes `"Shift zero-frequency component. Cost: 0 FLOPs (rearrangement)."`
- `"fft.ifftshift"`: notes `"Inverse fftshift. Cost: 0 FLOPs (rearrangement)."`

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_fft_transforms.py tests/test_fft_free.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/mechestim/fft/_transforms.py src/mechestim/fft/_free.py src/mechestim/fft/__init__.py src/mechestim/_registry.py tests/test_fft_transforms.py tests/test_fft_free.py
git commit -m "feat: add FFT operations (14 transforms + 4 free ops)"
```

---

## Task 7: Polynomial Ops

**Files:**
- Create: `src/mechestim/_polynomial.py`
- Create: `tests/test_polynomial.py`
- Modify: `src/mechestim/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_polynomial.py
"""Tests for polynomial operation wrappers with FLOP counting."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestPolyval:
    def test_result_matches_numpy(self):
        coeffs = numpy.array([1.0, -2.0, 3.0])  # x^2 - 2x + 3
        x = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import polyval
            result = polyval(coeffs, x)
            assert numpy.allclose(result, numpy.polyval(coeffs, x))

    def test_cost(self):
        coeffs = numpy.array([1.0, -2.0, 3.0])  # degree 2
        x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 points
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polyval
            polyval(coeffs, x)
            # 2 * m * deg = 2 * 5 * 2 = 20
            assert budget.flops_used == 20


class TestPolyadd:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import polyadd
            result = polyadd(a, b)
            assert numpy.allclose(result, numpy.polyadd(a, b))

    def test_cost(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polyadd
            polyadd(a, b)
            assert budget.flops_used == 3  # max(3, 2)


class TestPolysub:
    def test_cost(self):
        a = numpy.array([1.0, 2.0])
        b = numpy.array([3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polysub
            polysub(a, b)
            assert budget.flops_used == 3  # max(2, 3)


class TestPolyder:
    def test_result_matches_numpy(self):
        coeffs = numpy.array([1.0, -2.0, 3.0])  # x^2 - 2x + 3 -> 2x - 2
        with BudgetContext(flop_budget=10**6):
            from mechestim import polyder
            result = polyder(coeffs)
            assert numpy.allclose(result, numpy.polyder(coeffs))

    def test_cost(self):
        coeffs = numpy.array([1.0, -2.0, 3.0, 4.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polyder
            polyder(coeffs)
            assert budget.flops_used == 4  # n = len(coeffs)


class TestPolyint:
    def test_result_matches_numpy(self):
        coeffs = numpy.array([2.0, -2.0])  # 2x - 2 -> x^2 - 2x
        with BudgetContext(flop_budget=10**6):
            from mechestim import polyint
            result = polyint(coeffs)
            assert numpy.allclose(result, numpy.polyint(coeffs))

    def test_cost(self):
        coeffs = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polyint
            polyint(coeffs)
            assert budget.flops_used == 3


class TestPolymul:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0])
        b = numpy.array([3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import polymul
            result = polymul(a, b)
            assert numpy.allclose(result, numpy.polymul(a, b))

    def test_cost(self):
        a = numpy.array([1.0, 2.0])       # len 2
        b = numpy.array([3.0, 4.0, 5.0])  # len 3
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polymul
            polymul(a, b)
            assert budget.flops_used == 6  # 2 * 3


class TestPolydiv:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, -1.0, 0.0])
        b = numpy.array([1.0, 1.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import polydiv
            q, r = polydiv(a, b)
            q_np, r_np = numpy.polydiv(a, b)
            assert numpy.allclose(q, q_np)

    def test_cost(self):
        a = numpy.array([1.0, -1.0, 0.0])  # len 3
        b = numpy.array([1.0, 1.0])         # len 2
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polydiv
            polydiv(a, b)
            assert budget.flops_used == 6  # 3 * 2


class TestPolyfit:
    def test_result_matches_numpy(self):
        x = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import polyfit
            result = polyfit(x, y, 2)
            expected = numpy.polyfit(x, y, 2)
            assert numpy.allclose(result, expected, atol=1e-10)

    def test_cost(self):
        x = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])  # m = 5
        y = numpy.ones(5)
        deg = 2  # (deg+1) = 3
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import polyfit
            polyfit(x, y, deg)
            # 2 * m * (deg+1)^2 = 2 * 5 * 9 = 90
            assert budget.flops_used == 90


class TestPoly:
    def test_result_matches_numpy(self):
        roots = numpy.array([1.0, 2.0, 3.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import poly
            result = poly(roots)
            assert numpy.allclose(result, numpy.poly(roots))

    def test_cost(self):
        roots = numpy.array([1.0, 2.0, 3.0, 4.0])  # n = 4
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import poly
            poly(roots)
            assert budget.flops_used == 16  # n^2


class TestRoots:
    def test_result_matches_numpy(self):
        coeffs = numpy.array([1.0, -6.0, 11.0, -6.0])  # (x-1)(x-2)(x-3)
        with BudgetContext(flop_budget=10**6):
            from mechestim import roots
            result = roots(coeffs)
            expected = numpy.roots(coeffs)
            assert numpy.allclose(sorted(numpy.real(result)), sorted(numpy.real(expected)))

    def test_cost(self):
        coeffs = numpy.array([1.0, -6.0, 11.0, -6.0])  # n = 3 roots
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import roots
            roots(coeffs)
            # n = len(coeffs) - 1 = 3, cost = 10 * n^3 = 270
            assert budget.flops_used == 270
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_polynomial.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write implementation**

```python
# src/mechestim/_polynomial.py
"""Legacy polynomial operation wrappers with FLOP counting.

Each operation has a co-located cost function documenting the formula,
source, and assumptions. Cost functions are pure (shape params) -> int.
"""
from __future__ import annotations

import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


# ---------------------------------------------------------------------------
# polyval
# ---------------------------------------------------------------------------

def polyval_cost(deg: int, m: int) -> int:
    """FLOP cost of evaluating a polynomial at m points.

    Formula: 2 * m * deg
    Source: Horner's method — deg multiply-adds per evaluation point.
    """
    return max(2 * m * deg, 1)


def polyval(p, x):
    """Evaluate polynomial with FLOP counting."""
    budget = require_budget()
    p = _np.asarray(p)
    x = _np.asarray(x)
    deg = len(p) - 1
    m = max(x.size, 1)
    cost = polyval_cost(deg, m)
    budget.deduct("polyval", flop_cost=cost, subscripts=None, shapes=(p.shape, x.shape))
    return _np.polyval(p, x)


# ---------------------------------------------------------------------------
# polyadd / polysub
# ---------------------------------------------------------------------------

def polyadd_cost(n1: int, n2: int) -> int:
    """FLOP cost of polynomial addition.

    Formula: max(n1, n2)
    Source: Element-wise addition of coefficient vectors.
    """
    return max(n1, n2, 1)


def polyadd(a1, a2):
    """Add two polynomials with FLOP counting."""
    budget = require_budget()
    a1, a2 = _np.asarray(a1), _np.asarray(a2)
    cost = polyadd_cost(len(a1), len(a2))
    budget.deduct("polyadd", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape))
    return _np.polyadd(a1, a2)


def polysub_cost(n1: int, n2: int) -> int:
    """FLOP cost of polynomial subtraction.

    Formula: max(n1, n2)
    Source: Element-wise subtraction of coefficient vectors.
    """
    return max(n1, n2, 1)


def polysub(a1, a2):
    """Subtract two polynomials with FLOP counting."""
    budget = require_budget()
    a1, a2 = _np.asarray(a1), _np.asarray(a2)
    cost = polysub_cost(len(a1), len(a2))
    budget.deduct("polysub", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape))
    return _np.polysub(a1, a2)


# ---------------------------------------------------------------------------
# polyder / polyint
# ---------------------------------------------------------------------------

def polyder_cost(n: int) -> int:
    """FLOP cost of polynomial differentiation.

    Formula: n
    Source: One multiply per coefficient (multiply by power).
    """
    return max(n, 1)


def polyder(p, m=1):
    """Differentiate polynomial with FLOP counting."""
    budget = require_budget()
    p = _np.asarray(p)
    cost = polyder_cost(len(p))
    budget.deduct("polyder", flop_cost=cost, subscripts=None, shapes=(p.shape,))
    return _np.polyder(p, m=m)


def polyint_cost(n: int) -> int:
    """FLOP cost of polynomial integration.

    Formula: n
    Source: One divide per coefficient (divide by new power).
    """
    return max(n, 1)


def polyint(p, m=1, k=None):
    """Integrate polynomial with FLOP counting."""
    budget = require_budget()
    p = _np.asarray(p)
    cost = polyint_cost(len(p))
    budget.deduct("polyint", flop_cost=cost, subscripts=None, shapes=(p.shape,))
    if k is not None:
        return _np.polyint(p, m=m, k=k)
    return _np.polyint(p, m=m)


# ---------------------------------------------------------------------------
# polymul / polydiv
# ---------------------------------------------------------------------------

def polymul_cost(n1: int, n2: int) -> int:
    """FLOP cost of polynomial multiplication.

    Formula: n1 * n2
    Source: Convolution of two coefficient vectors.
    """
    return max(n1 * n2, 1)


def polymul(a1, a2):
    """Multiply two polynomials with FLOP counting."""
    budget = require_budget()
    a1, a2 = _np.asarray(a1), _np.asarray(a2)
    cost = polymul_cost(len(a1), len(a2))
    budget.deduct("polymul", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape))
    return _np.polymul(a1, a2)


def polydiv_cost(n1: int, n2: int) -> int:
    """FLOP cost of polynomial division.

    Formula: n1 * n2
    Source: Polynomial long division.
    """
    return max(n1 * n2, 1)


def polydiv(u, v):
    """Divide polynomials with FLOP counting."""
    budget = require_budget()
    u, v = _np.asarray(u), _np.asarray(v)
    cost = polydiv_cost(len(u), len(v))
    budget.deduct("polydiv", flop_cost=cost, subscripts=None, shapes=(u.shape, v.shape))
    return _np.polydiv(u, v)


# ---------------------------------------------------------------------------
# polyfit
# ---------------------------------------------------------------------------

def polyfit_cost(m: int, deg: int) -> int:
    """FLOP cost of polynomial least-squares fit.

    Formula: 2 * m * (deg + 1)^2
    Source: QR-based least squares on (m, deg+1) Vandermonde matrix.
    """
    return max(2 * m * (deg + 1) ** 2, 1)


def polyfit(x, y, deg, **kwargs):
    """Polynomial fit with FLOP counting."""
    budget = require_budget()
    x, y = _np.asarray(x), _np.asarray(y)
    m = len(x)
    cost = polyfit_cost(m, deg)
    budget.deduct("polyfit", flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape))
    return _np.polyfit(x, y, deg, **kwargs)


# ---------------------------------------------------------------------------
# poly / roots
# ---------------------------------------------------------------------------

def poly_cost(n: int) -> int:
    """FLOP cost of computing polynomial coefficients from n roots.

    Formula: n^2
    Source: Expanding n roots via repeated convolution.
    """
    return max(n ** 2, 1)


def poly(seq_of_zeros):
    """Polynomial from roots with FLOP counting."""
    budget = require_budget()
    seq_of_zeros = _np.asarray(seq_of_zeros)
    if seq_of_zeros.ndim == 2:
        # Matrix input — characteristic polynomial, n = shape[0]
        n = seq_of_zeros.shape[0]
    else:
        n = len(seq_of_zeros)
    cost = poly_cost(n)
    budget.deduct("poly", flop_cost=cost, subscripts=None, shapes=(seq_of_zeros.shape,))
    return _np.poly(seq_of_zeros)


def roots_cost(n: int) -> int:
    """FLOP cost of finding roots of a polynomial with n+1 coefficients.

    Formula: 10 * n^3
    Source: Companion matrix eigendecomposition (same cost model as eig).
    """
    return max(10 * n ** 3, 1)


def roots(p):
    """Polynomial roots with FLOP counting."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p) - 1  # degree = number of roots
    cost = roots_cost(n)
    budget.deduct("roots", flop_cost=cost, subscripts=None, shapes=(p.shape,))
    return _np.roots(p)
```

- [ ] **Step 4: Update top-level `__init__.py`**

Add to `src/mechestim/__init__.py`, after the existing `_free_ops` import block:

```python
# --- Polynomial (counted) ---
from mechestim._polynomial import (  # noqa: F401
    polyval, polyadd, polysub, polyder, polyint,
    polymul, polydiv, polyfit, poly, roots,
)
```

- [ ] **Step 5: Update registry — 10 polynomial entries**

Update all 10 from `"blacklisted"` to `"counted_custom"`:

- `"polyval"`: notes `"Evaluate polynomial. Cost: 2*m*deg (Horner's method)."`
- `"polyadd"`: notes `"Add polynomials. Cost: max(n1, n2) (element-wise add)."`
- `"polysub"`: notes `"Subtract polynomials. Cost: max(n1, n2) (element-wise sub)."`
- `"polyder"`: notes `"Differentiate polynomial. Cost: n (one multiply per coefficient)."`
- `"polyint"`: notes `"Integrate polynomial. Cost: n (one divide per coefficient)."`
- `"polymul"`: notes `"Multiply polynomials. Cost: n1*n2 (convolution)."`
- `"polydiv"`: notes `"Divide polynomials. Cost: n1*n2 (long division)."`
- `"polyfit"`: notes `"Polynomial fit. Cost: 2*m*(deg+1)^2 (QR least squares)."`
- `"poly"`: notes `"Polynomial from roots. Cost: n^2 (repeated convolution)."`
- `"roots"`: notes `"Polynomial roots. Cost: 10*n^3 (companion matrix eigendecomposition)."`

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_polynomial.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/_polynomial.py src/mechestim/__init__.py src/mechestim/_registry.py tests/test_polynomial.py
git commit -m "feat: add polynomial ops (polyval, polyadd, polysub, polymul, polydiv, polyfit, poly, roots, polyder, polyint)"
```

---

## Task 8: Window Functions & Unwrap

**Files:**
- Create: `src/mechestim/_window.py`
- Create: `src/mechestim/_unwrap.py`
- Create: `tests/test_window.py`
- Create: `tests/test_unwrap.py`
- Modify: `src/mechestim/__init__.py`
- Modify: `src/mechestim/_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_window.py
"""Tests for window function wrappers with FLOP counting."""
import numpy
import pytest
from mechestim._budget import BudgetContext


class TestBartlett:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from mechestim import bartlett
            result = bartlett(10)
            assert numpy.allclose(result, numpy.bartlett(10))

    def test_cost(self):
        n = 10
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import bartlett
            bartlett(n)
            assert budget.flops_used == n


class TestBlackman:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from mechestim import blackman
            result = blackman(10)
            assert numpy.allclose(result, numpy.blackman(10))

    def test_cost(self):
        n = 10
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import blackman
            blackman(n)
            assert budget.flops_used == 3 * n


class TestHamming:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from mechestim import hamming
            result = hamming(10)
            assert numpy.allclose(result, numpy.hamming(10))

    def test_cost(self):
        n = 10
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import hamming
            hamming(n)
            assert budget.flops_used == n


class TestHanning:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from mechestim import hanning
            result = hanning(10)
            assert numpy.allclose(result, numpy.hanning(10))

    def test_cost(self):
        n = 10
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import hanning
            hanning(n)
            assert budget.flops_used == n


class TestKaiser:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from mechestim import kaiser
            result = kaiser(10, 5.0)
            assert numpy.allclose(result, numpy.kaiser(10, 5.0))

    def test_cost(self):
        n = 10
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import kaiser
            kaiser(n, 5.0)
            assert budget.flops_used == 3 * n
```

```python
# tests/test_unwrap.py
"""Tests for unwrap wrapper with FLOP counting."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestUnwrap:
    def test_result_matches_numpy(self):
        phase = numpy.array([0.0, 1.0, 2.0, 3.0, -3.0, -2.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim import unwrap
            result = unwrap(phase)
            assert numpy.allclose(result, numpy.unwrap(phase))

    def test_cost(self):
        x = numpy.random.randn(20)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim import unwrap
            unwrap(x)
            assert budget.flops_used == 20  # numel

    def test_outside_context_raises(self):
        from mechestim import unwrap
        with pytest.raises(NoBudgetContextError):
            unwrap(numpy.ones(5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_window.py tests/test_unwrap.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write window implementation**

```python
# src/mechestim/_window.py
"""Window function wrappers with FLOP counting.

Each function generates a deterministic window of length n.
Cost functions document the per-sample computation involved.
"""
from __future__ import annotations

import numpy as _np

from mechestim._validation import require_budget


def bartlett_cost(n: int) -> int:
    """FLOP cost of Bartlett (triangular) window.

    Formula: n
    Source: One linear evaluation per sample.
    """
    return max(n, 1)


def bartlett(M):
    """Bartlett window with FLOP counting."""
    budget = require_budget()
    cost = bartlett_cost(M)
    budget.deduct("bartlett", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.bartlett(M)


def blackman_cost(n: int) -> int:
    """FLOP cost of Blackman window.

    Formula: 3 * n
    Source: Three cosine terms per sample: 0.42 - 0.5*cos + 0.08*cos.
    """
    return max(3 * n, 1)


def blackman(M):
    """Blackman window with FLOP counting."""
    budget = require_budget()
    cost = blackman_cost(M)
    budget.deduct("blackman", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.blackman(M)


def hamming_cost(n: int) -> int:
    """FLOP cost of Hamming window.

    Formula: n
    Source: One cosine evaluation per sample: 0.54 - 0.46*cos.
    """
    return max(n, 1)


def hamming(M):
    """Hamming window with FLOP counting."""
    budget = require_budget()
    cost = hamming_cost(M)
    budget.deduct("hamming", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.hamming(M)


def hanning_cost(n: int) -> int:
    """FLOP cost of Hanning window.

    Formula: n
    Source: One cosine evaluation per sample: 0.5*(1 - cos).
    """
    return max(n, 1)


def hanning(M):
    """Hanning window with FLOP counting."""
    budget = require_budget()
    cost = hanning_cost(M)
    budget.deduct("hanning", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.hanning(M)


def kaiser_cost(n: int) -> int:
    """FLOP cost of Kaiser window.

    Formula: 3 * n
    Source: Modified Bessel function I0 evaluation per sample (~3 ops each).
    """
    return max(3 * n, 1)


def kaiser(M, beta):
    """Kaiser window with FLOP counting."""
    budget = require_budget()
    cost = kaiser_cost(M)
    budget.deduct("kaiser", flop_cost=cost, subscripts=None, shapes=((M,),))
    return _np.kaiser(M, beta)
```

- [ ] **Step 4: Write unwrap implementation**

```python
# src/mechestim/_unwrap.py
"""Unwrap wrapper with FLOP counting."""
from __future__ import annotations

import numpy as _np

from mechestim._validation import require_budget, validate_ndarray


def unwrap_cost(shape: tuple[int, ...]) -> int:
    """FLOP cost of phase unwrapping.

    Formula: numel(input)
    Source: Diff + conditional period adjustment = O(n) pass.
    """
    numel = 1
    for d in shape:
        numel *= d
    return max(numel, 1)


def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):
    """Unwrap phase angles with FLOP counting."""
    budget = require_budget()
    validate_ndarray(p)
    cost = unwrap_cost(p.shape)
    budget.deduct("unwrap", flop_cost=cost, subscripts=None, shapes=(p.shape,))
    kwargs = {"axis": axis, "period": period}
    if discont is not None:
        kwargs["discont"] = discont
    return _np.unwrap(p, **kwargs)
```

- [ ] **Step 5: Update top-level `__init__.py`**

Add to `src/mechestim/__init__.py`, after the polynomial import block:

```python
# --- Window functions (counted) ---
from mechestim._window import (  # noqa: F401
    bartlett, blackman, hamming, hanning, kaiser,
)

# --- Unwrap (counted) ---
from mechestim._unwrap import unwrap  # noqa: F401
```

- [ ] **Step 6: Update registry — 6 entries**

Update these from `"blacklisted"` to `"counted_custom"`:

- `"bartlett"`: notes `"Bartlett window. Cost: n (one linear eval per sample)."`
- `"blackman"`: notes `"Blackman window. Cost: 3*n (three cosine terms per sample)."`
- `"hamming"`: notes `"Hamming window. Cost: n (one cosine per sample)."`
- `"hanning"`: notes `"Hanning window. Cost: n (one cosine per sample)."`
- `"kaiser"`: notes `"Kaiser window. Cost: 3*n (Bessel function eval per sample)."`
- `"unwrap"`: notes `"Phase unwrap. Cost: numel(input) (diff + conditional adjustment)."`

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_window.py tests/test_unwrap.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/mechestim/_window.py src/mechestim/_unwrap.py src/mechestim/__init__.py src/mechestim/_registry.py tests/test_window.py tests/test_unwrap.py
git commit -m "feat: add window functions (bartlett, blackman, hamming, hanning, kaiser) and unwrap"
```

---

## Task 9: Update `flops.py` Public API

**Files:**
- Modify: `src/mechestim/flops.py`

- [ ] **Step 1: Update `flops.py` to re-export all cost functions**

Replace the full contents of `src/mechestim/flops.py` with:

```python
"""Public FLOP cost query API.

All cost functions are pure (shape params) -> int. They can be used for
pre-flight cost estimation without running any computation.
"""
# Existing
from mechestim._flops import einsum_cost, pointwise_cost, reduction_cost, svd_cost  # noqa: F401

# Linalg — decompositions
from mechestim.linalg._decompositions import (  # noqa: F401
    cholesky_cost, qr_cost, eig_cost, eigh_cost,
    eigvals_cost, eigvalsh_cost, svdvals_cost,
)

# Linalg — solvers
from mechestim.linalg._solvers import (  # noqa: F401
    solve_cost, inv_cost, lstsq_cost, pinv_cost,
    tensorsolve_cost, tensorinv_cost,
)

# Linalg — properties
from mechestim.linalg._properties import (  # noqa: F401
    trace_cost, det_cost, slogdet_cost, norm_cost,
    vector_norm_cost, matrix_norm_cost, cond_cost, matrix_rank_cost,
)

# Linalg — compound
from mechestim.linalg._compound import multi_dot_cost, matrix_power_cost  # noqa: F401

# FFT
from mechestim.fft._transforms import (  # noqa: F401
    fft_cost, rfft_cost, fftn_cost, rfftn_cost, hfft_cost,
)

# Polynomial
from mechestim._polynomial import (  # noqa: F401
    polyval_cost, polyadd_cost, polysub_cost, polymul_cost,
    polydiv_cost, polyfit_cost, poly_cost, roots_cost,
    polyder_cost, polyint_cost,
)

# Window
from mechestim._window import (  # noqa: F401
    bartlett_cost, blackman_cost, hamming_cost, hanning_cost, kaiser_cost,
)

# Other
from mechestim._unwrap import unwrap_cost  # noqa: F401

__all__ = [
    # Existing
    "einsum_cost", "pointwise_cost", "reduction_cost", "svd_cost",
    # Linalg
    "cholesky_cost", "qr_cost", "eig_cost", "eigh_cost",
    "eigvals_cost", "eigvalsh_cost", "svdvals_cost",
    "solve_cost", "inv_cost", "lstsq_cost", "pinv_cost",
    "tensorsolve_cost", "tensorinv_cost",
    "trace_cost", "det_cost", "slogdet_cost", "norm_cost",
    "vector_norm_cost", "matrix_norm_cost", "cond_cost", "matrix_rank_cost",
    "multi_dot_cost", "matrix_power_cost",
    # FFT
    "fft_cost", "rfft_cost", "fftn_cost", "rfftn_cost", "hfft_cost",
    # Polynomial
    "polyval_cost", "polyadd_cost", "polysub_cost", "polymul_cost",
    "polydiv_cost", "polyfit_cost", "poly_cost", "roots_cost",
    "polyder_cost", "polyint_cost",
    # Window
    "bartlett_cost", "blackman_cost", "hamming_cost", "hanning_cost", "kaiser_cost",
    # Other
    "unwrap_cost",
]
```

- [ ] **Step 2: Run flops tests**

Run: `pytest tests/test_flops.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/mechestim/flops.py
git commit -m "feat: update flops.py to re-export all new cost functions"
```

---

## Task 10: Full Integration Test & Audit

**Files:**
- Modify: `tests/test_getattr.py` (if it tests blacklisted ops that are now implemented)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All PASS. If any `test_getattr.py` or `test_linalg.py` tests fail because they expected `AttributeError` on now-implemented ops (e.g., `linalg.cholesky`), update those tests.

- [ ] **Step 2: Run audit script**

Run: `python scripts/numpy_audit.py --filter blacklisted`
Expected: Only 32 entries remain (IO, config, formatting, datetime, meta).

Run: `python scripts/numpy_audit.py --filter unclassified`
Expected: 0 unclassified entries.

- [ ] **Step 3: Fix any failing tests from previous tasks**

If `test_linalg.py::test_linalg_unsupported` fails because it tests `linalg.cholesky` as unsupported, update the test to use a permanently blacklisted op instead (e.g., test that `linalg` doesn't provide some non-existent function, or update the test to check one of the permanently blacklisted ops).

- [ ] **Step 4: Commit any test fixes**

```bash
git add -A tests/
git commit -m "fix: update existing tests for newly-implemented ops"
```

- [ ] **Step 5: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: All PASS with 0 failures.
