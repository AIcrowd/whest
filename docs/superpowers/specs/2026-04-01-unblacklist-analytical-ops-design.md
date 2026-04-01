# Design: Unblacklist Analytically-Calculable Operations

**Date:** 2026-04-01
**Status:** Approved (pending spec review)
**Supersedes:** `docs/superpowers/plans/2026-04-01-unblacklist-known-ops.md` (gap analysis found missing ops and formula corrections)

## Goal

Move 66 blacklisted numpy functions with known analytical FLOP costs from `blacklisted` to `counted_custom` or `free`, implementing proper wrappers with documented cost formulas. Each formula represents our best analytical estimate and can be surgically edited later by the theory team.

## Design Principles

1. **Best analytical estimate now** — use accepted textbook formulas; constants may be approximate
2. **Clearly documented** — every cost function includes the formula, its source, and assumptions
3. **Surgically editable** — cost functions are co-located with their wrappers in domain-grouped files; the theory team can update any formula by editing a single function
4. **`import mechestim as np` compatibility** — all ops accessible via the same paths as numpy (`np.linalg.cholesky`, `np.fft.fft`, `np.polyval`, etc.)
5. **No `check_nan_inf`** — new wrappers rely on native numpy error behavior

## Philosophy: Parameter-Dependent vs Algorithm-Dependent Costs

Many ops have costs that depend on runtime parameters (e.g., `norm(x, ord=2)` vs `norm(x, ord=1)`). These are **not** algorithm-dependent — we have the parameters at call time and dispatch to the correct formula.

For genuinely data-dependent algorithms (e.g., QR iteration convergence in `eig`), we use universally-accepted asymptotic estimates (e.g., `~10*n^3` for nonsymmetric eigendecomposition). This is the same approach the project already uses for `svd_cost = m * n * k`.

| Concern | Resolution |
|---|---|
| `norm(x, ord=...)` dispatches on `ord` | `ord` is a param — inspect and pick formula |
| `matrix_power(A, k)` naive vs squaring | NumPy uses squaring; `k` is a param |
| `lstsq` LAPACK driver | NumPy uses `gelsd` (SVD-based) by default |
| `eig` QR iteration convergence | Use accepted `~10*n^3` (Golub & Van Loan §7.5) |
| FFT constant factor `5` in `5*n*log2(n)` | Universally-cited Cooley-Tukey estimate |

---

## File Structure

```
src/mechestim/
  linalg/
    __init__.py            # imports & re-exports everything
    _svd.py                # existing (unchanged)
    _decompositions.py     # NEW: cholesky, qr, eig, eigh, eigvals, eigvalsh, svdvals
    _solvers.py            # NEW: solve, inv, lstsq, pinv, tensorsolve, tensorinv
    _properties.py         # NEW: det, slogdet, norm, vector_norm, matrix_norm, cond, matrix_rank, trace
    _compound.py           # NEW: multi_dot, matrix_power
    _aliases.py            # NEW: matmul, cross, outer, tensordot, vecdot, diagonal, matrix_transpose
  fft/
    __init__.py            # imports & re-exports everything
    _transforms.py         # NEW: fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn, rfft2, irfft2, rfftn, irfftn, hfft, ihfft
    _free.py               # NEW: fftfreq, rfftfreq, fftshift, ifftshift
  _polynomial.py           # NEW: polyval, polyadd, polysub, polymul, polydiv, polyfit, poly, roots, polyder, polyint
  _window.py               # NEW: bartlett, blackman, hamming, hanning, kaiser
  _unwrap.py               # NEW: unwrap
  flops.py                 # UPDATED: re-exports all cost functions for pre-flight queries
  _registry.py             # UPDATED: categories + notes with formulas
```

### Wrapper Pattern

Every counted wrapper follows this structure:

```python
def cholesky_cost(n: int) -> int:
    """FLOP cost of Cholesky decomposition of an (n, n) matrix.

    Formula: n^3 / 3
    Source: Golub & Van Loan, "Matrix Computations", 4th ed., §4.2
    Assumes: Standard column-outer-product algorithm.
    """
    return max(n ** 3 // 3, 1)


def cholesky(a):
    """Cholesky decomposition with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    cost = cholesky_cost(a.shape[-1])
    budget.deduct("linalg.cholesky", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.cholesky(a)
```

### Alias Pattern

For linalg ops that delegate to already-implemented top-level functions:

```python
import mechestim as _me

def matmul(a, b):
    """Matrix multiply (linalg namespace). Delegates to mechestim.matmul."""
    return _me.matmul(a, b)
```

---

## Complete Op Catalog

### linalg — Free Aliases (7 ops)

These delegate to existing top-level implementations. File: `linalg/_aliases.py`.

| Op | Delegates to | Category |
|---|---|---|
| `linalg.matmul` | `mechestim.matmul` | counted (via delegation) |
| `linalg.cross` | `mechestim.cross` | counted (via delegation) |
| `linalg.outer` | `mechestim.outer` | counted (via delegation) |
| `linalg.tensordot` | `mechestim.tensordot` | counted (via delegation) |
| `linalg.vecdot` | `mechestim.vecdot` | counted (via delegation) |
| `linalg.diagonal` | `mechestim.diagonal` | free (via delegation) |
| `linalg.matrix_transpose` | `mechestim.transpose` | free (via delegation) |

### linalg — Decompositions (7 ops)

File: `linalg/_decompositions.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `cholesky` | `cholesky_cost(n)` | `n^3 // 3` | Golub & Van Loan §4.2 |
| `qr` | `qr_cost(m, n)` | `2*m*n^2 - (2*n^3)//3` | Golub & Van Loan §5.2 |
| `eig` | `eig_cost(n)` | `10 * n^3` | Francis double-shift QR, G&VL §7.5 |
| `eigh` | `eigh_cost(n)` | `(4 * n^3) // 3` | Symmetric tridiag + QR, G&VL §8.3 |
| `eigvals` | `eigvals_cost(n)` | `10 * n^3` | Same algorithm as eig, no back-substitution |
| `eigvalsh` | `eigvalsh_cost(n)` | `(4 * n^3) // 3` | Same algorithm as eigh |
| `svdvals` | `svdvals_cost(m, n)` | `m * n * min(m, n)` | Golub-Reinsch bidiagonalization |

### linalg — Solvers & Inverses (6 ops)

File: `linalg/_solvers.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `solve` | `solve_cost(n)` | `n^3` | LU factorization + back-substitution |
| `inv` | `inv_cost(n)` | `n^3` | LU + solve for n right-hand sides |
| `lstsq` | `lstsq_cost(m, n)` | `m * n * min(m, n)` | LAPACK gelsd (SVD-based, numpy default) |
| `pinv` | `pinv_cost(m, n)` | `m * n * min(m, n)` | Pseudoinverse via SVD |
| `tensorsolve` | `tensorsolve_cost(shape)` | `n^3` after reshape | Reduces to `solve` |
| `tensorinv` | `tensorinv_cost(shape)` | `n^3` after reshape | Reduces to `inv` |

### linalg — Properties & Reductions (8 ops)

File: `linalg/_properties.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `trace` | `trace_cost(n)` | `n` | Sum of n diagonal elements |
| `det` | `det_cost(n)` | `n^3` | LU factorization |
| `slogdet` | `slogdet_cost(n)` | `n^3` | Same algorithm as det |
| `norm` | `norm_cost(shape, ord)` | Dispatches on `ord` (see below) | Direct analysis |
| `vector_norm` | `vector_norm_cost(shape, ord)` | Dispatches on `ord` | Direct analysis |
| `matrix_norm` | `matrix_norm_cost(shape, ord)` | Dispatches on `ord` | Direct analysis |
| `cond` | `cond_cost(m, n)` | `m * n * min(m, n)` | SVD + ratio of extremal singular values |
| `matrix_rank` | `matrix_rank_cost(m, n)` | `m * n * min(m, n)` | SVD + threshold |

**Norm dispatch:**

| `ord` value | Vector cost | Matrix cost |
|---|---|---|
| `None` (default) | `numel` (L2 = sqrt of sum of squares) | `2 * numel` (Frobenius = square + sum) |
| `'fro'` | N/A | `2 * numel` |
| `1`, `-1` | `numel` | `numel` (column sums) |
| `inf`, `-inf` | `numel` | `numel` (row sums) |
| `2`, `-2` | `numel` | `m * n * min(m, n)` (requires SVD) |
| numeric `p` | `2 * numel` (power + sum) | N/A |

### linalg — Compound Ops (2 ops)

File: `linalg/_compound.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `multi_dot` | `multi_dot_cost(shapes)` | Sum of matmul costs using optimal chain ordering | Dynamic programming for matrix chain order |
| `matrix_power` | `matrix_power_cost(n, k)` | `(floor(log2(k)) + popcount(k) - 1) * n^3` | Exponentiation by squaring |

Note: `matrix_power` for `k=0` returns identity (free), `k=1` returns copy (free), `k<0` requires inversion first (`n^3` for inv + power cost for `|k|`).

### fft — Transforms (14 ops)

File: `fft/_transforms.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `fft` | `fft_cost(n)` | `5 * n * ceil(log2(n))` | Cooley-Tukey radix-2 |
| `ifft` | — | `5 * n * ceil(log2(n))` | Same as forward |
| `rfft` | `rfft_cost(n)` | `5 * (n//2) * ceil(log2(n))` | Real-input half-spectrum |
| `irfft` | — | `5 * (n//2) * ceil(log2(n))` | Inverse real |
| `fft2` | `fftn_cost(shape)` | `5 * N * ceil(log2(N))`, N=prod(shape) | Row-then-column FFT |
| `ifft2` | — | Same as fft2 | Inverse |
| `fftn` | `fftn_cost(shape)` | `5 * N * ceil(log2(N))` | N-D generalization |
| `ifftn` | — | Same as fftn | Inverse |
| `rfft2` | `rfftn_cost(shape)` | `5 * (N//2) * ceil(log2(N))` | Real N-D |
| `irfft2` | — | Same as rfft2 | Inverse |
| `rfftn` | `rfftn_cost(shape)` | `5 * (N//2) * ceil(log2(N))` | Real N-D |
| `irfftn` | — | Same as rfftn | Inverse |
| `hfft` | `hfft_cost(n)` | `5 * n * ceil(log2(n))` | Hermitian FFT |
| `ihfft` | — | Same as hfft | Inverse |

### fft — Free Ops (4 ops)

File: `fft/_free.py`.

| Op | Category | Reason |
|---|---|---|
| `fftfreq` | free | Index generation, no arithmetic |
| `rfftfreq` | free | Index generation, no arithmetic |
| `fftshift` | free | Array rearrangement, no arithmetic |
| `ifftshift` | free | Array rearrangement, no arithmetic |

### Top-level — Polynomial Ops (10 ops)

File: `_polynomial.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `polyval` | `polyval_cost(deg, m)` | `2 * m * deg` | Horner's method: m points, deg degree |
| `polyadd` | `polyadd_cost(n1, n2)` | `max(n1, n2)` | Element-wise addition |
| `polysub` | `polysub_cost(n1, n2)` | `max(n1, n2)` | Element-wise subtraction |
| `polyder` | `polyder_cost(n)` | `n` | One multiply per coefficient |
| `polyint` | `polyint_cost(n)` | `n` | One divide per coefficient |
| `polymul` | `polymul_cost(n1, n2)` | `n1 * n2` | Convolution of coefficient vectors |
| `polydiv` | `polydiv_cost(n1, n2)` | `n1 * n2` | Polynomial long division |
| `polyfit` | `polyfit_cost(m, deg)` | `2 * m * (deg+1)^2` | QR-based least squares fit |
| `poly` | `poly_cost(n)` | `n^2` | Expanding n roots via repeated convolution |
| `roots` | `roots_cost(n)` | `10 * n^3` | Companion matrix eigendecomposition |

### Top-level — Window Functions (5 ops)

File: `_window.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `bartlett` | `bartlett_cost(n)` | `n` | One linear eval per sample |
| `blackman` | `blackman_cost(n)` | `3 * n` | Three cosine terms per sample |
| `hamming` | `hamming_cost(n)` | `n` | One cosine per sample |
| `hanning` | `hanning_cost(n)` | `n` | One cosine per sample |
| `kaiser` | `kaiser_cost(n)` | `3 * n` | Bessel function eval per sample |

### Top-level — Other (1 op)

File: `_unwrap.py`.

| Op | Cost Function | Formula | Source |
|---|---|---|---|
| `unwrap` | `unwrap_cost(shape)` | `numel(input)` | Diff + conditional period adjustment |

---

## Permanently Blacklisted (30 ops, no change)

These stay `blacklisted` — no analytical FLOP counting makes sense:

- **IO (7):** `load`, `loadtxt`, `genfromtxt`, `save`, `savetxt`, `savez`, `savez_compressed`
- **Config (7):** `seterr`, `geterr`, `seterrcall`, `geterrcall`, `setbufsize`, `getbufsize`, `get_include`
- **Runtime/meta (2):** `show_config`, `show_runtime`
- **Formatting (5):** `array2string`, `array_repr`, `array_str`, `format_float_positional`, `format_float_scientific`
- **Datetime (5):** `busday_count`, `busday_offset`, `is_busday`, `datetime_as_string`, `datetime_data`
- **Advanced/meta (6):** `apply_along_axis`, `apply_over_axes`, `piecewise`, `frompyfunc`, `nested_iters`, `asmatrix`

---

## Registry Updates

Each unblacklisted op gets its registry entry updated:

**Before:**
```python
"linalg.cholesky": {
    "category": "blacklisted",
    "module": "numpy.linalg",
    "notes": "Cholesky decomposition. Not yet supported.",
},
```

**After:**
```python
"linalg.cholesky": {
    "category": "counted_custom",
    "module": "numpy.linalg",
    "notes": "Cholesky decomposition. Cost: n^3/3 (Golub & Van Loan §4.2).",
},
```

Free aliases and views get `"category": "free"` with notes indicating delegation.

---

## Public API: `mechestim.flops`

The existing `mechestim/flops.py` module is updated to re-export all cost functions:

```python
# Existing
from mechestim._flops import einsum_cost, pointwise_cost, reduction_cost, svd_cost

# New — linalg
from mechestim.linalg._decompositions import (
    cholesky_cost, qr_cost, eig_cost, eigh_cost, eigvals_cost, eigvalsh_cost, svdvals_cost,
)
from mechestim.linalg._solvers import solve_cost, inv_cost, lstsq_cost, pinv_cost
from mechestim.linalg._properties import (
    det_cost, slogdet_cost, norm_cost, vector_norm_cost, matrix_norm_cost,
    cond_cost, matrix_rank_cost, trace_cost,
)
from mechestim.linalg._compound import multi_dot_cost, matrix_power_cost

# New — fft
from mechestim.fft._transforms import fft_cost, rfft_cost, fftn_cost, rfftn_cost, hfft_cost

# New — top-level
from mechestim._polynomial import (
    polyval_cost, polyadd_cost, polysub_cost, polymul_cost, polydiv_cost,
    polyfit_cost, poly_cost, roots_cost, polyder_cost, polyint_cost,
)
from mechestim._window import bartlett_cost, blackman_cost, hamming_cost, hanning_cost, kaiser_cost
from mechestim._unwrap import unwrap_cost
```

Users can query costs without running anything: `me.flops.cholesky_cost(n=1000)`.

---

## Testing Strategy

Test files mirror the implementation grouping:

```
tests/
  test_linalg_decompositions.py
  test_linalg_solvers.py
  test_linalg_properties.py
  test_linalg_compound.py
  test_linalg_aliases.py
  test_fft_transforms.py
  test_fft_free.py
  test_polynomial.py
  test_window.py
  test_unwrap.py
```

Each test file verifies per op:
1. **Cost function correctness** — known inputs produce expected FLOP counts
2. **Budget deduction** — wrapper deducts correct amount from `BudgetContext`
3. **Numpy delegation** — wrapper produces same result as direct numpy call
4. **Budget exhaustion** — `BudgetExhaustedError` raised before execution when budget insufficient

---

## Summary

| Category | Count | Description |
|---|---|---|
| linalg aliases | 7 | Delegate to existing top-level ops |
| linalg counted | 23 | Decompositions, solvers, properties, compound |
| fft transforms | 14 | All FFT/IFFT variants |
| fft free | 4 | Frequency generation, shift ops |
| polynomial | 10 | Legacy polynomial operations |
| window | 5 | Window functions |
| top-level other | 1 | `unwrap` |
| **Total unblacklisted** | **64** | |
| Permanently blacklisted | 32 | IO, config, formatting, datetime, meta |
| **Grand total blacklisted** | **96** | All accounted for |
