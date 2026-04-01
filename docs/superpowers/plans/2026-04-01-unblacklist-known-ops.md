# Plan: Implement Currently-Blacklisted Ops with Known Complexity

**Status:** Planned (not started)
**Prerequisite:** Exhaustive numpy coverage (done — all 482 functions registered)

## Goal

Move blacklisted functions with known analytical FLOP costs from `blacklisted` to `counted_custom`, implementing proper wrappers with cost formulas. This is incremental — each group can be a standalone PR.

---

## Group 1: Linear Algebra — Core Decompositions

These are the most impactful for competition participants.

| Function | Cost Formula | Notes |
|----------|-------------|-------|
| `linalg.cholesky` | `n^3 // 3` | Cholesky of (n, n) matrix |
| `linalg.qr` | `2 * m * n^2 - (2/3) * n^3` | QR of (m, n), m >= n |
| `linalg.eig` | `n^3` | Full eigendecomposition |
| `linalg.eigh` | `n^3` | Symmetric eigendecomposition (cheaper constant) |
| `linalg.eigvals` | `n^3` | Eigenvalues only |
| `linalg.eigvalsh` | `n^3` | Symmetric eigenvalues only |
| `linalg.svdvals` | `m * n * min(m,n)` | Singular values only (same as existing svd) |

**Implementation pattern:** Same as existing `linalg.svd` — custom wrapper in `linalg/` with `require_budget()`, `validate_ndarray()`, cost formula, `budget.deduct()`.

## Group 2: Linear Algebra — Solvers & Inverses

| Function | Cost Formula | Notes |
|----------|-------------|-------|
| `linalg.solve` | `n^3` | Solve Ax=b for (n, n) A |
| `linalg.inv` | `n^3` | Matrix inverse |
| `linalg.lstsq` | `2 * m * n^2` | Least squares via QR |
| `linalg.pinv` | `m * n * min(m,n)` | Pseudoinverse via SVD |
| `linalg.tensorsolve` | product of dims | Tensor solve |
| `linalg.tensorinv` | product of dims | Tensor inverse |

## Group 3: Linear Algebra — Scalar/Reduction Outputs

| Function | Cost Formula | Notes |
|----------|-------------|-------|
| `linalg.det` | `n^3` | Determinant via LU |
| `linalg.slogdet` | `n^3` | Sign + log determinant |
| `linalg.norm` | `numel(input)` | Just a reduction — could even be `counted_reduction` |
| `linalg.vector_norm` | `numel(input)` | Same as norm for vectors |
| `linalg.matrix_norm` | `numel(input)` | Same as norm for matrices |
| `linalg.cond` | `m * n * min(m,n)` | Condition number via SVD |
| `linalg.matrix_rank` | `m * n * min(m,n)` | Rank via SVD |

## Group 4: Linear Algebra — Free/Passthrough

These are aliases or views — zero FLOP cost, should be `free`:

| Function | Reason |
|----------|--------|
| `linalg.diagonal` | Same as `numpy.diagonal` — view operation |
| `linalg.trace` | Same as `numpy.trace` — sum of diagonal |
| `linalg.matrix_transpose` | Same as `numpy.transpose` — view |
| `linalg.matmul` | Already implemented as top-level `matmul` |
| `linalg.cross` | Already implemented as top-level `cross` |
| `linalg.outer` | Already implemented as top-level `outer` |
| `linalg.tensordot` | Already implemented as top-level `tensordot` |
| `linalg.vecdot` | Already implemented as top-level `vecdot` |
| `linalg.multi_dot` | Chain of matmuls — cost = sum of individual matmul costs |
| `linalg.matrix_power` | Repeated matmul — cost = (k-1) * n^3 for power k |

## Group 5: FFT — O(n log n) Operations

Lower priority since FFT is niche for the competition, but the cost model is well-known:

| Function | Cost Formula | Notes |
|----------|-------------|-------|
| `fft.fft` | `5 * n * log2(n)` | Standard radix-2 FFT estimate |
| `fft.ifft` | `5 * n * log2(n)` | Same cost as forward |
| `fft.rfft` | `5 * n/2 * log2(n)` | Real-input FFT |
| `fft.fft2` / `fftn` | `5 * N * log2(N)` where N = product of shape | Multi-dimensional |
| `fft.fftfreq` / `rfftfreq` | 0 | Just index generation — free |
| `fft.fftshift` / `ifftshift` | 0 | Just rearrangement — free |
| `fft.hfft` / `ihfft` | `5 * n * log2(n)` | Hermitian FFT |

## Group 6: Permanently Blacklisted

These should stay blacklisted — no FLOP counting makes sense:

- **IO:** `load`, `loadtxt`, `save`, `savetxt`, `savez`, `savez_compressed`, `genfromtxt`
- **Config:** `seterr`, `geterr`, `seterrcall`, `geterrcall`, `setbufsize`, `getbufsize`, `show_config`, `show_runtime`, `get_include`
- **Formatting:** `array2string`, `array_repr`, `array_str`, `format_float_positional`, `format_float_scientific`
- **Datetime:** `busday_count`, `busday_offset`, `is_busday`, `datetime_as_string`, `datetime_data`
- **Window functions:** `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser` (could be free but unlikely needed)
- **Polynomial (legacy):** `poly`, `roots`, `polyadd`, `polyder`, `polydiv`, `polyfit`, `polyint`, `polymul`, `polysub`, `polyval`
- **Meta/advanced:** `apply_along_axis`, `apply_over_axes`, `piecewise`, `frompyfunc`, `nested_iters`, `asmatrix`, `unwrap`

---

## Suggested Implementation Order

1. **Group 4 first** (free linalg aliases) — trivial, instant coverage boost
2. **Group 3** (scalar linalg outputs) — simple, high-value for norm/det users
3. **Groups 1 & 2** (decompositions & solvers) — the main linalg workload
4. **Group 5** (FFT) — if competition participants need it
5. **Group 6** — never, these stay blacklisted
