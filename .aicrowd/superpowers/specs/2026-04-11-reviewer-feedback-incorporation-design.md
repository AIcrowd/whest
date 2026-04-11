# Incorporating Reviewer Feedback into mechestim

## Context

A reviewer examined 453 of 482 operations in the FLOP weight calibration
spreadsheet and provided feedback across 5 categories: weight overrides,
formula changes, free-op costing, unblocking ops, and sparse SVD cost.
All changes are to be implemented in a single sprint.

## Tracking

A "Review Status" column will be added to the Google Sheet with values:
`pending`, `accepted`, `rejected`, `modified`. Updated programmatically
as each workstream completes. The sheet remains the single source of truth.

## Workstream A: Weight Tier System

**What:** Replace 291 precise empirical weights with reviewer's 4-tier
system.

| Tier | Weight | Operations |
|------|:------:|:-----------|
| 1 | 1 | Arithmetic, comparisons, reductions, sorting, FFT, contractions, polynomial, most misc, bitwise, complex |
| 2 | 2 | std, var, nanstd, nanvar |
| 4 | 4 | All linalg decompositions (svd, cholesky, qr, solve, etc.) |
| 16 | 16 | Transcendentals (sin/cos/exp/log), random distributions, gcd/lcm, median/percentile, window functions, roots, angle |

**Implementation:**
1. Read reviewer weights from Google Sheet column F
2. Build `op_name -> reviewer_weight` map
3. Overwrite `weights` dict in `weights.json` with reviewer values
4. Keep empirical values in `meta.validation.absolute_correction_factors` for reference
5. Regenerate CSV, markdown, spreadsheet
6. Accept all reviewer values exactly — including rounding sub-1 ops up to 1

**Files:** `src/mechestim/data/weights.json`, `src/mechestim/data/weights.csv`,
`docs/reference/empirical-weights.md`, Google Sheet

## Workstream B: Formula Changes

**What:** Update analytical cost formulas where the reviewer explicitly
requested a change. Linalg decompositions keep their current formulas —
the reviewer's weight=4 serves as the correction factor on top.

### Contraction ops — treat FMA as single op (drop factor of 2):
| Op | Current formula | New formula | Rationale |
|----|:---|:---|:---|
| `matmul`, `dot` | `2*M*N*K` | `M*N*K` | FMA = 1 op |
| `einsum` op_factor | `2` (multiply+add) | `1` (FMA) | Same |
| `tensordot` | `2*∏(free)*∏(contracted)` | `∏(free)*∏(contracted)` | Same |
| `kron` | current | `numel(output)` | Simplify |
| `inner`, `vdot` | `a.size` | Keep | Already charges N, no factor of 2 |

### Linalg — NO formula changes, weight=4 handles correction:
Keep current formulas (`n^3/3` for Cholesky, `2*m*n^2 - 2*n^3/3` for QR,
etc.). The reviewer's weight=4 applies on top as the correction factor.
Column G suggestions (`n^3`) were the reviewer's estimate of the total
cost, not a replacement formula.

### Other formula changes:
| Op | Current | New | Rationale |
|----|:---|:---|:---|
| `sort_complex` | `numel(output)` | `n*ceil(log2(n))` | It's a sort |
| `argpartition` | `n` | `n * len(k)` | Scales with kth count |
| `svd_cost(m,n,k)` | `m*n*k` | `4*m*n*k` | Reviewer's explicit request |

**Files:** `src/mechestim/_flops.py` (svd_cost, einsum op_factor),
`src/mechestim/_pointwise.py` (dot, matmul, tensordot, kron),
`src/mechestim/_counting_ops.py` (sort_complex, argpartition),
`src/mechestim/_opt_einsum/_contract.py` (op_factor),
`src/mechestim/_registry.py` (notes),
benchmark `_analytical_cost` functions

## Workstream C: Count Free Ops (75 ops)

**What:** Move 75 ops from `category: "free"` to `category: "counted_custom"`
and add cost functions. 63 ops stay free (views/metadata).

**Sub-groups:**

### C1: Element-scanning ops (weight = 1, cost = numel(input))
`isnan`, `isinf`, `isfinite`, `nonzero`, `argwhere`, `flatnonzero`,
`extract`, `place`, `put`, `put_along_axis`, `putmask`, `select`,
`where`, `compress`, `fill_diagonal`, `packbits`, `unpackbits`

### C2: Array-creation/copy ops (weight = 1, cost = numel(output))
`append`, `concatenate`, `concat`, `stack`, `vstack`, `dstack`, `block`,
`bmat`, `repeat`, `tile`, `resize`, `pad`, `insert`, `delete`,
`broadcast_arrays`, `broadcast_to`, `choose`, `take`, `take_along_axis`,
`roll`, `rollaxis`, `split`, `dsplit`, `unstack`, `vsplit`, `trim_zeros`

### C3: Generation ops (weight = 1, cost = numel(output))
`arange`, `linspace`, `indices`, `meshgrid`, `full`, `full_like`,
`from_dlpack`, `frombuffer`, `fromfile`, `fromfunction`, `fromiter`,
`fromregex`, `fromstring`, `ix_`, `mask_indices`

### C4: View/validation ops (weight = 1, cost = numel(input))
`array`, `asarray`, `asarray_chkfinite`, `base_repr`, `binary_repr`,
`ravel`, `diagflat`, `diag`, `diagonal`

### C5: Ops that stay free (no change)
`reshape`, `transpose`, `squeeze`, `expand_dims`, `flip*`, `rot90`,
`swapaxes`, `moveaxis`, `copy`, `empty*`, `ones*`, `zeros*`, `eye`,
`identity`, `tri*`, `atleast_*d`, `broadcast_shapes`, `can_cast`,
metadata queries, `fft.fftfreq`, `fft.rfftfreq`, `fft.fftshift`,
`fft.ifftshift`, `random.seed`, etc.

**Implementation:**
1. Update `_registry.py`: change category from `"free"` to `"counted_custom"`
   for each C1-C4 op
2. Add cost deduction in `_free_ops.py` (these functions currently just
   delegate to numpy — add `require_budget()` + `budget.deduct()` calls)
3. Weight = 1 for all (from tier system)

## Workstream D: Unblock 3 Ops

**What:** Move `apply_along_axis`, `apply_over_axes`, `piecewise` from
`blacklisted` to `counted_custom`.

**Cost formula:**
- `apply_along_axis`: `numel(output)` — inner function costs counted
  separately via mechestim tracking
- `apply_over_axes`: `numel(output)`
- `piecewise`: `n` (input size)

**Implementation:**
1. Change category in `_registry.py`
2. Add implementations in `_counting_ops.py` with budget deduction
3. Weight = 1

## Workstream E: SVD Cost Formula Update

Merged into Workstream B — `svd_cost(m, n, k)` changes from `m*n*k` to
`4*m*n*k` as part of the formula changes.

## Execution Plan

```
Phase 0: Add "Review Status" column to Google Sheet
    |
Phase 1 (3 parallel agents):
    A: Weight tiers     B: Formula changes     D+E: Unblock + SVD cost
    |                   |                       |
Phase 2 (sequential, after B):
    C: Count free ops (75 ops, split into C1-C4 sub-batches)
    |
Phase 3: Reconciliation
    - Run full test suite
    - Regenerate weights.json, CSV, markdown
    - Update Google Sheet with all changes + Review Status
    - Update benchmark analytical formulas to match new runtime formulas
```

## Verification

- All 1699+ tests pass
- `test_weights_coverage.py` updated for new counted ops
- `test_methodology_consistency.py` updated for new formulas
- Google Sheet "Review Status" column shows `accepted` for all 453 reviewed ops
- No ops remain `pending`
