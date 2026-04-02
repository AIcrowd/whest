# SymmetricTensor Support for mechestim

**Date:** 2026-04-02
**Status:** Implemented
**Context:** Slack discussions with Paul Christiano and Wilson Wu (March 27–31, 2026) on ops budgeting for the ARC Network Estimation Challenge

---

## Problem

mechestim counts FLOPs analytically to enforce computational budgets. Symmetry
currently only reduces costs inside `einsum` (via `symmetric_dims` and repeated
operand detection). But competition algorithms (Edgeworth expansion, covprop)
produce symmetric tensors that flow through pointwise ops, solvers, and
decompositions. Those downstream ops pay full price today.

**Requirements from Paul Christiano:**
- Einsum with optimal symmetrized FLOP count (done)
- Savings for pointwise ops on symmetric tensors (not done)
- Leading proposal: einsum + pointwise ops + SVD, all symmetry-aware

**Requirements from Wilson Wu:**
- Pointwise ops on symmetric tensors need savings, with runtime validation
- Partially symmetric tensors (e.g., 4-tensor where dims (0,1) and (2,3) are
  separately symmetric)
- Repeated-index / diagonal patterns like `'i->ii'`

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Approach | `SymmetricTensor` subclass of `np.ndarray` | Discoverable type, natural propagation, competition-friendly |
| Storage | Full dense array + metadata | FLOP counting is decoupled from actual computation; no need for compressed storage |
| Validation | Validate at creation boundaries; trust algebraic guarantees | Adversarial at entry points; no wasted CPU when math proves symmetry |
| Propagation | Algebraic rules per op category | Validate when symmetry isn't provable; pass through when it is |
| Cost interface | Pass full symmetry metadata to cost functions | Each cost function decides how to use (or ignore) symmetry info |

---

## 1. SymmetryInfo Descriptor

A lightweight, immutable descriptor passed to cost functions. Cost functions
never see `SymmetricTensor` objects directly — they receive `SymmetryInfo | None`.

```python
@dataclass(frozen=True)
class SymmetryInfo:
    symmetric_dims: list[tuple[int, ...]]  # e.g., [(0, 1)] or [(0, 1), (2, 3)]
    shape: tuple[int, ...]                  # full tensor shape

    @property
    def unique_elements(self) -> int:
        """Number of unique elements accounting for all symmetry groups.

        For a (0,1)-symmetric n x n matrix: n*(n+1)/2
        For partial symmetry [(0,1), (2,3)] on shape (n, n, m, m):
            (n*(n+1)/2) * (m*(m+1)/2)
        """
        ...

    @property
    def symmetry_factor(self) -> int:
        """Product of factorials of group sizes."""
        ...
```

## 2. SymmetricTensor Class

A thin `np.ndarray` subclass carrying symmetry metadata.

```python
class SymmetricTensor(np.ndarray):
    symmetric_dims: list[tuple[int, ...]]

    @property
    def symmetry_info(self) -> SymmetryInfo:
        return SymmetryInfo(
            symmetric_dims=self.symmetric_dims,
            shape=self.shape,
        )
```

**Creation:**

- `me.as_symmetric(data, dims=(0, 1))` — public API. The `dims` parameter
  specifies a single symmetry group as a tuple of dimension indices; for
  multiple groups, pass a list: `dims=[(0, 1), (2, 3)]`. Validates symmetry
  with `np.allclose(atol=1e-6, rtol=1e-5)`, returns `SymmetricTensor`. Raises
  `SymmetryError` on failure.
- No public skip-validation API. All paths validate. Internal code (e.g.,
  wrapping output of `eigh`) also calls `as_symmetric` with full validation.

**Interop with NumPy:**

- Subclasses `np.ndarray` via `__array_finalize__`
- Intercepts operations via `__array_ufunc__` and `__array_function__`
- `isinstance(S, np.ndarray)` returns `True`
- Falls back to plain `ndarray` when symmetry cannot be preserved

## 3. Cost Function Interface

Every cost function receives `symmetry_info: SymmetryInfo | None` for each
relevant input. The cost function decides how to use it.

```python
# Pointwise — uses unique_elements count
def pointwise_cost(shape, symmetry_info=None):
    if symmetry_info:
        return symmetry_info.unique_elements
    return math.prod(shape)

# Solve — switches to Cholesky cost for symmetric input
def solve_cost(n, nrhs, symmetry_info=None):
    if symmetry_info and (0, 1) in symmetry_info.symmetric_dims:
        return n**3 // 3 + n * nrhs      # Cholesky path
    return 2 * n**3 // 3 + n**2 * nrhs   # LU path

# SVD — no symmetry savings (ignores it)
def svd_cost(m, n, k, symmetry_info=None):
    return m * n * k
```

**Einsum cost function** receives per-operand symmetry info:

```python
def einsum_cost(
    subscripts: str,
    shapes: list[tuple[int, ...]],
    operand_symmetries: list[SymmetryInfo | None] = None,
    output_symmetric_dims: list[tuple[int, ...]] | None = None,
    repeated_operand_indices: list[int] | None = None,
) -> int:
    ...
```

This lets einsum reason about input symmetry (fewer unique elements to iterate)
independently from output symmetry (fewer unique results to compute).

## 4. Propagation Rules

When a result preserves symmetry, the wrapper validates the output and returns
`SymmetricTensor`. When it does not, it returns a plain `ndarray`.

### 4.1 Operations that preserve symmetry

| Operation | Output | Reasoning |
|---|---|---|
| `unary_pointwise(S)` (exp, log, abs, ...) | `SymmetricTensor` (same dims) | `f(S[i,j]) == f(S[j,i])` — algebraic guarantee, no validation needed |
| `S + T` (both symmetric, same dims) | `SymmetricTensor` | Algebraic guarantee, no validation needed |
| `S * scalar` | `SymmetricTensor` | Algebraic guarantee, no validation needed |
| `S.copy()` | `SymmetricTensor` | Inherits metadata, no validation needed |
| `S.T` (transpose of 2D symmetric) | `SymmetricTensor` | Algebraic guarantee, no validation needed |
| `einsum(...)` with `symmetric_dims` | `SymmetricTensor` | Validated (user-declared, not provable) |

### 4.2 Operations that lose symmetry

| Operation | Output | Reasoning |
|---|---|---|
| `S + T` (different symmetric_dims) | plain array | Mixed symmetry not guaranteed |
| `S @ T` (matmul) | plain array | `AB != (AB)^T` in general |
| `eigh(A)` eigenvalues | plain array | 1D vector, not a matrix |
| `cholesky(A)` | plain array | Triangular, not symmetric |
| `solve(S, b)` | plain array | Solution vector not symmetric |
| `S[slice]` | plain array | Slicing may break symmetry |
| `reduction(S, axis=...)` | plain array | Reduces rank |

### 4.3 Operations that benefit from symmetric input (cost savings)

These ops accept `SymmetricTensor` input for reduced cost but do not produce
symmetric output:

| Operation | Cost with symmetric input | Cost without |
|---|---|---|
| `solve(S, b)` | `n^3/3 + n*nrhs` (Cholesky) | `2n^3/3 + n^2*nrhs` (LU) |
| `eigh(S)` | `4n^3/3` | N/A (eigh requires symmetric) |
| `det(S)` | `n^3/3` (via Cholesky) | `2n^3/3` (via LU) |
| `inv(S)` | `n^3/3 + n^3/2` | `2n^3` |
| `einsum('ij,j->i', S, v)` | ~`n(n+1)/2` muls | `n^2` muls |

### 4.4 Validation Policy

**Validate at creation boundaries** — when a user or external input claims
symmetry:

- `me.as_symmetric(data, dims=...)` — always validates
- `einsum(..., symmetric_dims=...)` — always validates output
- `eigh(A)` — validates input is symmetric

**Trust algebraic guarantees** — when symmetry preservation is mathematically
provable, pass through metadata without re-checking:

- Unary pointwise: `f(S[i,j]) == f(S[j,i])` guaranteed for elementwise ops
- Binary pointwise with matching symmetric_dims: `S[i,j] + T[i,j] == S[j,i] + T[j,i]`
- Scalar multiplication, copy, transpose

Validation uses `np.allclose(result, transposed, atol=1e-6, rtol=1e-5)`, runs
as real CPU compute (not FLOP-budgeted), and raises `SymmetryError` on failure.

## 5. Integration with Existing Ops

### 5.1 Pointwise ops (215 ops)

The factory functions `_counted_unary`, `_counted_binary`, `_counted_reduction`
are updated:

- **Unary:** Check if input is `SymmetricTensor`. Pass `SymmetryInfo` to cost
  function. Validate output, return `SymmetricTensor`.
- **Binary:** Propagate symmetry only if both inputs share the same
  `symmetric_dims`. Otherwise return plain array.
- **Reduction:** Always return plain array (rank change).

### 5.2 Einsum

- If inputs are `SymmetricTensor`, extract their `SymmetryInfo` and pass to
  `einsum_cost` as `operand_symmetries`. User does not need to re-declare
  input symmetry.
- `symmetric_dims` parameter still controls output symmetry declaration.
- When output passes validation, return `SymmetricTensor`.

### 5.3 Linalg ops

- `eigh(A)`: Validate A is symmetric on input (or accept `SymmetricTensor`).
  Return plain arrays for eigenvalues/eigenvectors.
- `cholesky(A)`: Validate A is symmetric. Return plain array (triangular).
- `solve(A, b)`: If A is `SymmetricTensor`, pass symmetry info to cost
  function (Cholesky path). Return plain array.
- `inv(S)`: If input is `SymmetricTensor`, use cheaper cost. Output is
  symmetric — validate and return `SymmetricTensor`.
- `det(S)`: If input is `SymmetricTensor`, use Cholesky-based cost.
- Other ops (`eig`, `svd`, `qr`, `lstsq`, etc.): Accept `SymmetricTensor`
  as input (extract shape), no special cost treatment, return plain arrays.

### 5.4 FFT, polynomial, window

No symmetry interaction. These ops ignore `SymmetryInfo` entirely.

## 6. Public API Surface

```python
# Creation — single symmetry group
S = me.as_symmetric(data, dims=(0, 1))

# Creation — multiple symmetry groups (Wilson Wu's partial symmetry)
T = me.as_symmetric(data, dims=[(0, 1), (2, 3)])

# Inspection
isinstance(S, me.SymmetricTensor)  # True
S.symmetric_dims                    # [(0, 1)]
S.symmetry_info                     # SymmetryInfo(...)

# Everything else works as normal numpy
me.exp(S)              # SymmetricTensor, costs unique_elements
me.linalg.solve(S, b)  # plain array, Cholesky cost
me.einsum('ij,j->i', S, v)  # plain array, input symmetry savings
```

## 7. Error Handling

- `SymmetryError` (existing) raised when validation fails
- Error message includes: which dims were claimed symmetric, maximum deviation
  found, the tolerance used
- No silent fallback to plain array on validation failure — always raise

## 8. Backward Compatibility

- All existing code continues to work unchanged
- `einsum` `symmetric_dims` parameter still works as before
- `SymmetricTensor` is additive — plain `ndarray` usage is unaffected
- Cost functions gain an optional `symmetry_info` parameter (default `None`)

## 9. Out of Scope

- Compressed/packed storage (triangular storage for symmetric matrices)
- Sparse tensor support
- Automatic symmetry detection (user must explicitly create via `as_symmetric`
  or receive from symmetry-producing ops)
- Diagonal / repeated-index einsum patterns (`'i->ii'`) — noted by Wilson Wu,
  deferred to a follow-up design
