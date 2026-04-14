# Use Linear Algebra

## When to use this page

Use this page to learn how to use `we.linalg` operations and their FLOP costs.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Available operations

### Decompositions

| Operation | Cost | Notes |
|-----------|------|-------|
| `we.linalg.svd(A, k=k)` | $m \cdot n \cdot k$ | Truncated SVD |
| `we.linalg.eig(A)` | $10n^3$ | General eigendecomposition |
| `we.linalg.eigh(A)` | $4n^3/3$ | Symmetric eigendecomposition |
| `we.linalg.cholesky(A)` | $n^3/3$ | Cholesky (symmetric positive definite) |
| `we.linalg.qr(A)` | $mn^2 - n^3/3$ | Householder QR (FMA=1) |
| `we.linalg.eigvals(A)` | $10n^3$ | Eigenvalues only |
| `we.linalg.eigvalsh(A)` | $4n^3/3$ | Symmetric eigenvalues only |
| `we.linalg.svdvals(A)` | $m \cdot n \cdot \min(m,n)$ | Singular values only |

### Solvers

| Operation | Cost | Symmetric cost |
|-----------|------|----------------|
| `we.linalg.solve(A, b)` | $n^3/3 + n^2 \cdot n_{\text{rhs}}$ | $n^3/3 + n \cdot n_{\text{rhs}}$ |
| `we.linalg.inv(A)` | $n^3$ | $n^3/3 + n^3/2$ |
| `we.linalg.lstsq(A, b)` | $m \cdot n \cdot \min(m,n)$ | — |
| `we.linalg.pinv(A)` | $m \cdot n \cdot \min(m,n)$ | — |

When the input is a `SymmetricTensor`, `solve` and `inv` automatically use cheaper Cholesky-based costs. `inv` of a symmetric matrix returns a `SymmetricTensor`.

### Properties

| Operation | Cost | Symmetric cost |
|-----------|------|----------------|
| `we.linalg.det(A)` | $n^3$ | $n^3/3$ |
| `we.linalg.slogdet(A)` | $n^3$ | $n^3/3$ |
| `we.linalg.norm(x)` | depends on ord | — |
| `we.linalg.cond(A)` | $m \cdot n \cdot \min(m,n)$ | — |
| `we.linalg.matrix_rank(A)` | $m \cdot n \cdot \min(m,n)$ | — |
| `we.linalg.trace(A)` | $n$ | — |

### Compound

| Operation | Cost | Notes |
|-----------|------|-------|
| `we.linalg.multi_dot(arrays)` | Optimal chain ordering | Uses `np.linalg.multi_dot` |
| `we.linalg.matrix_power(A, n)` | $n^3 \times \text{exponent}$ | Repeated squaring |

## Symmetric input savings

Pass a `SymmetricTensor` to get automatic cost reductions:

```python
import whest as we
import numpy as np

with we.BudgetContext(flop_budget=10**8) as budget:
    A = we.as_symmetric(np.eye(10) * 2.0, symmetric_axes=(0, 1))

    # solve uses Cholesky cost: n^3/3 + n*nrhs = 343
    x = we.linalg.solve(A, np.ones(10))

    # inv returns SymmetricTensor, uses cheaper cost
    A_inv = we.linalg.inv(A)
    print(isinstance(A_inv, we.SymmetricTensor))  # True
```

See [Exploit Symmetry Savings](./exploit-symmetry.md) for full details.

## Query cost before running

```python
cost = we.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")  # 655,360

cost = we.flops.solve_cost(n=256, nrhs=1, symmetric=True)
print(f"Solve cost (symmetric): {cost:,}")
```

## Common pitfalls

**Symptom:** Using `numpy.linalg.svd` instead of `we.linalg.svd`

**Fix:** Operations called through `numpy` directly bypass FLOP counting. Always use `we.linalg.*`.

## Related pages

- [Exploit Symmetry Savings](./exploit-symmetry.md) — symmetry-aware cost reductions
- [Plan Your Budget](./plan-your-budget.md) — query costs before running
- [Operation Audit](../reference/operation-audit.md) — full list of supported operations
