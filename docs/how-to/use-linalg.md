# Use Linear Algebra

## When to use this page

Use this page to learn how to use `me.linalg` operations and their FLOP costs.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Available operations

### Decompositions

| Operation | Cost | Notes |
|-----------|------|-------|
| `me.linalg.svd(A, k=k)` | $m \cdot n \cdot k$ | Truncated SVD |
| `me.linalg.eig(A)` | $10n^3$ | General eigendecomposition |
| `me.linalg.eigh(A)` | $4n^3/3$ | Symmetric eigendecomposition |
| `me.linalg.cholesky(A)` | $n^3/3$ | Cholesky (symmetric positive definite) |
| `me.linalg.qr(A)` | $2mn^2 - 2n^3/3$ | Householder QR |
| `me.linalg.eigvals(A)` | $10n^3$ | Eigenvalues only |
| `me.linalg.eigvalsh(A)` | $4n^3/3$ | Symmetric eigenvalues only |
| `me.linalg.svdvals(A)` | $m \cdot n \cdot \min(m,n)$ | Singular values only |

### Solvers

| Operation | Cost | Symmetric cost |
|-----------|------|----------------|
| `me.linalg.solve(A, b)` | $2n^3/3 + n^2 \cdot n_{\text{rhs}}$ | $n^3/3 + n \cdot n_{\text{rhs}}$ |
| `me.linalg.inv(A)` | $n^3$ | $n^3/3 + n^3/2$ |
| `me.linalg.lstsq(A, b)` | $m \cdot n \cdot \min(m,n)$ | — |
| `me.linalg.pinv(A)` | $m \cdot n \cdot \min(m,n)$ | — |

When the input is a `SymmetricTensor`, `solve` and `inv` automatically use cheaper Cholesky-based costs. `inv` of a symmetric matrix returns a `SymmetricTensor`.

### Properties

| Operation | Cost | Symmetric cost |
|-----------|------|----------------|
| `me.linalg.det(A)` | $n^3$ | $n^3/3$ |
| `me.linalg.slogdet(A)` | $n^3$ | $n^3/3$ |
| `me.linalg.norm(x)` | depends on ord | — |
| `me.linalg.cond(A)` | $m \cdot n \cdot \min(m,n)$ | — |
| `me.linalg.matrix_rank(A)` | $m \cdot n \cdot \min(m,n)$ | — |
| `me.linalg.trace(A)` | $n$ | — |

### Compound

| Operation | Cost | Notes |
|-----------|------|-------|
| `me.linalg.multi_dot(arrays)` | Optimal chain ordering | Uses `np.linalg.multi_dot` |
| `me.linalg.matrix_power(A, n)` | $n^3 \times \text{exponent}$ | Repeated squaring |

## Symmetric input savings

Pass a `SymmetricTensor` to get automatic cost reductions:

```python
import mechestim as me
import numpy as np

with me.BudgetContext(flop_budget=10**8) as budget:
    A = me.as_symmetric(np.eye(10) * 2.0, dims=(0, 1))

    # solve uses Cholesky cost: n^3/3 + n*nrhs = 343
    x = me.linalg.solve(A, np.ones(10))

    # inv returns SymmetricTensor, uses cheaper cost
    A_inv = me.linalg.inv(A)
    print(isinstance(A_inv, me.SymmetricTensor))  # True
```

See [Exploit Symmetry Savings](./exploit-symmetry.md) for full details.

## Query cost before running

```python
cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")  # 655,360

cost = me.flops.solve_cost(n=256, nrhs=1, symmetric=True)
print(f"Solve cost (symmetric): {cost:,}")
```

## Common pitfalls

**Symptom:** Using `numpy.linalg.svd` instead of `me.linalg.svd`

**Fix:** Operations called through `numpy` directly bypass FLOP counting. Always use `me.linalg.*`.

## Related pages

- [Exploit Symmetry Savings](./exploit-symmetry.md) — symmetry-aware cost reductions
- [Plan Your Budget](./plan-your-budget.md) — query costs before running
- [Operation Audit](../reference/operation-audit.md) — full list of supported operations
