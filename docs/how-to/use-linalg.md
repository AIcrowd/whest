# Use Linear Algebra

## When to use this page

Use this page to learn how to use `me.linalg` operations.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Available operations

### me.linalg.svd — Truncated SVD

Compute the top-k singular value decomposition.

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    A = me.random.randn(256, 256)

    # Truncated SVD: returns top-k singular values/vectors
    U, S, Vt = me.linalg.svd(A, k=10)

    print(f"U shape: {U.shape}")    # (256, 10)
    print(f"S shape: {S.shape}")    # (10,)
    print(f"Vt shape: {Vt.shape}")  # (10, 256)
    print(f"Cost: {budget.flops_used:,} FLOPs")  # 655,360
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | ndarray | Input matrix of shape (m, n) |
| `k` | int | Number of singular values to compute |

**Cost:** m × n × k FLOPs

**Returns:** `(U, S, Vt)` where U is (m, k), S is (k,), Vt is (k, n)

### Query cost before running

```python
cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")  # 655,360
```

## Unsupported linalg operations

Calling a NumPy linalg function that isn't available in mechestim raises an `AttributeError` with a message explaining what operations are supported:

```python
me.linalg.solve(A, b)
# AttributeError: module 'mechestim.linalg' has no attribute 'solve'.
# mechestim.linalg currently supports: svd
```

## ⚠️ Common pitfalls

**Symptom:** Using `numpy.linalg.svd` instead of `me.linalg.svd`

**Fix:** Operations called through `numpy` directly bypass FLOP counting. Always use `me.linalg.svd`.

## 📎 Related pages

- [Plan Your Budget](./plan-your-budget.md) — query SVD cost with `me.flops.svd_cost()`
- [Operation Audit](../reference/operation-audit.md) — full list of supported operations
