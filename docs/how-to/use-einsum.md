# Use Einsum

## When to use this page

Use this page to understand `me.einsum` — the core computation primitive in mechestim.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Common patterns

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    A = me.ones((256, 256))
    B = me.ones((256, 256))
    x = me.ones((256,))

    # Matrix-vector multiply: cost = 2 × m × k
    y = me.einsum('ij,j->i', A, x)           # 2 × 256 × 256 = 131,072 FLOPs

    # Matrix multiply: cost = 2 × m × k × n
    C = me.einsum('ij,jk->ik', A, B)         # 2 × 256 × 256 × 256 = 33,554,432 FLOPs

    # Outer product: cost = i × j (no summation, no ×2)
    outer = me.einsum('i,j->ij', x, x)       # 256 × 256 = 65,536 FLOPs

    # Trace: cost = 2 × i
    tr = me.einsum('ii->', A)                 # 2 × 256 = 512 FLOPs

    # Batched matmul: cost = 2 × b × m × k × n
    batch = me.ones((4, 256, 256))
    out = me.einsum('bij,bjk->bik', batch, batch)  # 2 × 4 × 256 × 256 × 256 FLOPs

    print(budget.summary())
```

## Cost formula

The cost of an einsum is the sum of per-step costs along the optimal contraction path. Every einsum — even a simple two-operand one — goes through opt_einsum's path optimizer.

For each pairwise step:

```
cost = (product of all index dimensions) × op_factor
```

where `op_factor = 2` when indices are summed (multiply + add) and `op_factor = 1` when no indices are summed (outer product / pure assignment).

For `'ij,jk->ik'` with shapes `(256, 256)` and `(256, 256)`:
- Indices: i=256, j=256, k=256
- `j` is summed out, so op_factor = 2
- Cost: 2 x 256 x 256 x 256 = 33,554,432

For multi-operand einsums (3+ tensors), mechestim automatically decomposes the contraction into optimal pairwise steps. The total cost is the sum of per-step costs.

## me.dot and me.matmul

`me.dot(A, B)` and `me.matmul(A, B)` are equivalent to the corresponding einsum and have the same FLOP cost.

## Symmetric tensors

Wrap your tensor with `me.as_symmetric(data, dims)`. The optimizer automatically:
- Uses symmetry to choose the best contraction order
- Charges reduced costs based on unique elements

```python
with me.BudgetContext(flop_budget=10**8) as budget:
    S = me.as_symmetric(np.eye(10), dims=(0, 1))  # 55 unique elements
    v = me.ones((10,))

    result = me.einsum('ij,j->i', S, v)  # costs based on unique elements, not 100

    print(f"Cost: {budget.flops_used:,}")
```

Use `symmetric_dims` on `einsum()` when the output is symmetric — this wraps the result as a `SymmetricTensor` for downstream savings:

```python
with me.BudgetContext(flop_budget=10**8) as budget:
    X = me.array(np.random.randn(100, 10))

    # Covariance: output dims (0,1) are symmetric
    C = me.einsum('ki,kj->ij', X, X, symmetric_dims=[(0, 1)])

    print(type(C))  # <class 'SymmetricTensor'>
```

For the full symmetry guide, see [Exploit Symmetry Savings](./exploit-symmetry.md).

## Inspecting costs

`me.einsum_path()` previews the contraction plan without executing or spending any budget:

```python
path, info = me.einsum_path('ijk,ai,bj,ck->abc', T, A, B, C)

print(info)
# Step  Subscript         FLOPs  Dense FLOPs  Symmetry Savings
# ────  ────────────────  ─────  ───────────  ────────────────
# 0     ijk,ai->ajk       ...    ...          ...
# 1     ajk,bj->abk       ...    ...          ...
# 2     abk,ck->abc       ...    ...          ...
# ────  ────────────────  ─────  ───────────  ────────────────
# Total                   ...    ...          ...x speedup

print(f"Naive cost:     {info.naive_cost:,}")
print(f"Optimized cost: {info.optimized_cost:,}")
print(f"Speedup:        {info.speedup:.1f}x")
```

`me.flops.einsum_cost()` returns the same cost that `einsum()` would deduct — one source of truth:

```python
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")  # 33,554,432
```

## ⚠️ Common pitfalls

**Symptom:** Unexpectedly high FLOP cost

**Fix:** Check all index dimensions. A subscript like `'ijkl,jklm->im'` multiplies all five dimension sizes together (times op_factor). Use `me.flops.einsum_cost()` or `me.einsum_path()` to preview costs before executing.

## 📎 Related pages

- [Exploit Symmetry](./exploit-symmetry.md) — full guide to symmetry mechanisms
- [Symmetric Tensors API](../api/symmetric.md) — `SymmetricTensor`, `SymmetryInfo`, `as_symmetric`
- [Plan Your Budget](./plan-your-budget.md) — query costs before executing
- [FLOP Counting Model](../concepts/flop-counting-model.md) — how costs are computed
