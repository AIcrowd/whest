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

    # Matrix-vector multiply: cost = m × k
    y = me.einsum('ij,j->i', A, x)           # 256 × 256 = 65,536 FLOPs

    # Matrix multiply: cost = m × k × n
    C = me.einsum('ij,jk->ik', A, B)         # 256 × 256 × 256 = 16,777,216 FLOPs

    # Outer product: cost = i × j
    outer = me.einsum('i,j->ij', x, x)       # 256 × 256 = 65,536 FLOPs

    # Trace: cost = i
    tr = me.einsum('ii->', A)                 # 256 FLOPs

    # Batched matmul: cost = b × m × k × n
    batch = me.ones((4, 256, 256))
    out = me.einsum('bij,bjk->bik', batch, batch)  # 4 × 256 × 256 × 256 FLOPs

    print(budget.summary())
```

## Cost formula

The FLOP cost of `me.einsum` is the product of all index dimensions in the subscript string:

```
cost = product of all unique index sizes
```

For `'ij,jk->ik'` with shapes `(256, 256)` and `(256, 256)`:
- Indices: i=256, j=256, k=256
- Cost: 256 × 256 × 256 = 16,777,216

## me.dot and me.matmul

`me.dot(A, B)` and `me.matmul(A, B)` are equivalent to the corresponding einsum and have the same FLOP cost.

## ⚠️ Common pitfalls

**Symptom:** Unexpectedly high FLOP cost

**Fix:** Check all index dimensions. A subscript like `'ijkl,jklm->im'` multiplies all five dimension sizes together. Use `me.flops.einsum_cost()` to preview costs before executing.

## 📎 Related pages

- [Exploit Symmetry](./exploit-symmetry.md) — reduce einsum costs with repeated operands
- [Plan Your Budget](./plan-your-budget.md) — query costs before executing
