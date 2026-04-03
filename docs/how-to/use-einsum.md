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
- Cost: $256^3$ = 16,777,216

## me.dot and me.matmul

`me.dot(A, B)` and `me.matmul(A, B)` are equivalent to the corresponding einsum and have the same FLOP cost.

## Advanced: symmetric tensors in einsum

Einsum integrates with mechestim's symmetry system to reduce costs. There are
three mechanisms — each can halve (or more) the FLOP cost.

### Repeated operands (auto-detected)

When you pass the **same Python object** as multiple operands, mechestim
detects this and divides the cost by the factorial of the repeat count:

```python
with me.BudgetContext(flop_budget=10**8) as budget:
    x = me.ones((10, 256))
    A = me.ones((10, 10))

    # x appears twice — cost is divided by 2!
    result = me.einsum('ai,bi,ab->', x, x, A)

    print(f"Cost: {budget.flops_used:,}")  # 12,800 (half of 25,600)
```

Detection is by object identity (`id()`), not value equality — `x.copy()`
breaks the savings.

### Symmetric output dimensions

Use `symmetric_dims` when the output has interchangeable dimensions, like a
covariance matrix where `C[i,j] == C[j,i]`. The result is returned as a
`SymmetricTensor`:

```python
with me.BudgetContext(flop_budget=10**8) as budget:
    X = me.array(np.random.randn(100, 10))

    # Covariance: output dims (0,1) are symmetric — cost halved
    C = me.einsum('ki,kj->ij', X, X, symmetric_dims=[(0, 1)])

    print(f"Cost: {budget.flops_used:,}")      # 5,000 (half of 10,000)
    print(type(C))                              # <class 'SymmetricTensor'>
```

### Symmetric inputs

When an input is a `SymmetricTensor`, einsum automatically accounts for the
reduced number of unique elements:

```python
with me.BudgetContext(flop_budget=10**8) as budget:
    S = me.as_symmetric(np.eye(10), dims=(0, 1))  # 55 unique elements
    v = me.ones((10,))

    result = me.einsum('ij,j->i', S, v)  # costs 55, not 100

    print(f"Cost: {budget.flops_used:,}")  # 55
```

All three mechanisms stack multiplicatively. For the full details, see
[Exploit Symmetry Savings](./exploit-symmetry.md).

## Multi-operand contractions

For einsums with 3+ operands, mechestim automatically finds the optimal
pairwise contraction order using opt_einsum. This is controlled by the
`optimize` kwarg (defaults to `'auto'`):

```python
with me.BudgetContext(flop_budget=10**9) as budget:
    T = me.as_symmetric(T_data, dims=(0, 1, 2))  # S₃ symmetric 3-tensor
    A = me.ones((100, 100))
    B = me.ones((100, 100))
    C = me.ones((100, 100))

    # Automatic path optimization (default)
    result = me.einsum('ijk,ai,bj,ck->abc', T, A, B, C)

    # Disable path optimization (uses left-to-right order)
    result = me.einsum('ijk,ai,bj,ck->abc', T, A, B, C, optimize=False)
```

Symmetry is tracked through intermediates: contracting an S₃-symmetric
`ijk` with `ai` yields an intermediate `ajk` with S₂ symmetry on `(j,k)`,
reducing the cost of that step.

## Inspecting contraction paths

Use `me.einsum_path()` to preview the contraction plan without executing
or spending any budget:

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

`einsum_path` returns a `(path, PathInfo)` tuple. See
[PathInfo / StepInfo API](../api/symmetric.md) for full field reference.

## ⚠️ Common pitfalls

**Symptom:** Unexpectedly high FLOP cost

**Fix:** Check all index dimensions. A subscript like `'ijkl,jklm->im'` multiplies all five dimension sizes together. Use `me.flops.einsum_cost()` or `me.einsum_path()` to preview costs before executing.

## 📎 Related pages

- [Exploit Symmetry](./exploit-symmetry.md) — full guide to all symmetry mechanisms
- [Symmetric Tensors API](../api/symmetric.md) — `SymmetricTensor`, `SymmetryInfo`, `as_symmetric`
- [Plan Your Budget](./plan-your-budget.md) — query costs before executing
- [FLOP Counting Model](../concepts/flop-counting-model.md) — how multi-operand costs are computed
