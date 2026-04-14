---
sidebar_position: 2
sidebar_label: Use Einsum
---
# Use Einsum

## When to use this page

Use this page to understand `we.einsum` — the core computation primitive in whest.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Common patterns

```python
import whest as we

with we.BudgetContext(flop_budget=10**8) as budget:
    A = we.ones((256, 256))
    B = we.ones((256, 256))
    x = we.ones((256,))

    # Matrix-vector multiply: cost = m × k
    y = we.einsum('ij,j->i', A, x)           # 256 × 256 = 65,536 FLOPs

    # Matrix multiply: cost = m × k × n
    C = we.einsum('ij,jk->ik', A, B)         # 256 × 256 × 256 = 16,777,216 FLOPs

    # Outer product: cost = i × j
    outer = we.einsum('i,j->ij', x, x)       # 256 × 256 = 65,536 FLOPs

    # Trace: cost = i
    tr = we.einsum('ii->', A)                 # 256 FLOPs

    # Batched matmul: cost = b × m × k × n
    batch = we.ones((4, 256, 256))
    out = we.einsum('bij,bjk->bik', batch, batch)  # 4 × 256 × 256 × 256 FLOPs

    print(budget.summary())
```

## Cost formula

The cost of an einsum is the sum of per-step costs along the optimal contraction path. Every einsum — even a simple two-operand one — goes through the [opt_einsum path optimizer](../api/) (a symmetry-aware fork of [opt_einsum](https://github.com/dgasmith/opt_einsum)).

For each pairwise step:

```
cost = product of all index dimensions
```

Each FMA (fused multiply-add) counts as 1 operation, so the cost is
simply the product of all index dimensions with no factor-of-2.

For `'ij,jk->ik'` with shapes `(256, 256)` and `(256, 256)`:
- Indices: i=256, j=256, k=256
- Cost: 256 x 256 x 256 = 16,777,216

For multi-operand einsums (3+ tensors), whest automatically decomposes the contraction into optimal pairwise steps. The total cost is the sum of per-step costs.

When symmetric tensors are involved, each step's cost is further reduced by the ratio of unique output elements to total output elements. See [Exploit Symmetry Savings](exploit-symmetry.md#symmetry-in-einsum) for details.

## we.dot and we.matmul

`we.dot(A, B)` and `we.matmul(A, B)` are equivalent to the corresponding einsum and have the same FLOP cost.

## Symmetric tensors

There are two separate symmetry declarations — one for inputs, one for outputs:

**Input symmetry** — wrap with `we.as_symmetric()` before passing to einsum. The optimizer automatically uses symmetry to choose the best contraction order and charges reduced costs:

```python
with we.BudgetContext(flop_budget=10**8) as budget:
    S = we.as_symmetric(np.eye(10), symmetric_axes=(0, 1))  # 55 unique elements
    v = we.ones((10,))

    result = we.einsum('ij,j->i', S, v)  # cost reduced by input symmetry
```

**Output symmetry** — pass `symmetric_axes` to `einsum()` to declare that the result is symmetric. This wraps the output as a `SymmetricTensor` so downstream operations benefit from reduced costs. It does NOT affect the cost of this einsum — it's a declaration about the result's structure:

```python
with we.BudgetContext(flop_budget=10**8) as budget:
    X = we.array(np.random.randn(100, 10))

    # X^T X is always symmetric — declare output axes (0,1) as symmetric
    C = we.einsum('ki,kj->ij', X, X, symmetric_axes=[(0, 1)])

    print(type(C))  # <class 'SymmetricTensor'>
    # C can now be passed to other operations with automatic cost savings
```

For the full symmetry guide, see [Exploit Symmetry Savings](./exploit-symmetry.md).

## Inspecting costs

`we.einsum_path()` previews the contraction plan without executing or spending any budget:

```python
path, info = we.einsum_path('ijk,ai,bj,ck->abc', T, A, B, C)

print(info)
# Prints a multi-line table with a header (complete contraction, naive
# cost, optimized cost, speedup, largest intermediate, index sizes, and
# the name of the optimizer that ran) followed by one row per contraction
# step showing the path-supplied contract tuple, subscript, FLOPs, dense
# FLOPs, savings percentage, BLAS label, unique/dense element counts, and
# the symmetry transformation. See the Exploit Symmetry guide for a
# worked example:
# https://github.com/AIcrowd/whest/blob/main/docs/how-to/exploit-symmetry.md#example

# Call info.format_table(verbose=True) to get an indented detail row per
# step with the merged operand subset, the intermediate's output shape,
# and the running cumulative cost — useful when debugging why a particular
# step's savings are what they are.

print(f"Naive cost:     {info.naive_cost:,}")
print(f"Optimized cost: {info.optimized_cost:,}")
print(f"Speedup:        {info.speedup:.1f}x")
print(f"Optimizer used: {info.optimizer_used}")
```

`we.flops.einsum_cost()` returns the same cost that `einsum()` would deduct — one source of truth:

```python
cost = we.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")  # 16,777,216
```

## Custom contraction paths

By default whest finds the optimal contraction order automatically. You can override this by passing an explicit path — a list of int-tuples specifying which operand positions to contract at each step:

```python
import whest as we

A = we.ones((3, 4))
B = we.ones((4, 5))
C = we.ones((5, 6))

# Plan first, execute later
path, info = we.einsum_path('ij,jk,kl->il', A, B, C)
print(f"Optimal path: {path}")  # e.g. [(0, 1), (0, 1)]

# Execute with the planned path
with we.BudgetContext(flop_budget=10**8) as budget:
    result = we.einsum('ij,jk,kl->il', A, B, C, optimize=path)
```

You can also specify a completely custom path. Each tuple names the positions (in the current operand list) to contract; the result is appended to the end:

```python
# Force B×C first (positions 1,2), then A×result (positions 0,1)
result = we.einsum('ij,jk,kl->il', A, B, C, optimize=[(1, 2), (0, 1)])

# Force A×B first (positions 0,1), then result×C (positions 0,1)
result = we.einsum('ij,jk,kl->il', A, B, C, optimize=[(0, 1), (0, 1)])
```

Different paths may have different FLOP costs. Use `we.einsum_path()` to compare — it returns the cost without executing or spending budget.

## Path caching

Contraction paths are cached automatically in a module-level LRU cache.
When you call `we.einsum()` with the same subscripts, shapes, optimizer,
and symmetry structure, the path is reused from cache instead of being
recomputed. This makes repeated einsums in loops essentially free in
path-finding overhead:

```python
with we.BudgetContext(flop_budget=10**9) as budget:
    for i in range(1000):
        y = we.einsum('ij,j->i', A, x)  # path computed once, reused 999 times
```

`we.einsum_path()` shares the same cache, so planning a path warms the
cache for subsequent `we.einsum()` calls and vice versa.

### Cache management

```python
# Inspect cache statistics
info = we.einsum_cache_info()
print(f"Hits: {info.hits}, Misses: {info.misses}, Size: {info.currsize}/{info.maxsize}")

# Clear the cache (e.g., to free memory or force recomputation)
we.clear_einsum_cache()

# Change the cache size (default 4096 entries, rebuilds the cache)
we.configure(einsum_path_cache_size=8192)
```

## ⚠️ Common pitfalls

**Symptom:** Unexpectedly high FLOP cost

**Fix:** Check all index dimensions. A subscript like `'ijkl,jklm->im'` multiplies all five dimension sizes together. Use `we.flops.einsum_cost()` or `we.einsum_path()` to preview costs before executing.

## 📎 Related pages

- [Exploit Symmetry](./exploit-symmetry.md) — full guide to symmetry mechanisms
- [API Reference](../api/) — algorithms, symmetry support, and operation details
- [Plan Your Budget](./plan-your-budget.md) — query costs before executing
- [FLOP Counting Model](../concepts/flop-counting-model.md) — how costs are computed
