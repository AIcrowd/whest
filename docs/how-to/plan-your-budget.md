# Plan Your Budget

## When to use this page

Use this page to learn how to query operation costs before running them.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Cost query functions

These functions work **outside** a BudgetContext — they compute costs from shapes without executing anything.

```python
import whest as we

# Einsum cost
cost = we.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")         # 16,777,216 (256³, FMA=1)

# SVD cost
cost = we.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")            # 655,360

# Pointwise cost (unary/binary ops)
cost = we.flops.pointwise_cost(shape=(256, 256))
print(f"Pointwise cost: {cost:,}")      # 65,536

# Reduction cost
cost = we.flops.reduction_cost(input_shape=(256, 256))
print(f"Reduction cost: {cost:,}")      # 65,536
```

## Budget breakdown example

Plan a multi-step computation before executing:

```python
import whest as we

# Plan
steps = [
    ("einsum ij,j->i", we.flops.einsum_cost('ij,j->i', shapes=[(256, 256), (256,)])),
    ("ReLU (maximum)", we.flops.pointwise_cost(shape=(256,))),
    ("sum reduction", we.flops.reduction_cost(input_shape=(256,))),
]

total = sum(cost for _, cost in steps)
print(f"{'Operation':<20} {'FLOPs':>12}")
print("-" * 34)
for name, cost in steps:
    print(f"{name:<20} {cost:>12,}")
print("-" * 34)
print(f"{'Total':<20} {total:>12,}")
```

Output:

```
Operation                   FLOPs
----------------------------------
einsum ij,j->i            131,072
ReLU (maximum)                256
sum reduction                 256
----------------------------------
Total                     131,584
```

## Multi-operand einsum planning with `einsum_path`

For multi-operand einsums (3+ operands), `we.einsum_path()` is more
informative than `we.flops.einsum_cost()` because it shows the step-by-step
contraction breakdown with per-step symmetry savings:

```python
import whest as we
import numpy as np

T = we.as_symmetric(np.random.randn(50, 50, 50), symmetric_axes=(0, 1, 2))
A = we.ones((50, 50))
B = we.ones((50, 50))
C = we.ones((50, 50))

path, info = we.einsum_path('ijk,ai,bj,ck->abc', T, A, B, C)

print(f"Optimized cost: {info.optimized_cost:,}")
print(f"Naive cost:     {info.naive_cost:,}")
print(f"Speedup:        {info.speedup:.1f}x")
print(f"Largest intermediate: {info.largest_intermediate:,} elements")
print(info)  # full per-step table
```

`we.einsum_path()` has **zero budget cost** — it plans the contraction path
without executing anything. Use it alongside `we.flops.einsum_cost()` for
comprehensive planning.

## Using namespaces to track phases

Use the `namespace` parameter to label different computation phases:

```python
with we.BudgetContext(flop_budget=total, namespace="forward") as budget:
    # forward pass here
    ...

with we.BudgetContext(flop_budget=total, namespace="backward") as budget:
    # backward pass here
    ...

# Session-wide summary across all phases
we.budget_summary()
```

`we.budget_summary_dict(by_namespace=True)` returns a dict with per-namespace breakdowns for programmatic analysis.

## 📎 Related pages

- [Use Einsum](./use-einsum.md) — understand einsum cost formulas and multi-operand paths
- [Calibrate Weights](./calibrate-weights.md) — measure per-operation weights to refine cost estimates
- [Debug Budget Overruns](./debug-budget-overruns.md) — diagnose after the fact
- [Symmetric Tensors API](../api/symmetric.md) — `PathInfo` and `StepInfo` reference
