# Plan Your Budget

## When to use this page

Use this page to learn how to query operation costs before running them.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Cost query functions

These functions work **outside** a BudgetContext — they compute costs from shapes without executing anything.

```python
import mechestim as me

# Einsum cost
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")         # 16,777,216

# SVD cost
cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")            # 655,360

# Pointwise cost (unary/binary ops)
cost = me.flops.pointwise_cost(shape=(256, 256))
print(f"Pointwise cost: {cost:,}")      # 65,536

# Reduction cost
cost = me.flops.reduction_cost(input_shape=(256, 256))
print(f"Reduction cost: {cost:,}")      # 65,536
```

## Budget breakdown example

Plan a multi-step computation before executing:

```python
import mechestim as me

# Plan
steps = [
    ("einsum ij,j->i", me.flops.einsum_cost('ij,j->i', shapes=[(256, 256), (256,)])),
    ("ReLU (maximum)", me.flops.pointwise_cost(shape=(256,))),
    ("sum reduction", me.flops.reduction_cost(input_shape=(256,))),
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
einsum ij,j->i             65,536
ReLU (maximum)                256
sum reduction                 256
----------------------------------
Total                      66,048
```

## 📎 Related pages

- [Use Einsum](./use-einsum.md) — understand einsum cost formulas
- [Debug Budget Overruns](./debug-budget-overruns.md) — diagnose after the fact
