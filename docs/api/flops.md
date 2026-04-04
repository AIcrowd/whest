# FLOP Cost Query API

Pre-execution cost estimation functions. These are pure functions that compute
FLOP costs from shapes without executing anything or consuming budget.

## Quick example

```python
import mechestim as me

# Einsum cost
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul: {cost:,} FLOPs")  # 33,554,432

# SVD cost
cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD: {cost:,} FLOPs")  # 655,360

# Pointwise (unary/binary) cost
cost = me.flops.pointwise_cost(shape=(256, 256))
print(f"Pointwise: {cost:,} FLOPs")  # 65,536

# Reduction cost
cost = me.flops.reduction_cost(input_shape=(1000, 100))
print(f"Reduction: {cost:,} FLOPs")  # 100,000
```

## API Reference

::: mechestim._flops
