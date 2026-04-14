# FLOP Cost Query API

Pre-execution cost estimation functions. These are pure functions that compute
FLOP costs from shapes without executing anything or consuming budget.

## Quick example

```python
import whest as we

# Einsum cost
cost = we.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul: {cost:,} FLOPs")  # 16,777,216

# SVD cost
cost = we.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD: {cost:,} FLOPs")  # 655,360

# Pointwise (unary/binary) cost
cost = we.flops.pointwise_cost(shape=(256, 256))
print(f"Pointwise: {cost:,} FLOPs")  # 65,536

# Reduction cost
cost = we.flops.reduction_cost(input_shape=(1000, 100))
print(f"Reduction: {cost:,} FLOPs")  # 100,000
```

## API Reference

::: whest._flops
