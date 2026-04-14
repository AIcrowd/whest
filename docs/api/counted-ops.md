# Counted Operations

Operations that consume the FLOP budget. Every call deducts its analytical cost
from the active budget before execution.

**Cost rules of thumb:**

- **Unary** (exp, log, sqrt, ...): 1 FLOP per output element
- **Binary** (add, multiply, maximum, ...): 1 FLOP per output element (after broadcasting)
- **Reductions** (sum, mean, max, ...): 1 FLOP per input element
- **Einsum**: `product_of_all_index_dims` (FMA=1; no factor of 2)

See [FLOP Counting Model](../concepts/flop-counting-model.md) for full details.

## API Reference

::: whest._einsum.einsum

::: whest._einsum.einsum_path

::: whest._pointwise

::: whest._polynomial

::: whest._window

::: whest._unwrap

::: whest._sorting_ops

::: whest._counting_ops
