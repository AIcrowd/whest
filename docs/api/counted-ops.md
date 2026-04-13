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

::: mechestim._einsum.einsum

::: mechestim._einsum.einsum_path

::: mechestim._pointwise

::: mechestim._polynomial

::: mechestim._window

::: mechestim._unwrap

::: mechestim._sorting_ops

::: mechestim._counting_ops
