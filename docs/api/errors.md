# Errors

Exception classes raised by mechestim. See [Common Errors](../troubleshooting/common-errors.md) for symptoms and fixes.

| Exception | When raised |
|-----------|------------|
| `BudgetExhaustedError` | Operation would exceed remaining FLOP budget |
| `NoBudgetContextError` | No active budget (rare — global default usually prevents this) |
| `SymmetryError` | Tensor data is not symmetric within tolerance |

## API Reference

::: mechestim.errors
