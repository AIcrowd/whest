# Errors

Exception and warning classes raised by mechestim. See [Common Errors](../troubleshooting/common-errors.md) for symptoms and fixes.

| Class | Type | When raised |
|-------|------|------------|
| `MechEstimError` | Exception (base) | Base class for all mechestim exceptions |
| `BudgetExhaustedError` | Exception | Operation would exceed remaining FLOP budget |
| `NoBudgetContextError` | Exception | No active budget (rare — global default usually prevents this) |
| `SymmetryError` | Exception | Tensor data is not symmetric within tolerance |
| `MechEstimWarning` | Warning (base) | Base class for all mechestim warnings |
| `SymmetryLossWarning` | Warning | An operation caused loss of symmetry metadata |

## API Reference

::: mechestim.errors
