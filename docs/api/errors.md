# Errors

Exception and warning classes raised by whest. See [Common Errors](../troubleshooting/common-errors.md) for symptoms and fixes.

| Class | Type | When raised |
|-------|------|------------|
| `WhestError` | Exception (base) | Base class for all whest exceptions |
| `BudgetExhaustedError` | Exception | Operation would exceed remaining FLOP budget |
| `TimeExhaustedError` | Exception | Operation exceeded the wall-clock time limit |
| `NoBudgetContextError` | Exception | No active budget (rare — global default usually prevents this) |
| `SymmetryError` | Exception | Tensor data is not symmetric within tolerance |
| `WhestWarning` | Warning (base) | Base class for all whest warnings |
| `SymmetryLossWarning` | Warning | An operation caused loss of symmetry metadata |

## API Reference

::: whest.errors
