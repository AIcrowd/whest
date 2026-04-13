# Budget

Budget management for FLOP counting. `BudgetContext` is the core context
manager that tracks operation costs and enforces limits.

## Quick example

```python
import whest as we

# Explicit budget with namespace
with we.BudgetContext(flop_budget=10**7, namespace="forward") as budget:
    W = we.ones((256, 256))
    x = we.ones((256,))
    h = we.einsum('ij,j->i', W, x)

    print(f"Used: {budget.flops_used:,} / {budget.flop_budget:,}")

# Session-wide summary across all namespaces
we.budget_summary()

# Programmatic access
data = we.budget_summary_dict()
```

## API Reference

::: whest._budget.BudgetContext

::: whest._budget.OpRecord

::: whest._budget.budget

::: whest._budget.budget_summary_dict

::: whest._budget.budget_reset

::: whest._display.render_budget_summary

::: whest._display.budget_summary

::: whest._display.budget_live
