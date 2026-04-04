# Budget

Budget management for FLOP counting. `BudgetContext` is the core context
manager that tracks operation costs and enforces limits.

## Quick example

```python
import mechestim as me

# Explicit budget with namespace
with me.BudgetContext(flop_budget=10**7, namespace="forward") as budget:
    W = me.ones((256, 256))
    x = me.ones((256,))
    h = me.einsum('ij,j->i', W, x)

    print(f"Used: {budget.flops_used:,} / {budget.flop_budget:,}")

# Session-wide summary across all namespaces
me.budget_summary()

# Programmatic access
data = me.budget_summary_dict()
```

## API Reference

::: mechestim._budget.BudgetContext

::: mechestim._budget.OpRecord

::: mechestim._budget.budget

::: mechestim._budget.budget_summary_dict

::: mechestim._budget.budget_reset

::: mechestim._display.render_budget_summary

::: mechestim._display.budget_summary

::: mechestim._display.budget_live
