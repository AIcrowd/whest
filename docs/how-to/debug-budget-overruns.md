# Debug Budget Overruns

## When to use this page

Use this page when you hit a `BudgetExhaustedError` and need to find which operations are using the most FLOPs.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Reading the budget summary

Call `budget.summary()` at any point inside a `BudgetContext`, or call `me.budget_summary()` outside a context for a session-wide summary across all namespaces:

```python
import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    A = me.ones((256, 256))
    x = me.ones((256,))

    h = me.einsum('ij,j->i', A, x)
    h = me.exp(h)
    h = me.sum(h)

    print(budget.summary())

# Outside the context — summarises every namespace recorded this session
print(me.budget_summary())
```

The summary shows cost per operation type, sorted by highest cost first.

## Session-wide programmatic analysis

Use `me.budget_data()` to retrieve the raw operation records for automated analysis (e.g. in tests or notebooks):

```python
records = me.budget_data()
for record in records:
    print(record.op_name, record.namespace, record.flop_cost)
```

## Inspecting the operation log

For per-call detail, use `budget.op_log`:

```python
for record in budget.op_log:
    print(f"{record.op_name:<16} ns={record.namespace!r}  cost={record.flop_cost:>12,}  cumulative={record.cumulative:>12,}")
```

Each `OpRecord` contains:

| Field | Description |
|-------|-------------|
| `op_name` | Operation name (e.g., `"einsum"`, `"exp"`) |
| `namespace` | Namespace label from the BudgetContext, or `None` |
| `subscripts` | Einsum subscript string, or `None` |
| `shapes` | Tuple of input shapes |
| `flop_cost` | FLOP cost of this single call |
| `cumulative` | Running total after this call |

## ⚠️ Common pitfalls

**Symptom:** `BudgetExhaustedError` but summary shows budget was nearly full

**Fix:** The budget is checked **before** execution. The failing operation's cost is in the error message — compare it with `budget.flops_remaining`.

## 📎 Related pages

- [Plan Your Budget](./plan-your-budget.md) — predict costs before running
- [Common Errors](../troubleshooting/common-errors.md) — all error types explained
