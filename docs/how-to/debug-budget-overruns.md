# Debug Budget Overruns

## When to use this page

Use this page when you hit a `BudgetExhaustedError` and need to find which operations are using the most FLOPs.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Reading the budget summary

Call `budget.summary()` at any point inside a `BudgetContext`, or call `we.budget_summary()` outside a context for a session-wide summary across all namespaces:

```python
import whest as we

with we.BudgetContext(flop_budget=10_000_000) as budget:
    A = we.ones((256, 256))
    x = we.ones((256,))

    h = we.einsum('ij,j->i', A, x)
    h = we.exp(h)
    h = we.sum(h)

    print(budget.summary())

# Outside the context — summarises every namespace recorded this session
print(we.budget_summary())
```

The summary shows cost per operation type, sorted by highest cost first.

## Session-wide programmatic analysis

Use `we.budget_summary_dict()` to retrieve aggregated budget data as a dict for automated analysis (e.g. in tests or notebooks):

```python
data = we.budget_summary_dict()
print(f"Budget: {data['flop_budget']:,}")
print(f"Used:   {data['flops_used']:,}")
print(f"Left:   {data['flops_remaining']:,}")
for op_name, op_data in data["operations"].items():
    print(f"  {op_name}: {op_data['flops']:,} ({op_data['calls']} calls)")
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

## Live budget display

Use `we.budget_live()` as a context manager for a Rich-based live-updating display that refreshes as operations run. This is useful when running long computations and you want to watch budget consumption in real time:

```python
import whest as we

with we.budget_live():
    with we.BudgetContext(flop_budget=10**8, namespace="training") as budget:
        for i in range(100):
            W = we.ones((256, 256))
            x = we.ones((256,))
            h = we.einsum('ij,j->i', W, x)
            h = we.exp(h)
            # The live display updates automatically as FLOPs are consumed
```

If Rich is not installed, `budget_live()` falls back to printing a plain-text summary on exit.

## ⚠️ Common pitfalls

**Symptom:** `BudgetExhaustedError` but summary shows budget was nearly full

**Fix:** The budget is checked **before** execution. The failing operation's cost is in the error message — compare it with `budget.flops_remaining`.

## 📎 Related pages

- [Plan Your Budget](./plan-your-budget.md) — predict costs before running
- [Common Errors](../troubleshooting/common-errors.md) — all error types explained
