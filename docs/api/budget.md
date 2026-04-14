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

## Wall-clock time limits

Use `wall_time_limit_s` to set a wall-clock deadline on a `BudgetContext`. When the deadline is exceeded, the next operation raises `TimeExhaustedError`:

```python
import whest as we

with we.BudgetContext(flop_budget=10**9, wall_time_limit_s=5.0) as budget:
    # Your computation here — must complete within 5 seconds
    ...
```

The banner shows both limits when a time limit is set:

```
whest 0.2.0 (numpy 2.2.0 backend) | budget: 1.00e+09 FLOPs | time limit: 5.0s
```

### Timing properties

After the context exits, timing data is available:

| Property | Description |
|----------|-------------|
| `budget.wall_time_s` | Total wall-clock duration of the context |
| `budget.wall_time_limit_s` | Configured time limit (or `None`) |
| `budget.elapsed_s` | Live elapsed time (while context is active) |
| `budget.total_tracked_time` | Sum of wall-clock durations of all numpy calls |
| `budget.untracked_time` | Wall time not attributable to numpy calls (Python overhead) |

### How enforcement works

The deadline is checked cooperatively — before each operation starts (in `deduct()`) and after each operation completes (in `_OpTimer.__exit__()`). This means a single long-running numpy call can overshoot the deadline by its own duration, but the error fires immediately after that call returns.

Signal-based preemption (`SIGALRM`) is deliberately not used because Python signal handlers cannot interrupt C extensions (numpy/LAPACK/BLAS), making them ineffective for the operations where timing matters most. In competition evaluation, the container enforces a hard kernel-level time limit via cgroups as the ultimate backstop.

## API Reference

::: whest._budget.BudgetContext

::: whest._budget.OpRecord

::: whest._budget.budget

::: whest._budget.budget_summary_dict

::: whest._budget.budget_reset

::: whest._display.render_budget_summary

::: whest._display.budget_summary

::: whest._display.budget_live
