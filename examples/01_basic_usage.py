"""Basic mechestim usage — BudgetContext, free ops, counted ops, summary.

Run: uv run python examples/01_basic_usage.py
"""

import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    # Free operations (0 FLOPs)
    W = me.ones((256, 256))
    x = me.ones((256,))

    # Counted operations
    h = me.einsum("ij,j->i", W, x)  # matrix-vector multiply
    h = me.maximum(h, 0)  # ReLU
    h = me.exp(h)  # exponential
    total = me.sum(h)  # reduction

    print(budget.summary())
    print(f"\nResult: {total}")
