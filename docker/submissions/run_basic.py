"""Basic smoke test — verifies mechestim works under lockdown."""

import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    W = me.ones((256, 256))
    x = me.ones((256,))
    h = me.einsum("ij,j->i", W, x)
    h = me.maximum(h, 0)
    total = me.sum(h)
    print(budget.summary())
    print(f"Result: {total}")
