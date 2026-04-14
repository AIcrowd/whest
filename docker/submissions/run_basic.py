"""Basic smoke test — verifies whest works under lockdown."""

import whest as we

with we.BudgetContext(flop_budget=10_000_000) as budget:
    W = we.ones((256, 256))
    x = we.ones((256,))
    h = we.einsum("ij,j->i", W, x)
    h = we.maximum(h, 0)
    total = we.sum(h)
    print(budget.summary())
    print(f"Result: {total}")
