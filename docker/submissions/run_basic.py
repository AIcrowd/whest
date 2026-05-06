"""Basic smoke test — verifies flopscope works under lockdown."""

import flopscope as we

with we.BudgetContext(flop_budget=10_000_000) as budget:
    W = we.ones((256, 256))
    x = we.ones((256,))
    h = we.einsum("ij,j->i", W, x)
    h = we.maximum(h, 0)
    total = we.sum(h)
    print(budget.summary())
    print(f"Result: {total}")
