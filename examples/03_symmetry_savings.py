"""Symmetry detection in einsum — same-object savings.

Run: uv run python examples/03_symmetry_savings.py
"""

import whest as we

# With symmetry: x passed twice (same object)
with we.BudgetContext(flop_budget=10**8) as budget:
    x = we.ones((10, 256))
    A = we.ones((10, 10))
    result = we.einsum("ai,bi,ab->", x, x, A)
    cost_symmetric = budget.flops_used

# Without symmetry: y is a copy (different object)
with we.BudgetContext(flop_budget=10**8) as budget:
    x = we.ones((10, 256))
    y = x.copy()
    A = we.ones((10, 10))
    result = we.einsum("ai,bi,ab->", x, y, A)
    cost_no_symmetry = budget.flops_used

print(f"Cost with symmetry:    {cost_symmetric:>10,} FLOPs")
print(f"Cost without symmetry: {cost_no_symmetry:>10,} FLOPs")
print(
    f"Savings:               {cost_no_symmetry - cost_symmetric:>10,} FLOPs ({100 * (1 - cost_symmetric / cost_no_symmetry):.0f}%)"
)
