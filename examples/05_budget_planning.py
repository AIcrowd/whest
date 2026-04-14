"""Budget planning — query costs before executing.

Run: uv run python examples/05_budget_planning.py
"""

import whest as we

# Plan a two-layer forward pass
width = 256
budget_limit = 500_000

steps = [
    (
        "Layer 1: W1 @ x",
        we.flops.einsum_cost("ij,j->i", shapes=[(width, width), (width,)]),
    ),
    ("Layer 1: ReLU", we.flops.pointwise_cost(shape=(width,))),
    (
        "Layer 2: W2 @ h1",
        we.flops.einsum_cost("ij,j->i", shapes=[(width, width), (width,)]),
    ),
    ("Layer 2: ReLU", we.flops.pointwise_cost(shape=(width,))),
    (
        "Output: mean",
        we.flops.reduction_cost(input_shape=(width,)) + 1,
    ),  # +1 for division
]

total = sum(cost for _, cost in steps)
fits = "YES" if total <= budget_limit else "NO"

print(f"Budget: {budget_limit:,} FLOPs")
print(f"{'Operation':<25} {'FLOPs':>10}")
print("-" * 37)
for name, cost in steps:
    print(f"{name:<25} {cost:>10,}")
print("-" * 37)
print(f"{'Total':<25} {total:>10,}")
print(f"Fits in budget? {fits}")

# Now execute it
if total <= budget_limit:
    with we.BudgetContext(flop_budget=budget_limit) as budget:
        x = we.random.randn(width)
        W1 = we.random.randn(width, width)
        W2 = we.random.randn(width, width)

        h1 = we.einsum("ij,j->i", W1, x)
        h1 = we.maximum(h1, 0)
        h2 = we.einsum("ij,j->i", W2, h1)
        h2 = we.maximum(h2, 0)
        result = we.mean(h2)

        print(f"\nActual usage: {budget.flops_used:,} / {budget_limit:,} FLOPs")
