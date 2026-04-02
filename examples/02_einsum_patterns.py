"""Common einsum patterns and their FLOP costs.

Run: uv run python examples/02_einsum_patterns.py
"""

import mechestim as me

patterns = [
    ("Matrix-vector", "ij,j->i", [(256, 256), (256,)]),
    ("Matrix multiply", "ij,jk->ik", [(256, 256), (256, 256)]),
    ("Outer product", "i,j->ij", [(256,), (256,)]),
    ("Trace", "ii->", [(256, 256)]),
    ("Bilinear form", "ai,bi,ab->", [(10, 256), (10, 256), (10, 10)]),
]

print(f"{'Pattern':<20} {'Subscripts':<15} {'FLOPs':>12}")
print("-" * 50)
for name, subs, shapes in patterns:
    cost = me.flops.einsum_cost(subs, shapes=shapes)
    print(f"{name:<20} {subs:<15} {cost:>12,}")
