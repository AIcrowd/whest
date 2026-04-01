import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    x = me.ones((100,))
    y = me.exp(x)
    z = x + y
    print(f"Result shape: {z.shape}")
    print(f"First value: {z[0]}")
    print(f"FLOPs used: {budget.flops_used}")
    print(f"Summary: {budget.summary()}")
    print("SUCCESS: client-server working!")
