# Quick Start

## Installation

    pip install git+https://github.com/AIcrowd/mechestim.git

## Basic Usage

    import mechestim as me

    # All computation happens inside a BudgetContext
    with me.BudgetContext(flop_budget=10_000_000) as budget:
        # Free operations (0 FLOPs)
        A = me.ones((256, 256))
        B = me.eye(256)

        # Counted operations (deduct from budget)
        C = me.einsum('ij,jk->ik', A, B)   # costs 256^3 FLOPs
        D = me.exp(C)                       # costs 256^2 FLOPs

        # Check your budget
        print(f"Used: {budget.flops_used:,} FLOPs")
        print(f"Remaining: {budget.flops_remaining:,} FLOPs")
        print(budget.summary())

## Planning Your Budget

Query operation costs before executing them:

    import mechestim as me

    cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
    print(f"Matmul cost: {cost:,} FLOPs")

    cost = me.flops.svd_cost(m=256, n=256, k=10)
    print(f"SVD cost: {cost:,} FLOPs")
