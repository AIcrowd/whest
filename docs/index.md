# mechestim

NumPy-compatible math primitives with analytical FLOP counting for the
[Mechanistic Estimation Challenge](https://github.com/AIcrowd/mechestim).

## What is this?

mechestim is a drop-in replacement for a subset of NumPy that counts
FLOPs as you compute. Use it to develop algorithms where the goal is
to minimize computational cost, not wall-clock time.

## Quick example

    import mechestim as me

    with me.BudgetContext(flop_budget=1_000_000) as budget:
        W = me.array(weight_matrix)
        x = me.zeros((256,))
        h = me.einsum('ij,j->i', W, x)
        h = me.maximum(h, 0)
        print(budget.summary())

## Installation

    pip install git+https://github.com/AIcrowd/mechestim.git
