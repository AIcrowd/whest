"""End-to-end integration tests via BudgetContext for the new reduction model."""

import flopscope as fps
import flopscope.numpy as fnp


def _flops_used(bc):
    return bc.summary_dict()["flops_used"]


def test_dense_sum_charges_n_minus_1_end_to_end():
    a = fnp.zeros(10)
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.sum(a)
    assert _flops_used(bc) == 9


def test_symmetric_sum_charges_via_orbit_mapping():
    n = 4
    T = fps.as_symmetric(fnp.zeros((n, n, n)), symmetry=(0, 1, 2))
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.sum(T, axis=2)
    expected = fps.reduction_accumulation_cost(T, axis=2).total
    assert _flops_used(bc) == expected


def test_prod_uses_tier1_model():
    a = fnp.ones(5)
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.prod(a)
    # 5 - 1 = 4 multiplications
    assert _flops_used(bc) == 4


def test_max_uses_tier1_model():
    a = fnp.zeros(5)
    with fps.BudgetContext(flop_budget=10**12, quiet=True) as bc:
        fnp.max(a)
    # 5 - 1 = 4 comparisons
    assert _flops_used(bc) == 4
