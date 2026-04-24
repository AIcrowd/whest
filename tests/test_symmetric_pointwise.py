"""Tests for symmetry-aware pointwise operations."""

import numpy

from whest._budget import BudgetContext
from whest._symmetric import SymmetricTensor, as_symmetric


class TestUnarySymmetry:
    def test_exp_symmetric_cost(self):
        import whest as we

        data = numpy.eye(10)
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = we.exp(S)
            assert budget.flops_used == 55  # 10*11/2

    def test_exp_symmetric_returns_symmetric(self):
        import whest as we

        data = numpy.eye(4)
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = we.exp(S)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_axes == [(0, 1)]

    def test_log_symmetric_cost(self):
        import whest as we

        data = numpy.eye(5) + 1
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            we.log(S)
            assert budget.flops_used == 15

    def test_plain_array_unchanged(self):
        import whest as we

        data = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            we.exp(data)
            assert budget.flops_used == 100


class TestBinarySymmetry:
    def test_add_both_symmetric_same_axes(self):
        import whest as we

        A = as_symmetric(numpy.eye(5), symmetric_axes=(0, 1))
        B = as_symmetric(numpy.eye(5) * 2, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = we.add(A, B)
            assert budget.flops_used == 15
            assert isinstance(result, SymmetricTensor)

    def test_add_different_dims_no_symmetry(self):
        import whest as we

        A = as_symmetric(numpy.eye(5), symmetric_axes=(0, 1))
        B = numpy.ones((5, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = we.add(A, B)
            assert budget.flops_used == 25
            assert not isinstance(result, SymmetricTensor)

    def test_multiply_scalar_preserves_symmetry(self):
        import whest as we

        A = as_symmetric(numpy.eye(5), symmetric_axes=(0, 1))
        scalar = numpy.asarray(3.0)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = we.multiply(A, scalar)
            assert isinstance(result, SymmetricTensor)


class TestReductionSymmetry:
    def test_sum_symmetric_cost(self):
        import whest as we

        data = numpy.eye(10)
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            we.sum(S)
            # sym(10,10) has 55 unique elements; full reduction: 55 − 1 = 54.
            assert budget.flops_used == 54

    def test_sum_returns_plain(self):
        import whest as we

        data = numpy.eye(4)
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = we.sum(S)
            assert not isinstance(result, SymmetricTensor)

    def test_sum_axis_sym_split_pair_runtime(self):
        import whest as we

        # sym(5,5) reducing axis=0 → split group → 5 outputs × (5−1) = 20.
        data = numpy.ones((5, 5))
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            we.sum(S, axis=0)
            assert budget.flops_used == 20

    def test_sum_axis_sym_preserving_runtime(self):
        import whest as we

        # sym(5,5,10) with sym axes (0,1) reducing axis=2:
        # S_2 survives on output (kept axes {0,1}) → 15 unique outputs × (10−1) = 135.
        data = numpy.ones((5, 5, 10))
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            we.sum(S, axis=2)
            assert budget.flops_used == 135

    def test_sum_tuple_axis_inner_clean_runtime(self):
        import whest as we

        # sym(5,5,10) reducing both (0,1) → inner-clean group (g.axes ⊆ R).
        # u_R = 15 (Burnside on S_2 over 5×5); 10 outputs (axis 2 kept).
        # Cost = 10 × (15−1) = 140.
        data = numpy.ones((5, 5, 10))
        S = as_symmetric(data, symmetric_axes=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            we.sum(S, axis=(0, 1))
            assert budget.flops_used == 140
