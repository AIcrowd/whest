"""Tests for symmetry-aware pointwise operations."""

import numpy

from mechestim._budget import BudgetContext
from mechestim._symmetric import SymmetricTensor, as_symmetric


class TestUnarySymmetry:
    def test_exp_symmetric_cost(self):
        import mechestim as me

        data = numpy.eye(10)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.exp(S)
            assert budget.flops_used == 55  # 10*11/2

    def test_exp_symmetric_returns_symmetric(self):
        import mechestim as me

        data = numpy.eye(4)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = me.exp(S)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_dims == [(0, 1)]

    def test_log_symmetric_cost(self):
        import mechestim as me

        data = numpy.eye(5) + 1
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            me.log(S)
            assert budget.flops_used == 15

    def test_plain_array_unchanged(self):
        import mechestim as me

        data = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            me.exp(data)
            assert budget.flops_used == 100


class TestBinarySymmetry:
    def test_add_both_symmetric_same_dims(self):
        import mechestim as me

        A = as_symmetric(numpy.eye(5), dims=(0, 1))
        B = as_symmetric(numpy.eye(5) * 2, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.add(A, B)
            assert budget.flops_used == 15
            assert isinstance(result, SymmetricTensor)

    def test_add_different_dims_no_symmetry(self):
        import mechestim as me

        A = as_symmetric(numpy.eye(5), dims=(0, 1))
        B = numpy.ones((5, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.add(A, B)
            assert budget.flops_used == 25
            assert not isinstance(result, SymmetricTensor)

    def test_multiply_scalar_preserves_symmetry(self):
        import mechestim as me

        A = as_symmetric(numpy.eye(5), dims=(0, 1))
        scalar = numpy.asarray(3.0)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = me.multiply(A, scalar)
            assert isinstance(result, SymmetricTensor)


class TestReductionSymmetry:
    def test_sum_symmetric_cost(self):
        import mechestim as me

        data = numpy.eye(10)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            me.sum(S)
            assert budget.flops_used == 55

    def test_sum_returns_plain(self):
        import mechestim as me

        data = numpy.eye(4)
        S = as_symmetric(data, dims=(0, 1))
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = me.sum(S)
            assert not isinstance(result, SymmetricTensor)
