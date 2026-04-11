"""Tests for symmetry-aware einsum."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._einsum import einsum
from mechestim._symmetric import SymmetricTensor, as_symmetric


class TestEinsumSymmetricInput:
    def test_symmetric_input_reduces_cost(self):
        # Use a contraction where symmetric indices survive in the output:
        # "ijk,k->ij" with S2 on {i,j}. The symmetric group on {i,j}
        # provides a real cost reduction since both indices survive.
        S = as_symmetric(numpy.ones((10, 10, 5)), symmetric_axes=(0, 1))
        v = numpy.ones(5)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ijk,k->ij", S, v)
            dense_cost = 2 * 10 * 10 * 5  # 1000 with op_factor
            assert budget.flops_used < dense_cost
            assert budget.flops_used > 0

    def test_plain_input_unchanged(self):
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ij,j->i", A, v)
            assert budget.flops_used == 200  # 10*10 * op_factor(2)


class TestEinsumSymmetricOutput:
    def test_symmetric_axes_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetric_axes=[(0, 1)])
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_axes == [(0, 1)]

    def test_without_symmetric_axes_returns_plain(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk->ik", A, B)
            assert not isinstance(result, SymmetricTensor)


from mechestim._perm_group import PermutationGroup


class TestEinsumSymmetryParam:
    def test_symmetry_param_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        g = PermutationGroup.symmetric(2, axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=g)
            assert isinstance(result, SymmetricTensor)

    def test_symmetry_and_symmetric_axes_mutually_exclusive(self):
        X = numpy.ones((3, 4))
        g = PermutationGroup.symmetric(2, axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            with pytest.raises(ValueError, match="mutually exclusive"):
                einsum("ki,kj->ij", X, X, symmetric_axes=[(0, 1)], symmetry=g)
