"""Tests for symmetry-aware einsum."""

import numpy

import whest as we
from whest._budget import BudgetContext
from whest._einsum import einsum
from whest._symmetric import SymmetricTensor, as_symmetric


class TestEinsumSymmetricInput:
    def test_symmetric_input_reduces_cost(self):
        S = as_symmetric(
            numpy.ones((10, 10, 5)),
            symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
        )
        v = numpy.ones(5)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            einsum("ijk,k->ij", S, v)
            dense_cost = 2 * 10 * 10 * 5
            assert budget.flops_used < dense_cost
            assert budget.flops_used > 0

    def test_plain_input_unchanged(self):
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            einsum("ij,j->i", A, v)
            assert budget.flops_used == 100


class TestEinsumSymmetricOutput:
    def test_symmetry_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        target = we.SymmetryGroup.symmetric(axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=target)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == target

    def test_without_symmetry_returns_plain(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk->ik", A, B)
            assert not isinstance(result, SymmetricTensor)


class TestEinsumSymmetryParam:
    def test_symmetry_param_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        g = we.SymmetryGroup.symmetric(axes=(0, 1))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=g)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == g

    def test_symmetry_accepts_exact_group_shorthand(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetry=(0, 1))
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == we.SymmetryGroup.symmetric(axes=(0, 1))
