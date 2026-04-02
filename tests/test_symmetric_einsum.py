"""Tests for symmetry-aware einsum."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim._einsum import einsum
from mechestim._symmetric import SymmetricTensor, as_symmetric


class TestEinsumSymmetricInput:
    def test_symmetric_input_reduces_cost(self):
        S = as_symmetric(numpy.eye(10), dims=(0, 1))
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum('ij,j->i', S, v)
            assert budget.flops_used == 55

    def test_plain_input_unchanged(self):
        A = numpy.eye(10)
        v = numpy.ones(10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum('ij,j->i', A, v)
            assert budget.flops_used == 100


class TestEinsumSymmetricOutput:
    def test_symmetric_dims_returns_symmetric_tensor(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum('ki,kj->ij', X, X, symmetric_dims=[(0, 1)])
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_dims == [(0, 1)]

    def test_without_symmetric_dims_returns_plain(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum('ij,jk->ik', A, B)
            assert not isinstance(result, SymmetricTensor)
