"""Tests for mechestim.random counted wrappers."""

import numpy

from mechestim import random as merandom
from mechestim._budget import BudgetContext
from mechestim._flops import _ceil_log2


class TestSeed:
    def test_seed_is_free(self):
        with BudgetContext(flop_budget=10, quiet=True) as budget:
            merandom.seed(42)
            assert budget.flops_used == 0


class TestRandn:
    def test_shape(self):
        with BudgetContext(flop_budget=10**6, quiet=True):
            assert merandom.randn(3, 4).shape == (3, 4)

    def test_cost(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.randn(10, 20)
            assert budget.flops_used == 200


class TestNormal:
    def test_shape(self):
        with BudgetContext(flop_budget=10**6, quiet=True):
            assert merandom.normal(0, 1, size=(5,)).shape == (5,)

    def test_cost(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.normal(0, 1, size=(10, 10))
            assert budget.flops_used == 100


class TestRand:
    def test_cost(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.rand(5, 5)
            assert budget.flops_used == 25


class TestUniform:
    def test_cost(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.uniform(0, 1, size=50)
            assert budget.flops_used == 50


class TestRandint:
    def test_cost(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.randint(0, 10, size=(4, 5))
            assert budget.flops_used == 20


class TestPermutation:
    def test_cost(self):
        n = 16
        # Sheet formula: numel(output)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.permutation(n)
            assert budget.flops_used == n


class TestShuffle:
    def test_cost(self):
        a = numpy.arange(16)
        n = 16
        # Sheet formula: numel(output)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.shuffle(a)
            assert budget.flops_used == n


class TestChoiceWithReplacement:
    def test_cost(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.choice(100, size=20, replace=True)
            assert budget.flops_used == 20


class TestChoiceWithoutReplacement:
    def test_cost(self):
        n = 16
        expected = n * _ceil_log2(n)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.choice(n, size=5, replace=False)
            assert budget.flops_used == expected


class TestDefaultRng:
    def test_is_free(self):
        with BudgetContext(flop_budget=10, quiet=True) as budget:
            merandom.default_rng(42)
            assert budget.flops_used == 0

    def test_rng_passthrough(self):
        rng = merandom.default_rng(42)
        assert rng.standard_normal((3,)).shape == (3,)
