"""Tests for flopscope.numpy.random counted wrappers."""

import numpy

from flopscope._budget import BudgetContext
from flopscope._flops import _ceil_log2
from flopscope.numpy import random as merandom


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
    def test_constructor_is_free(self):
        with BudgetContext(flop_budget=10, quiet=True) as budget:
            merandom.default_rng(42)
            assert budget.flops_used == 0

    def test_returned_rng_charges_flops_on_sample(self):
        # Issue #18 regression: previously rng.standard_normal() was 0 FLOPs.
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng = merandom.default_rng(42)
            rng.standard_normal((3,))
        assert budget.flops_used == 3

    def test_returned_rng_returns_flopscope_array(self):
        from flopscope._ndarray import FlopscopeArray

        with BudgetContext(flop_budget=10**6, quiet=True):
            rng = merandom.default_rng(42)
            result = rng.standard_normal((3,))
        assert isinstance(result, FlopscopeArray)

    def test_returned_rng_is_numpy_generator_subclass(self):
        rng = merandom.default_rng(42)
        assert isinstance(rng, numpy.random.Generator)


class TestRandomStateCounted:
    def test_constructor_is_free(self):
        with BudgetContext(flop_budget=10, quiet=True) as budget:
            merandom.RandomState(42)
            assert budget.flops_used == 0

    def test_randn_charges_flops(self):
        # Issue #18 regression: was 0 FLOPs.
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rs = merandom.RandomState(42)
            rs.randn(10)
        assert budget.flops_used == 10

    def test_randn_returns_flopscope_array(self):
        from flopscope._ndarray import FlopscopeArray

        with BudgetContext(flop_budget=10**6, quiet=True):
            rs = merandom.RandomState(42)
            result = rs.randn(10)
        assert isinstance(result, FlopscopeArray)

    def test_isinstance_numpy_random_state(self):
        rs = merandom.RandomState(42)
        assert isinstance(rs, numpy.random.RandomState)


class TestGetattrFallback:
    def test_unknown_attr_raises(self):
        # Issue #18 regression: was silent forward to numpy.
        import pytest

        with pytest.raises(AttributeError, match="default_rng"):
            _ = merandom.completely_unknown_attribute

    def test_random_integers_raises_via_fallback(self):
        # numpy.random.random_integers is a deprecated alias not in our explicit list.
        # Was silently forwarded; now must raise AttributeError.
        import pytest

        with pytest.raises(AttributeError):
            _ = merandom.random_integers

    def test_bit_generator_classes_pass_through(self):
        # numpy bit-generator classes are pure machinery, no math; allowlisted.
        for name in (
            "BitGenerator",
            "MT19937",
            "PCG64",
            "PCG64DXSM",
            "Philox",
            "SFC64",
        ):
            cls = getattr(merandom, name)
            assert cls is getattr(numpy.random, name), name
