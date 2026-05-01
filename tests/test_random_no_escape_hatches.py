"""Regression tests pinned to flopscope#18.

Each test snapshots a specific bypass that was present before the fix.
If any test here regresses, the FLOP-accounting hole has reopened.
"""

import pickle

import numpy
import pytest

from flopscope import BudgetContext
from flopscope._ndarray import FlopscopeArray
from flopscope.numpy import random as merandom


class TestDefaultRngHatch:
    def test_default_rng_charges_flops(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng = merandom.default_rng(42)
            rng.standard_normal(100)
        assert budget.flops_used == 100

    def test_default_rng_returns_flopscope_array(self):
        with BudgetContext(flop_budget=10**6, quiet=True):
            rng = merandom.default_rng(42)
            x = rng.standard_normal(10)
        assert isinstance(x, FlopscopeArray)

    def test_default_rng_uniform_charges(self):
        # Replicate whestbench/starterkit canonical idiom.
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng = merandom.default_rng(42)
            rng.uniform(0.0, 1.0, size=(8, 8))
        assert budget.flops_used == 64


class TestRandomStateHatch:
    def test_RandomState_charges_flops(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rs = merandom.RandomState(42)
            rs.randn(10)
        assert budget.flops_used == 10

    def test_RandomState_returns_flopscope_array(self):
        with BudgetContext(flop_budget=10**6, quiet=True):
            rs = merandom.RandomState(42)
            z = rs.randn(10)
        assert isinstance(z, FlopscopeArray)


class TestGetattrHatch:
    def test_unknown_attribute_raises(self):
        # Was: silent forward to numpy.random with no counting.
        with pytest.raises(AttributeError, match="default_rng"):
            merandom.completely_made_up_attribute

    def test_random_integers_raises(self):
        # numpy.random.random_integers exists but is not in our explicit list.
        with pytest.raises(AttributeError):
            merandom.random_integers


class TestModuleLevelUnchanged:
    def test_randn_no_warning(self):
        # The fix must not introduce a DeprecationWarning on module-level samplers.
        # Use pytest.warns(None) by treating any warning as a failure via -W error
        # (project conftest doesn't enforce this; check explicitly).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn any warning into an error
            with BudgetContext(flop_budget=10**6, quiet=True):
                merandom.randn(5)  # must not raise

    def test_randn_still_counts(self):
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            merandom.randn(10, 20)
        assert budget.flops_used == 200


class TestPickleRoundtrip:
    def test_pickle_default_rng_preserves_counting(self):
        rng = merandom.default_rng(42)
        revived = pickle.loads(pickle.dumps(rng))
        # Type identity preserved
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        assert isinstance(revived, _CountedGenerator)
        # Counting still works
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            revived.standard_normal(7)
        assert budget.flops_used == 7


class TestSpawn:
    def test_spawned_children_are_counted(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        rng = merandom.default_rng(42)
        children = rng.spawn(3)
        assert all(isinstance(c, _CountedGenerator) for c in children)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            children[0].standard_normal(11)
        assert budget.flops_used == 11


class TestBitGeneratorPassthrough:
    def test_passthrough_classes_resolve(self):
        for name in ("BitGenerator", "MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"):
            cls = getattr(merandom, name)
            assert cls is getattr(numpy.random, name)

    def test_construction_via_explicit_bit_generator(self):
        # Generator(BitGenerator(seed)) is a documented numpy idiom.
        rng = merandom.Generator(merandom.PCG64(42))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.normal(0, 1, size=10)
        assert budget.flops_used == 10
