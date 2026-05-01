"""Tests for the counted Generator/RandomState subclasses."""

import numpy as np
import pytest

from flopscope._budget import BudgetContext
from flopscope.errors import UnsupportedFunctionError


class TestCountedGeneratorGate:
    def test_unknown_method_raises_unsupported(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with BudgetContext(flop_budget=1000, quiet=True):
            with pytest.raises(UnsupportedFunctionError, match="Generator"):
                rng.totally_fake_sampler()

    def test_unknown_attribute_access_raises(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with pytest.raises(UnsupportedFunctionError):
            rng.nonexistent_attr

    def test_dunder_passthrough(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        # __class__, __repr__ etc. must always be accessible
        assert rng.__class__ is _CountedGenerator
        repr(rng)  # must not raise

    def test_isinstance_numpy_generator(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        assert isinstance(rng, np.random.Generator)


class TestCountedGeneratorMethods:
    """Verify that registry-driven method generation produces working wrappers."""

    def test_standard_normal_charges_flops(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.standard_normal(10)
        assert budget.flops_used == 10

    def test_standard_normal_returns_flopscope_array(self):
        from flopscope._ndarray import FlopscopeArray
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = rng.standard_normal(10)
        assert isinstance(result, FlopscopeArray)

    def test_normal_with_size_kwarg(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.normal(0, 1, size=(10, 10))
        assert budget.flops_used == 100

    def test_choice_with_replacement(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.choice(100, size=20, replace=True)
        assert budget.flops_used == 20

    def test_choice_without_replacement(self):
        from flopscope._flops import _ceil_log2
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        n = 16
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.choice(n, size=5, replace=False)
        assert budget.flops_used == n * _ceil_log2(n)

    def test_shuffle_charges_input_size(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        a = np.arange(50)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.shuffle(a)
        assert budget.flops_used == 50

    def test_bytes_charges_length(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.bytes(42)
        assert budget.flops_used == 42

    def test_bit_generator_is_free(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        with BudgetContext(flop_budget=10, quiet=True) as budget:
            _ = rng.bit_generator
        assert budget.flops_used == 0

    def test_counted_set_populated(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        # standard_normal must be in _COUNTED for the gate to allow it
        assert "standard_normal" in _CountedGenerator._COUNTED
        # bit_generator and spawn must be in _FREE
        assert "bit_generator" in _CountedGenerator._FREE
        assert "spawn" in _CountedGenerator._FREE

    def test_shuffle_2d_charges_shape_axis_not_numel(self):
        """Issue #18 follow-up: shuffle on 2D charges shape[0], not numel.

        Fisher-Yates does shape[axis] RNG draws regardless of slice width.
        Memory moves don't count as FLOPs.
        """
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        a = np.arange(50).reshape(5, 10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.shuffle(a)
        assert budget.flops_used == 5  # shape[0], not 50

    def test_shuffle_2d_explicit_axis(self):
        """shuffle(arr, axis=1) charges shape[1] (per-column shuffle)."""
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        a = np.arange(50).reshape(5, 10)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rng.shuffle(a, axis=1)
        assert budget.flops_used == 10  # shape[1], not 50


class TestCountedGeneratorOverrides:
    def test_spawn_returns_counted_children(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        children = rng.spawn(3)
        assert len(children) == 3
        for c in children:
            assert isinstance(c, _CountedGenerator)

    def test_spawned_child_charges_flops(self):
        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        children = rng.spawn(1)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            children[0].standard_normal(7)
        assert budget.flops_used == 7

    def test_pickle_roundtrip_preserves_counted_class(self):
        import pickle

        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        revived = pickle.loads(pickle.dumps(rng))
        assert isinstance(revived, _CountedGenerator)

    def test_pickled_revival_still_counts(self):
        import pickle

        from flopscope.numpy.random._counted_classes import _CountedGenerator

        bg = np.random.default_rng(42).bit_generator
        rng = _CountedGenerator(bg)
        revived = pickle.loads(pickle.dumps(rng))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            revived.standard_normal(13)
        assert budget.flops_used == 13


class TestCountedRandomState:
    def test_randn_charges_flops(self):
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rs.randn(10)
        assert budget.flops_used == 10

    def test_randn_returns_flopscope_array(self):
        from flopscope._ndarray import FlopscopeArray
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = rs.randn(10)
        assert isinstance(result, FlopscopeArray)

    def test_seed_is_free(self):
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        with BudgetContext(flop_budget=10, quiet=True) as budget:
            rs.seed(99)
        assert budget.flops_used == 0

    def test_get_state_is_free(self):
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        with BudgetContext(flop_budget=10, quiet=True) as budget:
            rs.get_state()
        assert budget.flops_used == 0

    def test_unknown_method_raises(self):
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        with BudgetContext(flop_budget=1000, quiet=True):
            with pytest.raises(UnsupportedFunctionError, match="RandomState"):
                rs.totally_fake_sampler()

    def test_isinstance_numpy_random_state(self):
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        assert isinstance(rs, np.random.RandomState)

    def test_choice_without_replacement_uses_sort_cost(self):
        from flopscope._flops import _ceil_log2
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        n = 16
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rs.choice(n, size=5, replace=False)
        assert budget.flops_used == n * _ceil_log2(n)

    def test_shuffle_charges_input_size(self):
        from flopscope.numpy.random._counted_classes import _CountedRandomState

        rs = _CountedRandomState(42)
        a = np.arange(50)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            rs.shuffle(a)
        assert budget.flops_used == 50
