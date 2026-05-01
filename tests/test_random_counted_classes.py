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
        # bit_generator must be in _FREE
        assert "bit_generator" in _CountedGenerator._FREE
