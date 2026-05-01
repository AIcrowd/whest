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
