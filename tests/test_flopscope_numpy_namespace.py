"""Tests for the JAX-style ``flopscope.numpy`` namespace.

The top-level :mod:`flopscope` package exposes only flopscope-specific
primitives (budget, configure, symmetric tensors, …).  Numpy-shaped
operations live under :mod:`flopscope.numpy` (aliased ``fnp``).  Names
not explicitly implemented by flopscope fall back to raw ``numpy``.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp


class TestTopLevelSurfaceIsFlopscopeOnly:
    """Numpy-shaped names must not be accessible on top-level ``flopscope``."""

    @pytest.mark.parametrize(
        "name",
        ["einsum", "array", "linspace", "sin", "matmul", "pi", "float32"],
    )
    def test_numpy_shaped_names_are_not_on_top_level(self, name):
        with pytest.raises(AttributeError, match="does not provide"):
            getattr(flops, name)

    @pytest.mark.parametrize(
        "name",
        [
            "BudgetContext", "configure", "SymmetricTensor", "PermutationGroup",
            "FlopscopeArray", "FlopscopeError", "budget",
        ],
    )
    def test_flopscope_primitives_are_on_top_level(self, name):
        assert getattr(flops, name) is not None


class TestFlopscopeNumpyExposesCountedSurface:
    def test_einsum_is_counted(self):
        with flops.BudgetContext(flop_budget=10**9, quiet=True) as b:
            a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            x = fnp.array([1.0, 1.0])
            _ = fnp.einsum("ij,j->i", a, x)
        # 2x2 matmul by 2-vector: ~2*2 = 4 counted multiplies+adds → non-zero
        assert b.flops_used > 0

    def test_array_constructor(self):
        a = fnp.array([1.0, 2.0, 3.0])
        # Returned object is a FlopscopeArray which subclasses numpy.ndarray.
        assert isinstance(a, np.ndarray)
        assert isinstance(a, flops.FlopscopeArray)

    def test_ndarray_alias_points_to_flopscope_array(self):
        assert fnp.ndarray is flops.FlopscopeArray

    def test_numpy_constants_available(self):
        assert fnp.pi == np.pi
        assert fnp.e == np.e
        assert fnp.inf == np.inf
        assert fnp.newaxis is None or fnp.newaxis is np.newaxis


class TestFlopscopeNumpySubmodules:
    def test_linalg_subpackage_wired(self):
        import flopscope.numpy.linalg as fnp_linalg
        assert fnp.linalg is fnp_linalg
        assert callable(fnp.linalg.svd)

    def test_fft_subpackage_wired(self):
        assert callable(fnp.fft.fft)

    def test_random_subpackage_wired(self):
        assert hasattr(fnp.random, "randn")

    def test_testing_subpackage_is_free(self):
        # assert_allclose is a test utility — does not count against budget.
        fnp.testing.assert_allclose(np.array([1.0]), np.array([1.0]))

    def test_typing_subpackage(self):
        from flopscope.numpy.typing import NDArray
        assert NDArray is not None


class TestRawNumpyFallback:
    """Attributes flopscope doesn't implement should transparently fall back
    to the raw ``numpy`` module (unclouded by the registry, uncounted)."""

    def test_unknown_numpy_scalar_type_falls_back(self):
        # ``numpy.int_`` is a basic scalar alias we don't expose explicitly.
        assert fnp.int_ is np.int_

    def test_unknown_name_raises_descriptive_attribute_error(self):
        with pytest.raises(AttributeError, match="does not provide"):
            _ = fnp.totally_not_a_real_numpy_attribute_xyz


class TestTopLevelHintsAtNumpySurface:
    """Error message for missing top-level attr should steer users to fnp."""

    def test_hint_mentions_flopscope_numpy(self):
        with pytest.raises(AttributeError) as exc_info:
            _ = flops.einsum
        msg = str(exc_info.value)
        assert "flopscope.numpy" in msg
