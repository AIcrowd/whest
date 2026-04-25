"""Tests for the flopscope client API surface.

These tests verify that the public API is correctly wired up without
requiring a running server.  They test:

- Registry correctness (entry count, lookups, categories)
- Module-level constants and dtype exports
- Class exports (ndarray, BudgetContext, errors)
- Auto-generated proxy functions exist and are callable
- __getattr__ error messages for blacklisted and unknown names
- Submodule structure (fft, linalg, random)
- flops local cost functions
"""

from __future__ import annotations

import math

import pytest
from flopscope._remote_array import RemoteArray

import flopscope as flops
import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
from flopscope._registry import (
    BLACKLISTED,
    COUNTED_BINARY,
    COUNTED_CUSTOM,
    COUNTED_REDUCTION,
    COUNTED_UNARY,
    FREE,
    FUNCTION_CATEGORIES,
    get_category,
    is_blacklisted,
    is_valid_op,
    iter_proxyable,
)

# ===================================================================
# Registry tests
# ===================================================================


class TestRegistry:
    """Tests for _registry.py and _registry_data.py."""

    def test_registry_has_482_plus_entries(self):
        assert len(FUNCTION_CATEGORIES) >= 400, (
            f"Registry suspiciously small: {len(FUNCTION_CATEGORIES)} entries. "
            f"Run: uv run scripts/sync_client.py"
        )

    @pytest.mark.parametrize("op", ["add", "exp", "zeros", "einsum", "dot", "matmul"])
    def test_is_valid_op_known(self, op):
        assert is_valid_op(op) is True

    def test_is_valid_op_unknown(self):
        assert is_valid_op("totally_fake_function_xyz") is False

    def test_get_category_add(self):
        assert get_category("add") == COUNTED_BINARY

    def test_get_category_exp(self):
        assert get_category("exp") == COUNTED_UNARY

    def test_get_category_sum(self):
        assert get_category("sum") == COUNTED_REDUCTION

    def test_get_category_einsum(self):
        assert get_category("einsum") == COUNTED_CUSTOM

    def test_get_category_zeros(self):
        assert get_category("zeros") == FREE

    def test_get_category_unknown(self):
        assert get_category("totally_fake_xyz") is None

    @pytest.mark.parametrize("op", ["save", "load", "savetxt", "savez"])
    def test_blacklisted_category(self, op):
        assert get_category(op) == BLACKLISTED

    def test_is_blacklisted(self):
        assert is_blacklisted("save") is True
        assert is_blacklisted("add") is False
        assert is_blacklisted("unknown_xyz") is False

    def test_iter_proxyable_nonempty(self):
        ops = iter_proxyable()
        assert len(ops) > 100
        # Should not contain any blacklisted operations
        for op in ops:
            assert get_category(op) != BLACKLISTED

    def test_iter_proxyable_prefix_filter(self):
        random_ops = iter_proxyable(prefix="random.")
        assert len(random_ops) > 10
        for op in random_ops:
            assert op.startswith("random.")

    def test_fft_ops_are_registered(self):
        fft_ops = [k for k in FUNCTION_CATEGORIES if k.startswith("fft.")]
        assert len(fft_ops) > 0

    def test_linalg_svd_is_counted_custom(self):
        assert get_category("linalg.svd") == COUNTED_CUSTOM


# ===================================================================
# Constants and dtype exports
# ===================================================================


class TestConstants:
    """Tests for constants exported by the top-level package."""

    def test_pi(self):
        assert fnp.pi == pytest.approx(math.pi)

    def test_e(self):
        assert fnp.e == pytest.approx(math.e)

    def test_inf(self):
        assert fnp.inf == math.inf

    def test_nan(self):
        assert math.isnan(fnp.nan)

    def test_newaxis(self):
        assert fnp.newaxis is None


class TestDtypes:
    """Tests for dtype string exports."""

    def test_float64(self):
        assert fnp.float64 == "float64"

    def test_float32(self):
        assert fnp.float32 == "float32"

    def test_float16(self):
        assert fnp.float16 == "float16"

    def test_int64(self):
        assert fnp.int64 == "int64"

    def test_int32(self):
        assert fnp.int32 == "int32"

    def test_int16(self):
        assert fnp.int16 == "int16"

    def test_int8(self):
        assert fnp.int8 == "int8"

    def test_uint8(self):
        assert fnp.uint8 == "uint8"

    def test_bool_(self):
        assert fnp.bool_ == "bool"

    def test_complex64(self):
        assert fnp.complex64 == "complex64"

    def test_complex128(self):
        assert fnp.complex128 == "complex128"


# ===================================================================
# Class exports
# ===================================================================


class TestClassExports:
    """Tests for class/type exports."""

    def test_ndarray_is_RemoteArray(self):
        assert fnp.ndarray is RemoteArray

    def test_BudgetContext_is_class(self):
        from flopscope._budget import BudgetContext

        assert flops.BudgetContext is BudgetContext

    def test_OpRecord_is_class(self):
        from flopscope._budget import OpRecord

        assert flops.OpRecord is OpRecord

    def test_errors_exported(self):
        assert issubclass(flops.BudgetExhaustedError, flops.FlopscopeError)
        assert issubclass(flops.NoBudgetContextError, flops.FlopscopeError)
        assert issubclass(flops.SymmetryError, flops.FlopscopeError)
        assert issubclass(flops.FlopscopeServerError, flops.FlopscopeError)
        assert issubclass(flops.FlopscopeWarning, UserWarning)


# ===================================================================
# Proxy function existence
# ===================================================================


class TestProxyFunctions:
    """Tests that auto-generated proxy functions exist and are callable."""

    @pytest.mark.parametrize(
        "name",
        [
            "add",
            "subtract",
            "multiply",
            "divide",
            "exp",
            "log",
            "sqrt",
            "abs",
            "zeros",
            "ones",
            "eye",
            "arange",
            "dot",
            "matmul",
            "sum",
            "mean",
            "reshape",
            "transpose",
            "concatenate",
            "stack",
            "clip",
        ],
    )
    def test_proxy_exists_and_callable(self, name):
        fn = getattr(fnp, name)
        assert callable(fn), f"fnp.{name} should be callable"

    def test_array_is_callable(self):
        assert callable(fnp.array)

    def test_einsum_is_callable(self):
        assert callable(fnp.einsum)

    def test_proxy_has_correct_name(self):
        assert fnp.add.__name__ == "add"
        assert fnp.exp.__name__ == "exp"
        assert fnp.zeros.__name__ == "zeros"


# ===================================================================
# __getattr__ error messages
# ===================================================================


class TestGetattr:
    """Tests for __getattr__ error handling on blacklisted/unknown names."""

    def test_blacklisted_top_level(self):
        with pytest.raises(AttributeError, match="intentionally not supported"):
            _ = fnp.save

    def test_blacklisted_top_level_load(self):
        with pytest.raises(AttributeError, match="intentionally not supported"):
            _ = fnp.load

    def test_unknown_top_level(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = fnp.completely_nonexistent_function_xyz

    def test_fft_registered_not_blacklisted(self):
        # fft.fft is now counted_custom, not blacklisted
        with pytest.raises(AttributeError, match="registered but not yet implemented"):
            _ = fnp.fft.fft

    def test_fft_rfft_registered_not_blacklisted(self):
        # fft.rfft is now counted_custom, not blacklisted
        with pytest.raises(AttributeError, match="registered but not yet implemented"):
            _ = fnp.fft.rfft

    def test_unknown_fft(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = fnp.fft.nonexistent_xyz

    def test_linalg_eig_is_proxy(self):
        # linalg.eig is now a counted_custom proxy, not blacklisted
        assert callable(fnp.linalg.eig)

    def test_unknown_linalg(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = fnp.linalg.nonexistent_xyz

    def test_unknown_random(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = fnp.random.nonexistent_xyz


# ===================================================================
# Submodule structure
# ===================================================================


class TestSubmodules:
    """Tests for submodule accessibility."""

    def test_fft_is_module(self):
        import types

        assert isinstance(fnp.fft, types.ModuleType)

    def test_linalg_is_module(self):
        import types

        assert isinstance(fnp.linalg, types.ModuleType)

    def test_random_is_module(self):
        import types

        assert isinstance(fnp.random, types.ModuleType)

    def test_flops_is_module(self):
        import types

        assert isinstance(flops.accounting, types.ModuleType)

    def test_linalg_svd_callable(self):
        assert callable(fnp.linalg.svd)

    def test_random_functions_callable(self):
        for name in [
            "randn",
            "normal",
            "uniform",
            "rand",
            "seed",
            "choice",
            "permutation",
            "shuffle",
        ]:
            fn = getattr(fnp.random, name)
            assert callable(fn), f"fnp.random.{name} should be callable"


# ===================================================================
# flops local cost functions
# ===================================================================


class TestFlopsLocal:
    """Tests for locally computable flops cost functions."""

    def test_pointwise_cost(self):
        assert flops.accounting.pointwise_cost("exp", shape=(3, 4)) == 12
        assert flops.accounting.pointwise_cost("add", shape=(10,)) == 10
        assert flops.accounting.pointwise_cost("sqrt", shape=()) == 1

    def test_reduction_cost(self):
        assert flops.accounting.reduction_cost("sum", input_shape=(3, 4), axis=0) == 12
        assert flops.accounting.reduction_cost("mean", input_shape=(3, 4), axis=None) == 12
        assert flops.accounting.reduction_cost("max", input_shape=(10,)) == 10

    def test_einsum_cost_is_callable(self):
        assert callable(flops.accounting.einsum_cost)

    def test_svd_cost_is_callable(self):
        assert callable(flops.accounting.svd_cost)


# ===================================================================
# array() and einsum() special cases (structure only, no server)
# ===================================================================


class TestArraySpecialCase:
    """Tests for the array() function structure (no server needed)."""

    def test_array_returns_existing_remote_array(self):
        ra = RemoteArray(handle_id="test-id", shape=(2, 3), dtype="float64")
        assert fnp.array(ra) is ra

    def test_array_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Cannot create array"):
            fnp.array("not a valid input")

    def test_flatten_helper(self):
        from flopscope.numpy import _flatten

        flat, shape = _flatten([[1, 2], [3, 4]])
        assert flat == [1, 2, 3, 4]
        assert shape == (2, 2)

    def test_flatten_1d(self):
        from flopscope.numpy import _flatten

        flat, shape = _flatten([1, 2, 3])
        assert flat == [1, 2, 3]
        assert shape == (3,)

    def test_flatten_3d(self):
        from flopscope.numpy import _flatten

        flat, shape = _flatten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert flat == [1, 2, 3, 4, 5, 6, 7, 8]
        assert shape == (2, 2, 2)

    def test_flatten_inhomogeneous_raises(self):
        from flopscope.numpy import _flatten

        with pytest.raises(ValueError, match="Inhomogeneous"):
            _flatten([[1, 2], [3]])


# ===================================================================
# __all__ coverage sanity check
# ===================================================================


class TestModuleCompleteness:
    """Sanity checks for module completeness."""

    def test_no_submodule_ops_in_top_level(self):
        """Submodule ops (random.*, linalg.*, fft.*) should NOT be
        directly in the top-level namespace."""
        # random.randn should not be accessible as fnp.randn
        # (it is accessed as fnp.random.randn)
        for prefix in ("random.", "linalg.", "fft."):
            for key in FUNCTION_CATEGORIES:
                if key.startswith(prefix):
                    short = key.split(".", 1)[1]
                    # If a top-level function or submodule exists with the
                    # same name, skip the check (e.g. "random" is both a
                    # submodule and random.random).
                    if short in FUNCTION_CATEGORIES or short in (
                        "fft",
                        "linalg",
                        "random",
                    ):
                        continue
                    # Should NOT be in top-level namespace
                    assert not hasattr(fnp, short) or short in dir(fnp.__class__), (
                        f"{key} should not be accessible as fnp.{short}"
                    )
