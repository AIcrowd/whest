"""Tests for the mechestim client API surface.

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
from mechestim._remote_array import RemoteArray

import mechestim as me
from mechestim._registry import (
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

    def test_all_fft_are_blacklisted(self):
        fft_ops = [k for k in FUNCTION_CATEGORIES if k.startswith("fft.")]
        assert len(fft_ops) > 0
        for op in fft_ops:
            assert get_category(op) == BLACKLISTED

    def test_linalg_svd_is_counted_custom(self):
        assert get_category("linalg.svd") == COUNTED_CUSTOM


# ===================================================================
# Constants and dtype exports
# ===================================================================


class TestConstants:
    """Tests for constants exported by the top-level package."""

    def test_pi(self):
        assert me.pi == pytest.approx(math.pi)

    def test_e(self):
        assert me.e == pytest.approx(math.e)

    def test_inf(self):
        assert me.inf == math.inf

    def test_nan(self):
        assert math.isnan(me.nan)

    def test_newaxis(self):
        assert me.newaxis is None


class TestDtypes:
    """Tests for dtype string exports."""

    def test_float64(self):
        assert me.float64 == "float64"

    def test_float32(self):
        assert me.float32 == "float32"

    def test_float16(self):
        assert me.float16 == "float16"

    def test_int64(self):
        assert me.int64 == "int64"

    def test_int32(self):
        assert me.int32 == "int32"

    def test_int16(self):
        assert me.int16 == "int16"

    def test_int8(self):
        assert me.int8 == "int8"

    def test_uint8(self):
        assert me.uint8 == "uint8"

    def test_bool_(self):
        assert me.bool_ == "bool"

    def test_complex64(self):
        assert me.complex64 == "complex64"

    def test_complex128(self):
        assert me.complex128 == "complex128"


# ===================================================================
# Class exports
# ===================================================================


class TestClassExports:
    """Tests for class/type exports."""

    def test_ndarray_is_RemoteArray(self):
        assert me.ndarray is RemoteArray

    def test_BudgetContext_is_class(self):
        from mechestim._budget import BudgetContext

        assert me.BudgetContext is BudgetContext

    def test_OpRecord_is_class(self):
        from mechestim._budget import OpRecord

        assert me.OpRecord is OpRecord

    def test_errors_exported(self):
        assert issubclass(me.BudgetExhaustedError, me.MechEstimError)
        assert issubclass(me.NoBudgetContextError, me.MechEstimError)
        assert issubclass(me.SymmetryError, me.MechEstimError)
        assert issubclass(me.MechEstimServerError, me.MechEstimError)
        assert issubclass(me.MechEstimWarning, UserWarning)


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
        fn = getattr(me, name)
        assert callable(fn), f"me.{name} should be callable"

    def test_array_is_callable(self):
        assert callable(me.array)

    def test_einsum_is_callable(self):
        assert callable(me.einsum)

    def test_proxy_has_correct_name(self):
        assert me.add.__name__ == "add"
        assert me.exp.__name__ == "exp"
        assert me.zeros.__name__ == "zeros"


# ===================================================================
# __getattr__ error messages
# ===================================================================


class TestGetattr:
    """Tests for __getattr__ error handling on blacklisted/unknown names."""

    def test_blacklisted_top_level(self):
        with pytest.raises(AttributeError, match="intentionally not supported"):
            me.save

    def test_blacklisted_top_level_load(self):
        with pytest.raises(AttributeError, match="intentionally not supported"):
            me.load

    def test_unknown_top_level(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            me.completely_nonexistent_function_xyz

    def test_blacklisted_fft(self):
        with pytest.raises(AttributeError, match="intentionally not supported"):
            me.fft.fft

    def test_blacklisted_fft_rfft(self):
        with pytest.raises(AttributeError, match="intentionally not supported"):
            me.fft.rfft

    def test_unknown_fft(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            me.fft.nonexistent_xyz

    def test_blacklisted_linalg(self):
        with pytest.raises(AttributeError, match="intentionally not supported"):
            me.linalg.eig

    def test_unknown_linalg(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            me.linalg.nonexistent_xyz

    def test_unknown_random(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            me.random.nonexistent_xyz


# ===================================================================
# Submodule structure
# ===================================================================


class TestSubmodules:
    """Tests for submodule accessibility."""

    def test_fft_is_module(self):
        import types

        assert isinstance(me.fft, types.ModuleType)

    def test_linalg_is_module(self):
        import types

        assert isinstance(me.linalg, types.ModuleType)

    def test_random_is_module(self):
        import types

        assert isinstance(me.random, types.ModuleType)

    def test_flops_is_module(self):
        import types

        assert isinstance(me.flops, types.ModuleType)

    def test_linalg_svd_callable(self):
        assert callable(me.linalg.svd)

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
            fn = getattr(me.random, name)
            assert callable(fn), f"me.random.{name} should be callable"


# ===================================================================
# flops local cost functions
# ===================================================================


class TestFlopsLocal:
    """Tests for locally computable flops cost functions."""

    def test_pointwise_cost(self):
        assert me.flops.pointwise_cost((3, 4)) == 12
        assert me.flops.pointwise_cost((10,)) == 10
        assert me.flops.pointwise_cost(()) == 1

    def test_reduction_cost(self):
        assert me.flops.reduction_cost((3, 4), axis=0) == 12
        assert me.flops.reduction_cost((3, 4), axis=None) == 12
        assert me.flops.reduction_cost((10,)) == 10

    def test_einsum_cost_is_callable(self):
        assert callable(me.flops.einsum_cost)

    def test_svd_cost_is_callable(self):
        assert callable(me.flops.svd_cost)


# ===================================================================
# array() and einsum() special cases (structure only, no server)
# ===================================================================


class TestArraySpecialCase:
    """Tests for the array() function structure (no server needed)."""

    def test_array_returns_existing_remote_array(self):
        ra = RemoteArray(handle_id="test-id", shape=(2, 3), dtype="float64")
        assert me.array(ra) is ra

    def test_array_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Cannot create array"):
            me.array("not a valid input")

    def test_flatten_helper(self):
        from mechestim import _flatten

        flat, shape = _flatten([[1, 2], [3, 4]])
        assert flat == [1, 2, 3, 4]
        assert shape == (2, 2)

    def test_flatten_1d(self):
        from mechestim import _flatten

        flat, shape = _flatten([1, 2, 3])
        assert flat == [1, 2, 3]
        assert shape == (3,)

    def test_flatten_3d(self):
        from mechestim import _flatten

        flat, shape = _flatten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert flat == [1, 2, 3, 4, 5, 6, 7, 8]
        assert shape == (2, 2, 2)

    def test_flatten_inhomogeneous_raises(self):
        from mechestim import _flatten

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
        # random.randn should not be accessible as me.randn
        # (it is accessed as me.random.randn)
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
                    assert not hasattr(me, short) or short in dir(me.__class__), (
                        f"{key} should not be accessible as me.{short}"
                    )
