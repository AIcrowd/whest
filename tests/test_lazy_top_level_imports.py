"""Tests for top-level lazy loading of flopscope submodules."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from tests.numpy_compat import conftest as numpy_compat_conftest

LAZY_SUBMODULES = ("fft", "flops", "linalg", "random", "stats", "testing")


def _fresh_flopscope():
    for name in (
        "flopscope",
        *(f"flopscope.{submodule}" for submodule in LAZY_SUBMODULES),
    ):
        sys.modules.pop(name, None)
    return importlib.import_module("flopscope")


def test_top_level_import_does_not_eagerly_load_random_or_testing():
    fnp = _fresh_flopscope()

    assert "flopscope.numpy.random" not in sys.modules
    assert "flopscope.numpy.testing" not in sys.modules
    assert "random" in dir(fnp)
    assert "testing" in dir(fnp)


def test_top_level_random_is_loaded_lazily_on_attribute_access():
    fnp = _fresh_flopscope()

    assert hasattr(fnp, "random") is True
    assert "flopscope.numpy.random" in sys.modules
    assert fnp.random.__name__ == "flopscope.numpy.random"


def test_top_level_testing_is_loaded_lazily_on_attribute_access():
    fnp = _fresh_flopscope()

    assert hasattr(fnp, "testing") is True
    assert "flopscope.numpy.testing" in sys.modules
    fnp.testing.assert_allclose(np.array([1.0]), np.array([1.0]))


def test_from_import_random_still_works():
    _fresh_flopscope()

    from flopscope import random as merandom

    assert "flopscope.numpy.random" in sys.modules
    assert merandom.__name__ == "flopscope.numpy.random"
    assert callable(merandom.randn)


def test_registry_attribute_errors_are_unchanged_for_unknown_names():
    fnp = _fresh_flopscope()

    with pytest.raises(AttributeError, match="does not provide"):
        _ = fnp.totally_fake_function_xyz


def test_numpy_compat_rebind_imports_lazy_random_before_patching_numpy():
    fnp = _fresh_flopscope()

    fnp.__dict__.pop("random", None)
    sys.modules.pop("flopscope.numpy.random", None)
    assert "flopscope.numpy.random" not in sys.modules

    frozen = numpy_compat_conftest._freeze_numpy()
    numpy_compat_conftest._rebind_flopscope_np(frozen)
    try:
        assert "flopscope.numpy.random" in sys.modules
        numpy_compat_conftest._patch_numpy()
        result = np.random.choice(a=[False, True], size=8)
        assert result.shape == (8,)
    finally:
        numpy_compat_conftest._unpatch_numpy()
        numpy_compat_conftest._restore_flopscope_np()


def test_numpy_compat_rebinds_plain_np_aliases_in_internal_hot_paths():
    _fresh_flopscope()

    import flopscope._symmetric as symmetric_mod
    import flopscope._symmetry_utils as symmetry_utils_mod

    original_symmetry_utils_np = symmetry_utils_mod.np
    original_symmetric_np = symmetric_mod.np

    frozen = numpy_compat_conftest._freeze_numpy()
    numpy_compat_conftest._rebind_flopscope_np(frozen)
    try:
        assert symmetry_utils_mod.np is frozen
        assert symmetric_mod.np is frozen
    finally:
        numpy_compat_conftest._restore_flopscope_np()
        assert symmetry_utils_mod.np is original_symmetry_utils_np
        assert symmetric_mod.np is original_symmetric_np
