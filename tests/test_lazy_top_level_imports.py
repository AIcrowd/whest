"""Tests for lazy loading of flopscope submodules (JAX-style layout)."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from tests.numpy_compat import conftest as numpy_compat_conftest

# Submodules the top-level ``flopscope`` package lazy-loads on demand.
TOP_LEVEL_LAZY = ("numpy", "accounting", "stats")
# Submodules the ``flopscope.numpy`` package lazy-loads on demand.
NUMPY_LAZY = ("linalg", "fft", "random", "testing", "typing")


def _fresh_flopscope():
    names = (
        "flopscope",
        *(f"flopscope.{s}" for s in TOP_LEVEL_LAZY),
        *(f"flopscope.numpy.{s}" for s in NUMPY_LAZY),
    )
    for name in names:
        sys.modules.pop(name, None)
    return importlib.import_module("flopscope")


def test_top_level_import_does_not_eagerly_load_numpy_submodule():
    we = _fresh_flopscope()

    assert "flopscope.numpy" not in sys.modules
    assert "numpy" in dir(we)
    assert "accounting" in dir(we)
    assert "stats" in dir(we)


def test_top_level_numpy_is_loaded_lazily_on_attribute_access():
    we = _fresh_flopscope()

    assert hasattr(we, "numpy") is True
    assert "flopscope.numpy" in sys.modules
    assert we.numpy.__name__ == "flopscope.numpy"


def test_numpy_submodule_random_is_loaded_lazily():
    _fresh_flopscope()
    fnp = importlib.import_module("flopscope.numpy")
    assert "flopscope.numpy.random" not in sys.modules
    assert hasattr(fnp, "random")
    assert "flopscope.numpy.random" in sys.modules
    assert fnp.random.__name__ == "flopscope.numpy.random"


def test_numpy_submodule_testing_is_loaded_lazily():
    _fresh_flopscope()
    fnp = importlib.import_module("flopscope.numpy")
    sys.modules.pop("flopscope.numpy.testing", None)
    assert hasattr(fnp, "testing")
    assert "flopscope.numpy.testing" in sys.modules
    fnp.testing.assert_allclose(np.array([1.0]), np.array([1.0]))


def test_from_import_random_still_works():
    _fresh_flopscope()

    from flopscope.numpy import random as merandom

    assert "flopscope.numpy.random" in sys.modules
    assert merandom.__name__ == "flopscope.numpy.random"
    assert callable(merandom.randn)


def test_registry_attribute_errors_are_unchanged_for_unknown_names():
    we = _fresh_flopscope()

    with pytest.raises(AttributeError, match="does not provide"):
        _ = we.totally_fake_function_xyz


def test_numpy_compat_rebind_imports_lazy_random_before_patching_numpy():
    _fresh_flopscope()
    fnp = importlib.import_module("flopscope.numpy")

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
