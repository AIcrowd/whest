"""Tests for top-level lazy loading of whest submodules."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from tests.numpy_compat import conftest as numpy_compat_conftest

LAZY_SUBMODULES = ("fft", "flops", "linalg", "random", "stats", "testing")


def _fresh_whest():
    for name in ("whest", *(f"whest.{submodule}" for submodule in LAZY_SUBMODULES)):
        sys.modules.pop(name, None)
    return importlib.import_module("whest")


def test_top_level_import_does_not_eagerly_load_random_or_testing():
    we = _fresh_whest()

    assert "whest.random" not in sys.modules
    assert "whest.testing" not in sys.modules
    assert "random" in dir(we)
    assert "testing" in dir(we)


def test_top_level_random_is_loaded_lazily_on_attribute_access():
    we = _fresh_whest()

    assert hasattr(we, "random") is True
    assert "whest.random" in sys.modules
    assert we.random.__name__ == "whest.random"


def test_top_level_testing_is_loaded_lazily_on_attribute_access():
    we = _fresh_whest()

    assert hasattr(we, "testing") is True
    assert "whest.testing" in sys.modules
    we.testing.assert_allclose(np.array([1.0]), np.array([1.0]))


def test_from_import_random_still_works():
    _fresh_whest()

    from whest import random as merandom

    assert "whest.random" in sys.modules
    assert merandom.__name__ == "whest.random"
    assert callable(merandom.randn)


def test_registry_attribute_errors_are_unchanged_for_unknown_names():
    we = _fresh_whest()

    with pytest.raises(AttributeError, match="does not provide"):
        _ = we.totally_fake_function_xyz


def test_numpy_compat_rebind_imports_lazy_random_before_patching_numpy():
    we = _fresh_whest()

    we.__dict__.pop("random", None)
    sys.modules.pop("whest.random", None)
    assert "whest.random" not in sys.modules

    frozen = numpy_compat_conftest._freeze_numpy()
    numpy_compat_conftest._rebind_whest_np(frozen)
    try:
        assert "whest.random" in sys.modules
        numpy_compat_conftest._patch_numpy()
        result = np.random.choice(a=[False, True], size=8)
        assert result.shape == (8,)
    finally:
        numpy_compat_conftest._unpatch_numpy()
        numpy_compat_conftest._restore_whest_np()
