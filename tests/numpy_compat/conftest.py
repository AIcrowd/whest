"""Conftest that monkeypatches numpy with whest for compatibility testing.

This lets us run NumPy's own test suite against whest to verify
that our interface matches NumPy's. Tests that fail due to known
divergences are listed in xfails.py.

Key trick: before patching numpy, we freeze a copy of the original
numpy module and rebind whest's internal `_np` references to it.
This breaks the infinite recursion that would otherwise occur when
whest functions call _np.func() → numpy.func() → we.func() → ...
"""

import fnmatch
import sys
import types

import numpy as np
import pytest

import whest as we
from whest._budget import _reset_global_default, budget_reset
from whest._registry import REGISTRY

from .xfails import XFAIL_PATTERNS


class _NonDescriptor:
    """Callable wrapper that prevents Python descriptor auto-binding.

    Python functions implement ``__get__`` so when stored as a class
    attribute and accessed via ``self.func()``, Python auto-binds
    ``self`` as the first positional argument. C built-in functions
    don't do this.  Wrapping in ``_NonDescriptor`` makes our Python
    replacements behave like C built-ins for attribute access.
    """

    def __init__(self, fn):
        self._fn = fn
        # Copy key attributes so introspection (numpy.ma docstring
        # parsing, inspect.signature, etc.) sees the original metadata.
        self.__name__ = getattr(fn, "__name__", "")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = getattr(fn, "__module__", None)
        self.__signature__ = getattr(fn, "__signature__", None)
        self.__wrapped__ = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._fn, name)

# Functions we monkeypatch onto numpy
_PATCHED: dict[str, object] = {}

# whest modules whose _np we rebind, with their originals
_REBOUND: dict[str, object] = {}

# All whest submodules that import numpy as _np
_WHEST_MODULES_WITH_NP = [
    "whest._pointwise",
    "whest._free_ops",
    "whest._sorting_ops",
    "whest._counting_ops",
    "whest._einsum",
    "whest._polynomial",
    "whest._unwrap",
    "whest._window",
    "whest._version_check",
    "whest.__init__",
    "whest.fft._transforms",
    "whest.fft._free",
    "whest.linalg._decompositions",
    "whest.linalg._properties",
    "whest.linalg._solvers",
    "whest.linalg._compound",
    "whest.linalg._svd",
    "whest.linalg._aliases",
]

# Modules that also import numpy.random as _npr
_WHEST_MODULES_WITH_NPR = [
    "whest.random",
]


def _freeze_numpy():
    """Create a frozen copy of numpy that won't be affected by patching.

    Returns a module whose attributes are snapshots of numpy's current
    functions. Submodules (linalg, fft, random) are also frozen.
    """
    frozen = types.ModuleType("_frozen_numpy")
    frozen.__dict__.update(np.__dict__)

    # Freeze submodules so _np.linalg.solve etc. still work
    for submod_name in ("linalg", "fft", "random"):
        original_submod = getattr(np, submod_name)
        frozen_submod = types.ModuleType(f"_frozen_numpy.{submod_name}")
        frozen_submod.__dict__.update(original_submod.__dict__)
        setattr(frozen, submod_name, frozen_submod)

    return frozen


def _rebind_whest_np(frozen_np):
    """Replace _np in all whest modules with the frozen copy."""
    for mod_name in _WHEST_MODULES_WITH_NP:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_np"):
            _REBOUND[mod_name] = mod._np
            mod._np = frozen_np
    # Also rebind _npr (numpy.random) in modules that use it
    for mod_name in _WHEST_MODULES_WITH_NPR:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_npr"):
            _REBOUND[mod_name + "._npr"] = mod._npr
            mod._npr = frozen_np.random


def _restore_whest_np():
    """Restore original _np references in whest modules."""
    for key, original in _REBOUND.items():
        if key.endswith("._npr"):
            mod_name = key[: -len("._npr")]
            mod = sys.modules.get(mod_name)
            if mod is not None:
                mod._npr = original
        else:
            mod = sys.modules.get(key)
            if mod is not None:
                mod._np = original
    _REBOUND.clear()


def _patch_numpy():
    """Replace numpy functions with whest equivalents.

    Patches all non-blacklisted functions from the registry, including
    ufuncs, custom ops, submodule functions, and free ops. The frozen
    numpy copy prevents infinite recursion.
    """
    for name, meta in REGISTRY.items():
        cat = meta["category"]
        if cat == "blacklisted":
            continue

        # Resolve the whest function
        parts = name.split(".")
        try:
            if len(parts) == 1:
                we_fn = getattr(we, name)
            elif len(parts) == 2:
                submod = getattr(we, parts[0])
                we_fn = getattr(submod, parts[1])
            else:
                continue
        except AttributeError:
            continue

        # Skip ufuncs — our replacements are plain functions and lack
        # .reduce/.accumulate/.outer/.nargs/etc. which tests check at
        # collection time.
        try:
            if len(parts) == 1:
                np_obj = getattr(np, name, None)
            elif len(parts) == 2:
                np_obj = getattr(getattr(np, parts[0], None), parts[1], None)
            else:
                np_obj = None
            if isinstance(np_obj, np.ufunc):
                continue
        except (AttributeError, TypeError):
            pass

        # Skip functions where whest delegates to a different numpy function
        # than the one being patched (e.g., we.linalg.outer → np.outer, not
        # np.linalg.outer). Patching causes collection-time errors in tests that
        # check the real np.linalg.outer's behaviour at class-definition time.
        _SKIP_PATCH = {
            "linalg.outer",
            # numpy.ma parses np.arange.__doc__ at import time to replace
            # "ndarray" with "MaskedArray". Our wrapper docstring doesn't
            # have the expected format, causing RuntimeError.
            "arange",
        }
        if name in _SKIP_PATCH:
            continue

        # Wrap Python functions to prevent descriptor auto-binding.
        # When a Python function is stored as a class attribute and
        # accessed via self.func(), Python's descriptor protocol
        # auto-binds self as the first arg. C built-in functions
        # (like the originals in numpy) don't do this. We wrap our
        # replacements in a non-descriptor callable to match behavior.
        patched_fn = _NonDescriptor(we_fn)

        # Patch numpy
        try:
            if len(parts) == 1:
                if hasattr(np, name):
                    _PATCHED[name] = getattr(np, name)
                    setattr(np, name, patched_fn)
            elif len(parts) == 2:
                np_submod = getattr(np, parts[0])
                if hasattr(np_submod, parts[1]):
                    _PATCHED[name] = getattr(np_submod, parts[1])
                    setattr(np_submod, parts[1], patched_fn)
        except (AttributeError, TypeError):
            continue


def _unpatch_numpy():
    """Restore original numpy functions."""
    for name, original in _PATCHED.items():
        parts = name.split(".")
        if len(parts) == 1:
            setattr(np, name, original)
        elif len(parts) == 2:
            setattr(getattr(np, parts[0]), parts[1], original)
    _PATCHED.clear()


@pytest.fixture(autouse=True)
def reset_budget():
    """Reset global budget between tests to avoid cross-test leakage."""
    _reset_global_default()
    budget_reset()
    yield
    _reset_global_default()
    budget_reset()


def pytest_configure(config):
    """Freeze numpy, rebind whest internals, then patch."""
    frozen = _freeze_numpy()
    _rebind_whest_np(frozen)
    _patch_numpy()


def pytest_unconfigure(config):
    """Restore everything."""
    _unpatch_numpy()
    _restore_whest_np()


def pytest_collection_modifyitems(config, items):
    """Mark known-divergent tests as xfail."""
    for item in items:
        node_id = item.nodeid
        for pattern, reason in XFAIL_PATTERNS.items():
            if fnmatch.fnmatch(node_id, pattern) or pattern in node_id:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break
