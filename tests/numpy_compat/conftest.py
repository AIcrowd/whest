"""Conftest that monkeypatches numpy with mechestim for compatibility testing.

This lets us run NumPy's own test suite against mechestim to verify
that our interface matches NumPy's. Tests that fail due to known
divergences are listed in xfails.py.
"""

import fnmatch

import numpy as np
import pytest

import mechestim as me
from mechestim._budget import _reset_global_default, budget_reset
from mechestim._registry import REGISTRY

from .xfails import XFAIL_PATTERNS

# Functions we monkeypatch onto numpy
_PATCHED = {}


def _patch_numpy():
    """Replace numpy functions with mechestim equivalents."""
    for name, meta in REGISTRY.items():
        cat = meta["category"]
        if cat in ("blacklisted", "free"):
            # Skip blacklisted (unsupported) and free ops (pure pass-throughs
            # that delegate back to numpy — patching them causes infinite
            # recursion because they call _np.<name> via module attribute).
            continue

        # Resolve the mechestim function
        parts = name.split(".")
        try:
            if len(parts) == 1:
                me_fn = getattr(me, name)
            elif len(parts) == 2:
                submod = getattr(me, parts[0])
                me_fn = getattr(submod, parts[1])
            else:
                continue
        except AttributeError:
            continue

        # Resolve the numpy target and patch
        try:
            if len(parts) == 1:
                if hasattr(np, name):
                    _PATCHED[name] = getattr(np, name)
                    setattr(np, name, me_fn)
            elif len(parts) == 2:
                np_submod = getattr(np, parts[0])
                if hasattr(np_submod, parts[1]):
                    _PATCHED[name] = getattr(np_submod, parts[1])
                    setattr(np_submod, parts[1], me_fn)
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
    """Apply monkeypatch at session start."""
    _patch_numpy()


def pytest_unconfigure(config):
    """Restore numpy at session end."""
    _unpatch_numpy()


def pytest_collection_modifyitems(config, items):
    """Mark known-divergent tests as xfail."""
    for item in items:
        node_id = item.nodeid
        for pattern, reason in XFAIL_PATTERNS.items():
            if fnmatch.fnmatch(node_id, pattern) or pattern in node_id:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break
