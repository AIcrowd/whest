"""Tests for the numpy audit script."""

import sys


def test_audit_introspection_discovers_numpy_functions():
    """The audit introspector should find well-known numpy functions."""
    sys.path.insert(0, "scripts")
    from numpy_audit import introspect_numpy

    discovered = introspect_numpy()

    # Should discover well-known top-level functions
    assert "exp" in discovered
    assert "sin" in discovered
    assert "reshape" in discovered
    assert "zeros" in discovered
    assert "concatenate" in discovered

    # Should discover linalg functions
    assert "linalg.svd" in discovered
    assert "linalg.solve" in discovered
    assert "linalg.norm" in discovered

    # Should discover fft functions
    assert "fft.fft" in discovered
    assert "fft.ifft" in discovered

    # Should discover random functions
    assert "random.rand" in discovered
    assert "random.normal" in discovered

    # Should NOT discover private names
    for name in discovered:
        parts = name.split(".")
        assert not any(p.startswith("_") for p in parts), f"Private name found: {name}"

    # Should NOT discover names from excluded modules
    for name in discovered:
        assert not name.startswith("testing."), f"testing submodule found: {name}"
        assert not name.startswith("lib."), f"lib submodule found: {name}"


def test_audit_introspection_returns_metadata():
    """Each discovered function should have module and callable type info."""
    sys.path.insert(0, "scripts")
    from numpy_audit import introspect_numpy

    discovered = introspect_numpy()

    entry = discovered["exp"]
    assert entry["module"] == "numpy"
    assert entry["kind"] in ("function", "ufunc", "builtin", "class")

    entry_linalg = discovered["linalg.svd"]
    assert entry_linalg["module"] == "numpy.linalg"
