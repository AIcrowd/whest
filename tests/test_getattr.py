"""Tests for registry-driven __getattr__ across modules."""

import pytest

import whest as we


def test_blacklisted_top_level_gives_notes():
    from whest._registry import REGISTRY

    blacklisted = [
        n
        for n, e in REGISTRY.items()
        if e["category"] == "blacklisted"
        and e["module"] == "numpy"
        and n not in dir(we)
    ]
    if not blacklisted:
        pytest.skip("No blacklisted top-level functions that aren't implemented")
    name = blacklisted[0]
    with pytest.raises(AttributeError, match="blacklisted"):
        getattr(we, name)


def test_registered_not_implemented_gives_message():
    from whest._registry import REGISTRY

    # Find a counted function that isn't yet implemented
    not_impl = [
        n
        for n, e in REGISTRY.items()
        if e["category"] in ("counted_unary", "counted_binary", "counted_reduction")
        and e["module"] == "numpy"
        and n not in dir(we)
    ]
    if not not_impl:
        pytest.skip("All counted functions are implemented")
    name = not_impl[0]
    with pytest.raises(AttributeError, match="registered but not yet implemented"):
        getattr(we, name)


def test_unknown_name_still_errors():
    with pytest.raises(AttributeError, match="does not provide"):
        _ = we.totally_fake_function_xyz


def test_fft_submodule_exists():
    assert hasattr(we, "fft")


def test_fft_getattr_gives_blacklist_error():
    from whest._registry import REGISTRY

    blacklisted_fft = [
        n
        for n, e in REGISTRY.items()
        if e["module"] == "numpy.fft" and e["category"] == "blacklisted"
    ]
    if not blacklisted_fft:
        pytest.skip("No blacklisted fft functions")
    func_name = blacklisted_fft[0].replace("fft.", "")
    with pytest.raises(AttributeError, match="blacklisted"):
        getattr(we.fft, func_name)


def test_linalg_getattr_consults_registry():
    """Verify that accessing a non-existent linalg attr raises AttributeError."""
    with pytest.raises(AttributeError):
        _ = we.linalg.nonexistent_function_xyz
