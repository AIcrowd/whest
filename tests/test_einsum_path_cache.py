"""Tests for einsum path caching."""

import numpy
import pytest

from whest._config import configure, get_setting
from whest._einsum import _symmetry_fingerprint, _identity_pattern
from whest._symmetric import SymmetricTensor


def test_einsum_path_cache_size_default():
    assert get_setting("einsum_path_cache_size") == 4096


def test_einsum_path_cache_size_configurable():
    original = get_setting("einsum_path_cache_size")
    try:
        configure(einsum_path_cache_size=256)
        assert get_setting("einsum_path_cache_size") == 256
    finally:
        configure(einsum_path_cache_size=original)


def test_symmetry_fingerprint_no_symmetry():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    fp = _symmetry_fingerprint([A, B], ["ij", "jk"])
    assert fp == (None, None)


def test_symmetry_fingerprint_with_symmetry():
    data = numpy.ones((3, 3))
    S = SymmetricTensor(data, symmetric_axes=[(0, 1)])
    fp = _symmetry_fingerprint([S], ["ij"])
    assert fp != (None,)
    assert isinstance(fp, tuple)
    # Fingerprint is hashable
    hash(fp)


def test_symmetry_fingerprint_deterministic():
    data = numpy.ones((3, 3))
    S = SymmetricTensor(data, symmetric_axes=[(0, 1)])
    fp1 = _symmetry_fingerprint([S], ["ij"])
    fp2 = _symmetry_fingerprint([S], ["ij"])
    assert fp1 == fp2


def test_identity_pattern_distinct():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    pat = _identity_pattern([A, B])
    # Distinct objects → no grouping, all singletons → normalized to None
    assert pat is None


def test_identity_pattern_same_object():
    A = numpy.ones((3, 3))
    pat = _identity_pattern([A, A])
    # Same object at positions 0 and 1
    assert pat is not None
    assert isinstance(pat, tuple)
    hash(pat)  # Must be hashable


def test_identity_pattern_mixed():
    A = numpy.ones((3, 3))
    B = numpy.ones((3, 3))
    pat = _identity_pattern([A, B, A])
    # A is at positions 0 and 2 (same object), B at 1 (distinct)
    assert pat is not None
    hash(pat)
