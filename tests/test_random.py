"""Tests for mechestim.random passthrough."""
from mechestim import random as merandom


def test_seed():
    merandom.seed(42)

def test_randn():
    assert merandom.randn(3, 4).shape == (3, 4)

def test_normal():
    assert merandom.normal(0, 1, size=(5,)).shape == (5,)

def test_default_rng():
    rng = merandom.default_rng(42)
    assert rng.standard_normal((3,)).shape == (3,)
