"""Tests for _math_compat — pure-Python replacements for math.prod/pi/e/inf/nan."""

import math

from mechestim._math_compat import prod, pi, e, inf, nan


def test_prod_basic():
    assert prod((2, 3, 4)) == 24


def test_prod_empty():
    assert prod(()) == 1


def test_prod_single():
    assert prod((7,)) == 7


def test_prod_with_zero():
    assert prod((5, 0, 3)) == 0


def test_constants():
    assert pi == math.pi
    assert e == math.e
    assert inf == math.inf
    assert math.isnan(nan)
