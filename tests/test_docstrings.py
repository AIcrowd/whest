"""Tests for numpy docstring inheritance."""

import whest as we


def test_counted_unary_has_whest_cost():
    doc = we.exp.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_counted_unary_has_numpy_docstring():
    doc = we.exp.__doc__
    # The NumPy docstring content should be present directly (not behind a separator)
    assert "Calculate the exponential" in doc


def test_counted_binary_has_whest_cost():
    doc = we.add.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_free_op_has_zero_cost():
    doc = we.zeros.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc
    assert "0 FLOPs" in doc


def test_reduction_has_whest_cost():
    doc = we.sum.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_custom_op_has_whest_cost():
    doc = we.dot.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc
