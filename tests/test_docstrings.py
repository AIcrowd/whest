"""Tests for numpy docstring inheritance."""

import flopscope.numpy as fnp


def test_counted_unary_has_flopscope_cost():
    doc = fnp.exp.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_counted_unary_has_numpy_docstring():
    doc = fnp.exp.__doc__
    # The NumPy docstring content should be present directly (not behind a separator)
    assert "Calculate the exponential" in doc


def test_counted_binary_has_flopscope_cost():
    doc = fnp.add.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_free_op_has_zero_cost():
    doc = fnp.zeros.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc
    assert "0 FLOPs" in doc


def test_reduction_has_flopscope_cost():
    doc = fnp.sum.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc


def test_custom_op_has_flopscope_cost():
    doc = fnp.dot.__doc__
    assert doc is not None
    assert "FLOP Cost" in doc
