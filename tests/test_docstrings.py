"""Tests for numpy docstring inheritance."""
import mechestim as me


def test_counted_unary_has_mechestim_cost():
    doc = me.exp.__doc__
    assert doc is not None
    assert "**mechestim cost:**" in doc


def test_counted_unary_has_numpy_docstring():
    import numpy as np
    doc = me.exp.__doc__
    # The NumPy docstring content should be present directly (not behind a separator)
    assert "Calculate the exponential" in doc


def test_counted_binary_has_mechestim_cost():
    doc = me.add.__doc__
    assert doc is not None
    assert "**mechestim cost:**" in doc


def test_free_op_has_zero_cost():
    doc = me.zeros.__doc__
    assert doc is not None
    assert "**mechestim cost:**" in doc
    assert "0 FLOPs" in doc


def test_reduction_has_mechestim_cost():
    doc = me.sum.__doc__
    assert doc is not None
    assert "**mechestim cost:**" in doc


def test_custom_op_has_mechestim_cost():
    doc = me.dot.__doc__
    assert doc is not None
    assert "**mechestim cost:**" in doc
