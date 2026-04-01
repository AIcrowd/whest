"""Tests for numpy docstring inheritance."""
import mechestim as me


def test_counted_unary_has_mechestim_header():
    doc = me.exp.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "counted_unary" in doc


def test_counted_unary_has_numpy_docstring():
    import numpy as np
    doc = me.exp.__doc__
    assert "numpy docstring" in doc.lower()


def test_counted_binary_has_mechestim_header():
    doc = me.add.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "counted_binary" in doc


def test_free_op_has_zero_cost_header():
    doc = me.zeros.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "0 FLOPs" in doc


def test_reduction_has_mechestim_header():
    doc = me.sum.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "counted_reduction" in doc


def test_custom_op_has_mechestim_header():
    doc = me.dot.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "counted_custom" in doc
