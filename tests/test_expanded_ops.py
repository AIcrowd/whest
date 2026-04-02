"""Tests for expanded operations — verifies they match numpy and charge correct FLOPs."""

import numpy
import pytest

import mechestim as me
from mechestim._budget import BudgetContext


class TestNewUnaryOps:
    @pytest.mark.parametrize(
        "op_name",
        [
            "arcsin",
            "arccos",
            "arctan",
            "sinh",
            "cosh",
            "arcsinh",
            "arccosh",
            "arctanh",
            "exp2",
            "expm1",
            "log1p",
            "rint",
            "trunc",
            "degrees",
            "radians",
            "reciprocal",
            "positive",
            "cbrt",
            "signbit",
            "tan",
            "fabs",
        ],
    )
    def test_unary_matches_numpy(self, op_name):
        x = numpy.array([0.1, 0.5, 0.9])
        me_func = getattr(me, op_name)
        np_func = getattr(numpy, op_name)
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = me_func(x)
            assert numpy.allclose(result, np_func(x), equal_nan=True)

    @pytest.mark.parametrize(
        "op_name",
        [
            "arcsin",
            "sinh",
            "exp2",
            "rint",
            "degrees",
            "reciprocal",
            "tan",
        ],
    )
    def test_unary_charges_numel(self, op_name):
        x = numpy.ones((3, 4))
        me_func = getattr(me, op_name)
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            me_func(x)
            assert budget.flops_used == 12


class TestNewBinaryOps:
    @pytest.mark.parametrize(
        "op_name",
        [
            "fmod",
            "remainder",
            "logaddexp",
            "logaddexp2",
            "float_power",
            "true_divide",
            "floor_divide",
            "arctan2",
            "hypot",
            "copysign",
            "fmax",
            "fmin",
            "greater",
            "less",
            "equal",
            "not_equal",
            "logical_and",
            "logical_or",
        ],
    )
    def test_binary_matches_numpy(self, op_name):
        x = numpy.array([1.0, 2.0, 3.0])
        y = numpy.array([2.0, 1.0, 4.0])
        me_func = getattr(me, op_name)
        np_func = getattr(numpy, op_name)
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = me_func(x, y)
            assert numpy.allclose(result, np_func(x, y), equal_nan=True)


class TestNewReductionOps:
    @pytest.mark.parametrize(
        "op_name",
        [
            "any",
            "all",
            "nansum",
            "nanmean",
            "nanmax",
            "nanmin",
            "median",
            "average",
            "count_nonzero",
        ],
    )
    def test_reduction_matches_numpy(self, op_name):
        x = numpy.array([[1.0, 2.0, 3.0], [4.0, 0.0, 6.0]])
        me_func = getattr(me, op_name)
        np_func = getattr(numpy, op_name)
        with BudgetContext(flop_budget=10**6, quiet=True):
            result = me_func(x)
            expected = np_func(x)
            assert numpy.allclose(result, expected, equal_nan=True)


class TestCustomOps:
    def test_inner(self):
        a, b = numpy.array([1.0, 2.0, 3.0]), numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            assert numpy.allclose(me.inner(a, b), numpy.inner(a, b))
            assert budget.flops_used == 3

    def test_outer(self):
        a, b = numpy.array([1.0, 2.0]), numpy.array([3.0, 4.0, 5.0])
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            assert numpy.allclose(me.outer(a, b), numpy.outer(a, b))
            assert budget.flops_used == 6

    def test_diff(self):
        x = numpy.array([1.0, 3.0, 6.0, 10.0])
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            assert numpy.allclose(me.diff(x), numpy.diff(x))
            assert budget.flops_used == 3

    def test_vdot(self):
        a, b = numpy.array([1.0, 2.0, 3.0]), numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            assert numpy.allclose(me.vdot(a, b), numpy.vdot(a, b))


class TestNewFreeOps:
    def test_rot90(self):
        x = numpy.array([[1, 2], [3, 4]])
        with BudgetContext(flop_budget=1, quiet=True) as budget:
            assert numpy.array_equal(me.rot90(x), numpy.rot90(x))
            assert budget.flops_used == 0

    def test_atleast_1d(self):
        assert me.atleast_1d(1.0).shape == (1,)

    def test_shape(self):
        assert me.shape(numpy.eye(3)) == (3, 3)

    def test_free_ops_outside_context(self):
        me.rot90(numpy.eye(3))
        me.shape(numpy.eye(3))
        me.atleast_1d(1.0)
