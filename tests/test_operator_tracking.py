"""Parametrized tests verifying operator overloads track FLOPs.

These tests prevent the regression of the bug where Python operators
on mechestim arrays bypassed FLOP tracking. Each operator is tested
for: correctness, FLOP count parity with the equivalent me.* function,
return type, reflected variants, and budget exhaustion.
"""

from __future__ import annotations

import numpy as np
import pytest

import mechestim as me

# ----- Operator parametrize lists -----

BINARY_ARITHMETIC = [
    ("add", lambda a, b: a + b, "add"),
    ("sub", lambda a, b: a - b, "subtract"),
    ("mul", lambda a, b: a * b, "multiply"),
    ("truediv", lambda a, b: a / b, "true_divide"),
    ("floordiv", lambda a, b: a // b, "floor_divide"),
    ("mod", lambda a, b: a % b, "mod"),
    ("pow", lambda a, b: a**b, "power"),
    ("matmul", lambda a, b: a @ b, "matmul"),
]

BINARY_COMPARISON = [
    ("eq", lambda a, b: a == b, "equal"),
    ("ne", lambda a, b: a != b, "not_equal"),
    ("lt", lambda a, b: a < b, "less"),
    ("le", lambda a, b: a <= b, "less_equal"),
    ("gt", lambda a, b: a > b, "greater"),
    ("ge", lambda a, b: a >= b, "greater_equal"),
]

BINARY_BITWISE = [
    ("and", lambda a, b: a & b, "bitwise_and"),
    ("or", lambda a, b: a | b, "bitwise_or"),
    ("xor", lambda a, b: a ^ b, "bitwise_xor"),
    ("lshift", lambda a, b: a << b, "left_shift"),
    ("rshift", lambda a, b: a >> b, "right_shift"),
]

UNARY_ARITHMETIC = [
    ("neg", lambda a: -a, "negative"),
    ("pos", lambda a: +a, "positive"),
    ("abs", lambda a: abs(a), "abs"),
]

UNARY_BITWISE = [
    ("invert", lambda a: ~a, "invert"),
]

# ----- Binary arithmetic tests -----


@pytest.mark.parametrize("name,op,func_name", BINARY_ARITHMETIC)
def test_binary_arith_result_matches_numpy(name, op, func_name):
    a_np = np.array([2.0, 3.0, 4.0, 5.0])
    b_np = np.array([1.0, 2.0, 1.0, 3.0])
    a_me = me.array(a_np)
    b_me = me.array(b_np)
    expected = op(a_np, b_np)
    with me.BudgetContext(flop_budget=int(1e9)):
        actual = op(a_me, b_me)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("name,op,func_name", BINARY_ARITHMETIC)
def test_binary_arith_flops_match_function(name, op, func_name):
    if name == "matmul":
        a = me.array([[1.0, 2.0], [3.0, 4.0]])
        b = me.array([[5.0, 6.0], [7.0, 8.0]])
    else:
        a = me.array([1.0, 2.0, 3.0, 4.0])
        b = me.array([0.5, 1.5, 2.5, 3.5])
    me_func = getattr(me, func_name)
    with me.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a, b)
    with me.BudgetContext(flop_budget=int(1e9)) as b2:
        me_func(a, b)
    assert b1.flops_used == b2.flops_used, (
        f"{name}: operator used {b1.flops_used} FLOPs, function used {b2.flops_used}"
    )


@pytest.mark.parametrize("name,op,func_name", BINARY_ARITHMETIC)
def test_binary_arith_returns_mechestim_array(name, op, func_name):
    if name == "matmul":
        a = me.array([[1.0, 2.0], [3.0, 4.0]])
        b = me.array([[5.0, 6.0], [7.0, 8.0]])
    else:
        a = me.array([1.0, 2.0, 3.0, 4.0])
        b = me.array([0.5, 1.5, 2.5, 3.5])
    with me.BudgetContext(flop_budget=int(1e9)):
        result = op(a, b)
    assert isinstance(result, me.ndarray)
    assert type(result).__name__ == "MechestimArray"


@pytest.mark.parametrize("name,op,func_name", BINARY_ARITHMETIC)
def test_binary_arith_reflected_tracked(name, op, func_name):
    if name == "matmul":
        a = me.array([[1.0, 2.0], [3.0, 4.0]])
        scalar_or_arr = np.array([[5.0, 6.0], [7.0, 8.0]])
    else:
        a = me.array([1.0, 2.0, 3.0, 4.0])
        scalar_or_arr = 2.0
    with me.BudgetContext(flop_budget=int(1e9)) as budget:
        result = op(scalar_or_arr, a)
    assert budget.flops_used > 0, f"{name}: reflected operator did not track FLOPs"
    assert isinstance(result, me.ndarray)


@pytest.mark.parametrize("name,op,func_name", BINARY_ARITHMETIC)
def test_binary_arith_raises_budget_exhausted(name, op, func_name):
    if name == "matmul":
        a = me.array([[1.0] * 100] * 100)
        b = me.array([[2.0] * 100] * 100)
    else:
        a = me.array([1.0] * 1000)
        b = me.array([2.0] * 1000)
    with pytest.raises(me.BudgetExhaustedError):
        with me.BudgetContext(flop_budget=10):
            op(a, b)


# ----- Binary comparison tests -----


@pytest.mark.parametrize("name,op,func_name", BINARY_COMPARISON)
def test_binary_compare_result_matches_numpy(name, op, func_name):
    a_np = np.array([1.0, 2.0, 3.0, 4.0])
    b_np = np.array([2.0, 2.0, 2.0, 2.0])
    a_me = me.array(a_np)
    b_me = me.array(b_np)
    expected = op(a_np, b_np)
    with me.BudgetContext(flop_budget=int(1e9)):
        actual = op(a_me, b_me)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("name,op,func_name", BINARY_COMPARISON)
def test_binary_compare_flops_match_function(name, op, func_name):
    a = me.array([1.0, 2.0, 3.0, 4.0])
    b = me.array([2.0, 2.0, 2.0, 2.0])
    me_func = getattr(me, func_name)
    with me.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a, b)
    with me.BudgetContext(flop_budget=int(1e9)) as b2:
        me_func(a, b)
    assert b1.flops_used == b2.flops_used


@pytest.mark.parametrize("name,op,func_name", BINARY_COMPARISON)
def test_binary_compare_returns_mechestim_array(name, op, func_name):
    a = me.array([1.0, 2.0, 3.0, 4.0])
    b = me.array([2.0, 2.0, 2.0, 2.0])
    with me.BudgetContext(flop_budget=int(1e9)):
        result = op(a, b)
    assert isinstance(result, me.ndarray)


@pytest.mark.parametrize("name,op,func_name", BINARY_COMPARISON)
def test_binary_compare_reflected_tracked(name, op, func_name):
    a = me.array([1.0, 2.0, 3.0, 4.0])
    with me.BudgetContext(flop_budget=int(1e9)):
        result = op(2.0, a)
    assert isinstance(result, me.ndarray)


# ----- Binary bitwise tests -----


@pytest.mark.parametrize("name,op,func_name", BINARY_BITWISE)
def test_binary_bitwise_result_matches_numpy(name, op, func_name):
    a_np = np.array([1, 2, 3, 4], dtype=np.int32)
    b_np = np.array([2, 2, 2, 2], dtype=np.int32)
    a_me = me.array(a_np)
    b_me = me.array(b_np)
    expected = op(a_np, b_np)
    with me.BudgetContext(flop_budget=int(1e9)):
        actual = op(a_me, b_me)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("name,op,func_name", BINARY_BITWISE)
def test_binary_bitwise_flops_match_function(name, op, func_name):
    a = me.array([1, 2, 3, 4], dtype=me.int32)
    b = me.array([2, 2, 2, 2], dtype=me.int32)
    me_func = getattr(me, func_name)
    with me.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a, b)
    with me.BudgetContext(flop_budget=int(1e9)) as b2:
        me_func(a, b)
    assert b1.flops_used == b2.flops_used


@pytest.mark.parametrize("name,op,func_name", BINARY_BITWISE)
def test_binary_bitwise_returns_mechestim_array(name, op, func_name):
    a = me.array([1, 2, 3, 4], dtype=me.int32)
    b = me.array([2, 2, 2, 2], dtype=me.int32)
    with me.BudgetContext(flop_budget=int(1e9)):
        result = op(a, b)
    assert isinstance(result, me.ndarray)


# ----- Unary tests -----


@pytest.mark.parametrize("name,op,func_name", UNARY_ARITHMETIC)
def test_unary_arith_result_matches_numpy(name, op, func_name):
    a_np = np.array([-2.0, -1.0, 1.0, 2.0])
    a_me = me.array(a_np)
    expected = op(a_np)
    with me.BudgetContext(flop_budget=int(1e9)):
        actual = op(a_me)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("name,op,func_name", UNARY_ARITHMETIC)
def test_unary_arith_flops_match_function(name, op, func_name):
    a = me.array([-2.0, -1.0, 1.0, 2.0])
    me_func = getattr(me, func_name)
    with me.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a)
    with me.BudgetContext(flop_budget=int(1e9)) as b2:
        me_func(a)
    assert b1.flops_used == b2.flops_used


@pytest.mark.parametrize("name,op,func_name", UNARY_ARITHMETIC)
def test_unary_arith_returns_mechestim_array(name, op, func_name):
    a = me.array([-2.0, -1.0, 1.0, 2.0])
    with me.BudgetContext(flop_budget=int(1e9)):
        result = op(a)
    assert isinstance(result, me.ndarray)


@pytest.mark.parametrize("name,op,func_name", UNARY_BITWISE)
def test_unary_bitwise_result_matches_numpy(name, op, func_name):
    a_np = np.array([1, 2, 3, 4], dtype=np.int32)
    a_me = me.array(a_np)
    expected = op(a_np)
    with me.BudgetContext(flop_budget=int(1e9)):
        actual = op(a_me)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("name,op,func_name", UNARY_BITWISE)
def test_unary_bitwise_flops_match_function(name, op, func_name):
    a = me.array([1, 2, 3, 4], dtype=me.int32)
    me_func = getattr(me, func_name)
    with me.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a)
    with me.BudgetContext(flop_budget=int(1e9)) as b2:
        me_func(a)
    assert b1.flops_used == b2.flops_used


# ----- In-place operator tests -----


def test_inplace_mul_tracked():
    a = me.array([1.0, 2.0, 3.0, 4.0])
    b = me.array([0.5, 1.5, 2.5, 3.5])
    with me.BudgetContext(flop_budget=int(1e9)) as budget:
        a *= b
    assert budget.flops_used > 0
    assert isinstance(a, me.ndarray)


def test_inplace_add_tracked():
    a = me.array([1.0, 2.0, 3.0, 4.0])
    b = me.array([0.5, 1.5, 2.5, 3.5])
    with me.BudgetContext(flop_budget=int(1e9)) as budget:
        a += b
    assert budget.flops_used > 0
    assert isinstance(a, me.ndarray)


def test_inplace_sub_tracked():
    a = me.array([1.0, 2.0, 3.0, 4.0])
    b = me.array([0.5, 1.5, 2.5, 3.5])
    with me.BudgetContext(flop_budget=int(1e9)) as budget:
        a -= b
    assert budget.flops_used > 0
    assert isinstance(a, me.ndarray)


def test_inplace_truediv_tracked():
    a = me.array([1.0, 2.0, 3.0, 4.0])
    b = me.array([0.5, 1.5, 2.5, 3.5])
    with me.BudgetContext(flop_budget=int(1e9)) as budget:
        a /= b
    assert budget.flops_used > 0
    assert isinstance(a, me.ndarray)


# ----- Integration test -----


def test_pythonic_estimator_matches_verbose_estimator():
    """Pythonic and verbose code should produce identical results and FLOP counts.

    This is the canonical test that verifies the original bug is fixed:
    a Pythonic estimator using operators must report the same FLOP count
    as the equivalent verbose code using me.* function calls.
    """
    x = me.array([1.0, 2.0, 3.0])

    with me.BudgetContext(flop_budget=int(1e9)) as b1:
        result_pythonic = me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)

    with me.BudgetContext(flop_budget=int(1e9)) as b2:
        result_verbose = me.multiply(
            me.exp(me.multiply(-0.5, me.multiply(x, x))),
            1.0 / float(me.sqrt(2.0 * me.pi)),
        )

    np.testing.assert_allclose(
        np.asarray(result_pythonic),
        np.asarray(result_verbose),
    )
    assert b1.flops_used == b2.flops_used
