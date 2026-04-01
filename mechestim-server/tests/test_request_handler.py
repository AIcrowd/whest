"""Tests for RequestHandler — written first (TDD)."""

from __future__ import annotations

import numpy as np
import pytest

from mechestim_server._session import Session
from mechestim_server._request_handler import RequestHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def session():
    s = Session(flop_budget=1_000_000)
    yield s
    if s.is_open:
        s.close()


@pytest.fixture()
def handler(session):
    return RequestHandler(session)


# ---------------------------------------------------------------------------
# Free ops: zeros / ones
# ---------------------------------------------------------------------------

def test_handle_zeros(handler, session):
    resp = handler.handle({"op": "zeros", "args": [(3, 4)], "kwargs": {}})
    assert resp["status"] == "ok"
    handle = resp["result"]["id"]
    arr = session.get_array(handle)
    np.testing.assert_array_equal(arr, np.zeros((3, 4)))
    assert resp["result"]["shape"] == [3, 4]
    assert resp["result"]["dtype"] == "float64"


def test_handle_ones(handler, session):
    resp = handler.handle({"op": "ones", "args": [(2, 3)], "kwargs": {}})
    assert resp["status"] == "ok"
    handle = resp["result"]["id"]
    arr = session.get_array(handle)
    np.testing.assert_array_equal(arr, np.ones((2, 3)))


# ---------------------------------------------------------------------------
# Counted unary: exp
# ---------------------------------------------------------------------------

def test_handle_unary_exp(handler, session):
    # Create an input array first
    inp = np.array([0.0, 1.0, 2.0])
    h_in = session.store_array(inp)

    resp = handler.handle({"op": "exp", "args": [h_in], "kwargs": {}})
    assert resp["status"] == "ok"
    h_out = resp["result"]["id"]
    result = session.get_array(h_out)
    np.testing.assert_allclose(result, np.exp(inp))


# ---------------------------------------------------------------------------
# Counted binary: add (two handles)
# ---------------------------------------------------------------------------

def test_handle_binary_add(handler, session):
    a = session.store_array(np.array([1.0, 2.0, 3.0]))
    b = session.store_array(np.array([10.0, 20.0, 30.0]))

    resp = handler.handle({"op": "add", "args": [a, b], "kwargs": {}})
    assert resp["status"] == "ok"
    result = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])


# ---------------------------------------------------------------------------
# Binary with scalar: handle + float
# ---------------------------------------------------------------------------

def test_handle_binary_with_scalar(handler, session):
    a = session.store_array(np.array([1.0, 2.0, 3.0]))

    resp = handler.handle({"op": "add", "args": [a, 10.0], "kwargs": {}})
    assert resp["status"] == "ok"
    result = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(result, [11.0, 12.0, 13.0])


# ---------------------------------------------------------------------------
# Reduction: sum
# ---------------------------------------------------------------------------

def test_handle_reduction_sum(handler, session):
    a = session.store_array(np.array([1.0, 2.0, 3.0]))

    resp = handler.handle({"op": "sum", "args": [a], "kwargs": {}})
    assert resp["status"] == "ok"
    # sum returns a scalar (0-d array)
    result = resp["result"]
    # Could be stored as 0-d array or returned as scalar value
    if "id" in result:
        arr = session.get_array(result["id"])
        assert float(arr) == 6.0
    else:
        assert result["value"] == 6.0


# ---------------------------------------------------------------------------
# Einsum: string subscript + handle args
# ---------------------------------------------------------------------------

def test_handle_einsum(handler, session):
    W = session.store_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    x = session.store_array(np.array([1.0, 1.0]))

    resp = handler.handle({"op": "einsum", "args": ["ij,j->i", W, x], "kwargs": {}})
    assert resp["status"] == "ok"
    result = session.get_array(resp["result"]["id"])
    np.testing.assert_allclose(result, [3.0, 7.0])


# ---------------------------------------------------------------------------
# create_from_data
# ---------------------------------------------------------------------------

def test_handle_create_from_data(handler, session):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    resp = handler.handle({
        "op": "create_from_data",
        "data": arr.tobytes(),
        "shape": [3],
        "dtype": "float64",
    })
    assert resp["status"] == "ok"
    stored = session.get_array(resp["result"]["id"])
    np.testing.assert_array_equal(stored, arr)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def test_handle_fetch(handler, session):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    handle = session.store_array(arr)

    resp = handler.handle({"op": "fetch", "id": handle})
    assert resp["status"] == "ok"
    assert resp["data"] == arr.tobytes()
    assert resp["shape"] == [3]
    assert resp["dtype"] == "float64"


# ---------------------------------------------------------------------------
# fetch_slice
# ---------------------------------------------------------------------------

def test_handle_fetch_slice(handler, session):
    arr = np.arange(10, dtype=np.float64)
    handle = session.store_array(arr)

    resp = handler.handle({"op": "fetch_slice", "id": handle, "slices": [[2, 5]]})
    assert resp["status"] == "ok"
    expected = arr[2:5]
    assert resp["data"] == expected.tobytes()
    assert resp["shape"] == [3]


# ---------------------------------------------------------------------------
# free
# ---------------------------------------------------------------------------

def test_handle_free(handler, session):
    h1 = session.store_array(np.array([1.0]))
    h2 = session.store_array(np.array([2.0]))

    resp = handler.handle({"op": "free", "ids": [h1, h2]})
    assert resp["status"] == "ok"

    with pytest.raises(KeyError):
        session.get_array(h1)
    with pytest.raises(KeyError):
        session.get_array(h2)


# ---------------------------------------------------------------------------
# budget_status
# ---------------------------------------------------------------------------

def test_handle_budget_status(handler):
    resp = handler.handle({"op": "budget_status"})
    assert resp["status"] == "ok"
    result = resp["result"]
    assert "flop_budget" in result
    assert "flops_used" in result
    assert "flops_remaining" in result
    assert result["flop_budget"] == 1_000_000


# ---------------------------------------------------------------------------
# Error: unknown handle
# ---------------------------------------------------------------------------

def test_handle_unknown_handle_returns_error(handler):
    resp = handler.handle({"op": "exp", "args": ["a999"], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "KeyError"


# ---------------------------------------------------------------------------
# Error: budget exhausted
# ---------------------------------------------------------------------------

def test_handle_budget_exhausted_returns_error():
    # Tiny budget that will be exceeded
    s = Session(flop_budget=1)
    h = RequestHandler(s)

    # Store a large array
    big = np.ones((100, 100))
    handle = s.store_array(big)

    resp = h.handle({"op": "exp", "args": [handle], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "BudgetExhaustedError"
    assert "message" in resp

    s.close()


# ---------------------------------------------------------------------------
# Budget info included in operation responses
# ---------------------------------------------------------------------------

def test_budget_info_included_in_operation_responses(handler, session):
    resp = handler.handle({"op": "zeros", "args": [(3,)], "kwargs": {}})
    assert resp["status"] == "ok"
    assert "budget" in resp
    assert "flops_remaining" in resp["budget"]

    # Now do a counted op
    h = resp["result"]["id"]
    resp2 = handler.handle({"op": "exp", "args": [h], "kwargs": {}})
    assert resp2["status"] == "ok"
    assert "budget" in resp2
    # After exp, some flops should have been used
    assert resp2["budget"]["flops_remaining"] < resp["budget"]["flops_remaining"]
