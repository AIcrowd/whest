"""Integration tests for MechestimServer — real ZMQ over TCP."""

from __future__ import annotations

import threading
import time

import msgpack
import numpy as np
import pytest
import zmq

from mechestim_server._server import MechestimServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERVER_URL = "tcp://127.0.0.1:15555"


def _send(sock: zmq.Socket, msg: dict) -> dict:
    """Send a msgpack request and return the decoded response."""
    sock.send(msgpack.packb(msg, use_bin_type=True))
    raw = sock.recv()
    return msgpack.unpackb(raw, raw=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def server_and_client():
    """Start a MechestimServer in a daemon thread and yield a REQ client socket."""
    server = MechestimServer(url=SERVER_URL, session_timeout_s=60.0)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.2)  # let the server bind

    ctx = zmq.Context()
    client = ctx.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 5000)
    client.connect(SERVER_URL)

    yield server, client

    server.stop()
    client.close(linger=0)
    ctx.term()
    t.join(timeout=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_budget_open_and_status(server_and_client):
    """budget_open creates a session; budget_status returns remaining FLOPs."""
    _server, client = server_and_client

    resp = _send(client, {"op": "budget_open", "flop_budget": 500_000})
    assert resp["status"] == "ok"
    assert resp["result"]["session"] == "opened"
    assert resp["budget"] == 500_000

    resp = _send(client, {"op": "budget_status"})
    assert resp["status"] == "ok"
    assert resp["result"]["flops_remaining"] == 500_000


def test_create_array_and_fetch(server_and_client):
    """Create an array via create_from_data and fetch it back."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    arr = np.array([1.0, 2.0, 3.0], dtype="float64")
    resp = _send(
        client,
        {
            "op": "create_from_data",
            "data": arr.tobytes(),
            "dtype": "float64",
            "shape": [3],
        },
    )
    assert resp["status"] == "ok"
    handle = resp["result"]["id"]

    resp = _send(client, {"op": "fetch", "id": handle})
    assert resp["status"] == "ok"
    fetched = np.frombuffer(resp["data"], dtype=resp["dtype"]).reshape(resp["shape"])
    np.testing.assert_array_equal(fetched, arr)


def test_operation_chain_ones_exp_fetch(server_and_client):
    """ones -> exp -> fetch, verify values are exp(1)."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 10_000_000})

    # ones
    resp = _send(client, {"op": "ones", "args": [(4,)], "kwargs": {}})
    assert resp["status"] == "ok"
    ones_handle = resp["result"]["id"]

    # exp
    resp = _send(client, {"op": "exp", "args": [ones_handle], "kwargs": {}})
    assert resp["status"] == "ok"
    exp_handle = resp["result"]["id"]

    # fetch
    resp = _send(client, {"op": "fetch", "id": exp_handle})
    assert resp["status"] == "ok"
    result = np.frombuffer(resp["data"], dtype=resp["dtype"]).reshape(resp["shape"])
    np.testing.assert_allclose(result, np.e * np.ones(4), rtol=1e-7)


def test_budget_close_returns_summary(server_and_client):
    """budget_close returns a summary with budget and comms info."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    # Do some work so the summary is non-trivial
    _send(client, {"op": "ones", "args": [(5,)], "kwargs": {}})

    resp = _send(client, {"op": "budget_close"})
    assert resp["status"] == "ok"
    result = resp["result"]
    assert "budget_summary" in result
    assert "comms_summary" in result
    assert result["comms_summary"]["request_count"] >= 1


def test_error_no_session(server_and_client):
    """Operations without an active session return NoBudgetContextError."""
    _server, client = server_and_client

    resp = _send(client, {"op": "ones", "args": [(3,)], "kwargs": {}})
    assert resp["status"] == "error"
    assert resp["error_type"] == "NoBudgetContextError"


def test_error_unknown_op(server_and_client):
    """Unknown ops return InvalidRequestError."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 1_000_000})

    resp = _send(client, {"op": "nonexistent_banana_op"})
    assert resp["status"] == "error"
    assert resp["error_type"] == "InvalidRequestError"
    assert "unknown op" in resp["message"]


def test_error_invalid_msgpack(server_and_client):
    """Sending garbage bytes returns an InvalidRequestError."""
    _server, client = server_and_client

    client.send(b"\x00\x01\x02garbage")
    raw = client.recv()
    resp = msgpack.unpackb(raw, raw=False)
    assert resp["status"] == "error"
    assert resp["error_type"] == "InvalidRequestError"


def test_budget_close_without_session(server_and_client):
    """budget_close without an open session returns an error."""
    _server, client = server_and_client

    resp = _send(client, {"op": "budget_close"})
    assert resp["status"] == "error"
    assert resp["error_type"] == "NoBudgetContextError"


def test_session_reopen(server_and_client):
    """Opening a new session implicitly closes the old one."""
    _server, client = server_and_client

    _send(client, {"op": "budget_open", "flop_budget": 100_000})
    resp = _send(client, {"op": "ones", "args": [(2,)], "kwargs": {}})
    handle = resp["result"]["id"]

    # Open a new session — old one is closed, old handles are gone
    _send(client, {"op": "budget_open", "flop_budget": 200_000})

    resp = _send(client, {"op": "budget_status"})
    assert resp["result"]["flop_budget"] == 200_000

    # Old handle should not be accessible
    resp = _send(client, {"op": "fetch", "id": handle})
    assert resp["status"] == "error"
