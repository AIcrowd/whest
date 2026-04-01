"""End-to-end integration tests: real client talking to a real server.

Starts a MechestimServer in a subprocess, points the client at it,
and exercises the full request/response chain.
"""
from __future__ import annotations

import math
import os
import subprocess
import signal
import sys
import time

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "mechestim-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "mechestim-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")

# Ensure the CLIENT mechestim is importable in this process
for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Server fixture — runs in a subprocess so it gets the REAL mechestim
# ---------------------------------------------------------------------------

_SERVER_URL = "tcp://127.0.0.1:15556"

# Inline server launcher script
_SERVER_SCRIPT = f"""
import sys, os
# The server needs the REAL mechestim (src/) and the server package
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})

from mechestim_server._server import MechestimServer
server = MechestimServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
def _start_server():
    """Start a real MechestimServer in a subprocess for the entire test session."""
    os.environ["MECHESTIM_SERVER_URL"] = _SERVER_URL

    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait for the server to be ready
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"Server failed to start: {line}"
    # Give the socket a moment to bind
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the client connection between tests."""
    from mechestim._connection import reset_connection

    reset_connection()
    yield
    reset_connection()


# ---------------------------------------------------------------------------
# TestBasicOps
# ---------------------------------------------------------------------------


class TestBasicOps:
    def test_zeros_and_fetch(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            z = me.zeros((3, 4))
            assert z.shape == (3, 4)
            assert z.dtype == "float64"
            values = z.tolist()
            assert len(values) == 3
            assert len(values[0]) == 4
            assert all(v == 0.0 for row in values for v in row)

    def test_ones(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            o = me.ones((5,))
            values = o.tolist()
            assert values == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_array_from_list(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            a = me.array([1.0, 2.0, 3.0])
            assert a.shape == (3,)
            assert a.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# TestCountedOps
# ---------------------------------------------------------------------------


class TestCountedOps:
    def test_exp(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as ctx:
            o = me.ones((3,))
            result = me.exp(o)
            values = result.tolist()
            for v in values:
                assert abs(v - math.e) < 1e-6
            # Check that flops were tracked
            ctx.summary()  # updates flops_used from server
            assert ctx.flops_used > 0

    def test_add(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            o = me.ones((4,))
            result = me.add(o, o)
            values = result.tolist()
            assert values == [2.0, 2.0, 2.0, 2.0]

    def test_sum_reduction(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            a = me.array([1.0, 2.0, 3.0])
            s = me.sum(a)
            assert float(s) == 6.0


# ---------------------------------------------------------------------------
# TestOperators
# ---------------------------------------------------------------------------


class TestOperators:
    def test_add_operator(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, 2.0])
            y = me.array([3.0, 4.0])
            z = x + y
            assert z.tolist() == [4.0, 6.0]

    def test_mul_scalar(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, 2.0, 3.0])
            z = x * 2.0
            assert z.tolist() == [2.0, 4.0, 6.0]

    def test_neg(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, -2.0, 3.0])
            z = -x
            assert z.tolist() == [-1.0, 2.0, -3.0]

    def test_matmul(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            # 2x2 identity matrix
            ident = me.eye(2)
            v = me.array([5.0, 7.0])
            result = ident @ v
            assert result.tolist() == [5.0, 7.0]


# ---------------------------------------------------------------------------
# TestTransparency
# ---------------------------------------------------------------------------


class TestTransparency:
    def test_print_shows_values(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([1.0, 2.0, 3.0])
            r = repr(x)
            assert "array(" in r
            assert "1.0" in r

    def test_iteration(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([10.0, 20.0, 30.0])
            values = [v for v in x]
            assert values == [10.0, 20.0, 30.0]

    def test_indexing(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([10.0, 20.0, 30.0])
            assert x[0] == 10.0
            assert x[2] == 30.0

    def test_len(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([[1.0, 2.0], [3.0, 4.0]])
            assert len(x) == 2

    def test_float_conversion(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            x = me.array([42.0])
            assert float(x) == 42.0


# ---------------------------------------------------------------------------
# TestEinsum
# ---------------------------------------------------------------------------


class TestEinsum:
    def test_einsum_matvec(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000):
            ident = me.eye(3)
            v = me.array([1.0, 2.0, 3.0])
            result = me.einsum("ij,j->i", ident, v)
            assert result.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# TestErrors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_budget_exhausted(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1):
            a = me.ones((100,))
            with pytest.raises(me.BudgetExhaustedError):
                me.exp(a)

    def test_no_budget_context(self):
        import mechestim as me

        # Operations outside a BudgetContext should raise NoBudgetContextError
        with pytest.raises((me.NoBudgetContextError, me.MechEstimServerError)):
            me.ones((3,))

    def test_blacklisted_function(self):
        import mechestim as me

        with pytest.raises(AttributeError, match="blacklisted"):
            me.save

    def test_unknown_function(self):
        import mechestim as me

        with pytest.raises(AttributeError):
            me.nonexistent_function_xyz


# ---------------------------------------------------------------------------
# TestBudgetTracking
# ---------------------------------------------------------------------------


class TestBudgetTracking:
    def test_flops_tracked(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as ctx:
            a = me.ones((10,))
            _ = me.exp(a)
            ctx.summary()  # updates flops_used from server
            assert ctx.flops_used > 0

    def test_summary(self):
        import mechestim as me

        with me.BudgetContext(flop_budget=1_000_000) as ctx:
            _ = me.ones((10,))
            _ = me.exp(me.ones((10,)))
            s = ctx.summary()
            assert isinstance(s, str)
            assert "FLOPs" in s or "flop" in s.lower()
