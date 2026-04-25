"""End-to-end integration tests for all flopscope operation categories.

Starts a real server subprocess and tests pointwise ops, reductions, linalg,
random, stats, einsum, and error propagation.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

import flopscope as flops
import flopscope.numpy as fnp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "flopscope-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
# Prefer the server's own venv (which has msgpack/pyzmq) for the server subprocess;
# fall back to the worktree root venv if it doesn't exist.
_SERVER_VENV_PYTHON = os.path.join(
    _WORKTREE, "flopscope-server", ".venv", "bin", "python"
)
_ROOT_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")
_VENV_PYTHON = (
    _SERVER_VENV_PYTHON if os.path.exists(_SERVER_VENV_PYTHON) else _ROOT_VENV_PYTHON
)

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------

_SERVER_URL = "tcp://127.0.0.1:15560"

_SERVER_SCRIPT = f"""
import sys, os
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="module", autouse=True)
def _start_server():
    os.environ["FLOPSCOPE_SERVER_URL"] = _SERVER_URL
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-c", _SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    assert "SERVER_READY" in line, f"Server failed to start: {line}"
    time.sleep(0.3)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def _reset_client():
    from flopscope._connection import reset_connection

    from flopscope._budget import _reset_global_default

    reset_connection()
    _reset_global_default()
    yield
    reset_connection()
    _reset_global_default()


# ===========================================================================
# Category 1: Pointwise operations
# ===========================================================================


class TestPointwise:
    def test_add_lists(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1, 2, 3])
            b = fnp.array([4, 5, 6])
            result = fnp.add(a, b)
            assert result.tolist() == [5, 7, 9]

    def test_add_floats(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1.0, 2.0, 3.0])
            b = fnp.array([4.0, 5.0, 6.0])
            result = fnp.add(a, b)
            assert result.tolist() == [5.0, 7.0, 9.0]

    def test_subtract(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([10.0, 20.0, 30.0])
            b = fnp.array([1.0, 2.0, 3.0])
            result = fnp.subtract(a, b)
            assert result.tolist() == [9.0, 18.0, 27.0]

    def test_multiply(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([2.0, 3.0, 4.0])
            b = fnp.array([5.0, 6.0, 7.0])
            result = fnp.multiply(a, b)
            assert result.tolist() == [10.0, 18.0, 28.0]

    def test_exp(self):
        import math

        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([0.0, 1.0])
            result = fnp.exp(a)
            vals = result.tolist()
            assert abs(vals[0] - 1.0) < 1e-10
            assert abs(vals[1] - math.e) < 1e-10


# ===========================================================================
# Category 2: Reductions
# ===========================================================================


class TestReduction:
    def test_sum_1d(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1.0, 2.0, 3.0])
            result = fnp.sum(a)
            assert float(result) == 6.0

    def test_sum_returns_scalar(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1, 2, 3])
            result = fnp.sum(a)
            assert float(result) == 6.0

    def test_sum_2d_axis0(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            result = fnp.sum(a, axis=0)
            assert result.tolist() == [4.0, 6.0]

    def test_sum_2d_axis1(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            result = fnp.sum(a, axis=1)
            assert result.tolist() == [3.0, 7.0]

    def test_mean(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1.0, 2.0, 3.0, 4.0])
            result = fnp.mean(a)
            assert float(result) == 2.5


# ===========================================================================
# Category 3: Linear algebra (linalg)
# ===========================================================================


class TestLinalg:
    def test_svd_diagonal(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            A = fnp.array([[1.0, 0.0], [0.0, 2.0]])
            U, S, Vh = fnp.linalg.svd(A)
            sv = sorted(S.tolist(), reverse=True)
            assert abs(sv[0] - 2.0) < 1e-10
            assert abs(sv[1] - 1.0) < 1e-10

    def test_svd_shapes(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            A = fnp.array([[1.0, 0.0], [0.0, 2.0]])
            U, S, Vh = fnp.linalg.svd(A)
            assert S.shape == (2,)
            assert U.shape == (2, 2)
            assert Vh.shape == (2, 2)

    def test_norm(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            # L2 norm of [3, 4] = 5
            a = fnp.array([3.0, 4.0])
            result = fnp.linalg.norm(a)
            assert abs(float(result) - 5.0) < 1e-10

    def test_dot_matmul(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            A = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            B = fnp.array([[5.0, 6.0], [7.0, 8.0]])
            C = fnp.linalg.matmul(A, B)
            assert C.tolist() == [[19.0, 22.0], [43.0, 50.0]]


# ===========================================================================
# Category 4: Random
# ===========================================================================


class TestRandom:
    def test_normal_shape(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            result = fnp.random.normal(size=[100])
            assert result.shape == (100,)

    def test_normal_values_are_floats(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            result = fnp.random.normal(size=[10])
            vals = result.tolist()
            assert len(vals) == 10
            assert all(isinstance(v, float) for v in vals)

    def test_uniform_shape(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            result = fnp.random.uniform(size=[50])
            assert result.shape == (50,)

    def test_uniform_range(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            result = fnp.random.uniform(size=[100])
            vals = result.tolist()
            assert all(0.0 <= v <= 1.0 for v in vals)


# ===========================================================================
# Category 5: Stats distributions
# ===========================================================================


class TestStats:
    def test_norm_pdf_at_zero(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            x = fnp.array([0.0])
            result = flops.stats.norm.pdf(x)
            val = float(result)
            # PDF of standard normal at 0 = 1/sqrt(2*pi) ≈ 0.3989
            assert abs(val - 0.3989422804014327) < 1e-6

    def test_norm_cdf_at_zero(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            x = fnp.array([0.0])
            result = flops.stats.norm.cdf(x)
            val = float(result)
            # CDF of standard normal at 0 = 0.5
            assert abs(val - 0.5) < 1e-10

    def test_expon_pdf_at_zero(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            x = fnp.array([0.0])
            result = flops.stats.expon.pdf(x)
            val = float(result)
            # PDF of exponential(rate=1) at 0 = 1.0
            assert abs(val - 1.0) < 1e-10

    def test_norm_pdf_shape(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            x = fnp.array([0.0, 1.0, -1.0])
            result = flops.stats.norm.pdf(x)
            assert result.shape == (3,)

    def test_norm_cdf_monotone(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            x = fnp.array([-1.0, 0.0, 1.0])
            result = flops.stats.norm.cdf(x)
            vals = result.tolist()
            assert vals[0] < vals[1] < vals[2]


# ===========================================================================
# Category 6: Einsum
# ===========================================================================


class TestEinsum:
    def test_matmul_2x2(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            A = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            B = fnp.array([[5.0, 6.0], [7.0, 8.0]])
            C = fnp.einsum("ij,jk->ik", A, B)
            # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
            assert C.tolist() == [[19.0, 22.0], [43.0, 50.0]]

    def test_matmul_identity(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            A = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            eye = fnp.array([[1.0, 0.0], [0.0, 1.0]])
            C = fnp.einsum("ij,jk->ik", A, eye)
            assert C.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_dot_product(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1.0, 2.0, 3.0])
            b = fnp.array([4.0, 5.0, 6.0])
            result = fnp.einsum("i,i->", a, b)
            # 1*4 + 2*5 + 3*6 = 32
            assert float(result) == 32.0

    def test_outer_product(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1.0, 2.0])
            b = fnp.array([3.0, 4.0])
            result = fnp.einsum("i,j->ij", a, b)
            assert result.shape == (2, 2)
            assert result.tolist() == [[3.0, 4.0], [6.0, 8.0]]

    def test_trace(self):
        with flops.BudgetContext(flop_budget=1_000_000):
            A = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            trace = fnp.einsum("ii->", A)
            assert float(trace) == 5.0


# ===========================================================================
# Category 7: Error propagation
# ===========================================================================


class TestErrorPropagation:
    def test_budget_exhausted_on_matmul(self):
        with flops.BudgetContext(flop_budget=1):
            A = fnp.array([[1.0, 2.0], [3.0, 4.0]])
            B = fnp.array([[5.0, 6.0], [7.0, 8.0]])
            with pytest.raises(flops.BudgetExhaustedError):
                fnp.einsum("ij,jk->ik", A, B)

    def test_budget_exhausted_error_type(self):
        with flops.BudgetContext(flop_budget=1):
            a = fnp.array([1.0, 2.0, 3.0])
            b = fnp.array([4.0, 5.0, 6.0])
            with pytest.raises(flops.BudgetExhaustedError):
                # Keep calling ops until budget is exhausted
                for _ in range(100):
                    fnp.add(a, b)

    def test_no_budget_context_raises(self):
        with pytest.raises((flops.NoBudgetContextError, flops.FlopscopeServerError)):
            fnp.array([1.0, 2.0, 3.0])

    def test_budget_context_isolates_errors(self):
        """Verify a new context works after a previous one exhausted budget."""
        # First context: exhaust budget
        try:
            with flops.BudgetContext(flop_budget=1):
                a = fnp.array([1.0] * 1000)
                fnp.exp(a)
        except flops.BudgetExhaustedError:
            pass

        # Second context: should work normally
        with flops.BudgetContext(flop_budget=1_000_000):
            a = fnp.array([1.0, 2.0, 3.0])
            b = fnp.array([4.0, 5.0, 6.0])
            result = fnp.add(a, b)
            assert result.tolist() == [5.0, 7.0, 9.0]
