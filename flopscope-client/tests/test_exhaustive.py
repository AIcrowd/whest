"""Exhaustive smoke test: every non-blacklisted function via client-server proxy.

Starts a real FlopscopeServer in a subprocess, then calls every proxyable
function from the registry with valid arguments.

Run with:
    PYTHONPATH=flopscope-client/src:flopscope-server/src:src \
        .venv/bin/python -m pytest flopscope-client/tests/test_exhaustive.py -v --timeout=120
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLIENT_SRC = os.path.join(_WORKTREE, "flopscope-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "flopscope-server", "src")
_REAL_SRC = os.path.join(_WORKTREE, "src")
_VENV_PYTHON = os.path.join(_WORKTREE, ".venv", "bin", "python")

for _p in (_CLIENT_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------

_SERVER_URL = "tcp://127.0.0.1:15557"

_SERVER_SCRIPT = f"""
import sys, os
sys.path.insert(0, {_REAL_SRC!r})
sys.path.insert(0, {_SERVER_SRC!r})
from flopscope_server._server import FlopscopeServer
server = FlopscopeServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
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

    import flopscope._budget as _bmod

    def _force_cleanup():
        """Force-reset client state and try to close server session."""
        _bmod._active_context = None
        # Always reset connection first (kills any stuck ZMQ socket)
        reset_connection()
        # Now create a fresh connection and try to close any server session
        try:
            from flopscope._connection import get_connection
            from flopscope._protocol import encode_budget_close

            conn = get_connection()
            conn.send_recv(encode_budget_close())
        except Exception:
            pass
        # Reset again to clear the connection used for cleanup
        reset_connection()

    _force_cleanup()
    yield
    _force_cleanup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_we():
    import flopscope as flops
    import flopscope.numpy as fnp
    return fnp


# ---------------------------------------------------------------------------
# Test data (created fresh in a budget context)
# ---------------------------------------------------------------------------


class _TestData:
    """Lazily-created test data for all tests."""

    _instance = None

    @classmethod
    def get(cls, fnp):
        # We recreate each time since handles may expire between budget contexts
        d = type("D", (), {})()
        d.SMALL_POS = fnp.array([0.5, 1.0, 1.5, 2.0])
        d.SMALL_UNIT = fnp.array([0.1, 0.3, 0.5, 0.7])
        d.SMALL_NEG = fnp.array([-2.0, -1.0, 1.0, 2.0])
        d.SMALL_INT = fnp.array([1.0, 2.0, 3.0, 4.0])
        d.PAIR_A = fnp.array([1.0, 2.0, 3.0])
        d.PAIR_B = fnp.array([4.0, 5.0, 6.0])
        d.MATRIX = fnp.array([[1.0, 2.0], [3.0, 4.0]])
        d.MATRIX3x2 = fnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        d.VEC2 = fnp.array([1.0, 2.0])
        d.VEC3 = fnp.array([1.0, 2.0, 3.0])
        d.BOOL_ARR = fnp.greater(d.PAIR_A, fnp.array([2.0, 2.0, 2.0]))
        d.SMALL_GE1 = fnp.array([1.0, 1.5, 2.0, 3.0])
        d.INT_ARR = fnp.array([1, 2, 3, 4], dtype="int64")
        d.INT_PAIR_A = fnp.array([6, 12, 15], dtype="int64")
        d.INT_PAIR_B = fnp.array([4, 8, 10], dtype="int64")
        d.COMPLEX_ARR = fnp.array([1.0 + 2.0j, 3.0 - 1.0j], dtype="complex128")
        d.ZERO_TRIMMED = fnp.array([0.0, 1.0, 2.0, 0.0])
        return d


# ---------------------------------------------------------------------------
# Counted Unary
# ---------------------------------------------------------------------------

_UNARY_GENERAL = [
    "exp",
    "exp2",
    "expm1",
    "sqrt",
    "square",
    "cbrt",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arctan",
    "arcsinh",
    "sign",
    "ceil",
    "floor",
    "abs",
    "absolute",
    "fabs",
    "negative",
    "positive",
    "rint",
    "round",
    "around",
    "fix",
    "trunc",
    "deg2rad",
    "degrees",
    "rad2deg",
    "radians",
    "log",
    "log2",
    "log10",
    "log1p",
    "reciprocal",
    "signbit",
    "spacing",
    "sinc",
    "i0",
    "nan_to_num",
    "real",
    "imag",
    "conj",
    "conjugate",
    "iscomplex",
    "isreal",
    "real_if_close",
    "logical_not",
]


@pytest.mark.parametrize("op_name", _UNARY_GENERAL)
def test_unary_general(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.SMALL_POS)
        assert result is not None


_UNARY_UNIT = ["arcsin", "arccos", "asin", "acos"]


@pytest.mark.parametrize("op_name", _UNARY_UNIT)
def test_unary_unit(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_ARCTANH = ["arctanh", "atanh", "atan"]


@pytest.mark.parametrize("op_name", _UNARY_ARCTANH)
def test_unary_arctanh(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_GE1 = ["acosh", "arccosh"]


@pytest.mark.parametrize("op_name", _UNARY_GE1)
def test_unary_ge1(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.SMALL_GE1)
        assert result is not None


def test_modf():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.modf(d.SMALL_POS)
        assert result is not None


def test_frexp():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.frexp(d.SMALL_POS)
        assert result is not None


def test_sort_complex():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.sort_complex(d.COMPLEX_ARR)
        assert result is not None


def test_isclose():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.isclose(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_angle():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.angle(d.COMPLEX_ARR)
        assert result is not None


def test_iscomplexobj():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.iscomplexobj(d.SMALL_POS)
        assert result is not None


def test_isrealobj():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.isrealobj(d.SMALL_POS)
        assert result is not None


_BITWISE_UNARY = ["bitwise_invert", "bitwise_not", "bitwise_count", "invert"]


@pytest.mark.parametrize("op_name", _BITWISE_UNARY)
def test_bitwise_unary(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.INT_ARR)
        assert result is not None


def test_isneginf():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.isneginf(d.SMALL_NEG)
        assert result is not None


def test_isposinf():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.isposinf(d.SMALL_POS)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Binary
# ---------------------------------------------------------------------------

_BINARY_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "power",
    "pow",
    "float_power",
    "mod",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logaddexp",
    "logaddexp2",
    "arctan2",
    "atan2",
    "hypot",
    "copysign",
    "nextafter",
    "heaviside",
]


@pytest.mark.parametrize("op_name", _BINARY_OPS)
def test_binary(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_ldexp():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.ldexp(d.PAIR_A, fnp.array([1, 2, 3], dtype="int64"))
        assert result is not None


_BITWISE_BINARY = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "left_shift",
    "right_shift",
]


@pytest.mark.parametrize("op_name", _BITWISE_BINARY)
def test_bitwise_binary(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.INT_PAIR_A, fnp.array([1, 2, 1], dtype="int64"))
        assert result is not None


def test_gcd():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.gcd(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_lcm():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.lcm(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_divmod():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.divmod(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vecdot():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.vecdot(d.PAIR_A, d.PAIR_B)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Reduction
# ---------------------------------------------------------------------------

_REDUCTION_OPS = [
    "sum",
    "prod",
    "mean",
    "std",
    "var",
    "max",
    "min",
    "amax",
    "amin",
    "all",
    "any",
    "argmax",
    "argmin",
    "cumsum",
    "cumprod",
    "count_nonzero",
    "median",
    "nansum",
    "nanprod",
    "nanmean",
    "nanstd",
    "nanvar",
    "nanmax",
    "nanmin",
    "nanmedian",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "ptp",
]


@pytest.mark.parametrize("op_name", _REDUCTION_OPS)
def test_reduction(op_name):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        fn = getattr(fnp, op_name)
        result = fn(d.SMALL_INT)
        assert result is not None


def test_average():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.average(d.SMALL_INT)
        assert result is not None


def test_percentile():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        # percentile server-side is wrapped as reduction(a, axis=None, **kw)
        # so q must be passed as keyword
        result = fnp.percentile(d.SMALL_INT, q=50)
        assert result is not None


def test_quantile():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.quantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_nanpercentile():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.nanpercentile(d.SMALL_INT, q=50)
        assert result is not None


def test_nanquantile():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.nanquantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_cumulative_sum():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.cumulative_sum(d.SMALL_INT, axis=0)
        assert result is not None


def test_cumulative_prod():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.cumulative_prod(d.SMALL_INT, axis=0)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Custom
# ---------------------------------------------------------------------------


def test_clip():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.clip(d.SMALL_INT, 1.5, 3.5)
        assert result is not None


def test_dot():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.dot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_matmul():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.matmul(d.MATRIX, d.VEC2)
        assert result is not None


def test_inner():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.inner(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_outer():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.outer(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vdot():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.vdot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_tensordot():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.tensordot(d.MATRIX, d.MATRIX, 1)
        assert result is not None


def test_kron():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.kron(d.VEC2, d.VEC3)
        assert result is not None


def test_cross():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.cross(d.VEC3, fnp.array([4.0, 5.0, 6.0]))
        assert result is not None


def test_diff():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.diff(d.SMALL_INT)
        assert result is not None


def test_ediff1d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.ediff1d(d.SMALL_INT)
        assert result is not None


def test_gradient():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.gradient(d.SMALL_INT)
        assert result is not None


def test_convolve():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.convolve(d.PAIR_A, d.VEC2)
        assert result is not None


def test_correlate():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.correlate(d.PAIR_A, d.VEC2)
        assert result is not None


def test_corrcoef():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.corrcoef(d.SMALL_INT)
        assert result is not None


def test_cov():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.cov(d.SMALL_INT)
        assert result is not None


def test_einsum():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.einsum("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


def test_einsum_path():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.einsum_path("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


def test_trapezoid():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.trapezoid(d.SMALL_INT)
        assert result is not None


def test_trapz():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        result = fnp.trapz(d.SMALL_INT)
        assert result is not None


def test_interp():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        result = fnp.interp(
            fnp.array([1.5, 2.5]),
            fnp.array([1.0, 2.0, 3.0]),
            fnp.array([10.0, 20.0, 30.0]),
        )
        assert result is not None


def test_linalg_svd():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        U, S, Vt = fnp.linalg.svd(d.MATRIX3x2)
        assert U is not None
        assert S is not None
        assert Vt is not None


# ---------------------------------------------------------------------------
# Free Ops: Creation
# ---------------------------------------------------------------------------


def test_array_creation():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.array([1.0, 2.0, 3.0]) is not None


def test_zeros():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.zeros((3, 3)) is not None


def test_ones():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.ones((3, 3)) is not None


def test_full():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.full((3, 3), 7.0) is not None


def test_empty():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.empty((3, 3)) is not None


def test_eye():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.eye(3) is not None


def test_identity():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.identity(3) is not None


def test_arange():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.arange(0, 10, 2) is not None


def test_linspace():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.linspace(0, 1, 5) is not None


def test_logspace():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.logspace(0, 2, 5) is not None


def test_geomspace():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.geomspace(1, 100, 5) is not None


def test_zeros_like():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.zeros_like(d.SMALL_INT) is not None


def test_ones_like():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.ones_like(d.SMALL_INT) is not None


def test_full_like():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.full_like(d.SMALL_INT, 9.0) is not None


def test_empty_like():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.empty_like(d.SMALL_INT) is not None


def test_diag():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.diag(d.VEC3) is not None


def test_diagflat():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.diagflat(d.VEC2) is not None


def test_tri():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.tri(3, 3) is not None


def test_tril():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.tril(d.MATRIX) is not None


def test_triu():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.triu(d.MATRIX) is not None


def test_vander():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.vander(d.PAIR_A, 3) is not None


# ---------------------------------------------------------------------------
# Free Ops: Manipulation
# ---------------------------------------------------------------------------


def test_reshape():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.reshape(d.SMALL_INT, (2, 2)) is not None


def test_transpose():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.transpose(d.MATRIX) is not None


def test_swapaxes():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.swapaxes(d.MATRIX, 0, 1) is not None


def test_moveaxis():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.moveaxis(d.MATRIX, 0, 1) is not None


def test_concatenate():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.concatenate([d.PAIR_A, d.PAIR_B]) is not None


def test_stack():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.stack([d.PAIR_A, d.PAIR_B]) is not None


def test_vstack():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.vstack([d.PAIR_A, d.PAIR_B]) is not None


def test_hstack():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.hstack([d.PAIR_A, d.PAIR_B]) is not None


def test_column_stack():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.column_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_row_stack():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.row_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_dstack():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.dstack([d.PAIR_A, d.PAIR_B]) is not None


def test_split():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.split(fnp.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_hsplit():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.hsplit(fnp.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_vsplit():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.vsplit(fnp.array([[1.0, 2.0], [3.0, 4.0]]), 2) is not None


def test_dsplit():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.dsplit(fnp.array([[[1.0, 2.0], [3.0, 4.0]]]), 2) is not None


def test_array_split():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.array_split(d.SMALL_INT, 2) is not None


def test_squeeze():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.squeeze(fnp.array([[[1.0, 2.0]]])) is not None


def test_expand_dims():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.expand_dims(d.PAIR_A, 0) is not None


def test_ravel():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.ravel(d.MATRIX) is not None


def test_copy():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.copy(d.PAIR_A) is not None


def test_flip():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.flip(d.PAIR_A) is not None


def test_fliplr():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.fliplr(d.MATRIX) is not None


def test_flipud():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.flipud(d.MATRIX) is not None


def test_rot90():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.rot90(d.MATRIX) is not None


def test_roll():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.roll(d.PAIR_A, 1) is not None


def test_rollaxis():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.rollaxis(d.MATRIX, 1) is not None


def test_tile():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.tile(d.PAIR_A, 2) is not None


def test_repeat():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.repeat(d.PAIR_A, 2) is not None


def test_resize():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.resize(d.PAIR_A, (2, 3)) is not None


def test_append():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.append(d.PAIR_A, d.PAIR_B) is not None


def test_insert():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.insert(d.PAIR_A, 1, 99.0) is not None


def test_delete():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.delete(d.PAIR_A, 1) is not None


def test_unique():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.unique(fnp.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_trim_zeros():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.trim_zeros(d.ZERO_TRIMMED) is not None


def test_sort():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.sort(fnp.array([3.0, 1.0, 2.0])) is not None


def test_argsort():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.argsort(fnp.array([3.0, 1.0, 2.0])) is not None


def test_partition():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.partition(fnp.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_argpartition():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.argpartition(fnp.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_take():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.take(d.PAIR_A, [0, 2]) is not None


def test_take_along_axis():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.take_along_axis(
                fnp.array([[1.0, 2.0], [3.0, 4.0]]),
                fnp.array([[0, 1], [1, 0]], dtype="int64"),
                axis=1,
            )
            is not None
        )


def test_compress():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.compress([True, False, True], d.PAIR_A) is not None


def test_extract():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.extract(d.BOOL_ARR, d.PAIR_A) is not None


def test_diagonal():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.diagonal(d.MATRIX) is not None


def test_trace():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.trace(d.MATRIX) is not None


def test_pad():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.pad(d.PAIR_A, 1) is not None


def test_searchsorted():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.searchsorted(fnp.array([1.0, 3.0, 5.0]), 2.0) is not None


def test_where():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.where(d.BOOL_ARR, d.PAIR_A, d.PAIR_B) is not None


def test_select():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.select([d.BOOL_ARR], [d.PAIR_A]) is not None


def test_nonzero():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.nonzero(fnp.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_argwhere():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.argwhere(fnp.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_flatnonzero():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.flatnonzero(fnp.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_isin():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.isin(d.PAIR_A, [1.0, 3.0]) is not None


def test_in1d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.in1d(d.PAIR_A, fnp.array([1.0, 3.0])) is not None


def test_intersect1d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.intersect1d(d.PAIR_A, fnp.array([2.0, 3.0, 7.0])) is not None


def test_union1d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.union1d(d.PAIR_A, fnp.array([2.0, 5.0])) is not None


def test_setdiff1d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.setdiff1d(d.PAIR_A, fnp.array([2.0])) is not None


def test_setxor1d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.setxor1d(d.PAIR_A, fnp.array([2.0, 5.0])) is not None


def test_allclose():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.allclose(d.PAIR_A, d.PAIR_A) is not None


def test_array_equal():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.array_equal(d.PAIR_A, d.PAIR_A) is not None


def test_array_equiv():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.array_equiv(d.PAIR_A, d.PAIR_A) is not None


def test_isfinite():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.isfinite(d.SMALL_POS) is not None


def test_isinf():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.isinf(d.SMALL_POS) is not None


def test_isnan():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.isnan(d.SMALL_POS) is not None


def test_isscalar():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.isscalar(5.0) is not None


def test_shape():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.shape(d.MATRIX) is not None


def test_ndim():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.ndim(d.MATRIX) is not None


def test_size():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.size(d.MATRIX) is not None


def test_broadcast_to():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.broadcast_to(d.PAIR_A, (2, 3)) is not None


def test_broadcast_arrays():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.broadcast_arrays(d.PAIR_A, d.PAIR_B) is not None


def test_broadcast_shapes():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.broadcast_shapes((3,), (3,)) is not None


def test_histogram():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.histogram(d.SMALL_INT, 3) is not None


def test_histogram_bin_edges():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.histogram_bin_edges(d.SMALL_INT, 3) is not None


def test_histogram2d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.histogram2d(
                fnp.array([1.0, 2.0, 3.0]),
                fnp.array([4.0, 5.0, 6.0]),
                3,
            )
            is not None
        )


def test_histogramdd():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.histogramdd(
                fnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                3,
            )
            is not None
        )


def test_asarray():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.asarray(d.PAIR_A) is not None


def test_asarray_chkfinite():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.asarray_chkfinite(d.PAIR_A) is not None


def test_astype():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.astype(d.PAIR_A, "float32") is not None


def test_can_cast():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.can_cast("float32", "float64") is not None


def test_result_type():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.result_type("float32", "float64") is not None


def test_promote_types():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.promote_types("float32", "float64") is not None


def test_atleast_1d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.atleast_1d(d.PAIR_A) is not None


def test_atleast_2d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.atleast_2d(d.PAIR_A) is not None


def test_atleast_3d():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.atleast_3d(d.PAIR_A) is not None


def test_diag_indices():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.diag_indices(3) is not None


def test_diag_indices_from():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.diag_indices_from(d.MATRIX) is not None


def test_tril_indices():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.tril_indices(3) is not None


def test_triu_indices():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.triu_indices(3) is not None


def test_tril_indices_from():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.tril_indices_from(d.MATRIX) is not None


def test_triu_indices_from():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.triu_indices_from(d.MATRIX) is not None


def test_indices():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.indices((2, 3)) is not None


def test_ix_():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.ix_(
                fnp.array([0, 1], dtype="int64"),
                fnp.array([0, 1], dtype="int64"),
            )
            is not None
        )


def test_unravel_index():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.unravel_index(5, (3, 3)) is not None


def test_ravel_multi_index():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.ravel_multi_index(
                (
                    fnp.array([0, 1, 2], dtype="int64"),
                    fnp.array([0, 1, 2], dtype="int64"),
                ),
                (3, 3),
            )
            is not None
        )


def test_mask_indices():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.mask_indices(3, fnp.triu) is not None


def test_meshgrid():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.meshgrid(fnp.array([1.0, 2.0]), fnp.array([3.0, 4.0])) is not None


def test_lexsort():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.lexsort((fnp.array([1.0, 2.0, 1.0]), fnp.array([3.0, 1.0, 2.0])))
            is not None
        )


def test_digitize():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.digitize(fnp.array([0.5, 1.5, 2.5]), fnp.array([1.0, 2.0, 3.0]))
            is not None
        )


def test_bincount():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.bincount(fnp.array([0, 1, 1, 2, 3, 3, 3], dtype="int64")) is not None


def test_packbits():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.packbits(fnp.array([1, 0, 1, 1, 0, 0, 0, 1], dtype="uint8")) is not None
        )


def test_unpackbits():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.unpackbits(fnp.array([177], dtype="uint8")) is not None


def test_block():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.block([[d.PAIR_A, d.PAIR_B]]) is not None


def test_concat():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.concat([d.PAIR_A, d.PAIR_B]) is not None


def test_iterable():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.iterable(d.PAIR_A) is not None


def test_isfortran():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.isfortran(d.MATRIX) is not None


def test_typename():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.typename("float64") is not None


def test_mintypecode():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.mintypecode(["f", "d"]) is not None


def test_base_repr():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.base_repr(10, 2) is not None


def test_binary_repr():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.binary_repr(10) is not None


def test_put():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        arr = fnp.array([1.0, 2.0, 3.0, 4.0])
        fnp.put(arr, [0, 2], [99.0, 88.0])


def test_place():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        arr = fnp.array([1.0, 2.0, 3.0, 4.0])
        fnp.place(arr, fnp.array([True, False, True, False], dtype="bool"), [99.0, 88.0])


def test_putmask():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        arr = fnp.array([1.0, 2.0, 3.0, 4.0])
        fnp.putmask(arr, fnp.array([True, False, True, False], dtype="bool"), 0.0)


def test_fill_diagonal():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        mat = fnp.array([[1.0, 2.0], [3.0, 4.0]])
        fnp.fill_diagonal(mat, 0.0)


def test_copyto():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        arr = fnp.array([1.0, 2.0, 3.0])
        fnp.copyto(arr, fnp.array([9.0, 8.0, 7.0]))


def test_choose():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert (
            fnp.choose(
                fnp.array([0, 1, 0], dtype="int64"),
                [fnp.array([10.0, 20.0, 30.0]), fnp.array([40.0, 50.0, 60.0])],
            )
            is not None
        )


def test_require():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.require(d.PAIR_A) is not None


def test_matrix_transpose():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.matrix_transpose(fnp.array([[[1.0, 2.0], [3.0, 4.0]]])) is not None


def test_permute_dims():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.permute_dims(d.MATRIX, (1, 0)) is not None


def test_unique_all():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.unique_all(fnp.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_counts():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.unique_counts(fnp.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_inverse():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.unique_inverse(fnp.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_values():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.unique_values(fnp.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unstack():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.unstack(d.MATRIX) is not None


def test_shares_memory():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.shares_memory(d.PAIR_A, d.PAIR_B) is not None


def test_may_share_memory():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.may_share_memory(d.PAIR_A, d.PAIR_B) is not None


def test_min_scalar_type():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.min_scalar_type(10) is not None


def test_issubdtype():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        assert fnp.issubdtype("float64", "float64") is not None


def test_common_type():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        d = _TestData.get(fnp)
        assert fnp.common_type(d.PAIR_A) is not None


def test_put_along_axis():
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        fnp.put_along_axis(
            fnp.array([[1.0, 2.0], [3.0, 4.0]]),
            fnp.array([[0], [1]], dtype="int64"),
            fnp.array([[99.0], [88.0]]),
            axis=1,
        )


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

_RANDOM_SIMPLE = [
    ("random.rand", lambda fnp: fnp.random.rand(3, 3)),
    ("random.randn", lambda fnp: fnp.random.randn(3, 3)),
    ("random.random", lambda fnp: fnp.random.random((3,))),
    ("random.randint", lambda fnp: fnp.random.randint(0, 10, (3,))),
    ("random.uniform", lambda fnp: fnp.random.uniform(0.0, 1.0, (3,))),
    ("random.normal", lambda fnp: fnp.random.normal(0.0, 1.0, (3,))),
    ("random.seed", lambda fnp: fnp.random.seed(42)),
    ("random.choice", lambda fnp: fnp.random.choice(10, 3)),
    ("random.permutation", lambda fnp: fnp.random.permutation(5)),
    ("random.beta", lambda fnp: fnp.random.beta(2.0, 5.0, (3,))),
    ("random.binomial", lambda fnp: fnp.random.binomial(10, 0.5, (3,))),
    ("random.chisquare", lambda fnp: fnp.random.chisquare(2, (3,))),
    ("random.exponential", lambda fnp: fnp.random.exponential(1.0, (3,))),
    ("random.gamma", lambda fnp: fnp.random.gamma(2.0, 1.0, (3,))),
    ("random.geometric", lambda fnp: fnp.random.geometric(0.5, (3,))),
    ("random.gumbel", lambda fnp: fnp.random.gumbel(0.0, 1.0, (3,))),
    ("random.laplace", lambda fnp: fnp.random.laplace(0.0, 1.0, (3,))),
    ("random.logistic", lambda fnp: fnp.random.logistic(0.0, 1.0, (3,))),
    ("random.lognormal", lambda fnp: fnp.random.lognormal(0.0, 1.0, (3,))),
    ("random.logseries", lambda fnp: fnp.random.logseries(0.9, (3,))),
    ("random.multinomial", lambda fnp: fnp.random.multinomial(10, [0.5, 0.3, 0.2])),
    ("random.negative_binomial", lambda fnp: fnp.random.negative_binomial(5, 0.5, (3,))),
    (
        "random.noncentral_chisquare",
        lambda fnp: fnp.random.noncentral_chisquare(2, 1.0, (3,)),
    ),
    ("random.noncentral_f", lambda fnp: fnp.random.noncentral_f(5, 10, 1.0, (3,))),
    ("random.pareto", lambda fnp: fnp.random.pareto(2.0, (3,))),
    ("random.poisson", lambda fnp: fnp.random.poisson(5.0, (3,))),
    ("random.power", lambda fnp: fnp.random.power(2.0, (3,))),
    ("random.rayleigh", lambda fnp: fnp.random.rayleigh(1.0, (3,))),
    ("random.standard_cauchy", lambda fnp: fnp.random.standard_cauchy((3,))),
    ("random.standard_exponential", lambda fnp: fnp.random.standard_exponential((3,))),
    ("random.standard_gamma", lambda fnp: fnp.random.standard_gamma(2.0, (3,))),
    ("random.standard_normal", lambda fnp: fnp.random.standard_normal((3,))),
    ("random.standard_t", lambda fnp: fnp.random.standard_t(5.0, (3,))),
    ("random.triangular", lambda fnp: fnp.random.triangular(0.0, 0.5, 1.0, (3,))),
    ("random.vonmises", lambda fnp: fnp.random.vonmises(0.0, 1.0, (3,))),
    ("random.wald", lambda fnp: fnp.random.wald(1.0, 1.0, (3,))),
    ("random.weibull", lambda fnp: fnp.random.weibull(2.0, (3,))),
    ("random.zipf", lambda fnp: fnp.random.zipf(2.0, (3,))),
    ("random.dirichlet", lambda fnp: fnp.random.dirichlet([1.0, 1.0, 1.0])),
    (
        "random.multivariate_normal",
        lambda fnp: fnp.random.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
    ),
    ("random.f", lambda fnp: fnp.random.f(5, 10, (3,))),
    ("random.hypergeometric", lambda fnp: fnp.random.hypergeometric(10, 5, 7, (3,))),
    ("random.random_sample", lambda fnp: fnp.random.random_sample((3,))),
    ("random.ranf", lambda fnp: fnp.random.ranf((3,))),
    ("random.sample", lambda fnp: fnp.random.sample((3,))),
    ("random.get_state", lambda fnp: fnp.random.get_state()),
    (
        "random.shuffle",
        lambda fnp: fnp.random.shuffle(fnp.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    ),
]


@pytest.mark.parametrize(
    "name,call", _RANDOM_SIMPLE, ids=[n for n, _ in _RANDOM_SIMPLE]
)
def test_random(name, call):
    fnp = _import_we()
    with flops.BudgetContext(flop_budget=10**9):
        call(fnp)
