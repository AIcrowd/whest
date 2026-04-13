"""Exhaustive smoke test: every non-blacklisted function via client-server proxy.

Starts a real WhestServer in a subprocess, then calls every proxyable
function from the registry with valid arguments.

Run with:
    PYTHONPATH=whest-client/src:whest-server/src:src \
        .venv/bin/python -m pytest whest-client/tests/test_exhaustive.py -v --timeout=120
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
_CLIENT_SRC = os.path.join(_WORKTREE, "whest-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "whest-server", "src")
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
from whest_server._server import WhestServer
server = WhestServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
def _start_server():
    os.environ["WHEST_SERVER_URL"] = _SERVER_URL
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
    from whest._connection import reset_connection

    import whest._budget as _bmod

    def _force_cleanup():
        """Force-reset client state and try to close server session."""
        _bmod._active_context = None
        # Always reset connection first (kills any stuck ZMQ socket)
        reset_connection()
        # Now create a fresh connection and try to close any server session
        try:
            from whest._connection import get_connection
            from whest._protocol import encode_budget_close

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


def _import_me():
    import whest as we

    return me


# ---------------------------------------------------------------------------
# Test data (created fresh in a budget context)
# ---------------------------------------------------------------------------


class _TestData:
    """Lazily-created test data for all tests."""

    _instance = None

    @classmethod
    def get(cls, me):
        # We recreate each time since handles may expire between budget contexts
        d = type("D", (), {})()
        d.SMALL_POS = we.array([0.5, 1.0, 1.5, 2.0])
        d.SMALL_UNIT = we.array([0.1, 0.3, 0.5, 0.7])
        d.SMALL_NEG = we.array([-2.0, -1.0, 1.0, 2.0])
        d.SMALL_INT = we.array([1.0, 2.0, 3.0, 4.0])
        d.PAIR_A = we.array([1.0, 2.0, 3.0])
        d.PAIR_B = we.array([4.0, 5.0, 6.0])
        d.MATRIX = we.array([[1.0, 2.0], [3.0, 4.0]])
        d.MATRIX3x2 = we.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        d.VEC2 = we.array([1.0, 2.0])
        d.VEC3 = we.array([1.0, 2.0, 3.0])
        d.BOOL_ARR = we.greater(d.PAIR_A, we.array([2.0, 2.0, 2.0]))
        d.SMALL_GE1 = we.array([1.0, 1.5, 2.0, 3.0])
        d.INT_ARR = we.array([1, 2, 3, 4], dtype="int64")
        d.INT_PAIR_A = we.array([6, 12, 15], dtype="int64")
        d.INT_PAIR_B = we.array([4, 8, 10], dtype="int64")
        d.COMPLEX_ARR = we.array([1.0 + 2.0j, 3.0 - 1.0j], dtype="complex128")
        d.ZERO_TRIMMED = we.array([0.0, 1.0, 2.0, 0.0])
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
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_POS)
        assert result is not None


_UNARY_UNIT = ["arcsin", "arccos", "asin", "acos"]


@pytest.mark.parametrize("op_name", _UNARY_UNIT)
def test_unary_unit(op_name):
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_ARCTANH = ["arctanh", "atanh", "atan"]


@pytest.mark.parametrize("op_name", _UNARY_ARCTANH)
def test_unary_arctanh(op_name):
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_GE1 = ["acosh", "arccosh"]


@pytest.mark.parametrize("op_name", _UNARY_GE1)
def test_unary_ge1(op_name):
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_GE1)
        assert result is not None


def test_modf():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.modf(d.SMALL_POS)
        assert result is not None


def test_frexp():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.frexp(d.SMALL_POS)
        assert result is not None


def test_sort_complex():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.sort_complex(d.COMPLEX_ARR)
        assert result is not None


def test_isclose():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.isclose(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_angle():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.angle(d.COMPLEX_ARR)
        assert result is not None


def test_iscomplexobj():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.iscomplexobj(d.SMALL_POS)
        assert result is not None


def test_isrealobj():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.isrealobj(d.SMALL_POS)
        assert result is not None


_BITWISE_UNARY = ["bitwise_invert", "bitwise_not", "bitwise_count", "invert"]


@pytest.mark.parametrize("op_name", _BITWISE_UNARY)
def test_bitwise_unary(op_name):
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.INT_ARR)
        assert result is not None


def test_isneginf():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.isneginf(d.SMALL_NEG)
        assert result is not None


def test_isposinf():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.isposinf(d.SMALL_POS)
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
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_ldexp():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.ldexp(d.PAIR_A, we.array([1, 2, 3], dtype="int64"))
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
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.INT_PAIR_A, we.array([1, 2, 1], dtype="int64"))
        assert result is not None


def test_gcd():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.gcd(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_lcm():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.lcm(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_divmod():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.divmod(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vecdot():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.vecdot(d.PAIR_A, d.PAIR_B)
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
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_INT)
        assert result is not None


def test_average():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.average(d.SMALL_INT)
        assert result is not None


def test_percentile():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        # percentile server-side is wrapped as reduction(a, axis=None, **kw)
        # so q must be passed as keyword
        result = we.percentile(d.SMALL_INT, q=50)
        assert result is not None


def test_quantile():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.quantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_nanpercentile():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.nanpercentile(d.SMALL_INT, q=50)
        assert result is not None


def test_nanquantile():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.nanquantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_cumulative_sum():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.cumulative_sum(d.SMALL_INT, axis=0)
        assert result is not None


def test_cumulative_prod():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.cumulative_prod(d.SMALL_INT, axis=0)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Custom
# ---------------------------------------------------------------------------


def test_clip():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.clip(d.SMALL_INT, 1.5, 3.5)
        assert result is not None


def test_dot():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.dot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_matmul():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.matmul(d.MATRIX, d.VEC2)
        assert result is not None


def test_inner():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.inner(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_outer():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.outer(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vdot():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.vdot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_tensordot():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.tensordot(d.MATRIX, d.MATRIX, 1)
        assert result is not None


def test_kron():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.kron(d.VEC2, d.VEC3)
        assert result is not None


def test_cross():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.cross(d.VEC3, we.array([4.0, 5.0, 6.0]))
        assert result is not None


def test_diff():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.diff(d.SMALL_INT)
        assert result is not None


def test_ediff1d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.ediff1d(d.SMALL_INT)
        assert result is not None


def test_gradient():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.gradient(d.SMALL_INT)
        assert result is not None


def test_convolve():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.convolve(d.PAIR_A, d.VEC2)
        assert result is not None


def test_correlate():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.correlate(d.PAIR_A, d.VEC2)
        assert result is not None


def test_corrcoef():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.corrcoef(d.SMALL_INT)
        assert result is not None


def test_cov():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.cov(d.SMALL_INT)
        assert result is not None


def test_einsum():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.einsum("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


def test_einsum_path():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.einsum_path("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


def test_trapezoid():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.trapezoid(d.SMALL_INT)
        assert result is not None


def test_trapz():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = we.trapz(d.SMALL_INT)
        assert result is not None


def test_interp():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        result = we.interp(
            we.array([1.5, 2.5]),
            we.array([1.0, 2.0, 3.0]),
            we.array([10.0, 20.0, 30.0]),
        )
        assert result is not None


def test_linalg_svd():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        U, S, Vt = we.linalg.svd(d.MATRIX3x2)
        assert U is not None
        assert S is not None
        assert Vt is not None


# ---------------------------------------------------------------------------
# Free Ops: Creation
# ---------------------------------------------------------------------------


def test_array_creation():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.array([1.0, 2.0, 3.0]) is not None


def test_zeros():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.zeros((3, 3)) is not None


def test_ones():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.ones((3, 3)) is not None


def test_full():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.full((3, 3), 7.0) is not None


def test_empty():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.empty((3, 3)) is not None


def test_eye():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.eye(3) is not None


def test_identity():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.identity(3) is not None


def test_arange():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.arange(0, 10, 2) is not None


def test_linspace():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.linspace(0, 1, 5) is not None


def test_logspace():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.logspace(0, 2, 5) is not None


def test_geomspace():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.geomspace(1, 100, 5) is not None


def test_zeros_like():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.zeros_like(d.SMALL_INT) is not None


def test_ones_like():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.ones_like(d.SMALL_INT) is not None


def test_full_like():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.full_like(d.SMALL_INT, 9.0) is not None


def test_empty_like():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.empty_like(d.SMALL_INT) is not None


def test_diag():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.diag(d.VEC3) is not None


def test_diagflat():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.diagflat(d.VEC2) is not None


def test_tri():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.tri(3, 3) is not None


def test_tril():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.tril(d.MATRIX) is not None


def test_triu():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.triu(d.MATRIX) is not None


def test_vander():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.vander(d.PAIR_A, 3) is not None


# ---------------------------------------------------------------------------
# Free Ops: Manipulation
# ---------------------------------------------------------------------------


def test_reshape():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.reshape(d.SMALL_INT, (2, 2)) is not None


def test_transpose():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.transpose(d.MATRIX) is not None


def test_swapaxes():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.swapaxes(d.MATRIX, 0, 1) is not None


def test_moveaxis():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.moveaxis(d.MATRIX, 0, 1) is not None


def test_concatenate():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.concatenate([d.PAIR_A, d.PAIR_B]) is not None


def test_stack():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.stack([d.PAIR_A, d.PAIR_B]) is not None


def test_vstack():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.vstack([d.PAIR_A, d.PAIR_B]) is not None


def test_hstack():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.hstack([d.PAIR_A, d.PAIR_B]) is not None


def test_column_stack():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.column_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_row_stack():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.row_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_dstack():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.dstack([d.PAIR_A, d.PAIR_B]) is not None


def test_split():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.split(we.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_hsplit():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.hsplit(we.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_vsplit():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.vsplit(we.array([[1.0, 2.0], [3.0, 4.0]]), 2) is not None


def test_dsplit():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.dsplit(we.array([[[1.0, 2.0], [3.0, 4.0]]]), 2) is not None


def test_array_split():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.array_split(d.SMALL_INT, 2) is not None


def test_squeeze():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.squeeze(we.array([[[1.0, 2.0]]])) is not None


def test_expand_dims():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.expand_dims(d.PAIR_A, 0) is not None


def test_ravel():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.ravel(d.MATRIX) is not None


def test_copy():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.copy(d.PAIR_A) is not None


def test_flip():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.flip(d.PAIR_A) is not None


def test_fliplr():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.fliplr(d.MATRIX) is not None


def test_flipud():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.flipud(d.MATRIX) is not None


def test_rot90():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.rot90(d.MATRIX) is not None


def test_roll():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.roll(d.PAIR_A, 1) is not None


def test_rollaxis():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.rollaxis(d.MATRIX, 1) is not None


def test_tile():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.tile(d.PAIR_A, 2) is not None


def test_repeat():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.repeat(d.PAIR_A, 2) is not None


def test_resize():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.resize(d.PAIR_A, (2, 3)) is not None


def test_append():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.append(d.PAIR_A, d.PAIR_B) is not None


def test_insert():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.insert(d.PAIR_A, 1, 99.0) is not None


def test_delete():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.delete(d.PAIR_A, 1) is not None


def test_unique():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_trim_zeros():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.trim_zeros(d.ZERO_TRIMMED) is not None


def test_sort():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.sort(we.array([3.0, 1.0, 2.0])) is not None


def test_argsort():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.argsort(we.array([3.0, 1.0, 2.0])) is not None


def test_partition():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.partition(we.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_argpartition():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.argpartition(we.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_take():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.take(d.PAIR_A, [0, 2]) is not None


def test_take_along_axis():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.take_along_axis(
                we.array([[1.0, 2.0], [3.0, 4.0]]),
                we.array([[0, 1], [1, 0]], dtype="int64"),
                axis=1,
            )
            is not None
        )


def test_compress():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.compress([True, False, True], d.PAIR_A) is not None


def test_extract():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.extract(d.BOOL_ARR, d.PAIR_A) is not None


def test_diagonal():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.diagonal(d.MATRIX) is not None


def test_trace():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.trace(d.MATRIX) is not None


def test_pad():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.pad(d.PAIR_A, 1) is not None


def test_searchsorted():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.searchsorted(we.array([1.0, 3.0, 5.0]), 2.0) is not None


def test_where():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.where(d.BOOL_ARR, d.PAIR_A, d.PAIR_B) is not None


def test_select():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.select([d.BOOL_ARR], [d.PAIR_A]) is not None


def test_nonzero():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.nonzero(we.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_argwhere():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.argwhere(we.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_flatnonzero():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.flatnonzero(we.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_isin():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.isin(d.PAIR_A, [1.0, 3.0]) is not None


def test_in1d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.in1d(d.PAIR_A, we.array([1.0, 3.0])) is not None


def test_intersect1d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.intersect1d(d.PAIR_A, we.array([2.0, 3.0, 7.0])) is not None


def test_union1d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.union1d(d.PAIR_A, we.array([2.0, 5.0])) is not None


def test_setdiff1d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.setdiff1d(d.PAIR_A, we.array([2.0])) is not None


def test_setxor1d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.setxor1d(d.PAIR_A, we.array([2.0, 5.0])) is not None


def test_allclose():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.allclose(d.PAIR_A, d.PAIR_A) is not None


def test_array_equal():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.array_equal(d.PAIR_A, d.PAIR_A) is not None


def test_array_equiv():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.array_equiv(d.PAIR_A, d.PAIR_A) is not None


def test_isfinite():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.isfinite(d.SMALL_POS) is not None


def test_isinf():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.isinf(d.SMALL_POS) is not None


def test_isnan():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.isnan(d.SMALL_POS) is not None


def test_isscalar():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.isscalar(5.0) is not None


def test_shape():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.shape(d.MATRIX) is not None


def test_ndim():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.ndim(d.MATRIX) is not None


def test_size():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.size(d.MATRIX) is not None


def test_broadcast_to():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.broadcast_to(d.PAIR_A, (2, 3)) is not None


def test_broadcast_arrays():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.broadcast_arrays(d.PAIR_A, d.PAIR_B) is not None


def test_broadcast_shapes():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.broadcast_shapes((3,), (3,)) is not None


def test_histogram():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.histogram(d.SMALL_INT, 3) is not None


def test_histogram_bin_edges():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.histogram_bin_edges(d.SMALL_INT, 3) is not None


def test_histogram2d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.histogram2d(
                we.array([1.0, 2.0, 3.0]),
                we.array([4.0, 5.0, 6.0]),
                3,
            )
            is not None
        )


def test_histogramdd():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.histogramdd(
                we.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                3,
            )
            is not None
        )


def test_asarray():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.asarray(d.PAIR_A) is not None


def test_asarray_chkfinite():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.asarray_chkfinite(d.PAIR_A) is not None


def test_astype():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.astype(d.PAIR_A, "float32") is not None


def test_can_cast():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.can_cast("float32", "float64") is not None


def test_result_type():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.result_type("float32", "float64") is not None


def test_promote_types():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.promote_types("float32", "float64") is not None


def test_atleast_1d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.atleast_1d(d.PAIR_A) is not None


def test_atleast_2d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.atleast_2d(d.PAIR_A) is not None


def test_atleast_3d():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.atleast_3d(d.PAIR_A) is not None


def test_diag_indices():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.diag_indices(3) is not None


def test_diag_indices_from():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.diag_indices_from(d.MATRIX) is not None


def test_tril_indices():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.tril_indices(3) is not None


def test_triu_indices():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.triu_indices(3) is not None


def test_tril_indices_from():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.tril_indices_from(d.MATRIX) is not None


def test_triu_indices_from():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.triu_indices_from(d.MATRIX) is not None


def test_indices():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.indices((2, 3)) is not None


def test_ix_():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.ix_(
                we.array([0, 1], dtype="int64"),
                we.array([0, 1], dtype="int64"),
            )
            is not None
        )


def test_unravel_index():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unravel_index(5, (3, 3)) is not None


def test_ravel_multi_index():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.ravel_multi_index(
                (
                    we.array([0, 1, 2], dtype="int64"),
                    we.array([0, 1, 2], dtype="int64"),
                ),
                (3, 3),
            )
            is not None
        )


def test_mask_indices():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.mask_indices(3, we.triu) is not None


def test_meshgrid():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.meshgrid(we.array([1.0, 2.0]), we.array([3.0, 4.0])) is not None


def test_lexsort():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.lexsort((we.array([1.0, 2.0, 1.0]), we.array([3.0, 1.0, 2.0])))
            is not None
        )


def test_digitize():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.digitize(we.array([0.5, 1.5, 2.5]), we.array([1.0, 2.0, 3.0]))
            is not None
        )


def test_bincount():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.bincount(we.array([0, 1, 1, 2, 3, 3, 3], dtype="int64")) is not None


def test_packbits():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.packbits(we.array([1, 0, 1, 1, 0, 0, 0, 1], dtype="uint8")) is not None
        )


def test_unpackbits():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unpackbits(we.array([177], dtype="uint8")) is not None


def test_block():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.block([[d.PAIR_A, d.PAIR_B]]) is not None


def test_concat():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.concat([d.PAIR_A, d.PAIR_B]) is not None


def test_iterable():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.iterable(d.PAIR_A) is not None


def test_isfortran():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.isfortran(d.MATRIX) is not None


def test_typename():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.typename("float64") is not None


def test_mintypecode():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.mintypecode(["f", "d"]) is not None


def test_base_repr():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.base_repr(10, 2) is not None


def test_binary_repr():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.binary_repr(10) is not None


def test_put():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0, 4.0])
        we.put(arr, [0, 2], [99.0, 88.0])


def test_place():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0, 4.0])
        we.place(arr, we.array([True, False, True, False], dtype="bool"), [99.0, 88.0])


def test_putmask():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0, 4.0])
        we.putmask(arr, we.array([True, False, True, False], dtype="bool"), 0.0)


def test_fill_diagonal():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        mat = we.array([[1.0, 2.0], [3.0, 4.0]])
        we.fill_diagonal(mat, 0.0)


def test_copyto():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        arr = we.array([1.0, 2.0, 3.0])
        we.copyto(arr, we.array([9.0, 8.0, 7.0]))


def test_choose():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert (
            we.choose(
                we.array([0, 1, 0], dtype="int64"),
                [we.array([10.0, 20.0, 30.0]), we.array([40.0, 50.0, 60.0])],
            )
            is not None
        )


def test_require():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.require(d.PAIR_A) is not None


def test_matrix_transpose():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.matrix_transpose(we.array([[[1.0, 2.0], [3.0, 4.0]]])) is not None


def test_permute_dims():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.permute_dims(d.MATRIX, (1, 0)) is not None


def test_unique_all():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_all(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_counts():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_counts(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_inverse():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_inverse(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_values():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.unique_values(we.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unstack():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.unstack(d.MATRIX) is not None


def test_shares_memory():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.shares_memory(d.PAIR_A, d.PAIR_B) is not None


def test_may_share_memory():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.may_share_memory(d.PAIR_A, d.PAIR_B) is not None


def test_min_scalar_type():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.min_scalar_type(10) is not None


def test_issubdtype():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        assert we.issubdtype("float64", "float64") is not None


def test_common_type():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert we.common_type(d.PAIR_A) is not None


def test_put_along_axis():
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        we.put_along_axis(
            we.array([[1.0, 2.0], [3.0, 4.0]]),
            we.array([[0], [1]], dtype="int64"),
            we.array([[99.0], [88.0]]),
            axis=1,
        )


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

_RANDOM_SIMPLE = [
    ("random.rand", lambda me: we.random.rand(3, 3)),
    ("random.randn", lambda me: we.random.randn(3, 3)),
    ("random.random", lambda me: we.random.random((3,))),
    ("random.randint", lambda me: we.random.randint(0, 10, (3,))),
    ("random.uniform", lambda me: we.random.uniform(0.0, 1.0, (3,))),
    ("random.normal", lambda me: we.random.normal(0.0, 1.0, (3,))),
    ("random.seed", lambda me: we.random.seed(42)),
    ("random.choice", lambda me: we.random.choice(10, 3)),
    ("random.permutation", lambda me: we.random.permutation(5)),
    ("random.beta", lambda me: we.random.beta(2.0, 5.0, (3,))),
    ("random.binomial", lambda me: we.random.binomial(10, 0.5, (3,))),
    ("random.chisquare", lambda me: we.random.chisquare(2, (3,))),
    ("random.exponential", lambda me: we.random.exponential(1.0, (3,))),
    ("random.gamma", lambda me: we.random.gamma(2.0, 1.0, (3,))),
    ("random.geometric", lambda me: we.random.geometric(0.5, (3,))),
    ("random.gumbel", lambda me: we.random.gumbel(0.0, 1.0, (3,))),
    ("random.laplace", lambda me: we.random.laplace(0.0, 1.0, (3,))),
    ("random.logistic", lambda me: we.random.logistic(0.0, 1.0, (3,))),
    ("random.lognormal", lambda me: we.random.lognormal(0.0, 1.0, (3,))),
    ("random.logseries", lambda me: we.random.logseries(0.9, (3,))),
    ("random.multinomial", lambda me: we.random.multinomial(10, [0.5, 0.3, 0.2])),
    ("random.negative_binomial", lambda me: we.random.negative_binomial(5, 0.5, (3,))),
    (
        "random.noncentral_chisquare",
        lambda me: we.random.noncentral_chisquare(2, 1.0, (3,)),
    ),
    ("random.noncentral_f", lambda me: we.random.noncentral_f(5, 10, 1.0, (3,))),
    ("random.pareto", lambda me: we.random.pareto(2.0, (3,))),
    ("random.poisson", lambda me: we.random.poisson(5.0, (3,))),
    ("random.power", lambda me: we.random.power(2.0, (3,))),
    ("random.rayleigh", lambda me: we.random.rayleigh(1.0, (3,))),
    ("random.standard_cauchy", lambda me: we.random.standard_cauchy((3,))),
    ("random.standard_exponential", lambda me: we.random.standard_exponential((3,))),
    ("random.standard_gamma", lambda me: we.random.standard_gamma(2.0, (3,))),
    ("random.standard_normal", lambda me: we.random.standard_normal((3,))),
    ("random.standard_t", lambda me: we.random.standard_t(5.0, (3,))),
    ("random.triangular", lambda me: we.random.triangular(0.0, 0.5, 1.0, (3,))),
    ("random.vonmises", lambda me: we.random.vonmises(0.0, 1.0, (3,))),
    ("random.wald", lambda me: we.random.wald(1.0, 1.0, (3,))),
    ("random.weibull", lambda me: we.random.weibull(2.0, (3,))),
    ("random.zipf", lambda me: we.random.zipf(2.0, (3,))),
    ("random.dirichlet", lambda me: we.random.dirichlet([1.0, 1.0, 1.0])),
    (
        "random.multivariate_normal",
        lambda me: we.random.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
    ),
    ("random.f", lambda me: we.random.f(5, 10, (3,))),
    ("random.hypergeometric", lambda me: we.random.hypergeometric(10, 5, 7, (3,))),
    ("random.random_sample", lambda me: we.random.random_sample((3,))),
    ("random.ranf", lambda me: we.random.ranf((3,))),
    ("random.sample", lambda me: we.random.sample((3,))),
    ("random.get_state", lambda me: we.random.get_state()),
    (
        "random.shuffle",
        lambda me: we.random.shuffle(we.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    ),
]


@pytest.mark.parametrize(
    "name,call", _RANDOM_SIMPLE, ids=[n for n, _ in _RANDOM_SIMPLE]
)
def test_random(name, call):
    me = _import_me()
    with we.BudgetContext(flop_budget=10**9):
        call(me)
