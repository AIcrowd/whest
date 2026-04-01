"""Exhaustive smoke test: every non-blacklisted function via client-server proxy.

Starts a real MechestimServer in a subprocess, then calls every proxyable
function from the registry with valid arguments.

Run with:
    PYTHONPATH=mechestim-client/src:mechestim-server/src:src \
        .venv/bin/python -m pytest mechestim-client/tests/test_exhaustive.py -v --timeout=120
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
_CLIENT_SRC = os.path.join(_WORKTREE, "mechestim-client", "src")
_SERVER_SRC = os.path.join(_WORKTREE, "mechestim-server", "src")
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
from mechestim_server._server import MechestimServer
server = MechestimServer(url={_SERVER_URL!r})
print("SERVER_READY", flush=True)
server.run()
"""


@pytest.fixture(scope="session", autouse=True)
def _start_server():
    os.environ["MECHESTIM_SERVER_URL"] = _SERVER_URL
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
    import mechestim._budget as _bmod
    from mechestim._connection import reset_connection

    def _force_cleanup():
        """Force-reset client state and try to close server session."""
        _bmod._active_context = None
        # Always reset connection first (kills any stuck ZMQ socket)
        reset_connection()
        # Now create a fresh connection and try to close any server session
        try:
            from mechestim._connection import get_connection
            from mechestim._protocol import encode_budget_close
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
    import mechestim as me
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
        d.SMALL_POS = me.array([0.5, 1.0, 1.5, 2.0])
        d.SMALL_UNIT = me.array([0.1, 0.3, 0.5, 0.7])
        d.SMALL_NEG = me.array([-2.0, -1.0, 1.0, 2.0])
        d.SMALL_INT = me.array([1.0, 2.0, 3.0, 4.0])
        d.PAIR_A = me.array([1.0, 2.0, 3.0])
        d.PAIR_B = me.array([4.0, 5.0, 6.0])
        d.MATRIX = me.array([[1.0, 2.0], [3.0, 4.0]])
        d.MATRIX3x2 = me.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        d.VEC2 = me.array([1.0, 2.0])
        d.VEC3 = me.array([1.0, 2.0, 3.0])
        d.BOOL_ARR = me.greater(d.PAIR_A, me.array([2.0, 2.0, 2.0]))
        d.SMALL_GE1 = me.array([1.0, 1.5, 2.0, 3.0])
        d.INT_ARR = me.array([1, 2, 3, 4], dtype="int64")
        d.INT_PAIR_A = me.array([6, 12, 15], dtype="int64")
        d.INT_PAIR_B = me.array([4, 8, 10], dtype="int64")
        d.COMPLEX_ARR = me.array([1.0 + 2.0j, 3.0 - 1.0j], dtype="complex128")
        d.ZERO_TRIMMED = me.array([0.0, 1.0, 2.0, 0.0])
        return d


# ---------------------------------------------------------------------------
# Counted Unary
# ---------------------------------------------------------------------------

_UNARY_GENERAL = [
    "exp", "exp2", "expm1", "sqrt", "square", "cbrt",
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "arctan", "arcsinh",
    "sign", "ceil", "floor", "abs", "absolute", "fabs",
    "negative", "positive", "rint", "round", "around",
    "fix", "trunc", "deg2rad", "degrees", "rad2deg", "radians",
    "log", "log2", "log10", "log1p",
    "reciprocal", "signbit", "spacing",
    "sinc", "i0", "nan_to_num", "real", "imag",
    "conj", "conjugate", "iscomplex", "isreal",
    "real_if_close", "logical_not",
]


@pytest.mark.parametrize("op_name", _UNARY_GENERAL)
def test_unary_general(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_POS)
        assert result is not None


_UNARY_UNIT = ["arcsin", "arccos", "asin", "acos"]


@pytest.mark.parametrize("op_name", _UNARY_UNIT)
def test_unary_unit(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_ARCTANH = ["arctanh", "atanh", "atan"]


@pytest.mark.parametrize("op_name", _UNARY_ARCTANH)
def test_unary_arctanh(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_UNIT)
        assert result is not None


_UNARY_GE1 = ["acosh", "arccosh"]


@pytest.mark.parametrize("op_name", _UNARY_GE1)
def test_unary_ge1(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_GE1)
        assert result is not None


def test_modf():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.modf(d.SMALL_POS)
        assert result is not None


def test_frexp():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.frexp(d.SMALL_POS)
        assert result is not None


def test_sort_complex():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.sort_complex(d.COMPLEX_ARR)
        assert result is not None


def test_isclose():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.isclose(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_angle():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.angle(d.COMPLEX_ARR)
        assert result is not None


def test_iscomplexobj():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.iscomplexobj(d.SMALL_POS)
        assert result is not None


def test_isrealobj():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.isrealobj(d.SMALL_POS)
        assert result is not None


_BITWISE_UNARY = ["bitwise_invert", "bitwise_not", "bitwise_count", "invert"]


@pytest.mark.parametrize("op_name", _BITWISE_UNARY)
def test_bitwise_unary(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.INT_ARR)
        assert result is not None


def test_isneginf():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.isneginf(d.SMALL_NEG)
        assert result is not None


def test_isposinf():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.isposinf(d.SMALL_POS)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Binary
# ---------------------------------------------------------------------------

_BINARY_OPS = [
    "add", "subtract", "multiply", "divide", "true_divide",
    "floor_divide", "power", "pow", "float_power",
    "mod", "remainder", "fmod",
    "maximum", "minimum", "fmax", "fmin",
    "greater", "greater_equal", "less", "less_equal",
    "equal", "not_equal",
    "logical_and", "logical_or", "logical_xor",
    "logaddexp", "logaddexp2",
    "arctan2", "atan2", "hypot", "copysign",
    "nextafter", "heaviside",
]


@pytest.mark.parametrize("op_name", _BINARY_OPS)
def test_binary(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_ldexp():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.ldexp(d.PAIR_A, me.array([1, 2, 3], dtype="int64"))
        assert result is not None


_BITWISE_BINARY = [
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_left_shift", "bitwise_right_shift",
    "left_shift", "right_shift",
]


@pytest.mark.parametrize("op_name", _BITWISE_BINARY)
def test_bitwise_binary(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.INT_PAIR_A, me.array([1, 2, 1], dtype="int64"))
        assert result is not None


def test_gcd():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.gcd(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_lcm():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.lcm(d.INT_PAIR_A, d.INT_PAIR_B)
        assert result is not None


def test_divmod():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.divmod(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vecdot():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.vecdot(d.PAIR_A, d.PAIR_B)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Reduction
# ---------------------------------------------------------------------------

_REDUCTION_OPS = [
    "sum", "prod", "mean", "std", "var", "max", "min",
    "amax", "amin", "all", "any",
    "argmax", "argmin", "cumsum", "cumprod",
    "count_nonzero", "median",
    "nansum", "nanprod", "nanmean", "nanstd", "nanvar",
    "nanmax", "nanmin", "nanmedian",
    "nanargmax", "nanargmin",
    "nancumprod", "nancumsum",
    "ptp",
]


@pytest.mark.parametrize("op_name", _REDUCTION_OPS)
def test_reduction(op_name):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        fn = getattr(me, op_name)
        result = fn(d.SMALL_INT)
        assert result is not None


def test_average():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.average(d.SMALL_INT)
        assert result is not None


def test_percentile():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        # percentile server-side is wrapped as reduction(a, axis=None, **kw)
        # so q must be passed as keyword
        result = me.percentile(d.SMALL_INT, q=50)
        assert result is not None


def test_quantile():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.quantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_nanpercentile():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.nanpercentile(d.SMALL_INT, q=50)
        assert result is not None


def test_nanquantile():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.nanquantile(d.SMALL_INT, q=0.5)
        assert result is not None


def test_cumulative_sum():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.cumulative_sum(d.SMALL_INT, axis=0)
        assert result is not None


def test_cumulative_prod():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.cumulative_prod(d.SMALL_INT, axis=0)
        assert result is not None


# ---------------------------------------------------------------------------
# Counted Custom
# ---------------------------------------------------------------------------

def test_clip():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.clip(d.SMALL_INT, 1.5, 3.5)
        assert result is not None


def test_dot():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.dot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_matmul():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.matmul(d.MATRIX, d.VEC2)
        assert result is not None


def test_inner():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.inner(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_outer():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.outer(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_vdot():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.vdot(d.PAIR_A, d.PAIR_B)
        assert result is not None


def test_tensordot():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.tensordot(d.MATRIX, d.MATRIX, 1)
        assert result is not None


def test_kron():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.kron(d.VEC2, d.VEC3)
        assert result is not None


def test_cross():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.cross(d.VEC3, me.array([4.0, 5.0, 6.0]))
        assert result is not None


def test_diff():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.diff(d.SMALL_INT)
        assert result is not None


def test_ediff1d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.ediff1d(d.SMALL_INT)
        assert result is not None


def test_gradient():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.gradient(d.SMALL_INT)
        assert result is not None


def test_convolve():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.convolve(d.PAIR_A, d.VEC2)
        assert result is not None


def test_correlate():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.correlate(d.PAIR_A, d.VEC2)
        assert result is not None


def test_corrcoef():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.corrcoef(d.SMALL_INT)
        assert result is not None


def test_cov():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.cov(d.SMALL_INT)
        assert result is not None


def test_einsum():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.einsum("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


def test_einsum_path():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.einsum_path("ij,jk->ik", d.MATRIX, d.MATRIX)
        assert result is not None


def test_trapezoid():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.trapezoid(d.SMALL_INT)
        assert result is not None


def test_trapz():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        result = me.trapz(d.SMALL_INT)
        assert result is not None


def test_interp():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        result = me.interp(
            me.array([1.5, 2.5]),
            me.array([1.0, 2.0, 3.0]),
            me.array([10.0, 20.0, 30.0]),
        )
        assert result is not None


def test_linalg_svd():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        U, S, Vt = me.linalg.svd(d.MATRIX3x2)
        assert U is not None
        assert S is not None
        assert Vt is not None


# ---------------------------------------------------------------------------
# Free Ops: Creation
# ---------------------------------------------------------------------------

def test_array_creation():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.array([1.0, 2.0, 3.0]) is not None


def test_zeros():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.zeros((3, 3)) is not None


def test_ones():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.ones((3, 3)) is not None


def test_full():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.full((3, 3), 7.0) is not None


def test_empty():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.empty((3, 3)) is not None


def test_eye():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.eye(3) is not None


def test_identity():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.identity(3) is not None


def test_arange():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.arange(0, 10, 2) is not None


def test_linspace():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.linspace(0, 1, 5) is not None


def test_logspace():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.logspace(0, 2, 5) is not None


def test_geomspace():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.geomspace(1, 100, 5) is not None


def test_zeros_like():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.zeros_like(d.SMALL_INT) is not None


def test_ones_like():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.ones_like(d.SMALL_INT) is not None


def test_full_like():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.full_like(d.SMALL_INT, 9.0) is not None


def test_empty_like():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.empty_like(d.SMALL_INT) is not None


def test_diag():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.diag(d.VEC3) is not None


def test_diagflat():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.diagflat(d.VEC2) is not None


def test_tri():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.tri(3, 3) is not None


def test_tril():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.tril(d.MATRIX) is not None


def test_triu():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.triu(d.MATRIX) is not None


def test_vander():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.vander(d.PAIR_A, 3) is not None


# ---------------------------------------------------------------------------
# Free Ops: Manipulation
# ---------------------------------------------------------------------------

def test_reshape():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.reshape(d.SMALL_INT, (2, 2)) is not None


def test_transpose():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.transpose(d.MATRIX) is not None


def test_swapaxes():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.swapaxes(d.MATRIX, 0, 1) is not None


def test_moveaxis():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.moveaxis(d.MATRIX, 0, 1) is not None


def test_concatenate():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.concatenate([d.PAIR_A, d.PAIR_B]) is not None


def test_stack():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.stack([d.PAIR_A, d.PAIR_B]) is not None


def test_vstack():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.vstack([d.PAIR_A, d.PAIR_B]) is not None


def test_hstack():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.hstack([d.PAIR_A, d.PAIR_B]) is not None


def test_column_stack():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.column_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_row_stack():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.row_stack([d.PAIR_A, d.PAIR_B]) is not None


def test_dstack():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.dstack([d.PAIR_A, d.PAIR_B]) is not None


def test_split():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.split(me.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_hsplit():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.hsplit(me.array([1.0, 2.0, 3.0, 4.0]), 2) is not None


def test_vsplit():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.vsplit(me.array([[1.0, 2.0], [3.0, 4.0]]), 2) is not None


def test_dsplit():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.dsplit(me.array([[[1.0, 2.0], [3.0, 4.0]]]), 2) is not None


def test_array_split():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.array_split(d.SMALL_INT, 2) is not None


def test_squeeze():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.squeeze(me.array([[[1.0, 2.0]]])) is not None


def test_expand_dims():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.expand_dims(d.PAIR_A, 0) is not None


def test_ravel():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.ravel(d.MATRIX) is not None


def test_copy():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.copy(d.PAIR_A) is not None


def test_flip():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.flip(d.PAIR_A) is not None


def test_fliplr():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.fliplr(d.MATRIX) is not None


def test_flipud():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.flipud(d.MATRIX) is not None


def test_rot90():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.rot90(d.MATRIX) is not None


def test_roll():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.roll(d.PAIR_A, 1) is not None


def test_rollaxis():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.rollaxis(d.MATRIX, 1) is not None


def test_tile():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.tile(d.PAIR_A, 2) is not None


def test_repeat():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.repeat(d.PAIR_A, 2) is not None


def test_resize():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.resize(d.PAIR_A, (2, 3)) is not None


def test_append():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.append(d.PAIR_A, d.PAIR_B) is not None


def test_insert():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.insert(d.PAIR_A, 1, 99.0) is not None


def test_delete():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.delete(d.PAIR_A, 1) is not None


def test_unique():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.unique(me.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_trim_zeros():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.trim_zeros(d.ZERO_TRIMMED) is not None


def test_sort():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.sort(me.array([3.0, 1.0, 2.0])) is not None


def test_argsort():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.argsort(me.array([3.0, 1.0, 2.0])) is not None


def test_partition():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.partition(me.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_argpartition():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.argpartition(me.array([3.0, 1.0, 2.0, 4.0]), 2) is not None


def test_take():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.take(d.PAIR_A, [0, 2]) is not None


def test_take_along_axis():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.take_along_axis(
            me.array([[1.0, 2.0], [3.0, 4.0]]),
            me.array([[0, 1], [1, 0]], dtype="int64"),
            axis=1,
        ) is not None


def test_compress():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.compress([True, False, True], d.PAIR_A) is not None


def test_extract():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.extract(d.BOOL_ARR, d.PAIR_A) is not None


def test_diagonal():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.diagonal(d.MATRIX) is not None


def test_trace():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.trace(d.MATRIX) is not None


def test_pad():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.pad(d.PAIR_A, 1) is not None


def test_searchsorted():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.searchsorted(me.array([1.0, 3.0, 5.0]), 2.0) is not None


def test_where():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.where(d.BOOL_ARR, d.PAIR_A, d.PAIR_B) is not None


def test_select():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.select([d.BOOL_ARR], [d.PAIR_A]) is not None


def test_nonzero():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.nonzero(me.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_argwhere():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.argwhere(me.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_flatnonzero():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.flatnonzero(me.array([0.0, 1.0, 0.0, 2.0])) is not None


def test_isin():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.isin(d.PAIR_A, [1.0, 3.0]) is not None


def test_in1d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.in1d(d.PAIR_A, me.array([1.0, 3.0])) is not None


def test_intersect1d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.intersect1d(d.PAIR_A, me.array([2.0, 3.0, 7.0])) is not None


def test_union1d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.union1d(d.PAIR_A, me.array([2.0, 5.0])) is not None


def test_setdiff1d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.setdiff1d(d.PAIR_A, me.array([2.0])) is not None


def test_setxor1d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.setxor1d(d.PAIR_A, me.array([2.0, 5.0])) is not None


def test_allclose():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.allclose(d.PAIR_A, d.PAIR_A) is not None


def test_array_equal():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.array_equal(d.PAIR_A, d.PAIR_A) is not None


def test_array_equiv():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.array_equiv(d.PAIR_A, d.PAIR_A) is not None


def test_isfinite():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.isfinite(d.SMALL_POS) is not None


def test_isinf():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.isinf(d.SMALL_POS) is not None


def test_isnan():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.isnan(d.SMALL_POS) is not None


def test_isscalar():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.isscalar(5.0) is not None


def test_shape():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.shape(d.MATRIX) is not None


def test_ndim():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.ndim(d.MATRIX) is not None


def test_size():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.size(d.MATRIX) is not None


def test_broadcast_to():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.broadcast_to(d.PAIR_A, (2, 3)) is not None


def test_broadcast_arrays():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.broadcast_arrays(d.PAIR_A, d.PAIR_B) is not None


def test_broadcast_shapes():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.broadcast_shapes((3,), (3,)) is not None


def test_histogram():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.histogram(d.SMALL_INT, 3) is not None


def test_histogram_bin_edges():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.histogram_bin_edges(d.SMALL_INT, 3) is not None


def test_histogram2d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.histogram2d(
            me.array([1.0, 2.0, 3.0]),
            me.array([4.0, 5.0, 6.0]),
            3,
        ) is not None


def test_histogramdd():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.histogramdd(
            me.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            3,
        ) is not None


def test_asarray():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.asarray(d.PAIR_A) is not None


def test_asarray_chkfinite():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.asarray_chkfinite(d.PAIR_A) is not None


def test_astype():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.astype(d.PAIR_A, "float32") is not None


def test_can_cast():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.can_cast("float32", "float64") is not None


def test_result_type():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.result_type("float32", "float64") is not None


def test_promote_types():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.promote_types("float32", "float64") is not None


def test_atleast_1d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.atleast_1d(d.PAIR_A) is not None


def test_atleast_2d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.atleast_2d(d.PAIR_A) is not None


def test_atleast_3d():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.atleast_3d(d.PAIR_A) is not None


def test_diag_indices():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.diag_indices(3) is not None


def test_diag_indices_from():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.diag_indices_from(d.MATRIX) is not None


def test_tril_indices():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.tril_indices(3) is not None


def test_triu_indices():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.triu_indices(3) is not None


def test_tril_indices_from():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.tril_indices_from(d.MATRIX) is not None


def test_triu_indices_from():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.triu_indices_from(d.MATRIX) is not None


def test_indices():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.indices((2, 3)) is not None


def test_ix_():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.ix_(
            me.array([0, 1], dtype="int64"),
            me.array([0, 1], dtype="int64"),
        ) is not None


def test_unravel_index():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.unravel_index(5, (3, 3)) is not None


def test_ravel_multi_index():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.ravel_multi_index(
            (me.array([0, 1, 2], dtype="int64"), me.array([0, 1, 2], dtype="int64")),
            (3, 3),
        ) is not None


def test_mask_indices():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.mask_indices(3, me.triu) is not None


def test_meshgrid():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.meshgrid(me.array([1.0, 2.0]), me.array([3.0, 4.0])) is not None


def test_lexsort():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.lexsort((me.array([1.0, 2.0, 1.0]), me.array([3.0, 1.0, 2.0]))) is not None


def test_digitize():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.digitize(me.array([0.5, 1.5, 2.5]), me.array([1.0, 2.0, 3.0])) is not None


def test_bincount():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.bincount(me.array([0, 1, 1, 2, 3, 3, 3], dtype="int64")) is not None


def test_packbits():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.packbits(me.array([1, 0, 1, 1, 0, 0, 0, 1], dtype="uint8")) is not None


def test_unpackbits():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.unpackbits(me.array([177], dtype="uint8")) is not None


def test_block():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.block([[d.PAIR_A, d.PAIR_B]]) is not None


def test_concat():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.concat([d.PAIR_A, d.PAIR_B]) is not None


def test_iterable():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.iterable(d.PAIR_A) is not None


def test_isfortran():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.isfortran(d.MATRIX) is not None


def test_typename():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.typename("float64") is not None


def test_mintypecode():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.mintypecode(["f", "d"]) is not None


def test_base_repr():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.base_repr(10, 2) is not None


def test_binary_repr():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.binary_repr(10) is not None


def test_put():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        arr = me.array([1.0, 2.0, 3.0, 4.0])
        me.put(arr, [0, 2], [99.0, 88.0])


def test_place():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        arr = me.array([1.0, 2.0, 3.0, 4.0])
        me.place(arr, me.array([True, False, True, False], dtype="bool"), [99.0, 88.0])


def test_putmask():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        arr = me.array([1.0, 2.0, 3.0, 4.0])
        me.putmask(arr, me.array([True, False, True, False], dtype="bool"), 0.0)


def test_fill_diagonal():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        mat = me.array([[1.0, 2.0], [3.0, 4.0]])
        me.fill_diagonal(mat, 0.0)


def test_copyto():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        arr = me.array([1.0, 2.0, 3.0])
        me.copyto(arr, me.array([9.0, 8.0, 7.0]))


def test_choose():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.choose(
            me.array([0, 1, 0], dtype="int64"),
            [me.array([10.0, 20.0, 30.0]), me.array([40.0, 50.0, 60.0])],
        ) is not None


def test_require():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.require(d.PAIR_A) is not None


def test_matrix_transpose():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.matrix_transpose(me.array([[[1.0, 2.0], [3.0, 4.0]]])) is not None


def test_permute_dims():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.permute_dims(d.MATRIX, (1, 0)) is not None


def test_unique_all():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.unique_all(me.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_counts():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.unique_counts(me.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_inverse():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.unique_inverse(me.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unique_values():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.unique_values(me.array([3.0, 1.0, 2.0, 1.0])) is not None


def test_unstack():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.unstack(d.MATRIX) is not None


def test_shares_memory():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.shares_memory(d.PAIR_A, d.PAIR_B) is not None


def test_may_share_memory():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.may_share_memory(d.PAIR_A, d.PAIR_B) is not None


def test_min_scalar_type():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.min_scalar_type(10) is not None


def test_issubdtype():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        assert me.issubdtype("float64", "float64") is not None


def test_common_type():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        d = _TestData.get(me)
        assert me.common_type(d.PAIR_A) is not None


def test_put_along_axis():
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        me.put_along_axis(
            me.array([[1.0, 2.0], [3.0, 4.0]]),
            me.array([[0], [1]], dtype="int64"),
            me.array([[99.0], [88.0]]),
            axis=1,
        )


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

_RANDOM_SIMPLE = [
    ("random.rand", lambda me: me.random.rand(3, 3)),
    ("random.randn", lambda me: me.random.randn(3, 3)),
    ("random.random", lambda me: me.random.random((3,))),
    ("random.randint", lambda me: me.random.randint(0, 10, (3,))),
    ("random.uniform", lambda me: me.random.uniform(0.0, 1.0, (3,))),
    ("random.normal", lambda me: me.random.normal(0.0, 1.0, (3,))),
    ("random.seed", lambda me: me.random.seed(42)),
    ("random.choice", lambda me: me.random.choice(10, 3)),
    ("random.permutation", lambda me: me.random.permutation(5)),
    ("random.beta", lambda me: me.random.beta(2.0, 5.0, (3,))),
    ("random.binomial", lambda me: me.random.binomial(10, 0.5, (3,))),
    ("random.chisquare", lambda me: me.random.chisquare(2, (3,))),
    ("random.exponential", lambda me: me.random.exponential(1.0, (3,))),
    ("random.gamma", lambda me: me.random.gamma(2.0, 1.0, (3,))),
    ("random.geometric", lambda me: me.random.geometric(0.5, (3,))),
    ("random.gumbel", lambda me: me.random.gumbel(0.0, 1.0, (3,))),
    ("random.laplace", lambda me: me.random.laplace(0.0, 1.0, (3,))),
    ("random.logistic", lambda me: me.random.logistic(0.0, 1.0, (3,))),
    ("random.lognormal", lambda me: me.random.lognormal(0.0, 1.0, (3,))),
    ("random.logseries", lambda me: me.random.logseries(0.9, (3,))),
    ("random.multinomial", lambda me: me.random.multinomial(10, [0.5, 0.3, 0.2])),
    ("random.negative_binomial", lambda me: me.random.negative_binomial(5, 0.5, (3,))),
    ("random.noncentral_chisquare", lambda me: me.random.noncentral_chisquare(2, 1.0, (3,))),
    ("random.noncentral_f", lambda me: me.random.noncentral_f(5, 10, 1.0, (3,))),
    ("random.pareto", lambda me: me.random.pareto(2.0, (3,))),
    ("random.poisson", lambda me: me.random.poisson(5.0, (3,))),
    ("random.power", lambda me: me.random.power(2.0, (3,))),
    ("random.rayleigh", lambda me: me.random.rayleigh(1.0, (3,))),
    ("random.standard_cauchy", lambda me: me.random.standard_cauchy((3,))),
    ("random.standard_exponential", lambda me: me.random.standard_exponential((3,))),
    ("random.standard_gamma", lambda me: me.random.standard_gamma(2.0, (3,))),
    ("random.standard_normal", lambda me: me.random.standard_normal((3,))),
    ("random.standard_t", lambda me: me.random.standard_t(5.0, (3,))),
    ("random.triangular", lambda me: me.random.triangular(0.0, 0.5, 1.0, (3,))),
    ("random.vonmises", lambda me: me.random.vonmises(0.0, 1.0, (3,))),
    ("random.wald", lambda me: me.random.wald(1.0, 1.0, (3,))),
    ("random.weibull", lambda me: me.random.weibull(2.0, (3,))),
    ("random.zipf", lambda me: me.random.zipf(2.0, (3,))),
    ("random.dirichlet", lambda me: me.random.dirichlet([1.0, 1.0, 1.0])),
    ("random.multivariate_normal", lambda me: me.random.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])),
    ("random.f", lambda me: me.random.f(5, 10, (3,))),
    ("random.hypergeometric", lambda me: me.random.hypergeometric(10, 5, 7, (3,))),
    ("random.random_sample", lambda me: me.random.random_sample((3,))),
    ("random.ranf", lambda me: me.random.ranf((3,))),
    ("random.sample", lambda me: me.random.sample((3,))),
    ("random.get_state", lambda me: me.random.get_state()),
    ("random.shuffle", lambda me: me.random.shuffle(me.array([1.0, 2.0, 3.0, 4.0, 5.0]))),
]


@pytest.mark.parametrize("name,call", _RANDOM_SIMPLE, ids=[n for n, _ in _RANDOM_SIMPLE])
def test_random(name, call):
    me = _import_me()
    with me.BudgetContext(flop_budget=10**9):
        call(me)
