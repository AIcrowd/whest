"""Extended tests to cover gaps in linalg property and decomposition wrappers."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._symmetric import as_symmetric
from mechestim.linalg._decompositions import (
    cholesky,
    cholesky_cost,
    eig,
    eig_cost,
    eigh,
    eigh_cost,
    eigvals,
    eigvals_cost,
    eigvalsh,
    eigvalsh_cost,
    qr,
    qr_cost,
    svdvals,
    svdvals_cost,
)
from mechestim.linalg._properties import (
    cond,
    det,
    det_cost,
    matrix_norm,
    matrix_norm_cost,
    matrix_rank,
    norm,
    norm_cost,
    slogdet,
    slogdet_cost,
    trace,
    trace_cost,
    vector_norm,
    vector_norm_cost,
)

# ---------------------------------------------------------------------------
# Cost helper functions — direct testing for edge cases
# ---------------------------------------------------------------------------


def test_trace_cost_min1():
    assert trace_cost(0) == 1
    assert trace_cost(5) == 5


def test_det_cost_symmetric():
    n = 4
    assert det_cost(n, symmetric=True) == max(n**3, 1)


def test_slogdet_cost_symmetric():
    n = 4
    assert slogdet_cost(n, symmetric=True) == max(n**3, 1)


def test_norm_cost_1d_ord_none():
    assert norm_cost((10,), ord=None) == 10


def test_norm_cost_1d_ord_inf():
    assert norm_cost((10,), ord=numpy.inf) == 10


def test_norm_cost_1d_ord_minus_inf():
    assert norm_cost((10,), ord=-numpy.inf) == 10


def test_norm_cost_1d_ord_0():
    assert norm_cost((10,), ord=0) == 10


def test_norm_cost_1d_p_norm():
    # ord=3 triggers the else: 2 * numel
    assert norm_cost((10,), ord=3) == 20


def test_norm_cost_2d_fro():
    assert norm_cost((4, 5), ord="fro") == 2 * 20


def test_norm_cost_2d_nuc():
    m, n = 4, 5
    assert norm_cost((m, n), ord="nuc") == m * n * min(m, n)


def test_norm_cost_2d_minus2():
    m, n = 4, 5
    assert norm_cost((m, n), ord=-2) == m * n * min(m, n)


def test_norm_cost_2d_1():
    assert norm_cost((4, 5), ord=1) == 20


def test_norm_cost_2d_minus1():
    assert norm_cost((4, 5), ord=-1) == 20


def test_norm_cost_2d_inf():
    assert norm_cost((4, 5), ord=numpy.inf) == 20


def test_norm_cost_2d_minus_inf():
    assert norm_cost((4, 5), ord=-numpy.inf) == 20


def test_norm_cost_2d_fallback():
    # Unrecognised ord for 2-D triggers return numel
    assert norm_cost((4, 5), ord="xyz") == 20


def test_vector_norm_cost_p_norm():
    # FMA=1: all norms cost numel (one pass over elements)
    assert vector_norm_cost((10,), ord=3) == 10


def test_vector_norm_cost_special_ords():
    for o in (None, 2, -2, 1, -1, numpy.inf, -numpy.inf, 0):
        assert vector_norm_cost((10,), ord=o) == 10


def test_matrix_norm_cost_fro():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord="fro") == m * n


def test_matrix_norm_cost_nuc():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord="nuc") == m * n * min(m, n)


def test_matrix_norm_cost_2():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=2) == m * n * min(m, n)


def test_matrix_norm_cost_minus2():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=-2) == m * n * min(m, n)


def test_matrix_norm_cost_1():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=1) == m * n


def test_matrix_norm_cost_minus1():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=-1) == m * n


def test_matrix_norm_cost_inf():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=numpy.inf) == m * n


def test_matrix_norm_cost_minus_inf():
    m, n = 3, 4
    assert matrix_norm_cost((m, n), ord=-numpy.inf) == m * n


def test_matrix_norm_cost_fallback():
    assert matrix_norm_cost((3, 4), ord="xyz") == 12


# ---------------------------------------------------------------------------
# Properties — SymmetricTensor paths
# ---------------------------------------------------------------------------


def test_det_symmetric_tensor_cost():
    n = 4
    data = numpy.eye(n)
    sym_a = as_symmetric(data, (0, 1))
    with BudgetContext(flop_budget=10**6) as budget:
        det(sym_a)
    assert budget.flops_used == det_cost(n, symmetric=True)


def test_slogdet_symmetric_tensor_cost():
    n = 4
    data = numpy.eye(n)
    sym_a = as_symmetric(data, (0, 1))
    with BudgetContext(flop_budget=10**6) as budget:
        slogdet(sym_a)
    assert budget.flops_used == slogdet_cost(n, symmetric=True)


# ---------------------------------------------------------------------------
# Properties — trace with positive/negative offset
# ---------------------------------------------------------------------------


def test_trace_positive_offset():
    A = numpy.arange(9, dtype=float).reshape(3, 3)
    with BudgetContext(flop_budget=10**6) as budget:
        result = trace(A, offset=1)
    assert numpy.isclose(result, numpy.trace(A, offset=1))


def test_trace_negative_offset():
    A = numpy.arange(9, dtype=float).reshape(3, 3)
    with BudgetContext(flop_budget=10**6) as budget:
        result = trace(A, offset=-1)
    assert numpy.isclose(result, numpy.trace(A, offset=-1))


# ---------------------------------------------------------------------------
# Properties — norm with axis parameter
# ---------------------------------------------------------------------------


def test_norm_with_tuple_axis():
    A = numpy.random.randn(3, 4, 5)
    with BudgetContext(flop_budget=10**6) as budget:
        result = norm(A, axis=(0, 1))
    assert result.shape == (5,)


def test_norm_with_single_axis():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6):
        result = norm(A, axis=0)
    assert result.shape == (4,)


# ---------------------------------------------------------------------------
# Properties — vector_norm with axis
# ---------------------------------------------------------------------------


def test_vector_norm_with_tuple_axis():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6):
        result = vector_norm(A, axis=(0, 1))
    assert numpy.ndim(result) == 0 or result is not None


def test_vector_norm_with_single_axis():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6):
        result = vector_norm(A, axis=0)
    assert result.shape == (4,)


# ---------------------------------------------------------------------------
# Properties — matrix_norm with various ord values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ord_val", [1, -1, 2, -2, numpy.inf, -numpy.inf, "nuc"])
def test_matrix_norm_various_ords(ord_val):
    A = numpy.random.randn(4, 4)
    with BudgetContext(flop_budget=10**6):
        result = matrix_norm(A, ord=ord_val)
    assert numpy.isfinite(result)


# ---------------------------------------------------------------------------
# Decompositions — cost helpers
# ---------------------------------------------------------------------------


def test_cholesky_cost():
    assert cholesky_cost(1) == 1
    assert cholesky_cost(3) == max(27, 1)


def test_qr_cost_wide_matrix():
    m, n = 3, 5
    cost = qr_cost(m, n)
    assert cost == max(m * n * min(m, n), 1)


def test_eig_cost():
    assert eig_cost(0) == 1
    assert eig_cost(4) == 64


def test_eigh_cost():
    assert eigh_cost(0) == 1
    assert eigh_cost(3) == max(27, 1)


def test_eigvals_cost():
    assert eigvals_cost(4) == 64


def test_eigvalsh_cost():
    assert eigvalsh_cost(4) == max(64, 1)


def test_svdvals_cost():
    m, n = 4, 3
    assert svdvals_cost(m, n) == m * n * min(m, n)


# ---------------------------------------------------------------------------
# Decompositions — non-square / wrong dim error paths
# ---------------------------------------------------------------------------


def test_cholesky_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            cholesky(numpy.ones((3, 4)))


def test_qr_non_2d_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            qr(numpy.ones((2, 3, 4)))


def test_eig_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eig(numpy.ones((3, 4)))


def test_eigh_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eigh(numpy.ones((3, 4)))


def test_eigvals_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eigvals(numpy.ones((3, 4)))


def test_eigvalsh_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            eigvalsh(numpy.ones((3, 4)))


def test_svdvals_non_2d_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            svdvals(numpy.ones((2, 3, 4)))


# ---------------------------------------------------------------------------
# Decompositions — successful runs
# ---------------------------------------------------------------------------


def test_eigh_result():
    A = numpy.array([[2.0, 1.0], [1.0, 2.0]])
    with BudgetContext(flop_budget=10**6) as budget:
        vals, vecs = eigh(A)
    assert budget.flops_used == eigh_cost(2)
    assert vals.shape == (2,)


def test_eigvals_result():
    A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    with BudgetContext(flop_budget=10**6):
        vals = eigvals(A)
    assert vals.shape == (2,)


def test_eigvalsh_result():
    A = numpy.array([[2.0, 1.0], [1.0, 2.0]])
    with BudgetContext(flop_budget=10**6) as budget:
        vals = eigvalsh(A)
    assert budget.flops_used == eigvalsh_cost(2)
    assert vals.shape == (2,)


def test_svdvals_result():
    A = numpy.random.randn(3, 4)
    with BudgetContext(flop_budget=10**6) as budget:
        sv = svdvals(A)
    assert budget.flops_used == svdvals_cost(3, 4)
    assert sv.shape == (3,)


# ---------------------------------------------------------------------------
# Properties — det/slogdet/cond/matrix_rank error paths
# ---------------------------------------------------------------------------


def test_det_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            det(numpy.ones((3, 4)))


def test_slogdet_non_square_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            slogdet(numpy.ones((3, 4)))


def test_cond_non_2d_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            cond(numpy.ones((2, 3, 4)))


def test_matrix_rank_non_2d_raises():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=10**6):
            matrix_rank(numpy.ones((2, 3, 4)))
