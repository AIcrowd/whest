# tests/test_linalg_decompositions.py
"""Tests for linalg decomposition wrappers with FLOP counting."""

import numpy

from whest._budget import BudgetContext


class TestCholesky:
    def test_result_matches_numpy(self):
        A = numpy.array([[4.0, 2.0], [2.0, 3.0]])
        with BudgetContext(flop_budget=10**6):
            from whest.linalg import cholesky

            result = cholesky(A)
            assert numpy.allclose(result, numpy.linalg.cholesky(A))

    def test_cost(self):
        n = 10
        A = numpy.random.randn(n, n)
        A = A @ A.T + numpy.eye(n) * 10
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import cholesky

            cholesky(A)
            assert budget.flops_used == n**3

    def test_op_log(self):
        A = numpy.eye(3) * 10
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import cholesky

            cholesky(A)
            assert budget.op_log[-1].op_name == "linalg.cholesky"

    def test_outside_context_uses_global_default(self):
        from whest.linalg import cholesky

        # Operations now auto-activate the global default budget instead of raising
        result = cholesky(numpy.eye(3))
        assert result.shape == (3, 3)


class TestQR:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(6, 4)
        with BudgetContext(flop_budget=10**6):
            from whest.linalg import qr

            Q, R = qr(A)
            Q_np, R_np = numpy.linalg.qr(A)
            assert numpy.allclose(numpy.abs(Q), numpy.abs(Q_np))
            assert numpy.allclose(numpy.abs(R), numpy.abs(R_np))

    def test_cost(self):
        m, n = 6, 4
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import qr

            qr(A)
            expected = m * n * min(m, n)
            assert budget.flops_used == expected

    def test_op_log(self):
        A = numpy.random.randn(4, 3)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import qr

            qr(A)
            assert budget.op_log[-1].op_name == "linalg.qr"


class TestEig:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from whest.linalg import eig

            w, v = eig(A)
            w_np, v_np = numpy.linalg.eig(A)
            assert numpy.allclose(sorted(numpy.abs(w)), sorted(numpy.abs(w_np)))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import eig

            eig(A)
            assert budget.flops_used == n**3


class TestEigh:
    def test_result_matches_numpy(self):
        A = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        with BudgetContext(flop_budget=10**6):
            from whest.linalg import eigh

            w, v = eigh(A)
            w_np, v_np = numpy.linalg.eigh(A)
            assert numpy.allclose(w, w_np)

    def test_cost(self):
        n = 6
        A = numpy.random.randn(n, n)
        A = A + A.T
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import eigh

            eigh(A)
            assert budget.flops_used == n**3


class TestEigvals:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from whest.linalg import eigvals

            w = eigvals(A)
            w_np = numpy.linalg.eigvals(A)
            assert numpy.allclose(sorted(numpy.abs(w)), sorted(numpy.abs(w_np)))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import eigvals

            eigvals(A)
            assert budget.flops_used == n**3


class TestEigvalsh:
    def test_cost(self):
        n = 6
        A = numpy.random.randn(n, n)
        A = A + A.T
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import eigvalsh

            eigvalsh(A)
            assert budget.flops_used == n**3


class TestSvdvals:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(6, 4)
        with BudgetContext(flop_budget=10**6):
            from whest.linalg import svdvals

            s = svdvals(A)
            s_np = numpy.linalg.svdvals(A)
            assert numpy.allclose(s, s_np)

    def test_cost(self):
        m, n = 6, 4
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest.linalg import svdvals

            svdvals(A)
            assert budget.flops_used == m * n * min(m, n)
