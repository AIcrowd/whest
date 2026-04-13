import numpy

from whest._budget import BudgetContext


class TestBartlett:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from whest import bartlett

            assert numpy.allclose(bartlett(10), numpy.bartlett(10))

    def test_cost(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from whest import bartlett

            bartlett(10)
            assert budget.flops_used == 10


class TestBlackman:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from whest import blackman

            assert numpy.allclose(blackman(10), numpy.blackman(10))

    def test_cost(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from whest import blackman

            blackman(10)
            assert budget.flops_used == 30


class TestHamming:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from whest import hamming

            assert numpy.allclose(hamming(10), numpy.hamming(10))

    def test_cost(self):
        # Sheet formula: n (FMA=1)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest import hamming

            hamming(10)
            assert budget.flops_used == 10


class TestHanning:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from whest import hanning

            assert numpy.allclose(hanning(10), numpy.hanning(10))

    def test_cost(self):
        # Sheet formula: n (FMA=1)
        with BudgetContext(flop_budget=10**6) as budget:
            from whest import hanning

            hanning(10)
            assert budget.flops_used == 10


class TestKaiser:
    def test_result_matches_numpy(self):
        with BudgetContext(flop_budget=10**6):
            from whest import kaiser

            assert numpy.allclose(kaiser(10, 5.0), numpy.kaiser(10, 5.0))

    def test_cost(self):
        with BudgetContext(flop_budget=10**6) as budget:
            from whest import kaiser

            kaiser(10, 5.0)
            assert budget.flops_used == 30
