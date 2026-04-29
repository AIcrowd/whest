# tests/test_fft_transforms.py
import math

import numpy

from flopscope._budget import BudgetContext


class TestFft:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.fft import fft

            assert numpy.allclose(fft(x), numpy.fft.fft(x))

    def test_cost(self):
        n = 16
        x = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fft

            fft(x)
            assert budget.flops_used == 5 * n * math.ceil(math.log2(n))

    def test_cost_with_n_param(self):
        x = numpy.random.randn(10)
        n = 32
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fft

            fft(x, n=n)
            assert budget.flops_used == 5 * n * math.ceil(math.log2(n))

    def test_op_log(self):
        x = numpy.random.randn(8)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fft

            fft(x)
            assert budget.op_log[-1].op_name == "fft.fft"

    def test_outside_context_uses_global_default(self):
        from flopscope.numpy.fft import fft

        # Operations now auto-activate the global default budget instead of raising
        result = fft(numpy.ones(8))
        assert result.shape == (8,)


class TestIfft:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(16) + 1j * numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.fft import ifft

            assert numpy.allclose(ifft(x), numpy.fft.ifft(x))

    def test_cost(self):
        n = 16
        x = numpy.random.randn(n) + 1j * numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import ifft

            ifft(x)
            assert budget.flops_used == 5 * n * math.ceil(math.log2(n))


class TestRfft:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(16)
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.fft import rfft

            assert numpy.allclose(rfft(x), numpy.fft.rfft(x))

    def test_cost(self):
        n = 16
        x = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import rfft

            rfft(x)
            assert budget.flops_used == 5 * (n // 2) * math.ceil(math.log2(n))


class TestIrfft:
    def test_cost(self):
        n = 16
        x = numpy.fft.rfft(numpy.random.randn(n))
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import irfft

            irfft(x, n=n)
            assert budget.flops_used == 5 * (n // 2) * math.ceil(math.log2(n))


class TestFft2:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(8, 8)
        with BudgetContext(flop_budget=10**6):
            from flopscope.numpy.fft import fft2

            assert numpy.allclose(fft2(x), numpy.fft.fft2(x))

    def test_cost(self):
        x = numpy.random.randn(8, 8)
        N = 64
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import fft2

            fft2(x)
            assert budget.flops_used == 5 * N * math.ceil(math.log2(N))


class TestFftn:
    def test_result_matches_numpy(self):
        x = numpy.random.randn(4, 4, 4)
        with BudgetContext(flop_budget=10**8):
            from flopscope.numpy.fft import fftn

            assert numpy.allclose(fftn(x), numpy.fft.fftn(x))


class TestHfft:
    def test_cost(self):
        n = 16
        x = numpy.random.randn(n) + 1j * numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from flopscope.numpy.fft import hfft

            hfft(x)
            out_n = 2 * (n - 1)
            assert budget.flops_used == 5 * out_n * math.ceil(math.log2(out_n))
