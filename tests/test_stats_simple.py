"""Tests for simple closed-form distributions against scipy."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from whest._budget import BudgetContext

# ============================================================
# uniform
# ============================================================


class TestUniformPdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import uniform

        x = np.linspace(loc - 1, loc + scale + 1, 500)
        result = np.asarray(uniform.pdf(x, loc=loc, scale=scale))
        expected = sp_stats.uniform.pdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestUniformCdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import uniform

        x = np.linspace(loc - 1, loc + scale + 1, 500)
        result = np.asarray(uniform.cdf(x, loc=loc, scale=scale))
        expected = sp_stats.uniform.cdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestUniformPpf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import uniform

        q = np.linspace(0, 1, 100)
        result = np.asarray(uniform.ppf(q, loc=loc, scale=scale))
        expected = sp_stats.uniform.ppf(q, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)

    def test_flop_cost(self):
        from whest.stats import uniform

        q = np.random.rand(100)
        with BudgetContext(flop_budget=10**6) as b:
            uniform.ppf(q)
            assert b.flops_used == 100


# ============================================================
# expon
# ============================================================


class TestExponPdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (1, 2), (0, 0.5)])
    def test_accuracy(self, loc, scale):
        from whest.stats import expon

        x = np.linspace(loc - 0.5, loc + 5 * scale, 500)
        result = np.asarray(expon.pdf(x, loc=loc, scale=scale))
        expected = sp_stats.expon.pdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestExponCdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (1, 2), (0, 0.5)])
    def test_accuracy(self, loc, scale):
        from whest.stats import expon

        x = np.linspace(loc - 0.5, loc + 5 * scale, 500)
        result = np.asarray(expon.cdf(x, loc=loc, scale=scale))
        expected = sp_stats.expon.cdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestExponPpf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (1, 2), (0, 0.5)])
    def test_accuracy(self, loc, scale):
        from whest.stats import expon

        q = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
        result = np.asarray(expon.ppf(q, loc=loc, scale=scale))
        expected = sp_stats.expon.ppf(q, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)

    def test_flop_cost(self):
        from whest.stats import expon

        x = np.random.rand(200)
        with BudgetContext(flop_budget=10**6) as b:
            expon.pdf(x)
            assert b.flops_used == 200


# ============================================================
# cauchy
# ============================================================


class TestCauchyPdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import cauchy

        x = np.linspace(-20, 20, 500)
        result = np.asarray(cauchy.pdf(x, loc=loc, scale=scale))
        expected = sp_stats.cauchy.pdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestCauchyCdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import cauchy

        x = np.linspace(-20, 20, 500)
        result = np.asarray(cauchy.cdf(x, loc=loc, scale=scale))
        expected = sp_stats.cauchy.cdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestCauchyPpf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import cauchy

        q = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        result = np.asarray(cauchy.ppf(q, loc=loc, scale=scale))
        expected = sp_stats.cauchy.ppf(q, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)

    def test_flop_cost(self):
        from whest.stats import cauchy

        x = np.random.randn(50)
        with BudgetContext(flop_budget=10**6) as b:
            cauchy.cdf(x)
            assert b.flops_used == 50


# ============================================================
# logistic
# ============================================================


class TestLogisticPdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import logistic

        x = np.linspace(-10, 10, 500)
        result = np.asarray(logistic.pdf(x, loc=loc, scale=scale))
        expected = sp_stats.logistic.pdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestLogisticCdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import logistic

        x = np.linspace(-10, 10, 500)
        result = np.asarray(logistic.cdf(x, loc=loc, scale=scale))
        expected = sp_stats.logistic.cdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestLogisticPpf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import logistic

        q = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        result = np.asarray(logistic.ppf(q, loc=loc, scale=scale))
        expected = sp_stats.logistic.ppf(q, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)

    def test_flop_cost(self):
        from whest.stats import logistic

        q = np.random.rand(80)
        with BudgetContext(flop_budget=10**6) as b:
            logistic.ppf(q)
            assert b.flops_used == 80


# ============================================================
# laplace
# ============================================================


class TestLaplacePdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import laplace

        x = np.linspace(-10, 10, 500)
        result = np.asarray(laplace.pdf(x, loc=loc, scale=scale))
        expected = sp_stats.laplace.pdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestLaplaceCdf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import laplace

        x = np.linspace(-10, 10, 500)
        result = np.asarray(laplace.cdf(x, loc=loc, scale=scale))
        expected = sp_stats.laplace.cdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


class TestLaplacePpf:
    @pytest.mark.parametrize("loc,scale", [(0, 1), (2, 0.5), (-1, 3)])
    def test_accuracy(self, loc, scale):
        from whest.stats import laplace

        q = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        result = np.asarray(laplace.ppf(q, loc=loc, scale=scale))
        expected = sp_stats.laplace.ppf(q, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)

    def test_flop_cost(self):
        from whest.stats import laplace

        x = np.random.randn(150)
        with BudgetContext(flop_budget=10**6) as b:
            laplace.pdf(x)
            assert b.flops_used == 150
