"""Tests for mechestim.stats normal distribution (erf, ndtri, norm)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special
import scipy.stats

from mechestim._budget import BudgetContext
from mechestim.errors import BudgetExhaustedError
from mechestim.stats._erf import _erf, _erfc
from mechestim.stats._ndtri import _ndtri
from mechestim.stats._norm import norm

# ---------------------------------------------------------------------------
# TestErf
# ---------------------------------------------------------------------------


class TestErf:
    """Verify _erf against scipy.special.erf."""

    @pytest.fixture
    def xs(self):
        return np.concatenate(
            [
                np.linspace(-6, 6, 500),
                np.array([0.0, 0.25, 0.49, 0.5, 0.51, 1.0, 2.0, 3.99, 4.0, 4.01, 6.0]),
                np.array([-0.25, -0.5, -1.0, -4.0, -6.0]),
            ]
        )

    def test_accuracy(self, xs):
        result = _erf(xs)
        expected = scipy.special.erf(xs)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)

    def test_erfc_accuracy(self, xs):
        result = _erfc(xs)
        expected = scipy.special.erfc(xs)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)

    def test_scalar(self):
        assert isinstance(_erf(0.5), float)
        np.testing.assert_allclose(_erf(0.5), scipy.special.erf(0.5), atol=1e-15)

    def test_symmetry(self, xs):
        """erf(-x) == -erf(x)"""
        np.testing.assert_allclose(_erf(-xs), -_erf(xs), atol=1e-15)


# ---------------------------------------------------------------------------
# TestNdtri
# ---------------------------------------------------------------------------


class TestNdtri:
    """Verify _ndtri against scipy.special.ndtri."""

    @pytest.fixture
    def ps(self):
        return np.concatenate(
            [
                np.linspace(0.001, 0.999, 500),
                np.array([0.001, 0.01, 0.02425, 0.5, 0.97575, 0.99, 0.999]),
            ]
        )

    def test_accuracy(self, ps):
        result = _ndtri(ps)
        expected = scipy.special.ndtri(ps)
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)

    def test_edge_zero(self):
        assert _ndtri(0.0) == -np.inf

    def test_edge_one(self):
        assert _ndtri(1.0) == np.inf

    def test_edge_out_of_range(self):
        assert np.isnan(_ndtri(-0.1))
        assert np.isnan(_ndtri(1.1))

    def test_scalar(self):
        assert isinstance(_ndtri(0.5), float)
        np.testing.assert_allclose(_ndtri(0.5), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# TestNormPdf
# ---------------------------------------------------------------------------

_PARAMS = [(0, 1), (2, 0.5), (-1, 3)]
_PARAM_IDS = ["standard", "loc2_scale0.5", "loc-1_scale3"]


class TestNormPdf:
    """Verify norm.pdf against scipy.stats.norm.pdf."""

    @pytest.mark.parametrize("loc,scale", _PARAMS, ids=_PARAM_IDS)
    def test_accuracy(self, loc, scale):
        xs = np.linspace(loc - 4 * scale, loc + 4 * scale, 200)
        with BudgetContext(10**9, quiet=True):
            result = np.asarray(norm.pdf(xs, loc=loc, scale=scale))
        expected = scipy.stats.norm.pdf(xs, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


# ---------------------------------------------------------------------------
# TestNormCdf
# ---------------------------------------------------------------------------


class TestNormCdf:
    """Verify norm.cdf against scipy.stats.norm.cdf."""

    @pytest.mark.parametrize("loc,scale", _PARAMS, ids=_PARAM_IDS)
    def test_accuracy(self, loc, scale):
        xs = np.linspace(loc - 4 * scale, loc + 4 * scale, 200)
        with BudgetContext(10**9, quiet=True):
            result = np.asarray(norm.cdf(xs, loc=loc, scale=scale))
        expected = scipy.stats.norm.cdf(xs, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)

    def test_cdf_neg_inf(self):
        with BudgetContext(10**9, quiet=True):
            val = float(norm.cdf(-np.inf))
        assert val == 0.0

    def test_cdf_pos_inf(self):
        with BudgetContext(10**9, quiet=True):
            val = float(norm.cdf(np.inf))
        assert val == 1.0


# ---------------------------------------------------------------------------
# TestNormPpf
# ---------------------------------------------------------------------------


class TestNormPpf:
    """Verify norm.ppf against scipy.stats.norm.ppf."""

    @pytest.mark.parametrize("loc,scale", _PARAMS, ids=_PARAM_IDS)
    def test_accuracy(self, loc, scale):
        qs = np.linspace(0.001, 0.999, 200)
        with BudgetContext(10**9, quiet=True):
            result = np.asarray(norm.ppf(qs, loc=loc, scale=scale))
        expected = scipy.stats.norm.ppf(qs, loc=loc, scale=scale)
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)

    def test_boundary_zero(self):
        with BudgetContext(10**9, quiet=True):
            val = float(norm.ppf(0.0))
        assert val == -np.inf

    def test_boundary_one(self):
        with BudgetContext(10**9, quiet=True):
            val = float(norm.ppf(1.0))
        assert val == np.inf

    def test_roundtrip(self):
        """cdf(ppf(q)) should round-trip to q."""
        qs = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        with BudgetContext(10**9, quiet=True):
            xs = norm.ppf(qs)
            rt = norm.cdf(xs)
        np.testing.assert_allclose(np.asarray(rt), qs, atol=1e-12)


# ---------------------------------------------------------------------------
# FLOP cost tests
# ---------------------------------------------------------------------------


class TestFlopCosts:
    """Verify that FLOP deductions match expected costs."""

    def test_pdf_cost(self):
        xs = np.ones(100)
        with BudgetContext(10**9, quiet=True) as ctx:
            norm.pdf(xs)
        assert ctx.flops_used == 100

    def test_cdf_cost(self):
        xs = np.ones(100)
        with BudgetContext(10**9, quiet=True) as ctx:
            norm.cdf(xs)
        assert ctx.flops_used == 100

    def test_ppf_cost(self):
        qs = np.ones(100) * 0.5
        with BudgetContext(10**9, quiet=True) as ctx:
            norm.ppf(qs)
        assert ctx.flops_used == 100


# ---------------------------------------------------------------------------
# BudgetExhaustedError test
# ---------------------------------------------------------------------------


class TestBudgetExhausted:
    def test_raises(self):
        xs = np.ones(100)
        with pytest.raises(BudgetExhaustedError):
            with BudgetContext(1, quiet=True):
                norm.pdf(xs)


# ---------------------------------------------------------------------------
# repr test
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr(self):
        assert repr(norm) == "<mechestim.stats.norm>"
