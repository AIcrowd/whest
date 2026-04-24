"""Truncated normal distribution with FLOP counting.

Mimics ``scipy.stats.truncnorm`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
"""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution
from flopscope.stats._erf import _erf
from flopscope.stats._ndtri import _ndtri

_SQRT2 = _np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


def _std_norm_cdf(x):
    """Standard normal CDF (no budget deduction)."""
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _std_norm_pdf(x):
    """Standard normal PDF (no budget deduction)."""
    return _INV_SQRT_2PI * _np.exp(-0.5 * x * x)


class TruncnormDistribution(ContinuousDistribution):
    """Truncated normal continuous random variable.

    Equivalent to ``scipy.stats.truncnorm``.  Parameters ``a`` and ``b``
    are **standardised** bounds — the distribution is truncated to
    ``[a * scale + loc, b * scale + loc]``.

    .. note::

       ``a`` and ``b`` are the **first two positional arguments**,
       matching scipy's signature:
       ``truncnorm.pdf(x, a, b, loc=0, scale=1)``.

    Methods
    -------
    pdf(x, a, b, loc=0, scale=1)
        Probability density function.
    cdf(x, a, b, loc=0, scale=1)
        Cumulative distribution function.
    ppf(q, a, b, loc=0, scale=1)
        Percent-point function (inverse of CDF).
    """

    def __init__(self):
        super().__init__("truncnorm")

    def pdf(self, x, a, b, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.truncnorm.pdf(x, a, b, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs

        Parameters
        ----------
        x : array_like
            Quantiles.
        a : float
            Lower standardised bound.
        b : float
            Upper standardised bound.
        loc : float, optional
            Mean of the un-truncated normal (default 0).
        scale : float, optional
            Standard deviation of the un-truncated normal (default 1).

        Returns
        -------
        FlopscopeArray
            PDF evaluated at *x*.
        """
        return self._deduct_and_call("pdf", 1, x, a, b, loc=loc, scale=scale)

    def cdf(self, x, a, b, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.truncnorm.cdf(x, a, b, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("cdf", 1, x, a, b, loc=loc, scale=scale)

    def ppf(self, q, a, b, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.truncnorm.ppf(q, a, b, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(q) FLOPs
        """
        return self._deduct_and_call("ppf", 1, q, a, b, loc=loc, scale=scale)

    def _compute_pdf(self, x, a, b, loc=0, scale=1):
        z = (x - loc) / scale
        phi_a = _std_norm_cdf(a)
        phi_b = _std_norm_cdf(b)
        denom = scale * (phi_b - phi_a)
        result = _std_norm_pdf(z) / denom
        return _np.where((z >= a) & (z <= b), result, 0.0)

    def _compute_cdf(self, x, a, b, loc=0, scale=1):
        z = (x - loc) / scale
        phi_a = _std_norm_cdf(a)
        phi_b = _std_norm_cdf(b)
        result = (_std_norm_cdf(z) - phi_a) / (phi_b - phi_a)
        result = _np.where(z < a, 0.0, result)
        result = _np.where(z > b, 1.0, result)
        return result

    def _compute_ppf(self, q, a, b, loc=0, scale=1):
        phi_a = _std_norm_cdf(a)
        phi_b = _std_norm_cdf(b)
        inner = phi_a + q * (phi_b - phi_a)
        z = _ndtri(inner)
        return loc + scale * z


truncnorm = TruncnormDistribution()
