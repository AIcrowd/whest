"""Exponential distribution with FLOP counting.

Mimics ``scipy.stats.expon`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
"""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution


class ExponDistribution(ContinuousDistribution):
    """Exponential continuous random variable.

    Equivalent to ``scipy.stats.expon``.  The PDF is
    ``(1/scale) * exp(-(x - loc) / scale)`` for ``x >= loc``.

    Methods
    -------
    pdf(x, loc=0, scale=1)
        Probability density function.
    cdf(x, loc=0, scale=1)
        Cumulative distribution function.
    ppf(q, loc=0, scale=1)
        Percent-point function (inverse of CDF).
    """

    def __init__(self):
        super().__init__("expon")

    def pdf(self, x, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.expon.pdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("pdf", 1, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.expon.cdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("cdf", 1, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.expon.ppf(q, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(q) FLOPs
        """
        return self._deduct_and_call("ppf", 1, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _np.where(x >= loc, _np.exp(-z) / scale, 0.0)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _np.where(x >= loc, 1.0 - _np.exp(-z), 0.0)

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc - scale * _np.log1p(-q)
        result = _np.where((q >= 0) & (q <= 1), result, _np.nan)
        result = _np.where(q == 0, loc, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


expon = ExponDistribution()
