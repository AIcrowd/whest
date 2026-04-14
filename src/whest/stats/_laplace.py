"""Laplace distribution with FLOP counting.

Mimics ``scipy.stats.laplace`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html
"""

from __future__ import annotations

import numpy as _np

from whest.stats._base import ContinuousDistribution


class LaplaceDistribution(ContinuousDistribution):
    """Laplace (double-exponential) continuous random variable.

    Equivalent to ``scipy.stats.laplace``.

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
        super().__init__("laplace")

    def pdf(self, x, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.laplace.pdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("pdf", 1, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.laplace.cdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("cdf", 1, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.laplace.ppf(q, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(q) FLOPs
        """
        return self._deduct_and_call("ppf", 1, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = _np.abs(x - loc) / scale
        return _np.exp(-z) / (2.0 * scale)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _np.where(z <= 0, 0.5 * _np.exp(z), 1.0 - 0.5 * _np.exp(-z))

    def _compute_ppf(self, q, loc=0, scale=1):
        result = _np.where(
            q <= 0.5,
            loc + scale * _np.log(2.0 * _np.maximum(q, 1e-300)),
            loc - scale * _np.log(2.0 * _np.maximum(1.0 - q, 1e-300)),
        )
        result = _np.where((q >= 0) & (q <= 1), result, _np.nan)
        result = _np.where(q == 0, -_np.inf, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


laplace = LaplaceDistribution()
