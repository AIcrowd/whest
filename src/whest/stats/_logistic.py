"""Logistic distribution with FLOP counting.

Mimics ``scipy.stats.logistic`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html
"""

from __future__ import annotations

import numpy as _np

from whest.stats._base import ContinuousDistribution


class LogisticDistribution(ContinuousDistribution):
    """Logistic continuous random variable.

    Equivalent to ``scipy.stats.logistic``.  The CDF is the sigmoid
    function; the PPF is the logit function.

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
        super().__init__("logistic")

    def pdf(self, x, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.logistic.pdf(x, loc, scale)``.

        FLOP Cost
        ---------
        8 * numel(x) FLOPs
        """
        return self._deduct_and_call("pdf", 1, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.logistic.cdf(x, loc, scale)``.

        FLOP Cost
        ---------
        5 * numel(x) FLOPs
        """
        return self._deduct_and_call("cdf", 1, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.logistic.ppf(q, loc, scale)``.

        FLOP Cost
        ---------
        5 * numel(q) FLOPs
        """
        return self._deduct_and_call("ppf", 1, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        ez = _np.exp(-z)
        return ez / (scale * (1.0 + ez) ** 2)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return 1.0 / (1.0 + _np.exp(-z))

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc + scale * _np.log(q / (1.0 - q))
        result = _np.where((q > 0) & (q < 1), result, _np.nan)
        result = _np.where(q == 0, -_np.inf, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


logistic = LogisticDistribution()
