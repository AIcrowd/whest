"""Logistic distribution with FLOP counting.

Mimics ``scipy.stats.logistic`` API.
CDF is the sigmoid function; PPF is the logit function.
"""

from __future__ import annotations

import numpy as _np

from mechestim.stats._base import ContinuousDistribution

_LOGISTIC_PDF_COST = 8
_LOGISTIC_CDF_COST = 5
_LOGISTIC_PPF_COST = 5


class LogisticDistribution(ContinuousDistribution):
    """Logistic distribution (scipy.stats.logistic compatible)."""

    def __init__(self):
        super().__init__("logistic")

    def pdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("pdf", _LOGISTIC_PDF_COST, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("cdf", _LOGISTIC_CDF_COST, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        return self._deduct_and_call("ppf", _LOGISTIC_PPF_COST, q, loc=loc, scale=scale)

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
