"""Exponential distribution with FLOP counting.

Mimics ``scipy.stats.expon`` API. PDF: ``(1/scale) * exp(-(x-loc)/scale)``
for ``x >= loc``.
"""

from __future__ import annotations

import numpy as _np

from mechestim.stats._base import ContinuousDistribution

_EXPON_PDF_COST = 5
_EXPON_CDF_COST = 5
_EXPON_PPF_COST = 5


class ExponDistribution(ContinuousDistribution):
    """Exponential distribution (scipy.stats.expon compatible)."""

    def __init__(self):
        super().__init__("expon")

    def pdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("pdf", _EXPON_PDF_COST, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("cdf", _EXPON_CDF_COST, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        return self._deduct_and_call("ppf", _EXPON_PPF_COST, q, loc=loc, scale=scale)

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
