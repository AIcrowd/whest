"""Uniform distribution with FLOP counting.

Mimics ``scipy.stats.uniform`` API. The distribution is uniform on
``[loc, loc + scale]``.
"""

from __future__ import annotations

import numpy as _np

from mechestim.stats._base import ContinuousDistribution

_UNIFORM_PDF_COST = 3
_UNIFORM_CDF_COST = 3
_UNIFORM_PPF_COST = 3


class UniformDistribution(ContinuousDistribution):
    """Continuous uniform distribution (scipy.stats.uniform compatible)."""

    def __init__(self):
        super().__init__("uniform")

    def pdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("pdf", _UNIFORM_PDF_COST, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("cdf", _UNIFORM_CDF_COST, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        return self._deduct_and_call("ppf", _UNIFORM_PPF_COST, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        return _np.where((x >= loc) & (x <= loc + scale), 1.0 / scale, 0.0)

    def _compute_cdf(self, x, loc=0, scale=1):
        return _np.clip((x - loc) / scale, 0.0, 1.0)

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc + q * scale
        result = _np.where((q >= 0) & (q <= 1), result, _np.nan)
        return result


uniform = UniformDistribution()
