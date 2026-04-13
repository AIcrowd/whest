"""Normal (Gaussian) distribution with FLOP counting."""

from __future__ import annotations

import numpy as np

from mechestim.stats._base import ContinuousDistribution
from mechestim.stats._erf import _erf
from mechestim.stats._ndtri import _ndtri

_NORM_PDF_COST = 10
_NORM_CDF_COST = 20
_NORM_PPF_COST = 40

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class NormDistribution(ContinuousDistribution):
    """Standard normal distribution (loc/scale parameterisation)."""

    def __init__(self):
        super().__init__("norm")

    def pdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("pdf", _NORM_PDF_COST, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        return self._deduct_and_call("cdf", _NORM_CDF_COST, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        return self._deduct_and_call("ppf", _NORM_PPF_COST, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _INV_SQRT_2PI / scale * np.exp(-0.5 * z * z)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / (scale * _SQRT2)
        return 0.5 * (1.0 + _erf(z))

    def _compute_ppf(self, q, loc=0, scale=1):
        return loc + scale * _ndtri(q)


norm = NormDistribution()
