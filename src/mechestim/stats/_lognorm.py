"""Log-normal distribution with FLOP counting.

Mimics ``scipy.stats.lognorm`` API. Shape parameter ``s`` is the
standard deviation of the underlying normal distribution.

scipy signature: ``lognorm.pdf(x, s, loc=0, scale=1)``
"""

from __future__ import annotations

import numpy as _np

from mechestim.stats._base import ContinuousDistribution
from mechestim.stats._erf import _erf
from mechestim.stats._ndtri import _ndtri

_LOGNORM_PDF_COST = 15
_LOGNORM_CDF_COST = 25
_LOGNORM_PPF_COST = 45

_SQRT2 = _np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


class LognormDistribution(ContinuousDistribution):
    """Log-normal distribution (scipy.stats.lognorm compatible).

    Note: ``s`` (shape parameter) is the first positional argument,
    matching scipy's signature.
    """

    def __init__(self):
        super().__init__("lognorm")

    def pdf(self, x, s, loc=0, scale=1):
        return self._deduct_and_call(
            "pdf", _LOGNORM_PDF_COST, x, s, loc=loc, scale=scale
        )

    def cdf(self, x, s, loc=0, scale=1):
        return self._deduct_and_call(
            "cdf", _LOGNORM_CDF_COST, x, s, loc=loc, scale=scale
        )

    def ppf(self, q, s, loc=0, scale=1):
        return self._deduct_and_call(
            "ppf", _LOGNORM_PPF_COST, q, s, loc=loc, scale=scale
        )

    def _compute_pdf(self, x, s, loc=0, scale=1):
        y = (x - loc) / scale
        # PDF is 0 for y <= 0
        safe_y = _np.where(y > 0, y, 1.0)  # avoid log(0)
        lny = _np.log(safe_y)
        result = (
            _INV_SQRT_2PI / (s * safe_y * scale) * _np.exp(-0.5 * (lny / s) ** 2)
        )
        return _np.where(y > 0, result, 0.0)

    def _compute_cdf(self, x, s, loc=0, scale=1):
        y = (x - loc) / scale
        safe_y = _np.where(y > 0, y, 1.0)
        z = _np.log(safe_y) / (s * _SQRT2)
        result = 0.5 * (1.0 + _erf(z))
        return _np.where(y > 0, result, 0.0)

    def _compute_ppf(self, q, s, loc=0, scale=1):
        return loc + scale * _np.exp(s * _ndtri(q))


lognorm = LognormDistribution()
