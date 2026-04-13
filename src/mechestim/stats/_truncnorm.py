"""Truncated normal distribution with FLOP counting.

Mimics ``scipy.stats.truncnorm`` API. Parameters ``a`` and ``b`` are the
standardized lower and upper bounds: the distribution is truncated to
``[a*scale + loc, b*scale + loc]``.
"""

from __future__ import annotations

import numpy as _np

from mechestim.stats._base import ContinuousDistribution
from mechestim.stats._erf import _erf
from mechestim.stats._ndtri import _ndtri

_TRUNCNORM_PDF_COST = 30
_TRUNCNORM_CDF_COST = 30
_TRUNCNORM_PPF_COST = 50

_SQRT2 = _np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


def _std_norm_cdf(x):
    """Standard normal CDF (no budget deduction)."""
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _std_norm_pdf(x):
    """Standard normal PDF (no budget deduction)."""
    return _INV_SQRT_2PI * _np.exp(-0.5 * x * x)


class TruncnormDistribution(ContinuousDistribution):
    """Truncated normal distribution (scipy.stats.truncnorm compatible).

    ``a``, ``b`` are standardized bounds (first two positional args).
    """

    def __init__(self):
        super().__init__("truncnorm")

    def pdf(self, x, a, b, loc=0, scale=1):
        return self._deduct_and_call(
            "pdf", _TRUNCNORM_PDF_COST, x, a, b, loc=loc, scale=scale
        )

    def cdf(self, x, a, b, loc=0, scale=1):
        return self._deduct_and_call(
            "cdf", _TRUNCNORM_CDF_COST, x, a, b, loc=loc, scale=scale
        )

    def ppf(self, q, a, b, loc=0, scale=1):
        return self._deduct_and_call(
            "ppf", _TRUNCNORM_PPF_COST, q, a, b, loc=loc, scale=scale
        )

    def _compute_pdf(self, x, a, b, loc=0, scale=1):
        z = (x - loc) / scale
        phi_a = _std_norm_cdf(a)
        phi_b = _std_norm_cdf(b)
        denom = scale * (phi_b - phi_a)
        result = _std_norm_pdf(z) / denom
        # Zero out outside [a, b] in standardized space
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
        # ppf = Phi_inv(Phi(a) + q * (Phi(b) - Phi(a)))
        inner = phi_a + q * (phi_b - phi_a)
        z = _ndtri(inner)
        return loc + scale * z


truncnorm = TruncnormDistribution()
