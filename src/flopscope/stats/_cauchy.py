"""Cauchy distribution with FLOP counting.

Mimics ``scipy.stats.cauchy`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html
"""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution


class CauchyDistribution(ContinuousDistribution):
    """Cauchy (Lorentz) continuous random variable.

    Equivalent to ``scipy.stats.cauchy``.

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
        super().__init__("cauchy")

    def pdf(self, x, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.cauchy.pdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("pdf", 1, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.cauchy.cdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("cdf", 1, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.cauchy.ppf(q, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(q) FLOPs
        """
        return self._deduct_and_call("ppf", 1, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return 1.0 / (_np.pi * scale * (1.0 + z * z))

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return 0.5 + _np.arctan(z) / _np.pi

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc + scale * _np.tan(_np.pi * (q - 0.5))
        result = _np.where((q > 0) & (q < 1), result, _np.nan)
        result = _np.where(q == 0, -_np.inf, result)
        result = _np.where(q == 1, _np.inf, result)
        return result


cauchy = CauchyDistribution()
