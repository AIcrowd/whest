"""Log-normal distribution with FLOP counting.

Mimics ``scipy.stats.lognorm`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
"""

from __future__ import annotations

import numpy as _np

from flopscope.stats._base import ContinuousDistribution
from flopscope.stats._erf import _erf
from flopscope.stats._ndtri import _ndtri

_SQRT2 = _np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


class LognormDistribution(ContinuousDistribution):
    """Log-normal continuous random variable.

    Equivalent to ``scipy.stats.lognorm``.  The shape parameter ``s``
    is the standard deviation of the underlying normal distribution.

    .. note::

       ``s`` is the **first positional argument** (before ``loc``/``scale``),
       matching scipy's signature: ``lognorm.pdf(x, s, loc=0, scale=1)``.

    Methods
    -------
    pdf(x, s, loc=0, scale=1)
        Probability density function.
    cdf(x, s, loc=0, scale=1)
        Cumulative distribution function.
    ppf(q, s, loc=0, scale=1)
        Percent-point function (inverse of CDF).
    """

    def __init__(self):
        super().__init__("lognorm")

    def pdf(self, x, s, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.lognorm.pdf(x, s, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs

        Parameters
        ----------
        x : array_like
            Quantiles.
        s : float
            Shape parameter (std dev of the underlying normal).
        loc : float, optional
            Location parameter (default 0).
        scale : float, optional
            Scale parameter (default 1).

        Returns
        -------
        FlopscopeArray
            PDF evaluated at *x*.
        """
        return self._deduct_and_call("pdf", 1, x, s, loc=loc, scale=scale)

    def cdf(self, x, s, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.lognorm.cdf(x, s, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs
        """
        return self._deduct_and_call("cdf", 1, x, s, loc=loc, scale=scale)

    def ppf(self, q, s, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.lognorm.ppf(q, s, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(q) FLOPs
        """
        return self._deduct_and_call("ppf", 1, q, s, loc=loc, scale=scale)

    def _compute_pdf(self, x, s, loc=0, scale=1):
        y = (x - loc) / scale
        safe_y = _np.where(y > 0, y, 1.0)  # avoid log(0)
        lny = _np.log(safe_y)
        result = _INV_SQRT_2PI / (s * safe_y * scale) * _np.exp(-0.5 * (lny / s) ** 2)
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
