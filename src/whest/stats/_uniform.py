"""Uniform distribution with FLOP counting.

Mimics ``scipy.stats.uniform`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
"""

from __future__ import annotations

import numpy as _np

from whest.stats._base import ContinuousDistribution


class UniformDistribution(ContinuousDistribution):
    """Continuous uniform random variable on ``[loc, loc + scale]``.

    Equivalent to ``scipy.stats.uniform``.

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
        super().__init__("uniform")

    def pdf(self, x, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.uniform.pdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs

        Parameters
        ----------
        x : array_like
            Quantiles.
        loc : float, optional
            Lower bound (default 0).
        scale : float, optional
            Width of the interval (default 1).

        Returns
        -------
        WhestArray
            PDF evaluated at *x*.
        """
        return self._deduct_and_call("pdf", 1, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.uniform.cdf(x, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(x) FLOPs

        Parameters
        ----------
        x : array_like
            Quantiles.
        loc : float, optional
            Lower bound (default 0).
        scale : float, optional
            Width of the interval (default 1).

        Returns
        -------
        WhestArray
            CDF evaluated at *x*.
        """
        return self._deduct_and_call("cdf", 1, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.uniform.ppf(q, loc, scale)``.

        FLOP Cost
        ---------
        1 * numel(q) FLOPs

        Parameters
        ----------
        q : array_like
            Quantiles in [0, 1].
        loc : float, optional
            Lower bound (default 0).
        scale : float, optional
            Width of the interval (default 1).

        Returns
        -------
        WhestArray
            PPF evaluated at *q*.
        """
        return self._deduct_and_call("ppf", 1, q, loc=loc, scale=scale)

    def _compute_pdf(self, x, loc=0, scale=1):
        return _np.where((x >= loc) & (x <= loc + scale), 1.0 / scale, 0.0)

    def _compute_cdf(self, x, loc=0, scale=1):
        return _np.clip((x - loc) / scale, 0.0, 1.0)

    def _compute_ppf(self, q, loc=0, scale=1):
        result = loc + q * scale
        result = _np.where((q >= 0) & (q <= 1), result, _np.nan)
        return result


uniform = UniformDistribution()
