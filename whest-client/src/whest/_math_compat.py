"""Pure-Python replacements for math.prod and math constants.

The hardened participant Docker image blocks ``import math`` to prevent
local computation.  This module provides the tiny subset the client needs
without importing the ``math`` module.

Constants are imported from ``_constants.py``, which is generated at
Docker build time by ``docker/lockdown/generate_constants.py`` using
the real ``math`` module values.  For local development (outside Docker),
a fallback ``_constants.py`` is checked in with the same values.
"""

from __future__ import annotations

from functools import reduce
from collections.abc import Iterable

from whest._constants import e, inf, nan, pi

__all__ = ["prod", "pi", "e", "inf", "nan"]


def prod(iterable: Iterable[int], start: int = 1) -> int:
    """Return the product of all elements (like ``math.prod``)."""
    return reduce(lambda a, b: a * b, iterable, start)
