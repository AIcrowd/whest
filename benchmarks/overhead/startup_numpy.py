"""Pure-NumPy startup factories for isolated NumPy baseline timing."""

from __future__ import annotations

import numpy as np


def _numpy_add_api(a, b):
    return np.add(a, b)


def _numpy_add_operator(a, b):
    return a + b


def _numpy_matmul_api(a, b):
    return np.matmul(a, b)


def _numpy_matmul_operator(a, b):
    return a @ b
