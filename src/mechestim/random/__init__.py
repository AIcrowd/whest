"""Passthrough to ``numpy.random``.

All functions are re-exported directly from NumPy and cost **0 FLOPs**.
Any attribute not explicitly listed is forwarded via ``__getattr__``.
"""
from numpy.random import (
    RandomState,
    choice,
    default_rng,
    normal,
    permutation,
    rand,
    randn,
    seed,
    shuffle,
    uniform,
)


def __getattr__(name):
    import numpy.random as _npr
    if hasattr(_npr, name):
        return getattr(_npr, name)
    raise AttributeError(f"mechestim.random does not provide '{name}'")
