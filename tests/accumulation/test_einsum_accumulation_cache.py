"""Tests for _get_accumulation_cost and its cache."""

import numpy as np

import flopscope as fps
from flopscope._einsum import _accumulation_cache, _get_accumulation_cost


def test_get_accumulation_cost_returns_AccumulationCost():
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    cost = _get_accumulation_cost(
        canonical_subscripts='ij,jk->ik',
        input_parts=('ij', 'jk'),
        output_subscript='ik',
        shapes=((3, 3), (3, 3)),
        operands=(A, B),
    )
    assert isinstance(cost, fps.AccumulationCost)


def test_accumulation_cache_is_hit_on_repeat():
    _accumulation_cache.cache_clear()
    A = np.zeros((4, 4))
    _get_accumulation_cost(
        canonical_subscripts='ij,jk->ik',
        input_parts=('ij', 'jk'),
        output_subscript='ik',
        shapes=((4, 4), (4, 4)),
        operands=(A, A),
    )
    info1 = _accumulation_cache.cache_info()
    _get_accumulation_cost(
        canonical_subscripts='ij,jk->ik',
        input_parts=('ij', 'jk'),
        output_subscript='ik',
        shapes=((4, 4), (4, 4)),
        operands=(A, A),
    )
    info2 = _accumulation_cache.cache_info()
    assert info2.hits == info1.hits + 1
