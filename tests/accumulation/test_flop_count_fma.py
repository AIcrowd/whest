"""Tests that _opt_einsum/_helpers.flop_count respects the fma_cost setting."""

import pytest

from flopscope._config import get_setting, set_setting
from flopscope._opt_einsum._helpers import flop_count


@pytest.fixture
def reset_fma_cost():
    """Restore fma_cost after each test."""
    original = get_setting('fma_cost')
    yield
    set_setting('fma_cost', original)


def test_flop_count_default_fma_one_for_2_op_inner(reset_fma_cost):
    """With fma_cost=1, a 2-operand inner product (matmul) is 1*size."""
    set_setting('fma_cost', 1)
    # ij,jk -> ik with sizes i=4, j=4, k=4: idx_contraction = {i,j,k}, overall_size = 64.
    # 2 operands, inner=True. Expected with fma_cost=1: 64 * 1 = 64.
    cost = flop_count(
        idx_contraction=frozenset({'i', 'j', 'k'}),
        inner=True,
        num_terms=2,
        size_dictionary={'i': 4, 'j': 4, 'k': 4},
    )
    assert cost == 64


def test_flop_count_fma_two_for_2_op_inner(reset_fma_cost):
    """With fma_cost=2, a 2-operand inner product is 2*size (textbook)."""
    set_setting('fma_cost', 2)
    cost = flop_count(
        idx_contraction=frozenset({'i', 'j', 'k'}),
        inner=True,
        num_terms=2,
        size_dictionary={'i': 4, 'j': 4, 'k': 4},
    )
    assert cost == 128


def test_flop_count_fma_one_for_2_op_outer(reset_fma_cost):
    """Outer product (no inner sum): fma_cost doesn't matter — outer is just multiplies."""
    set_setting('fma_cost', 1)
    # i,j -> ij: idx_contraction = {i,j}, overall_size = i*j = 12, inner=False, num_terms=2.
    # op_factor = max(1, 2-1) = 1. fma_cost=1 doesn't add. Result: 12.
    cost = flop_count(
        idx_contraction=frozenset({'i', 'j'}),
        inner=False,
        num_terms=2,
        size_dictionary={'i': 3, 'j': 4},
    )
    assert cost == 12


def test_flop_count_fma_two_for_2_op_outer(reset_fma_cost):
    """Outer product: fma_cost=2 still adds 0 because inner=False."""
    set_setting('fma_cost', 2)
    cost = flop_count(
        idx_contraction=frozenset({'i', 'j'}),
        inner=False,
        num_terms=2,
        size_dictionary={'i': 3, 'j': 4},
    )
    assert cost == 12  # unchanged; fma_cost only affects inner steps


def test_flop_count_3_op_contraction_fma_one(reset_fma_cost):
    """3-operand step: ijk,ai,bj -> abk, sizes a=b=2, i=j=k=3.
    idx_contraction = {a,b,i,j,k}, overall_size = 2*2*3*3*3 = 108.
    num_terms=3, op_factor = max(1, 3-1) = 2. inner=True, fma_cost=1: no add.
    Result: 108 * 2 = 216."""
    set_setting('fma_cost', 1)
    cost = flop_count(
        idx_contraction=frozenset({'a', 'b', 'i', 'j', 'k'}),
        inner=True,
        num_terms=3,
        size_dictionary={'a': 2, 'b': 2, 'i': 3, 'j': 3, 'k': 3},
    )
    assert cost == 216


def test_flop_count_3_op_contraction_fma_two(reset_fma_cost):
    """Same shape as above but with fma_cost=2: 108 * 3 = 324."""
    set_setting('fma_cost', 2)
    cost = flop_count(
        idx_contraction=frozenset({'a', 'b', 'i', 'j', 'k'}),
        inner=True,
        num_terms=3,
        size_dictionary={'a': 2, 'b': 2, 'i': 3, 'j': 3, 'k': 3},
    )
    assert cost == 324


def test_flop_count_rejects_invalid_fma_cost(reset_fma_cost):
    """If fma_cost is somehow set to an invalid value (bypassing the
    setter), flop_count should raise rather than silently miscompute."""
    # Bypass the validator by directly mutating _SETTINGS (not recommended for
    # users; this is a defense-in-depth check).
    from flopscope._config import _SETTINGS
    original = _SETTINGS['fma_cost']
    _SETTINGS['fma_cost'] = 3
    try:
        with pytest.raises(ValueError, match='fma_cost'):
            flop_count(
                idx_contraction=frozenset({'i'}),
                inner=True,
                num_terms=2,
                size_dictionary={'i': 4},
            )
    finally:
        _SETTINGS['fma_cost'] = original
