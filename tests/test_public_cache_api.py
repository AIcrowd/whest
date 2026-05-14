"""Tests for the public einsum_clear_caches / einsum_cache_info / fma_cost API."""

from __future__ import annotations

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


def test_fma_cost_public_reexport():
    """flopscope.fma_cost() should match the internal _cost_model.fma_cost()."""
    from flopscope._cost_model import fma_cost as internal_fma_cost

    assert flops.fma_cost is internal_fma_cost
    assert flops.fma_cost() == internal_fma_cost()


def test_fma_cost_reflects_configure_changes():
    original = flops.fma_cost()
    try:
        flops.configure(fma_cost=2)
        assert flops.fma_cost() == 2
        flops.configure(fma_cost=1)
        assert flops.fma_cost() == 1
    finally:
        flops.configure(fma_cost=original)


def test_einsum_cache_info_keys():
    info = flops.einsum_cache_info()
    assert set(info.keys()) == {"path", "accumulation"}
    for key, ci in info.items():
        assert hasattr(ci, "hits"), f"{key}: missing .hits attribute"
        assert hasattr(ci, "misses"), f"{key}: missing .misses attribute"
        assert hasattr(ci, "maxsize"), f"{key}: missing .maxsize attribute"
        assert hasattr(ci, "currsize"), f"{key}: missing .currsize attribute"


def test_einsum_clear_caches_clears_both():
    """A call should drop both currsize counters to 0."""
    # Warm both caches.
    A = np.zeros((4, 4))
    flops.einsum_accumulation_cost("ij,jk->ik", A, A)
    with flops.BudgetContext(flop_budget=10**9):
        fnp.einsum_path("ij,jk->ik", A, A)

    before = flops.einsum_cache_info()
    assert before["path"].currsize >= 1, before
    assert before["accumulation"].currsize >= 1, before

    flops.einsum_clear_caches()

    after = flops.einsum_cache_info()
    assert after["path"].currsize == 0, after
    assert after["accumulation"].currsize == 0, after


def test_einsum_clear_caches_independent_of_fnp_clear():
    """The fnp.clear_einsum_cache() narrow version still only clears the path cache."""
    A = np.zeros((4, 4))
    flops.einsum_accumulation_cost("ij,jk->ik", A, A)
    with flops.BudgetContext(flop_budget=10**9):
        fnp.einsum_path("ij,jk->ik", A, A)

    # fnp.clear_einsum_cache only touches the path cache.
    fnp.clear_einsum_cache()
    info = flops.einsum_cache_info()
    assert info["path"].currsize == 0
    assert info["accumulation"].currsize >= 1, (
        "fnp.clear_einsum_cache must NOT touch the accumulation cache"
    )

    # The unified version clears both.
    flops.einsum_clear_caches()
    info = flops.einsum_cache_info()
    assert info["path"].currsize == 0
    assert info["accumulation"].currsize == 0


def test_public_api_in_all():
    assert "fma_cost" in flops.__all__
    assert "einsum_clear_caches" in flops.__all__
    assert "einsum_cache_info" in flops.__all__
