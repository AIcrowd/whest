"""Guardrail: partition budget exceeded emits CostFallbackWarning instead of
dropping symmetry silently.

Migrated from the deleted oracle-acceptance test: with the new accumulation
model, symmetry savings are computed per-expression. When the partition budget
is exceeded, a CostFallbackWarning fires rather than silently returning a dense
cost without warning.
"""

from __future__ import annotations

import warnings

import numpy as np

import flopscope as fps
from flopscope.errors import CostFallbackWarning


def test_partition_budget_zero_emits_fallback_warning():
    """When partition_budget=0, the accumulation model cannot enumerate partitions
    and must fall back to dense cost. A CostFallbackWarning must fire."""
    A = np.zeros((3, 3, 3))
    A_sym = fps.as_symmetric(A, symmetry=(0, 1, 2))
    fps.configure(partition_budget=0)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", CostFallbackWarning)
            cost = fps.einsum_accumulation_cost("ijk,abc->ic", A_sym, A_sym)
            assert any(
                issubclass(warning.category, CostFallbackWarning) for warning in w
            ), (
                f"Expected CostFallbackWarning when partition_budget=0, got: {[str(x.category) for x in w]}"
            )
            assert cost.fallback_used is True
    finally:
        fps.configure(partition_budget=100_000)


def test_normal_budget_does_not_emit_fallback_warning():
    """With sufficient budget, symmetry savings are applied without warning."""
    A = np.zeros((3, 3, 3))
    A_sym = fps.as_symmetric(A, symmetry=(0, 1, 2))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", CostFallbackWarning)
        cost = fps.einsum_accumulation_cost("ijk,abc->ic", A_sym, A_sym)
        assert not any(
            issubclass(warning.category, CostFallbackWarning) for warning in w
        )
        assert cost.fallback_used is False
