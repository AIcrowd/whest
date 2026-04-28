"""Smoke tests for ``flopscope.accounting`` (was ``flopscope.flops``).

The submodule was renamed to avoid colliding with the new
``import flopscope as flops`` alias the JAX-style API encourages.
"""

from __future__ import annotations

import flopscope as flops
from flopscope import accounting


def test_accounting_is_a_submodule_not_an_alias_for_flopscope():
    assert accounting is flops.accounting
    assert accounting is not flops


def test_accounting_exports_core_cost_helpers():
    for name in (
        "einsum_cost",
        "pointwise_cost",
        "reduction_cost",
        "svd_cost",
        "cholesky_cost",
        "qr_cost",
        "solve_cost",
        "fft_cost",
        "polyval_cost",
    ):
        assert callable(getattr(accounting, name)), f"accounting.{name} missing"


def test_einsum_cost_matches_simple_matmul():
    cost = accounting.einsum_cost("ij,jk->ik", shapes=[(8, 8), (8, 8)])
    assert cost > 0


def test_pointwise_cost_follows_numel():
    cost = accounting.pointwise_cost("add", shape=(10,))
    assert cost == 10


def test_reduction_cost_follows_input_numel():
    cost = accounting.reduction_cost("sum", input_shape=(3, 4), axis=None)
    assert cost == 12
