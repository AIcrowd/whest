"""Tests for FLOP weight integration in BudgetContext.deduct()."""

import json
import os
import tempfile

import pytest

from mechestim._budget import BudgetContext
from mechestim._weights import load_weights, reset_weights
from mechestim.errors import BudgetExhaustedError


def test_deduct_applies_weight():
    reset_weights()
    data = {"weights": {"exp": 10.0}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        load_weights(path)
        with BudgetContext(flop_budget=1_000_000) as budget:
            budget.deduct("exp", flop_cost=100, subscripts=None, shapes=((100,),))
            assert budget.flops_used == 1000  # 100 * 10.0
            rec = budget.op_log[0]
            assert rec.flop_cost == 1000
    finally:
        reset_weights()
        os.unlink(path)


def test_deduct_weight_default_is_one():
    reset_weights()
    with BudgetContext(flop_budget=1_000_000) as budget:
        budget.deduct("add", flop_cost=100, subscripts=None, shapes=((100,),))
        assert budget.flops_used == 100  # no weight loaded, default 1.0


def test_deduct_weight_stacks_with_flop_multiplier():
    reset_weights()
    data = {"weights": {"exp": 10.0}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        load_weights(path)
        with BudgetContext(flop_budget=1_000_000, flop_multiplier=2.0) as budget:
            budget.deduct("exp", flop_cost=100, subscripts=None, shapes=((100,),))
            assert budget.flops_used == 2000  # 100 * 2.0 * 10.0
    finally:
        reset_weights()
        os.unlink(path)


def test_deduct_exhaustion_accounts_for_weight():
    reset_weights()
    data = {"weights": {"exp": 10.0}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        load_weights(path)
        with pytest.raises(BudgetExhaustedError):
            with BudgetContext(flop_budget=500) as budget:
                # 100 * 10.0 = 1000 > 500
                budget.deduct("exp", flop_cost=100, subscripts=None, shapes=((100,),))
    finally:
        reset_weights()
        os.unlink(path)
