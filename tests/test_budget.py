"""Tests for BudgetContext and OpRecord."""

import pytest

from whest._budget import BudgetContext, OpRecord, get_active_budget
from whest.errors import BudgetExhaustedError


def test_budget_context_basic():
    with BudgetContext(flop_budget=1000) as budget:
        assert budget.flop_budget == 1000
        assert budget.flops_used == 0
        assert budget.flops_remaining == 1000
        assert budget.flop_multiplier == 1.0
        assert budget.op_log == []


def test_budget_context_deduct():
    with BudgetContext(flop_budget=1000) as budget:
        budget.deduct("test_op", flop_cost=300, subscripts=None, shapes=((10, 10),))
        assert budget.flops_used == 300
        assert budget.flops_remaining == 700
        assert len(budget.op_log) == 1
        rec = budget.op_log[0]
        assert rec.op_name == "test_op"
        assert rec.flop_cost == 300
        assert rec.cumulative == 300


def test_budget_context_deduct_with_multiplier():
    with BudgetContext(flop_budget=1000, flop_multiplier=2.0) as budget:
        budget.deduct("test_op", flop_cost=100, subscripts=None, shapes=())
        assert budget.flops_used == 200


def test_budget_exhausted():
    with pytest.raises(BudgetExhaustedError) as exc_info:
        with BudgetContext(flop_budget=100) as budget:
            budget.deduct("einsum", flop_cost=200, subscripts="ij,jk->ik", shapes=())
    assert exc_info.value.op_name == "einsum"
    assert exc_info.value.flop_cost == 200


def test_budget_exact_boundary():
    with BudgetContext(flop_budget=100) as budget:
        budget.deduct("op", flop_cost=100, subscripts=None, shapes=())
        assert budget.flops_remaining == 0


def test_no_nesting():
    with pytest.raises(RuntimeError, match="Cannot nest"):
        with BudgetContext(flop_budget=1000):
            with BudgetContext(flop_budget=500):
                pass


def test_invalid_budget():
    with pytest.raises(ValueError, match="must be > 0"):
        BudgetContext(flop_budget=0)
    with pytest.raises(ValueError, match="must be > 0"):
        BudgetContext(flop_budget=-5)


def test_get_active_budget_inside():
    with BudgetContext(flop_budget=1000) as budget:
        assert get_active_budget() is budget


def test_get_active_budget_outside():
    assert get_active_budget() is None


def test_context_cleans_up_on_exit():
    with BudgetContext(flop_budget=1000):
        pass
    assert get_active_budget() is None


def test_context_cleans_up_on_exception():
    with pytest.raises(ValueError):
        with BudgetContext(flop_budget=1000):
            raise ValueError("boom")
    assert get_active_budget() is None


def test_op_record_fields():
    rec = OpRecord(
        op_name="einsum",
        subscripts="ij,jk->ik",
        shapes=((3, 4), (4, 5)),
        flop_cost=60,
        cumulative=60,
        namespace=None,
    )
    assert rec.op_name == "einsum"
    assert rec.subscripts == "ij,jk->ik"
    assert rec.flop_cost == 60
    assert rec.namespace is None


def test_summary():
    with BudgetContext(flop_budget=1000) as budget:
        budget.deduct("einsum", flop_cost=500, subscripts="ij->i", shapes=())
        budget.deduct("exp", flop_cost=100, subscripts=None, shapes=())
        budget.deduct("einsum", flop_cost=200, subscripts="ij->j", shapes=())
        s = budget.summary()
        assert "1,000" in s
        assert "800" in s
        assert "einsum" in s
        assert "exp" in s


def test_time_exhausted_error_attributes():
    from whest.errors import TimeExhaustedError
    err = TimeExhaustedError("matmul", elapsed_s=1.5, limit_s=1.0)
    assert err.op_name == "matmul"
    assert err.elapsed_s == 1.5
    assert err.limit_s == 1.0
    assert "matmul" in str(err)
    assert "1.500" in str(err)
    assert "1.000" in str(err)


def test_time_exhausted_error_is_whest_error():
    from whest.errors import TimeExhaustedError, WhestError
    err = TimeExhaustedError("add", elapsed_s=2.0, limit_s=1.0)
    assert isinstance(err, WhestError)


import time as _time


def test_budget_context_tracks_wall_time():
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        _ = whest.ones((100,))
        _time.sleep(0.01)
    assert b.wall_time_s is not None
    assert b.wall_time_s >= 0.01


def test_budget_context_wall_time_none_before_exit():
    import whest
    b = whest.BudgetContext(flop_budget=int(1e9))
    assert b.wall_time_s is None


def test_budget_context_elapsed_s_live():
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        _time.sleep(0.01)
        assert b.elapsed_s >= 0.01


def test_budget_context_wall_time_limit_s_property():
    import whest
    b = whest.BudgetContext(flop_budget=int(1e9), wall_time_limit_s=5.0)
    assert b.wall_time_limit_s == 5.0
    b2 = whest.BudgetContext(flop_budget=int(1e9))
    assert b2.wall_time_limit_s is None


def test_budget_context_total_tracked_time():
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        _ = whest.add(whest.ones((1000,)), whest.ones((1000,)))
    assert b.total_tracked_time >= 0
    assert b.total_tracked_time <= b.wall_time_s


def test_budget_context_untracked_time():
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        _ = whest.add(whest.ones((1000,)), whest.ones((1000,)))
        _time.sleep(0.01)
    assert b.untracked_time is not None
    assert b.untracked_time >= 0


def test_wall_time_limit_raises_time_exhausted():
    import pytest
    import whest
    from whest.errors import TimeExhaustedError
    with pytest.raises(TimeExhaustedError) as exc_info:
        with whest.BudgetContext(flop_budget=int(1e15), wall_time_limit_s=0.001):
            a = whest.ones((10,))
            for _ in range(100_000):
                a = whest.add(a, a)
    assert exc_info.value.limit_s == 0.001
    assert exc_info.value.elapsed_s >= 0.001


def test_oprecord_timestamps_monotonic():
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        a = whest.ones((10,))
        for _ in range(5):
            a = whest.add(a, a)
    timestamps = [r.timestamp for r in b.op_log if r.timestamp is not None]
    assert len(timestamps) >= 5
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


def test_oprecord_has_timestamp_and_duration_fields():
    import whest
    rec = whest.OpRecord(
        op_name="add", subscripts=None, shapes=((3,),),
        flop_cost=3, cumulative=3,
    )
    assert rec.timestamp is None
    assert rec.duration is None


def test_budget_factory_passes_wall_time_limit():
    import whest
    b = whest.budget(flop_budget=int(1e9), wall_time_limit_s=2.0)
    assert b.wall_time_limit_s == 2.0
