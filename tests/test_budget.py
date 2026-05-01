"""Tests for BudgetContext and OpRecord."""

import pytest

from flopscope._budget import BudgetContext, OpRecord, get_active_budget
from flopscope.errors import BudgetExhaustedError


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
    from flopscope.errors import TimeExhaustedError

    err = TimeExhaustedError("matmul", elapsed_s=1.5, limit_s=1.0)
    assert err.op_name == "matmul"
    assert err.elapsed_s == 1.5
    assert err.limit_s == 1.0
    assert "matmul" in str(err)
    assert "1.500" in str(err)
    assert "1.000" in str(err)


def test_time_exhausted_error_is_flopscope_error():
    from flopscope.errors import FlopscopeError, TimeExhaustedError

    err = TimeExhaustedError("add", elapsed_s=2.0, limit_s=1.0)
    assert isinstance(err, FlopscopeError)


import time as _time


def test_budget_context_tracks_wall_time():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9)) as b:
        _ = flopscope.numpy.ones((100,))
        _time.sleep(0.01)
    assert b.wall_time_s is not None
    assert b.wall_time_s >= 0.01


def test_budget_context_wall_time_none_before_exit():
    import flopscope

    b = flopscope.BudgetContext(flop_budget=int(1e9))
    assert b.wall_time_s is None


def test_budget_context_elapsed_s_live():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9)) as b:
        _time.sleep(0.01)
        assert b.elapsed_s >= 0.01


def test_budget_context_wall_time_limit_s_property():
    import flopscope

    b = flopscope.BudgetContext(flop_budget=int(1e9), wall_time_limit_s=5.0)
    assert b.wall_time_limit_s == 5.0
    b2 = flopscope.BudgetContext(flop_budget=int(1e9))
    assert b2.wall_time_limit_s is None


def test_budget_context_total_tracked_time():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9)) as b:
        _ = flopscope.numpy.add(
            flopscope.numpy.ones((1000,)), flopscope.numpy.ones((1000,))
        )
    assert b.total_tracked_time >= 0
    assert b.total_tracked_time <= b.wall_time_s  # pyright: ignore[reportOperatorIssue]


def test_budget_context_untracked_time():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9)) as b:
        _ = flopscope.numpy.add(
            flopscope.numpy.ones((1000,)), flopscope.numpy.ones((1000,))
        )
        _time.sleep(0.01)
    assert b.untracked_time is not None
    assert b.untracked_time >= 0


def test_wall_time_limit_raises_time_exhausted():
    import pytest

    import flopscope
    from flopscope.errors import TimeExhaustedError

    with pytest.raises(TimeExhaustedError) as exc_info:
        with flopscope.BudgetContext(flop_budget=int(1e15), wall_time_limit_s=0.001):
            a = flopscope.numpy.ones((10,))
            for _ in range(100_000):
                a = flopscope.numpy.add(a, a)
    assert exc_info.value.limit_s == 0.001
    assert exc_info.value.elapsed_s >= 0.001


def test_oprecord_timestamps_monotonic():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9)) as b:
        a = flopscope.numpy.ones((10,))
        for _ in range(5):
            a = flopscope.numpy.add(a, a)
    timestamps = [r.timestamp for r in b.op_log if r.timestamp is not None]
    assert len(timestamps) >= 5
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


def test_oprecord_has_timestamp_and_duration_fields():
    import flopscope

    rec = flopscope.OpRecord(
        op_name="add",
        subscripts=None,
        shapes=((3,),),
        flop_cost=3,
        cumulative=3,
    )
    assert rec.timestamp is None
    assert rec.duration is None


def test_oprecord_durations_populated():
    """OpRecord durations are populated for ops using with-deduct pattern."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9)) as b:
        a = flopscope.numpy.ones((10,))
        _ = flopscope.numpy.add(a, a)
    add_records = [r for r in b.op_log if r.op_name == "add"]
    assert len(add_records) >= 1
    assert all(r.duration is not None for r in add_records)
    assert all(r.duration >= 0 for r in add_records)  # pyright: ignore[reportOptionalOperand]


def test_durations_populated_across_op_types():
    """Verify durations populated for various operation types."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e12)) as b:
        a = flopscope.numpy.array([1.0, 2.0, 3.0])  # free_ops (charged)
        c = flopscope.numpy.add(
            a, flopscope.numpy.ones((3,))
        )  # pointwise binary (Task 3)
        d = flopscope.numpy.exp(c)  # pointwise unary (Task 3)
        e = flopscope.numpy.sum(d)  # reduction (Task 3)
        f = flopscope.numpy.concatenate([a, c])  # free_ops (charged)
        g = flopscope.numpy.linspace(0, 1, 10)  # free_ops (charged)
        h = flopscope.numpy.dot(a, a)  # standalone pointwise
        i = flopscope.numpy.sort(a.copy())  # sorting

    records_without = [r for r in b.op_log if r.duration is None]
    assert len(records_without) == 0, (
        f"Ops missing duration: {[r.op_name for r in records_without]}"
    )


def test_budget_factory_passes_wall_time_limit():
    import flopscope

    b = flopscope.budget(flop_budget=int(1e9), wall_time_limit_s=2.0)
    assert b.wall_time_limit_s == 2.0


def test_plain_text_summary_includes_timing():
    """Plain-text summary should show wall time and tracked/untracked."""
    import flopscope
    from flopscope._display import _plain_text_summary

    flopscope.budget_reset()
    with flopscope.BudgetContext(flop_budget=int(1e12), namespace="test", quiet=True):
        a = flopscope.numpy.ones((100,))
        _ = flopscope.numpy.add(a, a)

    text = _plain_text_summary()
    assert "Wall time:" in text
    assert "Tracked time:" in text
    assert "Untracked time:" in text


def test_namespace_record_includes_time():
    import flopscope

    flopscope.budget_reset()
    with flopscope.BudgetContext(
        flop_budget=int(1e9), namespace="test", quiet=True
    ) as b:
        _ = flopscope.numpy.add(
            flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
        )
    data = flopscope.budget_summary_dict(by_namespace=True)
    assert "wall_time_s" in data
    assert data["wall_time_s"] is not None
    assert data["wall_time_s"] > 0
    assert "tracked_time_s" in data
    assert "untracked_time_s" in data
    ns_data = data["by_namespace"]["test"]
    assert "tracked_time_s" in ns_data
    assert "wall_time_s" not in ns_data
    flopscope.budget_reset()


def test_summary_includes_time_section():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        _ = flopscope.numpy.add(
            flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
        )
    summary = b.summary()
    assert "Wall time:" in summary
    assert "Tracked time:" in summary


def test_deduct_without_with_leaves_duration_none():
    """Calling deduct() without 'with' leaves OpRecord.duration as None."""
    from flopscope._budget import BudgetContext

    ctx = BudgetContext(flop_budget=int(1e9), quiet=True)
    ctx.__enter__()
    # Call deduct without using 'with' — discard the returned _OpTimer
    ctx.deduct("test_op", flop_cost=10, subscripts=None, shapes=((10,),))
    ctx.__exit__(None, None, None)
    assert len(ctx.op_log) == 1
    assert ctx.op_log[0].duration is None
    assert ctx.op_log[0].timestamp is not None


import threading


def test_thread_isolation_time_tracking():
    """Two threads with separate BudgetContexts track time independently."""
    import flopscope
    from flopscope._budget import _reset_global_default

    results = {}

    def worker(name, sleep_time):
        _reset_global_default()
        with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
            _ = flopscope.numpy.add(
                flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
            )
            _time.sleep(sleep_time)
            _ = flopscope.numpy.add(
                flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
            )
        results[name] = b.wall_time_s

    t1 = threading.Thread(target=worker, args=("fast", 0.01))
    t2 = threading.Thread(target=worker, args=("slow", 0.05))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["fast"] < results["slow"]
    assert results["fast"] >= 0.01
    assert results["slow"] >= 0.05


def test_pointwise_ops_have_duration():
    """Pointwise factory ops (add, exp, sum) must record duration."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e12)) as b:
        a = flopscope.numpy.ones((100,))
        _ = flopscope.numpy.add(a, a)
        _ = flopscope.numpy.exp(a)
        _ = flopscope.numpy.sum(a)

    for rec in b.op_log:
        if rec.op_name in ("add", "exp", "sum"):
            assert rec.duration is not None, f"{rec.op_name} missing duration"
            assert rec.duration >= 0, f"{rec.op_name} has negative duration"


def test_linalg_ops_have_duration():
    """Linalg ops must record duration."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e12)) as b:
        A = flopscope.numpy.array([[1.0, 2.0], [3.0, 4.0]])
        _ = flopscope.numpy.linalg.det(A)
        _ = flopscope.numpy.linalg.solve(A, flopscope.numpy.array([1.0, 2.0]))
        _ = flopscope.numpy.linalg.svd(A)
        _ = flopscope.numpy.linalg.cholesky(A @ A.T + 2 * flopscope.numpy.eye(2))

    linalg_records = [r for r in b.op_log if r.op_name.startswith("linalg.")]
    assert len(linalg_records) >= 4
    for rec in linalg_records:
        assert rec.duration is not None, f"{rec.op_name} missing duration"
        assert rec.duration >= 0, f"{rec.op_name} has negative duration"


def test_counting_ops_have_duration():
    """Counting ops (trace, histogram, bincount, etc.) must record duration."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e12)) as b:
        a = flopscope.numpy.array([1.0, 2.0, 3.0, 4.0])
        _ = flopscope.numpy.trace(flopscope.numpy.eye(3))
        _ = flopscope.numpy.histogram(a, bins=5)
        _ = flopscope.numpy.logspace(0, 1, 10)

    counting_ops = {"trace", "histogram", "logspace"}
    for rec in b.op_log:
        if rec.op_name in counting_ops:
            assert rec.duration is not None, f"{rec.op_name} missing duration"
            assert rec.duration >= 0, f"{rec.op_name} has negative duration"


def test_banner_shows_time_limit(capsys):
    """Banner should include time limit when wall_time_limit_s is set."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e6), wall_time_limit_s=5.0):
        pass
    captured = capsys.readouterr()
    assert "time limit: 5.0s" in captured.err


def test_banner_no_time_limit(capsys):
    """Banner should not mention time limit when wall_time_limit_s is None."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e6)):
        pass
    captured = capsys.readouterr()
    assert "time limit" not in captured.err


def test_post_op_deadline_check():
    """_OpTimer.__exit__ raises TimeExhaustedError if deadline passed during op."""
    import time

    import pytest

    import flopscope
    from flopscope.errors import TimeExhaustedError

    with pytest.raises(TimeExhaustedError) as exc_info:
        with flopscope.BudgetContext(
            flop_budget=int(1e15), wall_time_limit_s=0.05
        ) as b:
            a = flopscope.numpy.ones((10,))
            timer = b.deduct("test_op", flop_cost=1, subscripts=None, shapes=((10,),))
            with timer:
                time.sleep(0.1)  # Exceeds 0.05s limit
    assert exc_info.value.elapsed_s >= 0.05


def test_budget_summary_dict_includes_op_duration():
    """budget_summary_dict() should include per-op duration."""
    import flopscope

    flopscope.budget_reset()
    with flopscope.BudgetContext(flop_budget=int(1e12), namespace="test", quiet=True):
        a = flopscope.numpy.ones((100,))
        _ = flopscope.numpy.add(a, a)

    data = flopscope.budget_summary_dict(by_namespace=True)
    ops = data["operations"]
    assert "add" in ops
    assert "duration" in ops["add"]
    assert ops["add"]["duration"] >= 0

    ns_ops = data["by_namespace"]["test"]["operations"]
    assert "add" in ns_ops
    assert "duration" in ns_ops["add"]


def test_budget_context_summary_dict_live_and_closed():
    import time

    from flopscope._budget import _call_numpy

    budget = BudgetContext(flop_budget=100, quiet=True)
    with budget:
        with budget.deduct("add", flop_cost=10, subscripts=None, shapes=()):
            _call_numpy(time.sleep, 0.01)  # tracked via _call_numpy
        live = budget.summary_dict()
        assert live["flop_budget"] == 100
        assert live["flops_used"] == 10
        assert live["flops_remaining"] == 90
        assert live["wall_time_s"] is not None
        assert live["tracked_time_s"] >= 0.01
        assert live["untracked_time_s"] == pytest.approx(
            live["wall_time_s"]
            - live["tracked_time_s"]
            - live["flopscope_overhead_time_s"],
            abs=1e-6,
        )
        assert live["operations"]["add"]["calls"] == 1
        assert live["operations"]["add"]["duration"] >= 0.01

    closed = budget.summary_dict()
    assert closed["wall_time_s"] is not None
    assert closed["tracked_time_s"] >= 0.01
    assert closed["untracked_time_s"] == pytest.approx(
        closed["wall_time_s"]
        - closed["tracked_time_s"]
        - closed["flopscope_overhead_time_s"],
        abs=1e-6,
    )


def test_budget_summary_dict_shows_live_timing_for_active_context():
    import time

    import flopscope

    with flopscope.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct("add", flop_cost=10, subscripts=None, shapes=()):
            pass
        time.sleep(0.01)

        live = flopscope.budget_summary_dict()
        assert live["wall_time_s"] is not None
        assert live["wall_time_s"] >= 0.01
        assert live["tracked_time_s"] >= 0.0
        assert live["untracked_time_s"] == pytest.approx(
            live["wall_time_s"]
            - live["tracked_time_s"]
            - live["flopscope_overhead_time_s"],
            abs=1e-6,
        )


def test_budget_summary_dict_includes_global_default_while_explicit_context_is_open():
    import flopscope

    a = flopscope.numpy.ones((10,))
    _ = flopscope.numpy.add(a, a)

    with flopscope.BudgetContext(flop_budget=100, quiet=True) as budget:
        with budget.deduct("mul", flop_cost=7, subscripts=None, shapes=()):
            pass

        live = flopscope.budget_summary_dict()
        assert live["flops_used"] == 17
        assert live["operations"]["add"]["flop_cost"] == 10
        assert live["operations"]["mul"]["flop_cost"] == 7


def test_budget_context_summary_dict_by_namespace_uses_exact_op_namespace():
    import flopscope

    with flopscope.BudgetContext(
        flop_budget=1000, namespace="predict..raw", quiet=True
    ) as budget:
        with budget.deduct("mul", flop_cost=5, subscripts=None, shapes=()):
            pass
        with flopscope.namespace("precompute"):
            with budget.deduct("add", flop_cost=25, subscripts=None, shapes=()):
                pass

    data = budget.summary_dict(by_namespace=True)
    assert set(data["by_namespace"]) == {"predict..raw", "predict..raw.precompute"}
    assert data["by_namespace"]["predict..raw"]["flops_used"] == 5
    assert data["by_namespace"]["predict..raw"]["calls"] == 1
    assert data["by_namespace"]["predict..raw.precompute"]["flops_used"] == 25
    assert data["by_namespace"]["predict..raw.precompute"]["calls"] == 1
    assert "flop_budget" not in data["by_namespace"]["predict..raw.precompute"]


def test_oprecord_has_flopscope_overhead_field():
    import flopscope

    rec = flopscope.OpRecord(
        op_name="add",
        subscripts=None,
        shapes=((3,),),
        flop_cost=3,
        cumulative=3,
    )
    assert rec.flopscope_overhead is None

    rec2 = flopscope.OpRecord(
        op_name="add",
        subscripts=None,
        shapes=((3,),),
        flop_cost=3,
        cumulative=3,
        flopscope_overhead=0.001,
    )
    assert rec2.flopscope_overhead == 0.001


def test_budget_context_initializes_overhead_state():
    import flopscope

    b = flopscope.BudgetContext(flop_budget=int(1e9))
    assert b._total_flopscope_overhead_time == 0.0
    assert b._recorded_overhead_time == 0.0
    assert b._current_op_timer is None


def test_budget_context_flopscope_overhead_time_property():
    import flopscope

    b = flopscope.BudgetContext(flop_budget=int(1e9))
    assert b.flopscope_overhead_time == 0.0

    b._total_flopscope_overhead_time = 1.5
    assert b.flopscope_overhead_time == 1.5


def test_timing_summary_subtracts_overhead():
    from flopscope._budget import _timing_summary

    wall, tracked, overhead, untracked = _timing_summary(10.0, 4.0, 3.0)
    assert wall == 10.0
    assert tracked == 4.0
    assert overhead == 3.0
    assert untracked == 3.0

    wall, tracked, overhead, untracked = _timing_summary(None, 4.0, 3.0)
    assert wall is None
    assert tracked == 4.0
    assert overhead == 3.0
    assert untracked is None

    wall, tracked, overhead, untracked = _timing_summary(10.0, 7.0, 3.0 + 1e-13)
    assert untracked == 0.0


def test_summary_dict_includes_flopscope_overhead_time_s():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        _ = flopscope.numpy.add(
            flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
        )
    d = b.summary_dict()
    assert "flopscope_overhead_time_s" in d
    assert d["flopscope_overhead_time_s"] is not None
    assert d["flopscope_overhead_time_s"] >= 0


def test_call_numpy_returns_fn_result_and_records_duration():
    import time as _time

    import numpy as np

    import flopscope
    from flopscope._budget import _call_numpy

    # Without an active timer, _call_numpy is a no-op timing-wise (just calls fn)
    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        result = _call_numpy(np.add, np.ones((3,)), np.ones((3,)))
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [2.0, 2.0, 2.0]

        # With a fake "timer" object that has _np_duration, the duration is reported
        class _FakeTimer:
            def __init__(self):
                self._np_duration = 0.0

        fake = _FakeTimer()
        b._current_op_timer = fake  # type: ignore[assignment]
        try:
            _call_numpy(_time.sleep, 0.01)
        finally:
            b._current_op_timer = None
        assert fake._np_duration >= 0.01


def test_counted_wrapper_records_overhead():
    import time as _time

    import flopscope
    from flopscope._budget import _counted_wrapper

    @_counted_wrapper
    def fake_wrapper(x):
        # Pure preamble (no deduct, no numpy) — all time is overhead
        _time.sleep(0.01)
        return x + 1

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        result = fake_wrapper(5)
    assert result == 6
    assert b.flopscope_overhead_time >= 0.01


def test_counted_wrapper_handles_exceptions():
    import flopscope
    from flopscope._budget import _counted_wrapper

    @_counted_wrapper
    def boom():
        raise ValueError("nope")

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        with pytest.raises(ValueError):
            boom()
    # Overhead recorded despite the exception
    assert b.flopscope_overhead_time > 0


def test_optimer_splits_block_into_tracked_and_overhead():
    import time as _time

    import flopscope
    from flopscope._budget import _call_numpy

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        with b.deduct("x", flop_cost=1, subscripts=None, shapes=()):
            _time.sleep(0.005)  # in-block overhead (no _call_numpy)
            _call_numpy(_time.sleep, 0.01)  # tracked
            _time.sleep(0.005)  # more in-block overhead
    op = b.op_log[-1]
    # tracked = ONLY the _call_numpy duration. OS scheduling can oversleep
    # significantly, so use a generous upper bound.
    assert op.duration is not None
    assert op.duration >= 0.009
    assert b.total_tracked_time == op.duration
    # in-block overhead = block - tracked, includes the two 0.005 sleeps
    assert op.flopscope_overhead is not None
    assert op.flopscope_overhead >= 0.008


def test_deduct_self_charges_body_to_overhead_and_oprecord():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        baseline = b.flopscope_overhead_time
        # Call deduct directly without entering the timer
        timer = b.deduct("x", flop_cost=1, subscripts=None, shapes=())
        # The deduct body itself should have charged some overhead
        assert b.flopscope_overhead_time > baseline
        # And the OpRecord should reflect it
        assert b.op_log[-1].flopscope_overhead is not None
        assert b.op_log[-1].flopscope_overhead > 0


def test_namespace_scope_charges_push_pop_to_overhead():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        baseline = b.flopscope_overhead_time
        with flopscope.namespace("ns"):
            pass
        # Push + pop are tiny but non-zero
        assert b.flopscope_overhead_time > baseline


def test_per_namespace_summary_includes_flopscope_overhead():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        with b.deduct("x", flop_cost=1, subscripts=None, shapes=()):
            pass
        with flopscope.namespace("ns"):
            with b.deduct("y", flop_cost=1, subscripts=None, shapes=()):
                pass
    d = b.summary_dict(by_namespace=True)
    assert "flopscope_overhead_time_s" in d["by_namespace"][None]
    assert "flopscope_overhead_time_s" in d["by_namespace"]["ns"]
    assert d["by_namespace"][None]["flopscope_overhead_time_s"] >= 0
    assert d["by_namespace"]["ns"]["flopscope_overhead_time_s"] >= 0


def test_factory_wrapper_tracked_time_is_numpy_only():
    import time as _time

    import numpy as np

    import flopscope

    plain_a = np.ones((10000,))
    plain_b = np.ones((10000,))

    # Pure numpy timing
    t0 = _time.perf_counter()
    for _ in range(1000):
        np.add(plain_a, plain_b)
    pure_numpy_wall = (_time.perf_counter() - t0) / 1000

    # Flopscope timing
    fnp_a = flopscope.numpy.asarray(plain_a)
    fnp_b = flopscope.numpy.asarray(plain_b)
    with flopscope.BudgetContext(flop_budget=int(1e15), quiet=True) as b:
        for _ in range(1000):
            flopscope.numpy.add(fnp_a, fnp_b)
    flopscope_tracked = b.total_tracked_time / 1000

    # Generous bounds for noisy timing
    assert flopscope_tracked < pure_numpy_wall * 3.0
    assert flopscope_tracked > pure_numpy_wall * 0.3


def test_decomposition_invariant_after_single_op():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        _ = flopscope.numpy.add(
            flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
        )
    d = b.summary_dict()
    total = d["tracked_time_s"] + d["flopscope_overhead_time_s"] + d["untracked_time_s"]
    assert d["wall_time_s"] == pytest.approx(total, abs=1e-6)


def test_decomposition_invariant_under_mixed_workload():
    import time as _time

    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        a = flopscope.numpy.ones((50, 50))
        _ = flopscope.numpy.matmul(a, a)
        _ = flopscope.numpy.linalg.inv(a + flopscope.numpy.eye(50))
        _ = flopscope.numpy.fft.fft(flopscope.numpy.ones((128,)))
        _ = flopscope.numpy.sort(flopscope.numpy.ones((100,)))
        _time.sleep(0.005)  # genuinely untracked
    d = b.summary_dict()
    total = d["tracked_time_s"] + d["flopscope_overhead_time_s"] + d["untracked_time_s"]
    assert d["wall_time_s"] == pytest.approx(total, abs=1e-6)
    # The sleep should land in untracked, not overhead
    assert d["untracked_time_s"] >= 0.004


def test_per_op_overhead_recorded():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        _ = flopscope.numpy.add(
            flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
        )
    op = b.op_log[-1]
    assert op.flopscope_overhead is not None
    assert op.flopscope_overhead >= 0


def test_per_op_overhead_sums_to_global_no_namespace():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        for _ in range(5):
            _ = flopscope.numpy.add(
                flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
            )
    per_op_total = sum(op.flopscope_overhead or 0.0 for op in b.op_log)
    # OpRecord.flopscope_overhead captures overhead attributed to each op that
    # created an OpRecord.  Zero-FLOP ops (e.g. ones) still add to the global
    # counter without creating an OpRecord, so per_op_total <= global overhead.
    # We verify that the per-op sum is positive and does not exceed the global.
    assert per_op_total > 0
    assert per_op_total <= b.flopscope_overhead_time + 1e-9


def test_per_namespace_overhead_breakdown():
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        with flopscope.namespace("ns1"):
            _ = flopscope.numpy.add(
                flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
            )
        with flopscope.namespace("ns2"):
            _ = flopscope.numpy.multiply(
                flopscope.numpy.ones((10,)), flopscope.numpy.ones((10,))
            )
    d = b.summary_dict(by_namespace=True)
    assert d["by_namespace"]["ns1"]["flopscope_overhead_time_s"] > 0
    assert d["by_namespace"]["ns2"]["flopscope_overhead_time_s"] > 0


def test_overhead_recorded_on_budget_exhausted():
    import flopscope
    from flopscope.errors import BudgetExhaustedError

    with flopscope.BudgetContext(flop_budget=10, quiet=True) as b:
        with pytest.raises(BudgetExhaustedError):
            _ = flopscope.numpy.matmul(
                flopscope.numpy.ones((100, 100)),
                flopscope.numpy.ones((100, 100)),
            )
    assert b.flopscope_overhead_time > 0


def test_overhead_recorded_on_numpy_call_exception():
    import numpy as np

    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        # Singular matrix raises LinAlgError inside the numpy call
        singular = flopscope.numpy.zeros((3, 3))
        with pytest.raises(np.linalg.LinAlgError):
            _ = flopscope.numpy.linalg.inv(singular)
    # Decorator's try/finally captured the wrapper time
    assert b.flopscope_overhead_time > 0


def test_nested_wrapper_no_double_count():
    """Calling a wrapped op transitively (e.g., einsum internally calling
    np.tensordot through opt_einsum) does not double-count overhead."""
    import flopscope

    with flopscope.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
        # einsum's _execute_pairwise dispatches to multiple inner ops
        _ = flopscope.numpy.einsum(
            "ij,jk,kl->il",
            flopscope.numpy.ones((4, 4)),
            flopscope.numpy.ones((4, 4)),
            flopscope.numpy.ones((4, 4)),
        )
    d = b.summary_dict()
    total = d["tracked_time_s"] + d["flopscope_overhead_time_s"] + d["untracked_time_s"]
    assert d["wall_time_s"] == pytest.approx(total, abs=1e-6)


def test_check_nan_inf_opt_in_attributes_to_overhead():
    """Toggling check_nan_inf=True should grow flopscope overhead.

    Each call pays an extra np.any(np.isnan | np.isinf) scan — pure overhead.
    We measure the per-call overhead delta directly to avoid cross-run noise.
    """
    import flopscope

    N_CALLS = 200
    ARRAY_SIZE = 100_000  # large enough for np.any scan to be measurable

    # Warmup to ensure modules are loaded before timing
    for _ in range(3):
        with flopscope.BudgetContext(flop_budget=int(1e15), quiet=True):
            _ = flopscope.numpy.add(
                flopscope.numpy.ones((ARRAY_SIZE,)),
                flopscope.numpy.ones((ARRAY_SIZE,)),
            )

    flopscope.configure(check_nan_inf=False)
    with flopscope.BudgetContext(flop_budget=int(1e15), quiet=True) as b_off:
        for _ in range(N_CALLS):
            _ = flopscope.numpy.add(
                flopscope.numpy.ones((ARRAY_SIZE,)),
                flopscope.numpy.ones((ARRAY_SIZE,)),
            )

    flopscope.configure(check_nan_inf=True)
    try:
        with flopscope.BudgetContext(flop_budget=int(1e15), quiet=True) as b_on:
            for _ in range(N_CALLS):
                _ = flopscope.numpy.add(
                    flopscope.numpy.ones((ARRAY_SIZE,)),
                    flopscope.numpy.ones((ARRAY_SIZE,)),
                )
    finally:
        flopscope.configure(check_nan_inf=False)

    overhead_off = b_off.flopscope_overhead_time
    overhead_on = b_on.flopscope_overhead_time

    # check_nan_inf=True must add measurable overhead. We verify per-call
    # overhead grew by at least 5 µs per call (a very conservative lower bound
    # for np.any on a 100k-element array) to tolerate timing noise.
    per_call_delta = (overhead_on - overhead_off) / N_CALLS
    assert per_call_delta > 5e-6, (
        f"Expected per-call overhead to grow by >5µs with check_nan_inf=True; "
        f"got {per_call_delta * 1e6:.2f}µs "
        f"(overhead_on={overhead_on:.4f}s, overhead_off={overhead_off:.4f}s)"
    )


def test_untracked_time_property_agrees_with_summary_dict():
    """Regression: BudgetContext.untracked_time property must match
    summary_dict()['untracked_time_s']. Prior to the fix, the property
    returned `wall - tracked` (pre-#76 semantics) while summary_dict
    returned `wall - tracked - overhead` — the two disagreed by exactly
    flopscope_overhead_time.
    """
    import flopscope
    import flopscope.numpy as fnp

    ctx = flopscope.BudgetContext(flop_budget=int(1e12), quiet=True)
    with ctx:
        # Generate enough wrapper overhead that the bug would be visible
        a = fnp.eye(64)
        for _ in range(20):
            a = a @ a + a

    prop_value = ctx.untracked_time
    summary_value = ctx.summary_dict()["untracked_time_s"]
    assert prop_value is not None
    assert summary_value is not None
    assert prop_value == pytest.approx(summary_value, abs=1e-9)
    # Also verify both equal the documented identity.
    expected = (
        ctx.wall_time_s - ctx.total_tracked_time - ctx.flopscope_overhead_time  # type: ignore[operator]
    )
    assert prop_value == pytest.approx(max(expected, 0.0), abs=1e-9)
