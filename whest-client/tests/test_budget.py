"""Unit tests for BudgetContext and OpRecord.

All tests mock the connection — no server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import msgpack
import pytest


@pytest.fixture(autouse=True)
def _reset_active_context():
    """Reset the module-level _active_context guard between tests."""
    import whest._budget as bmod

    old = bmod._active_context
    bmod._active_context = None
    yield
    bmod._active_context = old


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack_response(payload: dict) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def _make_mock_conn(response: dict) -> MagicMock:
    """Return a mock Connection whose send_recv always returns *response*."""
    conn = MagicMock()
    conn.send_recv.return_value = response
    return conn


# ---------------------------------------------------------------------------
# OpRecord
# ---------------------------------------------------------------------------


class TestOpRecord:
    """OpRecord stores op metadata and is accessible via attributes."""

    def test_op_name(self):
        from whest._budget import OpRecord

        rec = OpRecord(op_name="dot", flop_cost=100, cumulative=500)
        assert rec.op_name == "dot"

    def test_flop_cost(self):
        from whest._budget import OpRecord

        rec = OpRecord(op_name="matmul", flop_cost=2000, cumulative=3000)
        assert rec.flop_cost == 2000

    def test_cumulative(self):
        from whest._budget import OpRecord

        rec = OpRecord(op_name="add", flop_cost=10, cumulative=110)
        assert rec.cumulative == 110


# ---------------------------------------------------------------------------
# BudgetContext – attribute defaults
# ---------------------------------------------------------------------------


class TestBudgetContextAttributes:
    """BudgetContext stores parameters without connecting."""

    def test_flop_budget_stored(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        assert ctx.flop_budget == 1000

    def test_flop_multiplier_default(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=500)
        assert ctx.flop_multiplier == 1.0

    def test_flop_multiplier_custom(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=500, flop_multiplier=2.5)
        assert ctx.flop_multiplier == 2.5

    def test_flops_used_starts_zero(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        assert ctx.flops_used == 0

    def test_flops_remaining_equals_budget_minus_used(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        assert ctx.flops_remaining == 1000

    def test_quiet_default_false(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=100)
        assert ctx.quiet is False

    def test_quiet_custom(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=100, quiet=True)
        assert ctx.quiet is True


# ---------------------------------------------------------------------------
# BudgetContext – _update_budget
# ---------------------------------------------------------------------------


class TestUpdateBudget:
    """_update_budget patches local flops_used from a server-response dict."""

    def test_update_flops_used(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({"flops_used": 300})
        assert ctx.flops_used == 300

    def test_flops_remaining_after_update(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({"flops_used": 400})
        assert ctx.flops_remaining == 600

    def test_update_ignores_missing_key(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({})  # no flops_used key — should not raise
        assert ctx.flops_used == 0

    def test_update_multiple_times(self):
        from whest._budget import BudgetContext

        ctx = BudgetContext(flop_budget=1000)
        ctx._update_budget({"flops_used": 100})
        ctx._update_budget({"flops_used": 250})
        assert ctx.flops_used == 250
        assert ctx.flops_remaining == 750


# ---------------------------------------------------------------------------
# BudgetContext – context manager (__enter__ / __exit__)
# ---------------------------------------------------------------------------


class TestBudgetContextManager:
    """__enter__ sends budget_open; __exit__ sends budget_close."""

    def test_enter_sends_budget_open(self):

        from whest._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        with patch("whest._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=500)
            result = ctx.__enter__()
            assert result is ctx
            mock_conn.send_recv.assert_called_once()
            # Verify the payload encodes budget_open
            sent_bytes = mock_conn.send_recv.call_args[0][0]
            decoded = msgpack.unpackb(sent_bytes, raw=False)
            assert decoded["op"] == "budget_open"
            assert decoded["kwargs"]["flop_budget"] == 500

    def test_enter_updates_flops_used_from_response(self):
        from whest._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 50})
        with patch("whest._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=1000)
            ctx.__enter__()
            assert ctx.flops_used == 50

    def test_enter_returns_self(self):
        from whest._budget import BudgetContext

        mock_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        with patch("whest._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=200)
            returned = ctx.__enter__()
            assert returned is ctx

    def test_exit_sends_budget_close(self):
        from whest._budget import BudgetContext

        open_conn = _make_mock_conn({"status": "ok", "flops_used": 0})
        close_resp = {"status": "ok", "flops_used": 75}
        open_conn.send_recv.side_effect = [
            {"status": "ok", "flops_used": 0},
            close_resp,
        ]
        with patch("whest._budget.get_connection", return_value=open_conn):
            ctx = BudgetContext(flop_budget=200)
            ctx.__enter__()
            ctx.__exit__(None, None, None)
            # Second call should be budget_close
            assert open_conn.send_recv.call_count == 2
            close_bytes = open_conn.send_recv.call_args_list[1][0][0]
            decoded = msgpack.unpackb(close_bytes, raw=False)
            assert decoded["op"] == "budget_close"

    def test_context_manager_with_statement(self):
        from whest._budget import BudgetContext

        responses = [
            {"status": "ok", "flops_used": 0},  # budget_open
            {"status": "ok", "flops_used": 100},  # budget_close
        ]
        mock_conn = MagicMock()
        mock_conn.send_recv.side_effect = responses

        with patch("whest._budget.get_connection", return_value=mock_conn):
            with BudgetContext(flop_budget=1000) as ctx:
                assert isinstance(ctx, BudgetContext)
            assert mock_conn.send_recv.call_count == 2


# ---------------------------------------------------------------------------
# BudgetContext – summary
# ---------------------------------------------------------------------------


class TestBudgetContextSummary:
    """summary() sends budget_status, updates cache, returns a string."""

    def test_summary_sends_budget_status(self):
        from whest._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "flops_used": 200,
                "flop_budget": 1000,
            }
        )
        with patch("whest._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=1000)
            ctx.summary()
            sent_bytes = mock_conn.send_recv.call_args[0][0]
            decoded = msgpack.unpackb(sent_bytes, raw=False)
            assert decoded["op"] == "budget_status"

    def test_summary_updates_flops_used(self):
        from whest._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"flops_used": 350, "flop_budget": 1000},
            }
        )
        with patch("whest._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=1000)
            ctx.summary()
            assert ctx.flops_used == 350

    def test_summary_returns_string(self):
        from whest._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "result": {"flops_used": 100, "flop_budget": 500},
            }
        )
        with patch("whest._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=500)
            result = ctx.summary()
            assert isinstance(result, str)

    def test_summary_contains_budget_info(self):
        from whest._budget import BudgetContext

        mock_conn = _make_mock_conn(
            {
                "status": "ok",
                "flops_used": 100,
                "flop_budget": 500,
            }
        )
        with patch("whest._budget.get_connection", return_value=mock_conn):
            ctx = BudgetContext(flop_budget=500)
            result = ctx.summary()
            # Should contain numbers related to budget usage
            assert "100" in result or "500" in result
