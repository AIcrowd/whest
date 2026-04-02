"""Tests for Rich budget display and plain-text fallback."""

from unittest.mock import patch

import pytest

from mechestim._budget import BudgetContext
from mechestim._display import render_budget_summary, _plain_text_summary, budget_live


def test_plain_text_summary_single_namespace():
    with BudgetContext(flop_budget=1000, namespace="train", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
        ctx.deduct("mul", flop_cost=200, subscripts=None, shapes=())

    text = _plain_text_summary()
    assert "train" in text
    assert "300" in text.replace(",", "")
    assert "add" in text
    assert "mul" in text


def test_plain_text_summary_multiple_namespaces():
    with BudgetContext(flop_budget=1000, namespace="train", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())
    with BudgetContext(flop_budget=500, namespace="eval", quiet=True) as ctx:
        ctx.deduct("mul", flop_cost=50, subscripts=None, shapes=())

    text = _plain_text_summary()
    assert "train" in text
    assert "eval" in text


def test_plain_text_summary_no_data():
    text = _plain_text_summary()
    assert "No budget data" in text


def test_render_budget_summary_falls_back_to_text():
    """When Rich is not available, render_budget_summary returns plain text."""
    with BudgetContext(flop_budget=1000, namespace="test", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())

    with patch.dict("sys.modules", {"rich": None, "rich.panel": None, "rich.table": None, "rich.text": None}):
        result = render_budget_summary()
        assert isinstance(result, str)
        assert "test" in result


def test_render_budget_summary_with_rich():
    """When Rich is available, render_budget_summary returns a Rich renderable."""
    pytest.importorskip("rich")

    with BudgetContext(flop_budget=1000, namespace="test", quiet=True) as ctx:
        ctx.deduct("add", flop_cost=100, subscripts=None, shapes=())

    result = render_budget_summary()
    from rich.table import Table
    assert isinstance(result, Table)


def test_budget_live_is_context_manager():
    """budget_live() returns a context manager."""
    live = budget_live()
    assert hasattr(live, "__enter__")
    assert hasattr(live, "__exit__")
