"""Tests for the new fma_cost setting."""

import pytest

from flopscope._config import get_setting, set_setting


def test_fma_cost_default_is_one():
    assert get_setting("fma_cost") == 1


def test_fma_cost_can_be_set_to_two():
    original = get_setting("fma_cost")
    try:
        set_setting("fma_cost", 2)
        assert get_setting("fma_cost") == 2
    finally:
        set_setting("fma_cost", original)


def test_fma_cost_rejects_zero():
    original = get_setting("fma_cost")
    try:
        with pytest.raises((ValueError, TypeError)):
            set_setting("fma_cost", 0)
    finally:
        set_setting("fma_cost", original)


def test_fma_cost_rejects_three():
    original = get_setting("fma_cost")
    try:
        with pytest.raises((ValueError, TypeError)):
            set_setting("fma_cost", 3)
    finally:
        set_setting("fma_cost", original)


def test_fma_cost_rejects_negative():
    original = get_setting("fma_cost")
    try:
        with pytest.raises((ValueError, TypeError)):
            set_setting("fma_cost", -1)
    finally:
        set_setting("fma_cost", original)


def test_fma_cost_rejects_string():
    original = get_setting("fma_cost")
    try:
        with pytest.raises((ValueError, TypeError)):
            set_setting("fma_cost", "1")
    finally:
        set_setting("fma_cost", original)
