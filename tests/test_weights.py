"""Tests for FLOP weight loading and lookup."""

import json
import os
import tempfile

import pytest

from mechestim._weights import get_weight, load_weights, reset_weights


def test_get_weight_default_is_one():
    reset_weights()
    assert get_weight("exp") == 1.0
    assert get_weight("nonexistent_op") == 1.0


def test_load_weights_from_file():
    reset_weights()
    data = {
        "meta": {"timestamp": "2026-04-06T00:00:00Z"},
        "weights": {"exp": 18.3, "log": 12.7, "add": 1.0},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        load_weights(path)
        assert get_weight("exp") == 18.3
        assert get_weight("log") == 12.7
        assert get_weight("add") == 1.0
        assert get_weight("sin") == 1.0  # not in file, falls back
    finally:
        os.unlink(path)


def test_load_weights_from_env_var(monkeypatch):
    reset_weights()
    data = {"weights": {"sqrt": 3.8}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        monkeypatch.setenv("MECHESTIM_WEIGHTS_FILE", path)
        load_weights()  # no explicit path — reads env var
        assert get_weight("sqrt") == 3.8
    finally:
        os.unlink(path)


def test_load_weights_missing_file_raises():
    reset_weights()
    with pytest.raises(FileNotFoundError):
        load_weights("/nonexistent/path.json")


def test_load_weights_invalid_json():
    reset_weights()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json")
        path = f.name
    try:
        with pytest.raises(json.JSONDecodeError):
            load_weights(path)
    finally:
        os.unlink(path)


def test_reset_weights_clears():
    data = {"weights": {"exp": 18.3}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        load_weights(path)
        assert get_weight("exp") == 18.3
        reset_weights()
        assert get_weight("exp") == 1.0
    finally:
        os.unlink(path)
