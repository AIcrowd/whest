"""Tests for einsum path caching."""

import numpy
import pytest

from whest._config import configure, get_setting


def test_einsum_path_cache_size_default():
    assert get_setting("einsum_path_cache_size") == 4096


def test_einsum_path_cache_size_configurable():
    original = get_setting("einsum_path_cache_size")
    try:
        configure(einsum_path_cache_size=256)
        assert get_setting("einsum_path_cache_size") == 256
    finally:
        configure(einsum_path_cache_size=original)
