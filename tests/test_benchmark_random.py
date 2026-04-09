"""Tests for random benchmark module."""

from unittest.mock import patch

import pytest

from benchmarks._perf import PerfResult
from benchmarks._random import RANDOM_OPS, benchmark_random


class TestOpsLists:
    def test_random_ops_non_empty(self):
        assert len(RANDOM_OPS) > 0

    def test_contains_expected_ops(self):
        for op in (
            "random.standard_normal",
            "random.uniform",
            "random.poisson",
            "random.shuffle",
        ):
            assert op in RANDOM_OPS, f"{op} missing from RANDOM_OPS"

    def test_contains_new_distribution_ops(self):
        for op in (
            "random.beta",
            "random.chisquare",
            "random.choice",
            "random.dirichlet",
            "random.exponential",
            "random.f",
            "random.gamma",
            "random.geometric",
            "random.gumbel",
            "random.hypergeometric",
            "random.laplace",
            "random.logistic",
            "random.lognormal",
            "random.logseries",
            "random.multinomial",
            "random.multivariate_normal",
            "random.negative_binomial",
            "random.noncentral_chisquare",
            "random.noncentral_f",
            "random.normal",
            "random.pareto",
            "random.power",
            "random.rand",
            "random.randint",
            "random.randn",
            "random.random",
            "random.random_sample",
            "random.rayleigh",
            "random.triangular",
            "random.vonmises",
            "random.wald",
            "random.weibull",
            "random.zipf",
        ):
            assert op in RANDOM_OPS, f"{op} missing from RANDOM_OPS"


class TestBenchmarkRandom:
    def test_returns_dict_with_all_ops(self):
        mock_result = PerfResult(
            scalar_double=1_000_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._random.measure_flops", return_value=mock_result):
            result = benchmark_random(n=1_000, dtype="float64", repeats=1)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(RANDOM_OPS)

    def test_values_are_floats(self):
        mock_result = PerfResult(
            scalar_double=500_000,
            packed_128_double=0,
            packed_256_double=0,
            packed_512_double=0,
        )
        with patch("benchmarks._random.measure_flops", return_value=mock_result):
            result = benchmark_random(n=1_000, dtype="float64", repeats=1)

        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not float"

    def test_flops_per_element_calculation(self):
        mock_result = PerfResult(
            scalar_double=0,
            packed_128_double=0,
            packed_256_double=2_000,
            packed_512_double=0,
        )
        with patch("benchmarks._random.measure_flops", return_value=mock_result):
            result = benchmark_random(n=1_000, dtype="float64", repeats=4)

        # total_flops = 2000 * 4 = 8000, per_element = 8000 / (1000 * 4) = 2.0
        for val in result.values():
            assert val == pytest.approx(2.0)
