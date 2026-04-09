"""Benchmark random number generation operations."""

from __future__ import annotations

import statistics

from benchmarks._perf import measure_flops

RANDOM_OPS: list[str] = [
    "random.standard_normal",
    "random.uniform",
    "random.standard_exponential",
    "random.standard_cauchy",
    "random.standard_gamma",
    "random.standard_t",
    "random.poisson",
    "random.binomial",
    "random.permutation",
    "random.shuffle",
    # --- added in Step 2.5 ---
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
]

# Ops that need extra arguments beyond size.
_EXTRA_ARGS: dict[str, str] = {
    "random.uniform": "0.0, 1.0, ",
    "random.standard_gamma": "1.0, ",
    "random.standard_t": "3, ",
    "random.poisson": "5.0, ",
    "random.binomial": "10, 0.5, ",
    # --- added in Step 2.5 ---
    "random.beta": "2.0, 5.0, ",
    "random.chisquare": "2, ",
    "random.exponential": "1.0, ",
    "random.f": "5, 10, ",
    "random.gamma": "2.0, 1.0, ",
    "random.geometric": "0.5, ",
    "random.gumbel": "0.0, 1.0, ",
    "random.hypergeometric": "100, 50, 20, ",
    "random.laplace": "0.0, 1.0, ",
    "random.logistic": "0.0, 1.0, ",
    "random.lognormal": "0.0, 1.0, ",
    "random.logseries": "0.5, ",
    "random.negative_binomial": "10, 0.5, ",
    "random.noncentral_chisquare": "2, 1.0, ",
    "random.noncentral_f": "5, 10, 1.0, ",
    "random.normal": "0.0, 1.0, ",
    "random.pareto": "3.0, ",
    "random.power": "5.0, ",
    "random.rayleigh": "1.0, ",
    "random.triangular": "-1.0, 0.0, 1.0, ",
    "random.vonmises": "0.0, 1.0, ",
    "random.wald": "1.0, 1.0, ",
    "random.weibull": "2.0, ",
    "random.zipf": "2.0, ",
    "random.randint": "0, 1000, ",
}

# Ops that need completely custom setup/bench code.
_CUSTOM_OPS = frozenset({
    "random.choice",
    "random.dirichlet",
    "random.multinomial",
    "random.multivariate_normal",
})


def _custom_bench(op: str, n: int, dtype: str, seed: int) -> tuple[str, str]:
    """Return (setup, bench) for ops with non-standard patterns."""
    base_setup = f"import numpy as np; np.random.seed({seed})"

    if op == "random.choice":
        setup = base_setup + "; _pool = np.arange(1000)"
        bench = f"np.random.choice(_pool, {n})"
    elif op == "random.dirichlet":
        setup = base_setup
        bench = f"np.random.dirichlet([1.0, 2.0, 3.0], {n})"
    elif op == "random.multinomial":
        setup = base_setup
        bench = f"np.random.multinomial(10, [0.2, 0.3, 0.5], {n})"
    elif op == "random.multivariate_normal":
        setup = (
            base_setup
            + "; _mean = np.zeros(10); _cov = np.eye(10)"
        )
        bench = f"np.random.multivariate_normal(_mean, _cov, {n})"
    else:
        raise ValueError(f"Unknown custom op: {op}")

    return setup, bench


def benchmark_random(
    n: int = 10_000_000,
    dtype: str = "float64",
    repeats: int = 10,
) -> dict[str, float]:
    """Benchmark random ops, returning FP ops per element.

    Parameters
    ----------
    n : int
        Array size.
    dtype : str
        NumPy dtype string.
    repeats : int
        Number of repetitions per measurement.

    Returns
    -------
    dict[str, float]
        Mapping from op name to median FP ops per element.
    """
    results: dict[str, float] = {}
    seeds = [42, 123, 999]

    for op in RANDOM_OPS:
        dist_values: list[float] = []
        extra = _EXTRA_ARGS.get(op, "")

        for seed in seeds:
            if op in _CUSTOM_OPS:
                setup, bench = _custom_bench(op, n, dtype, seed)
            elif op == "random.shuffle":
                setup = (
                    f"import numpy as np; np.random.seed({seed})"
                    f"; x = np.arange({n}, dtype=np.{dtype})"
                )
                bench = "np.random.shuffle(x)"
            elif op == "random.permutation":
                setup = f"import numpy as np; np.random.seed({seed})"
                bench = f"np.random.permutation({n})"
            else:
                setup = f"import numpy as np; np.random.seed({seed})"
                bench = f"np.{op}({extra}{n})"

            try:
                result = measure_flops(setup, bench, repeats=repeats)
            except RuntimeError:
                continue
            dist_values.append(result.total_flops / (n * repeats))

        if dist_values:
            results[op] = statistics.median(dist_values)

    return results
