# Calibrate Weights

## When to use this page

Use this page to measure per-operation FLOP weights empirically and produce
a weights config file for mechestim.

## Prerequisites

- [FLOP Counting Model](../concepts/flop-counting-model.md) -- understand
  what weights are and how they compose with analytical formulas

## Why calibrate?

mechestim's analytical cost formulas treat all pointwise operations equally
-- `exp`, `log`, `sin`, and `abs` all cost $\text{numel}(\text{output})$
FLOPs. Per-operation weights correct for the fact that transcendental
functions like `exp` decompose into many more basic floating-point
operations than simple functions like `abs`.

After calibration, a call to `me.exp(x)` on a 1000-element array might
cost 14,450 weighted FLOPs instead of 1,000 unweighted FLOPs -- reflecting
the approximately 14.4 basic FP instructions that the `exp` polynomial
approximation uses internally.

## Quick start

Run the benchmark suite and produce a JSON config:

```bash
python -m benchmarks.runner \
    --dtype float64 \
    --output weights.json \
    --html report.html \
    --repeats 5
```

This benchmarks all operation categories and writes:

- `weights.json` -- the weights config for mechestim
- `report.html` -- a human-readable HTML dashboard

To use the weights:

```bash
export MECHESTIM_WEIGHTS_FILE=weights.json
python your_code.py
```

## Measurement modes

The benchmark suite supports two measurement modes, chosen automatically:

| Mode | Platform | What it measures | Accuracy |
|------|----------|------------------|----------|
| **perf** | Linux (with `perf` installed) | Actual FP instructions via hardware counters (`fp_arith_inst_retired.*`), weighted by SIMD width | Exact FP op counts |
| **timing** | Any (macOS, Linux without `perf`) | Wall-clock time via `time.perf_counter_ns()`, normalized against `np.add` | Relative proxy -- correct ordering, approximate magnitudes |

The mode is reported at the start of every benchmark run:

```
Measurement mode: perf
```

or:

```
Measurement mode: timing
  (perf not available -- using wall-clock time as proxy)
```

## What gets measured

The suite benchmarks eleven categories of operations, all using the
same unified correction-factor formula:

| Category | Operations | Size | Normalization |
|----------|-----------|------|---------------|
| **Pointwise Unary** | 47 ops | 10M elements | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Pointwise Binary** | 32 ops | 10M elements | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Reductions** | 35 ops | 10M elements | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Sorting** | 17 ops | 10M elements | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **FFT** | 14 FFT variants | $2^{20}$ elements | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Linalg** | 14 decomposition/solver ops | 1024 x 1024 | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Contractions** | 9 BLAS contraction ops | 512 x 512 / 10K vectors | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Polynomial** | 10 polynomial ops | 1M elements, deg=100 | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Random** | 43 random generators | 10M elements | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Misc** | 25 misc ops | Varies per op | $\alpha(\text{op}) / \alpha(\text{add})$ |
| **Window** | 5 window functions | 10M elements | $\alpha(\text{op}) / \alpha(\text{add})$ |

**Total: 251 benchmarked operations.**

### Measurement methodology

Every operation uses the same unified formula:

$$
\alpha(\text{op}) = \text{median}_{D} \left[ \frac{\sum \texttt{fp\_arith\_inst\_retired.*} \times \texttt{simd\_width}}{C(\text{op}, \text{params}) \times R} \right]
$$

$$
\text{weight}(\text{op}) = \frac{\alpha(\text{op})}{\alpha(\text{add})}
$$

Where $C(\text{op}, \text{params})$ is mechestim's analytical FLOP formula
for the operation and $R$ is the number of repetitions. The $\alpha(\text{add})$
baseline is measured identically and used as the universal normalization
constant for all categories.

Each measurement uses:

1. **Pre-allocated output** -- all benchmarks use `out=` (where supported)
   to eliminate memory allocation overhead, isolating pure compute cost.
2. **Multiple input distributions** -- 3 distributions per operation (e.g.,
   standard normal, uniform positive, uniform wide range), with the
   **median** reported to smooth out data-dependent variance.
3. **Subprocess isolation** -- each measurement runs in a separate Python
   process to prevent interference from the benchmark framework's own work.
4. **Warmup** -- in timing mode, 2 warmup iterations are run and discarded
   before the timed loop.

## Available categories

To benchmark only certain categories:

```bash
# Only pointwise and linalg
python -m benchmarks.runner \
    --dtype float64 \
    --output weights.json \
    --category pointwise \
    --category linalg
```

Available categories: `pointwise`, `reductions`, `linalg`, `fft`,
`sorting`, `random`, `polynomial`, `contractions`, `misc`, `window`.

## Output format

The v2 JSON config has two top-level keys, with expanded `meta` containing
methodology and validation sections:

```json
{
  "meta": {
    "timestamp": "2026-04-09T16:39:18.621649+00:00",
    "duration_seconds": 2268.4,
    "hardware": {
      "cpu_model": "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
      "cpu_cores": 64,
      "cpu_threads": 128,
      "ram_gb": 251.7,
      "arch": "x86_64",
      "cache": {"L1d": 48, "L1i": 32, "L2": 1280, "L3": 55296}
    },
    "software": {
      "os": "Linux 6.1.166",
      "python": "3.11.14",
      "numpy": "2.1.3",
      "blas": "scipy-openblas 0.3.27"
    },
    "benchmark_config": {
      "dtype": "float64",
      "repeats": 5,
      "distributions": 3,
      "measurement_mode": "perf",
      "perf_events": ["fp_arith_inst_retired.scalar_double", "..."]
    },
    "methodology": {
      "version": "2.0",
      "formula": "weight(op) = alpha(op) / alpha(add), ...",
      "baseline_alpha": 1.564071,
      "note": "analytical_FLOPs from mechestim registry; ..."
    },
    "validation": {
      "absolute_correction_factors": {"add": 2.2001, "exp": 22.6001, "...": "..."},
      "timing_weights": {"add": 0.9689, "exp": 0.8427, "...": "..."},
      "perf_vs_timing": {
        "pearson_r": 0.7342,
        "spearman_rho": 0.5319,
        "max_divergence": {
          "op": "matmul",
          "perf_weight": 0.6426,
          "timing_weight": 0.0009,
          "ratio": 714.0071
        }
      }
    }
  },
  "weights": {
    "abs": 0.3837,
    "add": 1.4067,
    "exp": 14.4495,
    "sin": 25.8688,
    "matmul": 0.6426,
    "linalg.cholesky": 1.0699,
    "fft.fft": 0.534
  }
}
```

mechestim reads only the `"weights"` dict at runtime. The `"meta"` section
is for human reference, the HTML dashboard, and scientific reproducibility.

### How weights are interpreted

All weights -- regardless of operation category -- use the same
interpretation:

- **weight = 1.0** means the operation retires the same number of FP
  instructions per analytical FLOP as `np.add`.
- **weight > 1.0** means more FP instructions per analytical FLOP.
  Example: `exp` at 14.4 means each analytical FLOP of `exp` costs
  approximately 14 FP instructions, compared to 1 for `add`.
- **weight < 1.0** means fewer FP instructions per analytical FLOP.
  This is the **FMA effect** -- operations backed by BLAS (e.g., `matmul`,
  `dot`, `einsum`, linalg decompositions, some FFTs) use Fused Multiply-Add
  instructions that compute $a \times b + c$ in one instruction but count
  as 2 analytical FLOPs. A weight of 0.64 for `matmul` means the
  FMA-optimized inner loop fuses two analytical FLOPs into roughly one
  instruction.
- **Missing operations** default to 1.0. A partial config (e.g.,
  pointwise-only) works without error.

## Validation

The benchmark suite performs dual-mode validation by running every
operation in both perf mode (hardware counters) and timing mode (wall-clock
nanoseconds). Both modes use the identical formula, differing only in the
numerator (instruction count vs. elapsed nanoseconds).

The validation section in `meta.validation.perf_vs_timing` reports:

- **Pearson r** -- linear correlation between perf and timing weight
  vectors. Values above 0.99 indicate excellent agreement for pointwise-
  dominated workloads.
- **Spearman rho** -- rank correlation. Are the orderings consistent
  between modes? Values above 0.95 indicate consistent relative rankings.
- **Max divergence** -- the operation with the largest ratio between perf
  and timing weights. BLAS operations typically show the largest
  divergence because they achieve much higher FLOP throughput than scalar
  operations, making their timing weights disproportionately small.

!!! note "Expected divergence for BLAS/linalg"

    Perf mode counts FP instructions regardless of throughput, while timing
    mode measures wall-clock time that reflects memory bandwidth, cache
    effects, and instruction-level parallelism. BLAS operations achieve
    near-peak throughput, so their per-instruction timing is much lower than
    for scalar operations. This structural divergence is expected and does
    not indicate a measurement error.

## Programmatic usage

You can load and reset weights from Python:

```python
from mechestim._weights import load_weights, reset_weights, get_weight

# Load from a specific file
load_weights("/path/to/weights.json")

# Check a weight
print(get_weight("exp"))   # 14.4495
print(get_weight("add"))   # 1.4067
print(get_weight("foo"))   # 1.0 (unknown ops default to 1.0)

# Reset to defaults (all 1.0)
reset_weights()
```

## Caveats

- **Weights are approximate.** Many operations have internal branching
  based on input size, values, or array layout. A single constant measured
  at one size does not capture all regimes.

- **Weights are platform-specific.** Different CPUs, BLAS libraries, and
  libm implementations produce different constants. Always measure on the
  target platform.

- **Timing mode is a proxy.** Wall-clock measurements include cache
  effects, memory bandwidth, and SIMD efficiency -- not just FP compute.
  The relative ordering is correct, but magnitudes may differ from true FP
  op counts.

- **Symmetry is independent of weights.** Symmetry reduces the element
  count in the analytical formula. Weights scale the per-element cost.
  Both apply independently: `cost = unique_elements * weight(op_name)`.

## Related pages

- [FLOP Counting Model](../concepts/flop-counting-model.md) -- how weights fit into the cost model
- [Empirical Weights](../reference/empirical-weights.md) -- full weight tables and methodology reference
- [Plan Your Budget](./plan-your-budget.md) -- query costs before running
- [Operation Categories](../concepts/operation-categories.md) -- which operations are counted
