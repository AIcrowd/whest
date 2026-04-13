# Calibrate Weights

## When to use this page

Use this page to measure per-operation FLOP weights empirically and produce
a weights config file for whest.

## Prerequisites

- [FLOP Counting Model](../concepts/flop-counting-model.md) -- understand
  what weights are and how they compose with analytical formulas

## Why calibrate?

whest's analytical cost formulas treat all pointwise operations equally
-- `exp`, `log`, `sin`, and `abs` all cost $\text{numel}(\text{output})$
FLOPs. Per-operation weights correct for the fact that transcendental
functions like `exp` decompose into many more basic floating-point
operations than simple functions like `abs`.

After calibration, a call to `we.exp(x)` on a 1000-element array might
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

- `weights.json` -- the weights config for whest
- `report.html` -- a human-readable HTML dashboard

To use the weights:

```bash
export WHEST_WEIGHTS_FILE=weights.json
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

The suite benchmarks fourteen categories of operations, all using the
same correction formula:

| Category | Operations | Size | Normalization |
|----------|-----------|------|---------------|
| **Pointwise Unary** | 47 ops | 10M elements | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Pointwise Binary** | 32 ops | 10M elements | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Reductions** | 35 ops | 10M elements | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Sorting** | 17 ops | 10M elements | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **FFT** | 14 FFT variants | $2^{20}$ elements | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Linalg** | 14 decomposition/solver ops | 1024 x 1024 | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Contractions** | 9 BLAS contraction ops | 512 x 512 / 10K vectors | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Polynomial** | 10 polynomial ops | 1M elements, deg=100 | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Random** | 43 random generators | 10M elements | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Misc** | 25 misc ops | Varies per op | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Window** | 5 window functions | 10M elements | $\max(\alpha_{\text{raw}} - \text{overhead}, 0)$ |
| **Bitwise** | 14 bitwise/integer ops | 10M int64 elements | Timing only (integer ops produce 0 perf FP counts) |
| **Complex** | 11 complex ops | 10M complex128 elements | Perf on complex128 (iscomplexobj/isrealobj use timing) |
| **Linalg Delegates** | 15 linalg namespace ops | Various | Perf (same as linalg decompositions) |

**Total: 291 benchmarked operations.**

### Measurement methodology (v3)

Every operation's raw correction factor is measured as:

$$
\alpha_{\text{raw}}(\text{op}) = \mathrm{median}_{D} \left[ \frac{F(\text{op})}{C(\text{op}, \text{params}) \times R} \right]
$$

where $F(\text{op})$ is the SIMD-width-weighted count of retired FP instructions
(`fp_arith_inst_retired.*`: scalar x1, 128-bit x2, 256-bit x4, 512-bit x8),
$C$ is the analytical FLOP formula (FMA = 1 op), and $R$ is the number of repeats.

#### Ufunc overhead subtraction

Numpy's ufunc dispatch layer generates spurious FP instructions (type checking,
iterator setup, error-state management) that inflate measurements for
elementwise operations. We measure this overhead using `np.abs` (which is a
bitwise sign-bit clear, NOT an FP instruction -- all measured FP instructions
are pure ufunc overhead), then subtract it per category:

$$
w(\text{op}) = \max\bigl(\alpha_{\text{raw}}(\text{op}) - \text{overhead}_{\text{category}}, \ 0\bigr)
$$

| Category | Overhead source | Typical value |
|----------|----------------|---------------|
| `ufunc_unary` | $\alpha(\texttt{abs})$ | ~0.6 |
| `ufunc_binary` | $\alpha(\texttt{add}) - 1.0$ | ~1.2 |
| `ufunc_reduction` | same as unary | ~0.6 |
| `blas` / `linalg` | 0 (bypasses ufunc) | 0 |
| `custom` (fft, sort, etc.) | 0 | 0 |
| `instructions` (bitwise) | 0 (different counter) | 0 |

**Note on BLAS/linalg FMA ops:** `fp_arith_inst_retired` counts each FMA
as 2 retired operations (one multiply + one add). Pure-FMA ops like
matmul will therefore show empirical weight ≈ 2.0. The reviewer can
decide whether to keep this or override to 1.0.

This replaces the v2 formula $w = \alpha(\text{op}) / \alpha(\text{add})$
which penalized BLAS operations (that bypass the ufunc layer) with overhead
they don't have.

#### Measurement modes

| Mode | Counter | Used for |
|------|---------|----------|
| **perf** | `fp_arith_inst_retired.*` (SIMD-weighted) | FP operations (default) |
| **instructions** | `instructions` (total retired) | Integer/bitwise ops |
| **timing** | `time.perf_counter_ns()` | Validation; fallback when perf unavailable |

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

Available categories: `pointwise`, `reductions`, `linalg`, `linalg_delegates`,
`fft`, `sorting`, `random`, `polynomial`, `contractions`, `misc`, `window`,
`bitwise`, `complex`.

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
      "version": "3.0",
      "formula": "weight(op) = max(alpha_raw(op) - overhead_for_category, 0)",
      "baseline_alpha_add_raw": 1.6001,
      "baseline_alpha_abs_raw": 0.3001,
      "overhead_ufunc_unary": 0.3001,
      "overhead_ufunc_binary": 0.6001,
      "note": "ufunc overhead subtracted per category; FMA=1"
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
    "abs": 1.0,
    "add": 1.0,
    "exp": 16.0,
    "sin": 16.0,
    "matmul": 1.0,
    "linalg.cholesky": 4.0,
    "fft.fft": 1.0
  }
}
```

whest reads only the `"weights"` dict at runtime. The `"meta"` section
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
from whest._weights import load_weights, reset_weights, get_weight

# Load from a specific file
load_weights("/path/to/weights.json")

# Check a weight
print(get_weight("exp"))   # 16.0
print(get_weight("add"))   # 1.0
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
- [FLOP Weight Calibration Results](../reference/empirical-weights.md) -- full weight tables and methodology reference
- [Plan Your Budget](./plan-your-budget.md) -- query costs before running
- [Operation Categories](../concepts/operation-categories.md) -- which operations are counted
