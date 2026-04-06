# Calibrate Weights

## When to use this page

Use this page to measure per-operation FLOP weights empirically and produce
a weights config file for mechestim.

## Prerequisites

- [FLOP Counting Model](../concepts/flop-counting-model.md) — understand
  what weights are and how they compose with analytical formulas

## Why calibrate?

mechestim's analytical cost formulas treat all pointwise operations equally
— `exp`, `log`, `sin`, and `abs` all cost $\text{numel}(\text{output})$
FLOPs. Per-operation weights correct for the fact that transcendental
functions like `exp` decompose into many more basic floating-point
operations than simple functions like `abs`.

After calibration, a call to `me.exp(x)` on a 1000-element array might
cost 23,000 weighted FLOPs instead of 1,000 unweighted FLOPs — reflecting
the ~23 basic FP ops that a typical `exp` implementation uses internally.

## Quick start

Run the benchmark suite and produce a JSON config:

```bash
python -m benchmarks.runner \
    --dtype float64 \
    --output weights.json \
    --html report.html \
    --repeats 10
```

This benchmarks all operation categories and writes:

- `weights.json` — the weights config for mechestim
- `report.html` — a human-readable HTML dashboard

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
| **timing** | Any (macOS, Linux without `perf`) | Wall-clock time via `time.perf_counter_ns()`, normalized against `np.add` | Relative proxy — correct ordering, approximate magnitudes |

The mode is reported at the start of every benchmark run:

```
Measurement mode: perf
```

or:

```
Measurement mode: timing
  (perf not available — using wall-clock time as proxy)
```

## What gets measured

The suite benchmarks seven categories of operations:

| Category | Operations | Size | Normalization |
|----------|-----------|------|---------------|
| **pointwise** | 68 unary + binary ops | 10M elements | per element, vs `add` baseline |
| **reductions** | 28 reduction ops | 10M elements | per element, vs `add` baseline |
| **linalg** | 14 decomposition/solver ops | 1024 × 1024 | per analytical FLOP, vs `add` baseline |
| **fft** | 14 FFT variants | 2²⁰ elements | per analytical FLOP, vs `add` baseline |
| **sorting** | 7 sort/search ops | 10M elements | per element, vs `add` baseline |
| **random** | 10 random generators | 10M elements | per element, vs `add` baseline |
| **polynomial** | 10 polynomial ops | 1M elements | per element, vs `add` baseline |

### Measurement methodology

Each operation is measured with:

1. **Pre-allocated output** — all benchmarks use `out=` to eliminate memory
   allocation overhead, isolating pure compute cost
2. **Multiple input distributions** — 3 distributions per operation (e.g.,
   standard normal, uniform positive, uniform wide range), with the
   **median** reported to smooth out data-dependent variance
3. **Subprocess isolation** — each measurement runs in a separate Python
   process to prevent interference from the benchmark framework's own work
4. **Warmup** — in timing mode, 2 warmup iterations are run and discarded
   before the timed loop

## Benchmarking specific categories

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
`sorting`, `random`, `polynomial`.

## Output format

The JSON config has two top-level keys:

```json
{
  "meta": {
    "timestamp": "2026-04-06T14:32:01+00:00",
    "duration_seconds": 462.6,
    "hardware": {
      "cpu_model": "Intel Xeon Gold 6248R",
      "cpu_cores": 24,
      "arch": "x86_64",
      "ram_gb": 128.0
    },
    "software": {
      "os": "Linux 5.15.0",
      "python": "3.11.7",
      "numpy": "2.1.3",
      "blas": "OpenBLAS 0.3.24"
    },
    "benchmark_config": {
      "dtype": "float64",
      "repeats": 10,
      "distributions": 3,
      "measurement_mode": "perf"
    }
  },
  "weights": {
    "abs": 0.67,
    "add": 1.0,
    "exp": 23.43,
    "sin": 30.08,
    "linalg.cholesky": 0.10,
    "fft.fft": 0.64
  }
}
```

mechestim reads only the `"weights"` dict. The `"meta"` section is for
human reference and the HTML dashboard.

### How weights are interpreted

- **Pointwise / reduction / sorting / random / polynomial weights** are
  relative to `np.add`. A weight of 23.4 for `exp` means `exp` costs
  ~23.4× what `add` costs per element.

- **Linalg / FFT weights** are relative to the baseline `add` cost per
  analytical FLOP. A weight of 0.10 for `linalg.cholesky` means the
  measured cost per analytical FLOP of Cholesky is 10% of the baseline
  `add` cost — i.e., the textbook formula $n^3/3$ slightly overestimates
  relative to wall-clock measurement.

- **Missing operations** default to 1.0. A partial config (e.g.,
  pointwise-only) works without error.

## Programmatic usage

You can load and reset weights from Python:

```python
from mechestim._weights import load_weights, reset_weights, get_weight

# Load from a specific file
load_weights("/path/to/weights.json")

# Check a weight
print(get_weight("exp"))   # 23.43
print(get_weight("add"))   # 1.0
print(get_weight("foo"))   # 1.0 (unknown ops default to 1.0)

# Reset to defaults (all 1.0)
reset_weights()
```

## ⚠️ Caveats

- **Weights are approximate.** Many operations have internal branching
  based on input size, values, or array layout. A single constant measured
  at one size does not capture all regimes.

- **Weights are platform-specific.** Different CPUs, BLAS libraries, and
  libm implementations produce different constants. Always measure on the
  target platform.

- **Timing mode is a proxy.** Wall-clock measurements include cache
  effects, memory bandwidth, and SIMD efficiency — not just FP compute.
  The relative ordering is correct, but magnitudes may differ from true FP
  op counts.

- **Symmetry is independent of weights.** Symmetry reduces the element
  count in the analytical formula. Weights scale the per-element cost.
  Both apply independently: `cost = unique_elements × weight(op_name)`.

## 📎 Related pages

- [FLOP Counting Model](../concepts/flop-counting-model.md) — how weights fit into the cost model
- [Plan Your Budget](./plan-your-budget.md) — query costs before running
- [Operation Categories](../concepts/operation-categories.md) — which operations are counted
