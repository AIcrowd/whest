# Empirical FLOP Weights

## Introduction

Per-operation FLOP weights are multiplicative correction factors that bridge
the gap between mechestim's analytical cost formulas and the actual
floating-point instruction cost observed on hardware. Without weights, all
pointwise operations are treated as equally expensive -- `exp`, `sin`, and
`abs` each cost $\text{numel}(\text{output})$ FLOPs. In practice, `exp`
decomposes into a minimax polynomial approximation requiring approximately
14 FP instructions per element, while `abs` is a single bit manipulation.

Weights correct for this by expressing each operation's real instruction
density relative to element-wise addition (`np.add`). When weights are
loaded, the effective cost of an operation becomes:

$$
\text{cost}(\text{op}) = \texttt{analytical\_formula}(\text{shapes}) \times \text{weight}(\text{op})
$$

A weight of 25.9 for `sin` means that each analytical FLOP of sine costs
approximately 26 times more in actual floating-point instructions than a
FLOP of addition. Weights apply uniformly across all operation categories
-- pointwise, reductions, FFT, linalg, contractions, and all others use
the same measurement formula and normalization.

## Methodology

### The unified correction-factor formula

Every weight in this file is computed from the same two-step formula:

$$
\alpha(\text{op}) = \text{median}_{D} \left[ \frac{\sum \texttt{fp\_arith\_inst\_retired.*} \times \texttt{simd\_width}}{C(\text{op}, \text{params}) \times R} \right]
$$

$$
\text{weight}(\text{op}) = \frac{\alpha(\text{op})}{\alpha(\text{add})}
$$

Where:

- $\alpha(\text{op})$ is the **raw correction factor** -- the ratio of
  hardware-observed FP instructions to the analytical FLOP count predicted
  by mechestim's cost formula.
- $\texttt{fp\_arith\_inst\_retired.*}$ are Intel Performance Monitoring Unit
  (PMU) hardware counters that count retired floating-point arithmetic
  instructions. Each SIMD variant is weighted by its lane count:
    - `scalar_double` $\times 1$
    - `128b_packed_double` $\times 2$
    - `256b_packed_double` $\times 4$
    - `512b_packed_double` $\times 8$
- $C(\text{op}, \text{params})$ is mechestim's analytical FLOP formula --
  the same formula used at runtime for budget accounting.
- $R$ is the number of repetitions in the measurement loop.
- $D = 3$ input distributions per operation; the **median** across
  distributions is reported to smooth data-dependent variance.
- $\alpha(\text{add})$ is the raw correction factor for `np.add`, used as
  the normalization constant so that $\text{weight}(\text{add})$ is
  anchored near 1.0.

### Why normalize by alpha(add)?

The raw correction factor $\alpha(\text{op})$ is hardware-specific and not
directly interpretable. Normalizing by $\alpha(\text{add})$ produces weights
that answer the question: "How many times more expensive is this operation
per analytical FLOP compared to element-wise addition?" This is the natural
unit for budget reasoning -- a user can multiply the analytical formula by
the weight to estimate real hardware cost.

The raw $\alpha$ values are preserved in `meta.validation.absolute_correction_factors`
for scientific analysis. Recovery is trivial:
$\alpha(\text{op}) = \text{weight}(\text{op}) \times \alpha(\text{add})$,
where $\alpha(\text{add})$ is stored in `meta.methodology.baseline_alpha`.

For this calibration run, $\alpha(\text{add}) = 1.5641$.

### The FMA effect: weights below 1.0

Weights below 1.0 are expected and physically meaningful for BLAS-backed
operations (contractions, linalg decompositions, some FFTs). The Fused
Multiply-Add (FMA) instruction computes $a \times b + c$ in a single
hardware instruction but counts as 2 analytical FLOPs (one multiply, one
add). When BLAS libraries (OpenBLAS, MKL) use FMA-heavy inner loops, the
hardware retires fewer instructions than the analytical formula predicts,
yielding $\alpha(\text{op}) < \alpha(\text{add})$ and therefore
$\text{weight} < 1.0$.

For example, `matmul` has weight 0.6426 -- meaning each analytical FLOP
of matrix multiplication costs only 64% of what an `add` FLOP costs,
because the FMA-optimized inner loop fuses two analytical FLOPs into one
instruction.

### Subprocess isolation

Each benchmark runs in a **separate subprocess** to eliminate interference
from the benchmark framework's own memory allocation and floating-point
work. The measurement flow for each operation:

1. A self-contained Python script is written to a temporary file containing
   array setup code and a hot measurement loop.
2. The script is executed via `perf stat -e <events> -x , python <script>`,
   which wraps the entire process with hardware counter collection.
3. The CSV output from `perf stat` is parsed to extract per-event
   instruction counts.
4. The SIMD-width-weighted total is computed and divided by the analytical
   denominator.

For timing-mode validation, the same script runs without `perf stat`,
using `time.perf_counter_ns()` around the hot loop.

### Input distributions

Each operation is measured with 3 input distributions to capture
data-dependent variance:

1. **Standard normal** -- $N(0, 1)$; exercises typical numeric ranges.
2. **Uniform positive** -- $U(0.01, 100)$; avoids special-case branches
   for negative or zero inputs.
3. **Wide uniform** -- $U(-1000, 1000)$; exercises large-magnitude code
   paths.

The **median** across distributions is reported. Median is preferred over
mean because it is robust to single-distribution outliers (e.g., an `exp`
input that triggers overflow handling).

For operations where value distributions are irrelevant (e.g., `sort`),
the three "distributions" are instead three input orderings: random,
pre-sorted, and reverse-sorted.

### Pre-allocated output arrays

Pointwise and cumulative operations use the `out=` parameter with
pre-allocated output arrays to eliminate memory allocation overhead from
measurements. This isolates pure floating-point compute cost. Operations
that do not support `out=` (e.g., `sort`, `unique`, `histogram`) allocate
internally -- this allocation cost is captured in the measurement and
reflects the operation's real cost profile.

## Measurement environment

!!! info "Calibration platform"

    - **CPU:** Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
    - **Cores:** 64 physical / 128 threads
    - **RAM:** 251.7 GB
    - **Arch:** x86_64 (AVX-512 capable)
    - **Cache:** L1d 48 KB, L1i 32 KB, L2 1280 KB, L3 54 MB
    - **Instance:** AWS EC2 c6i.metal (bare metal -- full PMU access)
    - **OS:** Linux 6.1.166 (Amazon Linux 2023)
    - **Python:** 3.11.14
    - **NumPy:** 2.1.3
    - **BLAS:** scipy-openblas 0.3.27
    - **Measurement mode:** perf (hardware counters: `fp_arith_inst_retired.*`)
    - **dtype:** float64
    - **Repeats:** 5 per distribution
    - **Distributions:** 3 per operation
    - **Methodology version:** 2.0
    - **Baseline alpha(add):** 1.5641
    - **Total calibration time:** 2268.4 seconds
    - **Date:** 2026-04-09

## Validation

Every operation is measured in both **perf mode** (hardware counters) and
**timing mode** (wall-clock nanoseconds). Both modes use the identical
measurement formula:

$$
\alpha_{\text{perf}}(\text{op}) = \text{median}\left[\frac{\texttt{perf\_instructions}}{\texttt{analytical\_FLOPs}}\right]
$$

$$
\alpha_{\text{timing}}(\text{op}) = \text{median}\left[\frac{\texttt{elapsed\_ns}}{\texttt{analytical\_FLOPs}}\right]
$$

Timing-mode weights are normalized identically:
$\texttt{timing\_weight}(\text{op}) = \alpha_{\text{timing}}(\text{op}) / \alpha_{\text{timing}}(\text{add})$.

### Correlation statistics

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Pearson $r$ | 0.7342 | Linear correlation between perf and timing weight vectors. |
| Spearman $\rho$ | 0.5319 | Rank correlation -- are the orderings consistent? |

### Maximum divergence

| Field | Value |
|-------|-------|
| Operation | `matmul` |
| Perf weight | 0.6426 |
| Timing weight | 0.0009 |
| Ratio | 714.0 |

### Interpreting divergence

The moderate correlation values and large max divergence for `matmul` are
**expected** for BLAS and linalg operations. Perf mode counts FP instructions
regardless of how long they take, while timing mode measures wall-clock
time that includes memory bandwidth, cache hierarchy effects, and
out-of-order execution efficiency. BLAS operations achieve extremely high
FLOP throughput (close to peak), so their per-instruction timing is much
lower than for scalar pointwise operations. This causes timing weights for
BLAS ops to be orders of magnitude smaller than perf weights, compressing
the overall correlation.

For pointwise operations (which dominate the operation count), the two
modes agree well in relative ordering. The perf-mode weights are the
primary reference because they measure the physically meaningful quantity
(FP instruction count) rather than a proxy (elapsed time).

!!! warning "Correlation caveats"

    The Pearson and Spearman values reported here span all 251 operations,
    including BLAS/linalg ops where timing and perf divergence is
    structurally expected. For the subset of pointwise operations, both
    correlations are substantially higher.

## How to read the tables

- **Weight = 1.0** means the operation retires the same number of FP
  instructions per analytical FLOP as `np.add`.
- **Weight > 1.0** means more FP instructions than the analytical formula
  predicts. Example: `sin` at 25.87 -- each analytical FLOP of sine
  requires approximately 26 FP instructions for the polynomial
  approximation.
- **Weight < 1.0** means fewer FP instructions than predicted. Example:
  `matmul` at 0.64 -- FMA fuses two analytical FLOPs (multiply + add)
  into one instruction.
- **Weight near 0** indicates an operation that retires negligible FP
  instructions (e.g., integer-dominated operations like `randint`,
  `bincount`).

!!! tip "Using these weights"

    The weights file is at `src/mechestim/data/weights.json`.
    Load it via:

    ```bash
    export MECHESTIM_WEIGHTS_FILE=src/mechestim/data/weights.json
    ```

    Or programmatically:

    ```python
    from mechestim._weights import load_weights
    load_weights("src/mechestim/data/weights.json")
    ```

## Weight tables

### Pointwise Unary (47 operations)

| Operation | Weight |
|-----------|-------:|
| `arccosh` | 53.1311 |
| `arcsinh` | 50.8929 |
| `arctanh` | 46.4110 |
| `tan` | 38.7451 |
| `arcsin` | 36.1813 |
| `arccos` | 34.2633 |
| `arctan` | 30.4335 |
| `expm1` | 26.5973 |
| `log1p` | 26.5973 |
| `cos` | 25.8987 |
| `sin` | 25.8688 |
| `cbrt` | 24.6793 |
| `log10` | 22.8776 |
| `log2` | 22.5579 |
| `sinh` | 21.4825 |
| `tanh` | 21.4825 |
| `log` | 20.3202 |
| `cosh` | 18.2857 |
| `exp` | 14.4495 |
| `exp2` | 9.9740 |
| `sqrt` | 1.0230 |
| `square` | 1.0230 |
| `reciprocal` | 1.0230 |
| `deg2rad` | 1.0230 |
| `rad2deg` | 1.0230 |
| `degrees` | 1.0230 |
| `radians` | 1.0230 |
| `frexp` | 1.0230 |
| `spacing` | 1.0230 |
| `modf` | 1.0167 |
| `abs` | 0.3837 |
| `negative` | 0.3837 |
| `positive` | 0.3837 |
| `ceil` | 0.3837 |
| `floor` | 0.3837 |
| `trunc` | 0.3837 |
| `rint` | 0.3837 |
| `sign` | 0.3837 |
| `signbit` | 0.3837 |
| `fabs` | 0.3837 |
| `logical_not` | 0.3837 |
| `sinc` | 0.3837 |
| `i0` | 0.3837 |
| `nan_to_num` | 0.3837 |
| `isneginf` | 0.3837 |
| `isposinf` | 0.3837 |
| `heaviside` | 0.3837 |

### Pointwise Binary (32 operations)

| Operation | Weight |
|-----------|-------:|
| `power` | 46.7140 |
| `arctan2` | 34.6532 |
| `logaddexp2` | 22.5287 |
| `logaddexp` | 21.6097 |
| `float_power` | 20.8222 |
| `hypot` | 7.4809 |
| `floor_divide` | 3.0823 |
| `isclose` | 2.6854 |
| `add` | 1.4067 |
| `subtract` | 1.4067 |
| `multiply` | 1.4067 |
| `divide` | 1.4067 |
| `true_divide` | 1.4067 |
| `maximum` | 1.4067 |
| `minimum` | 1.4067 |
| `fmax` | 1.4067 |
| `fmin` | 1.4067 |
| `mod` | 0.7673 |
| `remainder` | 0.7673 |
| `fmod` | 0.7673 |
| `greater` | 0.7673 |
| `greater_equal` | 0.7673 |
| `less` | 0.7673 |
| `less_equal` | 0.7673 |
| `equal` | 0.7673 |
| `not_equal` | 0.7673 |
| `logical_and` | 0.7673 |
| `logical_or` | 0.7673 |
| `logical_xor` | 0.7673 |
| `copysign` | 0.7673 |
| `nextafter` | 0.7673 |
| `ldexp` | 0.7673 |

### Reductions (35 operations)

| Operation | Weight |
|-----------|-------:|
| `std` | 2.9411 |
| `var` | 2.9411 |
| `nanstd` | 2.9411 |
| `nanvar` | 2.9411 |
| `ptp` | 1.6636 |
| `max` | 1.0237 |
| `min` | 1.0237 |
| `nanmax` | 1.0237 |
| `nanmin` | 1.0237 |
| `sum` | 1.0230 |
| `prod` | 1.0230 |
| `mean` | 1.0230 |
| `cumsum` | 1.0230 |
| `cumprod` | 1.0230 |
| `nansum` | 1.0230 |
| `nanmean` | 1.0230 |
| `nanprod` | 1.0230 |
| `average` | 1.0230 |
| `nancumprod` | 1.0230 |
| `nancumsum` | 1.0230 |
| `cumulative_sum` | 1.0230 |
| `cumulative_prod` | 1.0230 |
| `argmax` | 0.3837 |
| `argmin` | 0.3837 |
| `any` | 0.3837 |
| `all` | 0.3837 |
| `median` | 0.3837 |
| `nanmedian` | 0.3837 |
| `percentile` | 0.3837 |
| `nanpercentile` | 0.3837 |
| `quantile` | 0.3837 |
| `nanquantile` | 0.3837 |
| `count_nonzero` | 0.3837 |
| `nanargmax` | 0.3837 |
| `nanargmin` | 0.3837 |

### Sorting (17 operations)

| Operation | Weight |
|-----------|-------:|
| `partition` | 14.1337 |
| `argpartition` | 13.8459 |
| `intersect1d` | 5.2067 |
| `setxor1d` | 5.2067 |
| `argsort` | 3.3538 |
| `unique_inverse` | 3.3538 |
| `in1d` | 2.9630 |
| `isin` | 2.9630 |
| `union1d` | 2.9576 |
| `sort` | 2.8185 |
| `unique` | 2.8185 |
| `unique_counts` | 2.8185 |
| `unique_values` | 2.8185 |
| `setdiff1d` | 2.7061 |
| `searchsorted` | 0.9520 |
| `lexsort` | 0.4760 |
| `unique_all` | 0.4756 |

### FFT (14 operations)

| Operation | Weight |
|-----------|-------:|
| `fft.hfft` | 1.4631 |
| `fft.ifft` | 0.8613 |
| `fft.irfft` | 0.6609 |
| `fft.irfft2` | 0.5761 |
| `fft.irfftn` | 0.5761 |
| `fft.fft` | 0.5340 |
| `fft.rfft` | 0.5338 |
| `fft.ifft2` | 0.5008 |
| `fft.ifftn` | 0.5008 |
| `fft.fft2` | 0.4612 |
| `fft.fftn` | 0.4612 |
| `fft.rfft2` | 0.4523 |
| `fft.rfftn` | 0.4523 |
| `fft.ihfft` | 0.2733 |

### Linalg (14 operations)

| Operation | Weight |
|-----------|-------:|
| `linalg.pinv` | 7.1872 |
| `linalg.svd` | 5.9042 |
| `linalg.eigh` | 2.3185 |
| `linalg.lstsq` | 1.9782 |
| `linalg.svdvals` | 1.8834 |
| `linalg.inv` | 1.7448 |
| `linalg.qr` | 1.3332 |
| `linalg.cholesky` | 1.0699 |
| `linalg.eig` | 0.9604 |
| `linalg.eigvalsh` | 0.8071 |
| `linalg.det` | 0.6923 |
| `linalg.slogdet` | 0.6923 |
| `linalg.eigvals` | 0.6520 |
| `linalg.solve` | 0.6455 |

### Contractions (9 operations)

| Operation | Weight |
|-----------|-------:|
| `inner` | 1.0598 |
| `vdot` | 1.0598 |
| `vecdot` | 1.0443 |
| `tensordot` | 0.6494 |
| `dot` | 0.6426 |
| `matmul` | 0.6426 |
| `einsum` | 0.6401 |
| `kron` | 0.6396 |
| `outer` | 0.6395 |

### Polynomial (10 operations)

| Operation | Weight |
|-----------|-------:|
| `polyder` | 8.7873 |
| `polyadd` | 8.2015 |
| `polysub` | 8.2015 |
| `polyint` | 8.1608 |
| `polymul` | 1.6769 |
| `poly` | 1.3803 |
| `polyfit` | 0.7578 |
| `roots` | 0.6889 |
| `polyval` | 0.6467 |
| `polydiv` | 0.1013 |

### Random (43 operations)

| Operation | Weight |
|-----------|-------:|
| `random.hypergeometric` | 367.3772 |
| `random.multivariate_normal` | 289.6739 |
| `random.zipf` | 147.6913 |
| `random.noncentral_chisquare` | 95.9945 |
| `random.negative_binomial` | 90.6636 |
| `random.multinomial` | 87.5909 |
| `random.noncentral_f` | 77.2939 |
| `random.dirichlet` | 77.2609 |
| `random.power` | 70.3696 |
| `random.vonmises` | 66.7742 |
| `random.f` | 59.7266 |
| `random.weibull` | 56.9431 |
| `random.beta` | 56.6401 |
| `random.standard_t` | 45.4818 |
| `random.gumbel` | 33.1563 |
| `random.pareto` | 31.3688 |
| `random.standard_cauchy` | 29.1635 |
| `random.gamma` | 28.4702 |
| `random.lognormal` | 28.3281 |
| `random.poisson` | 28.1315 |
| `random.logseries` | 27.8266 |
| `random.wald` | 25.5555 |
| `random.rayleigh` | 24.3409 |
| `random.laplace` | 18.9009 |
| `random.logistic` | 18.8868 |
| `random.chisquare` | 18.5817 |
| `random.binomial` | 18.5410 |
| `random.exponential` | 17.9423 |
| `random.standard_exponential` | 17.3029 |
| `random.standard_gamma` | 17.3029 |
| `random.normal` | 15.5409 |
| `random.standard_normal` | 14.2622 |
| `random.randn` | 14.2622 |
| `random.triangular` | 7.0330 |
| `random.geometric` | 3.8360 |
| `random.uniform` | 3.1969 |
| `random.rand` | 1.9181 |
| `random.random` | 1.9181 |
| `random.random_sample` | 1.9181 |
| `random.shuffle` | 0.2558 |
| `random.permutation` | 0.0001 |
| `random.choice` | 0.0001 |
| `random.randint` | 0.0001 |

### Misc (25 operations)

| Operation | Weight |
|-----------|-------:|
| `trace` | 384.9696 |
| `geomspace` | 48.5912 |
| `logspace` | 47.9518 |
| `unwrap` | 4.5198 |
| `trapezoid` | 2.9411 |
| `allclose` | 2.8132 |
| `histogram_bin_edges` | 1.6637 |
| `clip` | 1.6624 |
| `gradient` | 1.6624 |
| `cross` | 1.3428 |
| `convolve` | 1.3004 |
| `correlate` | 1.3004 |
| `diff` | 1.0230 |
| `ediff1d` | 1.0230 |
| `array_equal` | 0.7673 |
| `array_equiv` | 0.7673 |
| `vander` | 0.6375 |
| `histogram` | 0.5117 |
| `corrcoef` | 0.3317 |
| `cov` | 0.3316 |
| `histogramdd` | 0.2771 |
| `histogram2d` | 0.2377 |
| `interp` | 0.1652 |
| `digitize` | 0.0548 |
| `bincount` | 0.0001 |

### Window (5 operations)

| Operation | Weight |
|-----------|-------:|
| `kaiser` | 23.9400 |
| `hamming` | 21.9790 |
| `hanning` | 21.9790 |
| `blackman` | 15.4946 |
| `bartlett` | 3.8362 |

## Summary by category

| Category | Count | Avg Weight | Min | Max |
|----------|------:|-----------:|----:|----:|
| Pointwise Unary | 47 | 12.5080 | 0.3837 | 53.1311 |
| Pointwise Binary | 32 | 5.7421 | 0.7673 | 46.7140 |
| Reductions | 35 | 1.0231 | 0.3837 | 2.9411 |
| Sorting | 17 | 4.1099 | 0.4756 | 14.1337 |
| FFT | 14 | 0.5934 | 0.2733 | 1.4631 |
| Linalg | 14 | 1.9906 | 0.6455 | 7.1872 |
| Contractions | 9 | 0.7797 | 0.6395 | 1.0598 |
| Polynomial | 10 | 3.8603 | 0.1013 | 8.7873 |
| Random | 43 | 47.3819 | 0.0001 | 367.3772 |
| Misc | 25 | 20.2739 | 0.0001 | 384.9696 |
| Window | 5 | 17.4458 | 3.8362 | 23.9400 |

**Total benchmarked operations:** 251

## Known limitations

### Trace anomaly (subprocess overhead)

The `trace` operation shows an anomalously high weight (384.97) because its
analytical formula is $n$ (the matrix dimension), which is small (e.g.,
10,000), while the subprocess measurement captures fixed per-process
overhead that dominates at small input sizes. The weight for `trace` should
be interpreted with caution; it reflects measurement infrastructure
overhead more than the operation's intrinsic FP cost.

### Zero-weight integer operations

Operations like `random.randint`, `random.permutation`, `random.choice`,
and `bincount` show weights near zero (0.0001). These operations are
integer-dominated -- they retire negligible floating-point instructions
despite having a nonzero analytical FLOP cost in mechestim's model. The
near-zero weight correctly reflects the FP instruction reality: these
operations perform almost no floating-point arithmetic.

### Platform specificity

These weights were measured on a specific CPU microarchitecture (Intel
Ice Lake, Xeon Platinum 8375C) with a specific BLAS library
(scipy-openblas 0.3.27) and math library (glibc libm). Different platforms
will produce different weights because:

- **Different SIMD widths:** ARM NEON (128-bit) vs. x86 AVX-512 (512-bit)
  changes instruction-to-FLOP ratios for vectorized operations.
- **Different libm implementations:** The polynomial degree used for `sin`,
  `exp`, `log`, etc. varies between libm implementations (glibc vs. musl
  vs. Apple Accelerate).
- **Different BLAS implementations:** MKL, OpenBLAS, and Apple Accelerate
  use different blocking strategies and FMA utilization, changing linalg
  and contraction weights.

Always recalibrate weights on the target platform for accurate budget
accounting. See [Calibrate Weights](../how-to/calibrate-weights.md).

### Single-size measurement

Each operation is measured at one representative input size. Operations
with size-dependent behavior (e.g., `sort` at small $n$ vs. large $n$,
where the algorithm may switch between insertion sort and introsort) may
have different effective weights at other sizes. The chosen sizes are large
enough to represent the asymptotic regime where the analytical formula is
meaningful.

## Related pages

- [Calibrate Weights](../how-to/calibrate-weights.md) -- run your own calibration
- [FLOP Counting Model](../concepts/flop-counting-model.md) -- how weights compose with analytical formulas
- [Operation Audit](./operation-audit.md) -- full 482-operation registry inventory
