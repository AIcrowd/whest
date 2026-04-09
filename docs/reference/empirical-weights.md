# Empirical FLOP Weights

Per-operation weights measured via hardware performance counters
(`fp_arith_inst_retired`) on bare metal. These weights correct for the
fact that transcendental functions (e.g. `exp`, `sin`) decompose into
many more basic FP instructions than simple arithmetic (`add`, `multiply`).

!!! info "Measurement environment"

    - **CPU:** Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
    - **Cores:** 64 physical / 128 threads
    - **RAM:** 251.7 GB
    - **Arch:** x86_64 (AVX-512)
    - **Instance:** AWS EC2 c6i.metal (bare metal — full PMU access)
    - **Mode:** perf (hardware counters: `fp_arith_inst_retired.*`)
    - **dtype:** float64
    - **NumPy:** 2.1.3 (scipy-openblas 0.3.27)
    - **Date:** 2026-04-09
    - **Repeats:** 5

## How to read this table

- **Weight** = measured FP instructions per element, normalized so
  `np.add` = 1.0 (the baseline).
- A weight of 25.9 for `sin` means each `sin` element costs ~26×
  what an `add` element costs in actual retired FP instructions.
- **Linalg/FFT weights** are correction factors applied to the
  analytical formula — a weight of 9.2 for `svd` means the measured
  cost is 9.2× the textbook $2mn^2 + 2n^3$ formula per analytical FLOP.
- Weights below 1.0 indicate operations that retire fewer FP
  instructions than `add` (e.g. comparison ops like `greater`).

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

## Pointwise Unary

| Operation | Weight | Notes |
|-----------|-------:|-------|
| `arccosh` | 53.1311 | Element-wise inverse hyperbolic cosine. |
| `arcsinh` | 50.8929 | Element-wise inverse hyperbolic sine. |
| `arctanh` | 46.4110 | Element-wise inverse hyperbolic tangent. |
| `tan` | 38.7451 | Element-wise tangent. |
| `arcsin` | 36.1813 | Element-wise inverse sine. |
| `arccos` | 34.2633 | Element-wise inverse cosine. |
| `arctan` | 30.4335 | Element-wise inverse tangent. |
| `expm1` | 26.5973 | Element-wise e^x - 1 (accurate near zero). |
| `log1p` | 26.5973 | Element-wise log(1+x) (accurate near zero). |
| `cos` | 25.8987 | Element-wise cosine. |
| `sin` | 25.8688 | Element-wise sine. |
| `cbrt` | 24.6793 | Element-wise cube root. |
| `log10` | 22.8776 | Element-wise base-10 logarithm. |
| `log2` | 22.5579 | Element-wise base-2 logarithm. |
| `sinh` | 21.4825 | Element-wise hyperbolic sine. |
| `tanh` | 21.4825 | Element-wise hyperbolic tangent. |
| `log` | 20.3202 | Element-wise natural logarithm. |
| `cosh` | 18.2857 | Element-wise hyperbolic cosine. |
| `exp` | 14.4495 | Element-wise e^x. |
| `exp2` | 9.9740 | Element-wise 2^x. |
| `deg2rad` | 1.0230 | Alias for radians. |
| `degrees` | 1.0230 | Convert radians to degrees element-wise. |
| `rad2deg` | 1.0230 | Alias for degrees. |
| `radians` | 1.0230 | Convert degrees to radians element-wise. |
| `reciprocal` | 1.0230 | Element-wise 1/x. |
| `sqrt` | 1.0230 | Element-wise square root. |
| `square` | 1.0230 | Element-wise x^2. |
| `abs` | 0.3837 | Element-wise absolute value; alias for absolute. |
| `ceil` | 0.3837 | Element-wise ceiling. |
| `fabs` | 0.3837 | Element-wise absolute value (always float). |
| `floor` | 0.3837 | Element-wise floor. |
| `logical_not` | 0.3837 | Element-wise logical NOT. |
| `negative` | 0.3837 | Element-wise negation. |
| `positive` | 0.3837 | Element-wise unary plus (copy with sign preserved). |
| `rint` | 0.3837 | Round to nearest integer element-wise. |
| `sign` | 0.3837 | Element-wise sign function. |
| `signbit` | 0.3837 | Returns True for elements with negative sign bit. |
| `trunc` | 0.3837 | Truncate toward zero element-wise. |

## Pointwise Binary

| Operation | Weight | Notes |
|-----------|-------:|-------|
| `power` | 46.7140 | Element-wise exponentiation x**y. |
| `arctan2` | 34.6532 | Element-wise arctan(y/x) considering quadrant. |
| `logaddexp2` | 22.5287 | log2(2**x1 + 2**x2) element-wise. |
| `logaddexp` | 21.6097 | log(exp(x1) + exp(x2)) element-wise. |
| `float_power` | 20.8222 | Element-wise exponentiation in float64. |
| `hypot` | 7.4809 | Element-wise Euclidean norm sqrt(x1^2 + x2^2). |
| `floor_divide` | 3.0823 | Element-wise floor division. |
| `add` | 1.4067 | Element-wise addition. |
| `divide` | 1.4067 | Element-wise true division. |
| `fmax` | 1.4067 | Element-wise maximum ignoring NaN. |
| `fmin` | 1.4067 | Element-wise minimum ignoring NaN. |
| `maximum` | 1.4067 | Element-wise maximum (propagates NaN). |
| `minimum` | 1.4067 | Element-wise minimum (propagates NaN). |
| `multiply` | 1.4067 | Element-wise multiplication. |
| `subtract` | 1.4067 | Element-wise subtraction. |
| `true_divide` | 1.4067 | Element-wise true division (explicit). |
| `copysign` | 0.7673 | Copy sign of x2 to magnitude of x1 element-wise. |
| `equal` | 0.7673 | Element-wise x1 == x2. |
| `fmod` | 0.7673 | Element-wise C-style fmod (remainder toward zero). |
| `greater` | 0.7673 | Element-wise x1 > x2. |
| `greater_equal` | 0.7673 | Element-wise x1 >= x2. |
| `ldexp` | 0.7673 | Return x1 * 2**x2 element-wise. |
| `less` | 0.7673 | Element-wise x1 < x2. |
| `less_equal` | 0.7673 | Element-wise x1 <= x2. |
| `logical_and` | 0.7673 | Element-wise logical AND. |
| `logical_or` | 0.7673 | Element-wise logical OR. |
| `logical_xor` | 0.7673 | Element-wise logical XOR. |
| `mod` | 0.7673 | Element-wise modulo. |
| `nextafter` | 0.7673 | Return next float after x1 toward x2 element-wise. |
| `not_equal` | 0.7673 | Element-wise x1 != x2. |
| `remainder` | 0.7673 | Element-wise remainder (same as mod). |

## Reductions

| Operation | Weight | Notes |
|-----------|-------:|-------|
| `nanstd` | 2.9411 | Standard deviation ignoring NaNs. |
| `nanvar` | 2.9411 | Variance ignoring NaNs. |
| `std` | 2.9411 | Standard deviation; cost_multiplier=2 (two passes). |
| `var` | 2.9411 | Variance; cost_multiplier=2 (two passes). |
| `max` | 1.0237 | Maximum value of array. |
| `min` | 1.0237 | Minimum value of array. |
| `nanmax` | 1.0237 | Maximum ignoring NaNs. |
| `nanmin` | 1.0237 | Minimum ignoring NaNs. |
| `average` | 1.0230 | Weighted average of array elements. |
| `cumprod` | 1.0230 | Cumulative product of array elements. |
| `cumsum` | 1.0230 | Cumulative sum of array elements. |
| `mean` | 1.0230 | Arithmetic mean of array elements. |
| `nanmean` | 1.0230 | Mean ignoring NaNs. |
| `nanprod` | 1.0230 | Product ignoring NaNs. |
| `nansum` | 1.0230 | Sum ignoring NaNs. |
| `prod` | 1.0230 | Product of array elements. |
| `sum` | 1.0230 | Sum of array elements. |
| `all` | 0.3837 | Test whether all array elements are true. |
| `any` | 0.3837 | Test whether any array element is true. |
| `argmax` | 0.3837 | Index of maximum value. |
| `argmin` | 0.3837 | Index of minimum value. |
| `count_nonzero` | 0.3837 | Count non-zero elements. |
| `median` | 0.3837 | Median of array elements (sorts internally). |
| `nanmedian` | 0.3837 | Median ignoring NaNs. |
| `nanpercentile` | 0.3837 | q-th percentile ignoring NaNs. |
| `nanquantile` | 0.3837 | q-th quantile ignoring NaNs. |
| `percentile` | 0.3837 | q-th percentile of array elements. |
| `quantile` | 0.3837 | q-th quantile of array elements. |

## Custom Formula

| Operation | Weight | Notes |
|-----------|-------:|-------|
| `roots` | 68892.4486 | Return roots of polynomial with given coefficients. ... |
| `polyfit` | 15459.7653 | Least squares polynomial fit. Cost: 2 * m * (deg+1)^... |
| `polymul` | 169.3638 | Multiply polynomials. Cost: n1 * n2 FLOPs. |
| `poly` | 138.0333 | Polynomial from roots. Cost: $n^2$ FLOPs. |
| `polyval` | 129.3312 | Evaluate polynomial at given points. Cost: 2 * m * d... |
| `argsort` | 80.4905 | Indirect sort; cost = n*ceil(log2(n)) per slice. |
| `sort` | 67.6431 | Comparison sort; cost = n*ceil(log2(n)) per slice. |
| `unique` | 67.6431 | Sort-based unique; cost = n*ceil(log2(n)). |
| `random.standard_t` | 45.4818 | Sampling; cost = numel(output). |
| `random.standard_cauchy` | 29.1635 | Sampling; cost = numel(output). |
| `random.poisson` | 28.1315 | Sampling; cost = numel(output). |
| `lexsort` | 22.8490 | Multi-key sort; cost = k*n*ceil(log2(n)). |
| `searchsorted` | 22.8490 | Binary search; cost = m*ceil(log2(n)). |
| `random.binomial` | 18.5410 | Sampling; cost = numel(output). |
| `random.standard_exponential` | 17.3029 | Sampling; cost = numel(output). |
| `random.standard_gamma` | 17.3029 | Sampling; cost = numel(output). |
| `random.standard_normal` | 14.2622 | Sampling; cost = numel(output). |
| `partition` | 14.1337 | Quickselect; cost = n per slice. |
| `argpartition` | 13.8459 | Indirect partition; cost = n per slice. |
| `linalg.pinv` | 11.2414 | Pseudoinverse. Cost: m*n*min(m,n) (via SVD). |
| `polydiv` | 10.2335 | Divide one polynomial by another. Cost: n1 * n2 FLOPs. |
| `linalg.svd` | 9.2345 | Singular value decomposition; cost ~ O(min(m,n)*m*n). |
| `polyder` | 8.7003 | Differentiate polynomial. Cost: n FLOPs. |
| `polyadd` | 8.2015 | Add two polynomials. Cost: max(n1, n2) FLOPs. |
| `polysub` | 8.2015 | Difference (subtraction) of two polynomials. Cost: m... |
| `polyint` | 8.0800 | Integrate polynomial. Cost: n FLOPs. |
| `linalg.eigh` | 3.6263 | Symmetric eigendecomposition. Cost: $(4/3)n^3$ (Golu... |
| `random.uniform` | 3.1969 | Sampling; cost = numel(output). |
| `linalg.lstsq` | 3.0940 | Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD). |
| `linalg.svdvals` | 2.9458 | Singular values only. Cost: m*n*min(m,n) (Golub-Rein... |
| `linalg.inv` | 2.7290 | Matrix inverse. Cost: $n^3$ (LU + solve). |
| `fft.hfft` | 2.2883 | FFT of Hermitian-symmetric signal. Cost: 5*n_out*cei... |
| `linalg.qr` | 2.0853 | QR decomposition. Cost: $2mn^2 - (2/3)n^3$ (Golub & ... |
| `linalg.cholesky` | 1.6734 | Cholesky decomposition. Cost: $n^3/3$ (Golub & Van L... |
| `linalg.eig` | 1.5021 | Eigendecomposition. Cost: $10n^3$ (Francis QR, Golub... |
| `fft.ifft` | 1.3471 | Inverse 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Co... |
| `linalg.eigvalsh` | 1.2624 | Symmetric eigenvalues. Cost: $(4/3)n^3$ (same as eigh). |
| `linalg.det` | 1.0828 | Determinant. Cost: $n^3$ (LU factorization). |
| `linalg.slogdet` | 1.0828 | Sign + log determinant. Cost: $n^3$ (LU factorization). |
| `fft.irfft` | 1.0337 | Inverse 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (... |
| `linalg.eigvals` | 1.0197 | Eigenvalues only. Cost: $10n^3$ (same as eig). |
| `linalg.solve` | 1.0096 | Solve Ax=b. Cost: $2n^3/3$ (LU) + $n^2 \cdot n_{\tex... |
| `fft.irfft2` | 0.9010 | Inverse 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), ... |
| `fft.irfftn` | 0.9010 | Inverse N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), ... |
| `fft.fft` | 0.8351 | 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tuk... |
| `fft.rfft` | 0.8348 | 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-T... |
| `fft.ifft2` | 0.7833 | Inverse 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=... |
| `fft.ifftn` | 0.7833 | Inverse N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=... |
| `fft.fft2` | 0.7213 | 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) ... |
| `fft.fftn` | 0.7213 | N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) ... |
| `fft.rfft2` | 0.7074 | 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s... |
| `fft.rfftn` | 0.7074 | N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s... |
| `fft.ihfft` | 0.4274 | Inverse FFT of Hermitian signal. Cost: 5*n*ceil(log2... |
| `random.shuffle` | 0.2558 | Shuffle; cost = n*ceil(log2(n)). |
| `random.permutation` | 0.0001 | Shuffle; cost = n*ceil(log2(n)). |

## Summary by Category

| Category | Benchmarked | Avg Weight | Min | Max |
|----------|------------:|-----------:|----:|----:|
| counted_unary | 38 | 15.33 | 0.3837 | 53.1311 |
| counted_binary | 31 | 5.84 | 0.7673 | 46.7140 |
| counted_reduction | 28 | 1.05 | 0.3837 | 2.9411 |
| counted_custom | 55 | 1551.86 | 0.0001 | 68892.4486 |

**Total benchmarked operations:** 152

## Related pages

- [Calibrate Weights](../how-to/calibrate-weights.md) — run your own calibration
- [FLOP Counting Model](../concepts/flop-counting-model.md) — how weights compose with analytical formulas
- [Operation Audit](./operation-audit.md) — full 482-operation registry inventory

