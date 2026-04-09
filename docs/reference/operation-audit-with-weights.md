# mechestim Operation Audit with Perf Weights

**CPU:** Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
**Mode:** perf (hardware counters)
**dtype:** float64
**NumPy:** 2.1.3
**BLAS:** scipy-openblas 0.3.27
**Date:** 2026-04-09T14:32:52.219745+00:00
**Baseline:** np.add = 1.0 (raw FPE: N/A)

## All Benchmarked Operations (sorted by weight)

| Operation | Category | Weight | Notes |
|-----------|----------|-------:|-------|
| roots | counted_custom | 68892.4486 | Return roots of polynomial with given coefficients. Cost:... |
| polyfit | counted_custom | 15459.7653 | Least squares polynomial fit. Cost: 2 * m * (deg+1)^2 FLOPs. |
| polymul | counted_custom | 169.3638 | Multiply polynomials. Cost: n1 * n2 FLOPs. |
| poly | counted_custom | 138.0333 | Polynomial from roots. Cost: $n^2$ FLOPs. |
| polyval | counted_custom | 129.3312 | Evaluate polynomial at given points. Cost: 2 * m * deg FL... |
| argsort | counted_custom | 80.4905 | Indirect sort; cost = n*ceil(log2(n)) per slice. |
| sort | counted_custom | 67.6431 | Comparison sort; cost = n*ceil(log2(n)) per slice. |
| unique | counted_custom | 67.6431 | Sort-based unique; cost = n*ceil(log2(n)). |
| arccosh | counted_unary | 53.1311 | Element-wise inverse hyperbolic cosine. |
| arcsinh | counted_unary | 50.8929 | Element-wise inverse hyperbolic sine. |
| power | counted_binary | 46.7140 | Element-wise exponentiation x**y. |
| arctanh | counted_unary | 46.4110 | Element-wise inverse hyperbolic tangent. |
| random.standard_t | counted_custom | 45.4818 | Sampling; cost = numel(output). |
| tan | counted_unary | 38.7451 | Element-wise tangent. |
| arcsin | counted_unary | 36.1813 | Element-wise inverse sine. |
| arctan2 | counted_binary | 34.6532 | Element-wise arctan(y/x) considering quadrant. |
| arccos | counted_unary | 34.2633 | Element-wise inverse cosine. |
| arctan | counted_unary | 30.4335 | Element-wise inverse tangent. |
| random.standard_cauchy | counted_custom | 29.1635 | Sampling; cost = numel(output). |
| random.poisson | counted_custom | 28.1315 | Sampling; cost = numel(output). |
| expm1 | counted_unary | 26.5973 | Element-wise e^x - 1 (accurate near zero). |
| log1p | counted_unary | 26.5973 | Element-wise log(1+x) (accurate near zero). |
| cos | counted_unary | 25.8987 | Element-wise cosine. |
| sin | counted_unary | 25.8688 | Element-wise sine. |
| cbrt | counted_unary | 24.6793 | Element-wise cube root. |
| log10 | counted_unary | 22.8776 | Element-wise base-10 logarithm. |
| lexsort | counted_custom | 22.8490 | Multi-key sort; cost = k*n*ceil(log2(n)). |
| searchsorted | counted_custom | 22.8490 | Binary search; cost = m*ceil(log2(n)). |
| log2 | counted_unary | 22.5579 | Element-wise base-2 logarithm. |
| logaddexp2 | counted_binary | 22.5287 | log2(2**x1 + 2**x2) element-wise. |
| logaddexp | counted_binary | 21.6097 | log(exp(x1) + exp(x2)) element-wise. |
| sinh | counted_unary | 21.4825 | Element-wise hyperbolic sine. |
| tanh | counted_unary | 21.4825 | Element-wise hyperbolic tangent. |
| float_power | counted_binary | 20.8222 | Element-wise exponentiation in float64. |
| log | counted_unary | 20.3202 | Element-wise natural logarithm. |
| random.binomial | counted_custom | 18.5410 | Sampling; cost = numel(output). |
| cosh | counted_unary | 18.2857 | Element-wise hyperbolic cosine. |
| random.standard_exponential | counted_custom | 17.3029 | Sampling; cost = numel(output). |
| random.standard_gamma | counted_custom | 17.3029 | Sampling; cost = numel(output). |
| exp | counted_unary | 14.4495 | Element-wise e^x. |
| random.standard_normal | counted_custom | 14.2622 | Sampling; cost = numel(output). |
| partition | counted_custom | 14.1337 | Quickselect; cost = n per slice. |
| argpartition | counted_custom | 13.8459 | Indirect partition; cost = n per slice. |
| linalg.pinv | counted_custom | 11.2414 | Pseudoinverse. Cost: m*n*min(m,n) (via SVD). |
| polydiv | counted_custom | 10.2335 | Divide one polynomial by another. Cost: n1 * n2 FLOPs. |
| exp2 | counted_unary | 9.9740 | Element-wise 2^x. |
| linalg.svd | counted_custom | 9.2345 | Singular value decomposition; cost ~ O(min(m,n)*m*n). |
| polyder | counted_custom | 8.7003 | Differentiate polynomial. Cost: n FLOPs. |
| polyadd | counted_custom | 8.2015 | Add two polynomials. Cost: max(n1, n2) FLOPs. |
| polysub | counted_custom | 8.2015 | Difference (subtraction) of two polynomials. Cost: max(n1... |
| polyint | counted_custom | 8.0800 | Integrate polynomial. Cost: n FLOPs. |
| hypot | counted_binary | 7.4809 | Element-wise Euclidean norm sqrt(x1^2 + x2^2). |
| linalg.eigh | counted_custom | 3.6263 | Symmetric eigendecomposition. Cost: $(4/3)n^3$ (Golub & V... |
| random.uniform | counted_custom | 3.1969 | Sampling; cost = numel(output). |
| linalg.lstsq | counted_custom | 3.0940 | Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD). |
| floor_divide | counted_binary | 3.0823 | Element-wise floor division. |
| linalg.svdvals | counted_custom | 2.9458 | Singular values only. Cost: m*n*min(m,n) (Golub-Reinsch). |
| std | counted_reduction | 2.9411 | Standard deviation; cost_multiplier=2 (two passes). |
| var | counted_reduction | 2.9411 | Variance; cost_multiplier=2 (two passes). |
| nanstd | counted_reduction | 2.9411 | Standard deviation ignoring NaNs. |
| nanvar | counted_reduction | 2.9411 | Variance ignoring NaNs. |
| linalg.inv | counted_custom | 2.7290 | Matrix inverse. Cost: $n^3$ (LU + solve). |
| fft.hfft | counted_custom | 2.2883 | FFT of Hermitian-symmetric signal. Cost: 5*n_out*ceil(log... |
| linalg.qr | counted_custom | 2.0853 | QR decomposition. Cost: $2mn^2 - (2/3)n^3$ (Golub & Van L... |
| linalg.cholesky | counted_custom | 1.6734 | Cholesky decomposition. Cost: $n^3/3$ (Golub & Van Loan §... |
| linalg.eig | counted_custom | 1.5021 | Eigendecomposition. Cost: $10n^3$ (Francis QR, Golub & Va... |
| add | counted_binary | 1.4067 | Element-wise addition. |
| subtract | counted_binary | 1.4067 | Element-wise subtraction. |
| multiply | counted_binary | 1.4067 | Element-wise multiplication. |
| divide | counted_binary | 1.4067 | Element-wise true division. |
| true_divide | counted_binary | 1.4067 | Element-wise true division (explicit). |
| maximum | counted_binary | 1.4067 | Element-wise maximum (propagates NaN). |
| minimum | counted_binary | 1.4067 | Element-wise minimum (propagates NaN). |
| fmax | counted_binary | 1.4067 | Element-wise maximum ignoring NaN. |
| fmin | counted_binary | 1.4067 | Element-wise minimum ignoring NaN. |
| fft.ifft | counted_custom | 1.3471 | Inverse 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-... |
| linalg.eigvalsh | counted_custom | 1.2624 | Symmetric eigenvalues. Cost: $(4/3)n^3$ (same as eigh). |
| linalg.det | counted_custom | 1.0828 | Determinant. Cost: $n^3$ (LU factorization). |
| linalg.slogdet | counted_custom | 1.0828 | Sign + log determinant. Cost: $n^3$ (LU factorization). |
| fft.irfft | counted_custom | 1.0337 | Inverse 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Coole... |
| max | counted_reduction | 1.0237 | Maximum value of array. |
| min | counted_reduction | 1.0237 | Minimum value of array. |
| nanmax | counted_reduction | 1.0237 | Maximum ignoring NaNs. |
| nanmin | counted_reduction | 1.0237 | Minimum ignoring NaNs. |
| sqrt | counted_unary | 1.0230 | Element-wise square root. |
| square | counted_unary | 1.0230 | Element-wise x^2. |
| reciprocal | counted_unary | 1.0230 | Element-wise 1/x. |
| deg2rad | counted_unary | 1.0230 | Alias for radians. |
| rad2deg | counted_unary | 1.0230 | Alias for degrees. |
| degrees | counted_unary | 1.0230 | Convert radians to degrees element-wise. |
| radians | counted_unary | 1.0230 | Convert degrees to radians element-wise. |
| sum | counted_reduction | 1.0230 | Sum of array elements. |
| prod | counted_reduction | 1.0230 | Product of array elements. |
| mean | counted_reduction | 1.0230 | Arithmetic mean of array elements. |
| cumsum | counted_reduction | 1.0230 | Cumulative sum of array elements. |
| cumprod | counted_reduction | 1.0230 | Cumulative product of array elements. |
| nansum | counted_reduction | 1.0230 | Sum ignoring NaNs. |
| nanmean | counted_reduction | 1.0230 | Mean ignoring NaNs. |
| nanprod | counted_reduction | 1.0230 | Product ignoring NaNs. |
| average | counted_reduction | 1.0230 | Weighted average of array elements. |
| linalg.eigvals | counted_custom | 1.0197 | Eigenvalues only. Cost: $10n^3$ (same as eig). |
| linalg.solve | counted_custom | 1.0096 | Solve Ax=b. Cost: $2n^3/3$ (LU) + $n^2 \cdot n_{\text{rhs... |
| fft.irfft2 | counted_custom | 0.9010 | Inverse 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=pro... |
| fft.irfftn | counted_custom | 0.9010 | Inverse N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=pro... |
| fft.fft | counted_custom | 0.8351 | 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey ra... |
| fft.rfft | counted_custom | 0.8348 | 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey ... |
| fft.ifft2 | counted_custom | 0.7833 | Inverse 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(... |
| fft.ifftn | counted_custom | 0.7833 | Inverse N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(... |
| mod | counted_binary | 0.7673 | Element-wise modulo. |
| remainder | counted_binary | 0.7673 | Element-wise remainder (same as mod). |
| fmod | counted_binary | 0.7673 | Element-wise C-style fmod (remainder toward zero). |
| greater | counted_binary | 0.7673 | Element-wise x1 > x2. |
| greater_equal | counted_binary | 0.7673 | Element-wise x1 >= x2. |
| less | counted_binary | 0.7673 | Element-wise x1 < x2. |
| less_equal | counted_binary | 0.7673 | Element-wise x1 <= x2. |
| equal | counted_binary | 0.7673 | Element-wise x1 == x2. |
| not_equal | counted_binary | 0.7673 | Element-wise x1 != x2. |
| logical_and | counted_binary | 0.7673 | Element-wise logical AND. |
| logical_or | counted_binary | 0.7673 | Element-wise logical OR. |
| logical_xor | counted_binary | 0.7673 | Element-wise logical XOR. |
| copysign | counted_binary | 0.7673 | Copy sign of x2 to magnitude of x1 element-wise. |
| nextafter | counted_binary | 0.7673 | Return next float after x1 toward x2 element-wise. |
| ldexp | counted_binary | 0.7673 | Return x1 * 2**x2 element-wise. |
| fft.fft2 | counted_custom | 0.7213 | 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cool... |
| fft.fftn | counted_custom | 0.7213 | N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cool... |
| fft.rfft2 | counted_custom | 0.7074 | 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Co... |
| fft.rfftn | counted_custom | 0.7074 | N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Co... |
| fft.ihfft | counted_custom | 0.4274 | Inverse FFT of Hermitian signal. Cost: 5*n*ceil(log2(n)) ... |
| abs | counted_unary | 0.3837 | Element-wise absolute value; alias for absolute. |
| negative | counted_unary | 0.3837 | Element-wise negation. |
| positive | counted_unary | 0.3837 | Element-wise unary plus (copy with sign preserved). |
| ceil | counted_unary | 0.3837 | Element-wise ceiling. |
| floor | counted_unary | 0.3837 | Element-wise floor. |
| trunc | counted_unary | 0.3837 | Truncate toward zero element-wise. |
| rint | counted_unary | 0.3837 | Round to nearest integer element-wise. |
| sign | counted_unary | 0.3837 | Element-wise sign function. |
| signbit | counted_unary | 0.3837 | Returns True for elements with negative sign bit. |
| fabs | counted_unary | 0.3837 | Element-wise absolute value (always float). |
| logical_not | counted_unary | 0.3837 | Element-wise logical NOT. |
| argmax | counted_reduction | 0.3837 | Index of maximum value. |
| argmin | counted_reduction | 0.3837 | Index of minimum value. |
| any | counted_reduction | 0.3837 | Test whether any array element is true. |
| all | counted_reduction | 0.3837 | Test whether all array elements are true. |
| median | counted_reduction | 0.3837 | Median of array elements (sorts internally). |
| nanmedian | counted_reduction | 0.3837 | Median ignoring NaNs. |
| percentile | counted_reduction | 0.3837 | q-th percentile of array elements. |
| nanpercentile | counted_reduction | 0.3837 | q-th percentile ignoring NaNs. |
| quantile | counted_reduction | 0.3837 | q-th quantile of array elements. |
| nanquantile | counted_reduction | 0.3837 | q-th quantile ignoring NaNs. |
| count_nonzero | counted_reduction | 0.3837 | Count non-zero elements. |
| random.shuffle | counted_custom | 0.2558 | Shuffle; cost = n*ceil(log2(n)). |
| random.permutation | counted_custom | 0.0001 | Shuffle; cost = n*ceil(log2(n)). |

**Total benchmarked:** 152

## Registry Summary (all 482 operations)

| Category | Count | Benchmarked | Avg Weight |
|----------|------:|------------:|-----------:|
| blacklisted | 32 | 0 | 0.00 |
| counted_binary | 45 | 31 | 5.84 |
| counted_custom | 157 | 55 | 1551.86 |
| counted_reduction | 37 | 28 | 1.05 |
| counted_unary | 73 | 38 | 15.33 |
| free | 138 | 0 | 0.00 |

## Counted Operations Without Benchmark Weight

These are registered with a cost category but were not benchmarked (aliases, custom formulas, etc.):

- **absolute** (counted_unary): Element-wise absolute value.
- **acos** (counted_unary): Alias for arccos (NumPy 2.x).
- **acosh** (counted_unary): Alias for arccosh (NumPy 2.x).
- **allclose** (counted_custom): Element-wise tolerance check; cost = numel(a).
- **amax** (counted_reduction): Maximum value of array (alias for max/numpy.amax).
- **amin** (counted_reduction): Minimum value of array (alias for min/numpy.amin).
- **angle** (counted_unary): Return angle of complex argument element-wise.
- **around** (counted_unary): Alias for round.
- **array_equal** (counted_custom): Element-wise equality; cost = numel(a).
- **array_equiv** (counted_custom): Element-wise equivalence; cost = numel(a).
- **asin** (counted_unary): Alias for arcsin (NumPy 2.x).
- **asinh** (counted_unary): Alias for arcsinh (NumPy 2.x).
- **atan** (counted_unary): Alias for arctan (NumPy 2.x).
- **atan2** (counted_binary): Alias for arctan2 (NumPy 2.x).
- **atanh** (counted_unary): Alias for arctanh (NumPy 2.x).
- **bartlett** (counted_custom): Bartlett window. Cost: n (one linear eval per sample).
- **bincount** (counted_custom): Integer counting; cost = numel(x).
- **bitwise_and** (counted_binary): Element-wise bitwise AND.
- **bitwise_count** (counted_unary): Count set bits element-wise (popcount).
- **bitwise_invert** (counted_unary): Element-wise bitwise invert (alias for bitwise_not).
- **bitwise_left_shift** (counted_binary): Element-wise left bit shift.
- **bitwise_not** (counted_unary): Element-wise bitwise NOT.
- **bitwise_or** (counted_binary): Element-wise bitwise OR.
- **bitwise_right_shift** (counted_binary): Element-wise right bit shift.
- **bitwise_xor** (counted_binary): Element-wise bitwise XOR.
- **blackman** (counted_custom): Blackman window. Cost: 3*n (three cosine terms per sample).
- **clip** (counted_custom): Clip array to [a_min, a_max] element-wise.
- **conj** (counted_unary): Complex conjugate element-wise.
- **conjugate** (counted_unary): Complex conjugate element-wise.
- **convolve** (counted_custom): 1-D discrete convolution.
- **corrcoef** (counted_custom): Pearson correlation coefficients.
- **correlate** (counted_custom): 1-D cross-correlation.
- **cov** (counted_custom): Covariance matrix.
- **cross** (counted_custom): Cross product of two 3-D vectors.
- **cumulative_prod** (counted_reduction): Cumulative product (NumPy 2.x array API).
- **cumulative_sum** (counted_reduction): Cumulative sum (NumPy 2.x array API).
- **diff** (counted_custom): n-th discrete difference along axis.
- **digitize** (counted_custom): Bin search; cost = n*ceil(log2(bins)).
- **divmod** (counted_binary): Element-wise (quotient, remainder) tuple.
- **dot** (counted_custom): Dot product; cost = 2*M*N*K for matrix multiply.
- **ediff1d** (counted_custom): Differences between consecutive elements.
- **einsum** (counted_custom): Generalized Einstein summation.
- **einsum_path** (counted_custom): Optimize einsum contraction path (no numeric output).
- **fix** (counted_unary): Round toward zero element-wise (alias for trunc).
- **frexp** (counted_unary): Decompose x into mantissa and exponent element-wise.
- **gcd** (counted_binary): Element-wise greatest common divisor.
- **geomspace** (counted_custom): Geometric-spaced generation; cost = num.
- **gradient** (counted_custom): Gradient using central differences.
- **hamming** (counted_custom): Hamming window. Cost: n (one cosine per sample).
- **hanning** (counted_custom): Hanning window. Cost: n (one cosine per sample).
- **heaviside** (counted_binary): Heaviside step function element-wise.
- **histogram** (counted_custom): Binning; cost = n*ceil(log2(bins)).
- **histogram2d** (counted_custom): 2D binning; cost = n*(ceil(log2(bx))+ceil(log2(by))).
- **histogram_bin_edges** (counted_custom): Bin edge computation; cost = numel(a).
- **histogramdd** (counted_custom): ND binning; cost = n*sum(ceil(log2(b_i))).
- **i0** (counted_unary): Modified Bessel function of order 0, element-wise.
- **imag** (counted_unary): Return imaginary part of complex array.
- **in1d** (counted_custom): Set membership; cost = (n+m)*ceil(log2(n+m)).
- **inner** (counted_custom): Inner product; cost = 2*N for 1-D, 2*N*M for n-D.
- **interp** (counted_custom): 1-D linear interpolation.
- **intersect1d** (counted_custom): Set intersection; cost = (n+m)*ceil(log2(n+m)).
- **invert** (counted_unary): Bitwise NOT element-wise.
- **isclose** (counted_unary): Element-wise approximate equality test.
- **iscomplex** (counted_unary): Test if element is complex element-wise.
- **iscomplexobj** (counted_unary): Return True if input is a complex type or array.
- **isin** (counted_custom): Set membership; cost = (n+m)*ceil(log2(n+m)).
- **isnat** (counted_unary): Test for NaT (not-a-time) element-wise.
- **isneginf** (counted_unary): Test for negative infinity element-wise.
- **isposinf** (counted_unary): Test for positive infinity element-wise.
- **isreal** (counted_unary): Test if element is real (imag == 0) element-wise.
- **isrealobj** (counted_unary): Return True if x is a not complex type or array.
- **kaiser** (counted_custom): Kaiser window. Cost: 3*n (Bessel function eval per sample).
- **kron** (counted_custom): Kronecker product; cost proportional to output size.
- **lcm** (counted_binary): Element-wise least common multiple.
- **left_shift** (counted_binary): Element-wise left bit shift (legacy name).
- **linalg.cond** (counted_custom): Condition number. Cost: m*n*min(m,n) (via SVD).
- **linalg.cross** (counted_custom): Delegates to `me.cross` which charges `numel(output)` FLOPs.
- **linalg.matmul** (counted_custom): Delegates to `me.matmul` which charges `2*m*k*n` FLOPs.
- **linalg.matrix_norm** (counted_custom): Matrix norm. Cost depends on ord: 2*numel for Frobenius, m*n*min(m,n) for ord=2.
- **linalg.matrix_power** (counted_custom): Matrix power. Cost: $(\lfloor\log_2 k\rfloor + \text{popcount}(k) - 1) \cdot n^3$ (exponentiation by squaring).
- **linalg.matrix_rank** (counted_custom): Matrix rank. Cost: m*n*min(m,n) (via SVD).
- **linalg.multi_dot** (counted_custom): Chain matmul. Cost: sum of optimal chain matmul costs (CLRS §15.2).
- **linalg.norm** (counted_custom): Norm. Cost depends on ord: numel for L1/inf, 2*numel for Frobenius, m*n*min(m,n) for ord=2.
- **linalg.outer** (counted_custom): Delegates to `me.outer` which charges `m*n` FLOPs.
- **linalg.tensordot** (counted_custom): Delegates to `me.tensordot` which charges FLOPs based on contraction.
- **linalg.tensorinv** (counted_custom): Tensor inverse. Cost: $n^3$ after reshape (delegates to inv).
- **linalg.tensorsolve** (counted_custom): Tensor solve. Cost: $n^3$ after reshape (delegates to solve).
- **linalg.trace** (counted_custom): Matrix trace. Cost: n (sum of diagonal elements).
- **linalg.vecdot** (counted_custom): Delegates to `me.vecdot` which charges `2*n` FLOPs.
- **linalg.vector_norm** (counted_custom): Vector norm. Cost: numel (or 2*numel for general p-norm).
- **logspace** (counted_custom): Log-spaced generation; cost = num.
- **matmul** (counted_custom): Matrix multiplication; cost = 2*M*N*K.
- **modf** (counted_unary): Return fractional and integral parts element-wise.
- **nan_to_num** (counted_unary): Replace NaN/inf with finite numbers element-wise.
- **nanargmax** (counted_reduction): Index of maximum ignoring NaNs.
- **nanargmin** (counted_reduction): Index of minimum ignoring NaNs.
- **nancumprod** (counted_reduction): Cumulative product ignoring NaNs.
- **nancumsum** (counted_reduction): Cumulative sum ignoring NaNs.
- **outer** (counted_custom): Outer product of two vectors; cost = M*N.
- **pow** (counted_binary): Alias for power (NumPy 2.x).
- **ptp** (counted_reduction): Peak-to-peak (max - min) range of array.
- **random.beta** (counted_custom): Sampling; cost = numel(output).
- **random.bytes** (counted_custom): Sampling; cost = numel(output).
- **random.chisquare** (counted_custom): Sampling; cost = numel(output).
- **random.choice** (counted_custom): Sampling; cost = numel(output) if replace, n*ceil(log2(n)) if not.
- **random.dirichlet** (counted_custom): Sampling; cost = numel(output).
- **random.exponential** (counted_custom): Sampling; cost = numel(output).
- **random.f** (counted_custom): Sampling; cost = numel(output).
- **random.gamma** (counted_custom): Sampling; cost = numel(output).
- **random.geometric** (counted_custom): Sampling; cost = numel(output).
- **random.gumbel** (counted_custom): Sampling; cost = numel(output).
- **random.hypergeometric** (counted_custom): Sampling; cost = numel(output).
- **random.laplace** (counted_custom): Sampling; cost = numel(output).
- **random.logistic** (counted_custom): Sampling; cost = numel(output).
- **random.lognormal** (counted_custom): Sampling; cost = numel(output).
- **random.logseries** (counted_custom): Sampling; cost = numel(output).
- **random.multinomial** (counted_custom): Sampling; cost = numel(output).
- **random.multivariate_normal** (counted_custom): Sampling; cost = numel(output).
- **random.negative_binomial** (counted_custom): Sampling; cost = numel(output).
- **random.noncentral_chisquare** (counted_custom): Sampling; cost = numel(output).
- **random.noncentral_f** (counted_custom): Sampling; cost = numel(output).
- **random.normal** (counted_custom): Sampling; cost = numel(output).
- **random.pareto** (counted_custom): Sampling; cost = numel(output).
- **random.power** (counted_custom): Sampling; cost = numel(output).
- **random.rand** (counted_custom): Sampling; cost = numel(output).
- **random.randint** (counted_custom): Sampling; cost = numel(output).
- **random.randn** (counted_custom): Sampling; cost = numel(output).
- **random.random** (counted_custom): Sampling; cost = numel(output).
- **random.random_integers** (counted_custom): Sampling; cost = numel(output).
- **random.random_sample** (counted_custom): Sampling; cost = numel(output).
- **random.ranf** (counted_custom): Sampling; cost = numel(output).
- **random.rayleigh** (counted_custom): Sampling; cost = numel(output).
- **random.sample** (counted_custom): Sampling; cost = numel(output).
- **random.triangular** (counted_custom): Sampling; cost = numel(output).
- **random.vonmises** (counted_custom): Sampling; cost = numel(output).
- **random.wald** (counted_custom): Sampling; cost = numel(output).
- **random.weibull** (counted_custom): Sampling; cost = numel(output).
- **random.zipf** (counted_custom): Sampling; cost = numel(output).
- **real** (counted_unary): Return real part of complex array.
- **real_if_close** (counted_unary): Return real array if imaginary part is negligible.
- **right_shift** (counted_binary): Element-wise right bit shift (legacy name).
- **round** (counted_unary): Round to given number of decimals element-wise.
- **setdiff1d** (counted_custom): Set difference; cost = (n+m)*ceil(log2(n+m)).
- **setxor1d** (counted_custom): Symmetric set difference; cost = (n+m)*ceil(log2(n+m)).
- **sinc** (counted_unary): Normalized sinc function element-wise.
- **sort_complex** (counted_unary): Sort complex array by real then imaginary part.
- **spacing** (counted_unary): Return ULP spacing for each element.
- **tensordot** (counted_custom): Tensor dot product along specified axes.
- **trace** (counted_custom): Diagonal sum; cost = min(n,m).
- **trapezoid** (counted_custom): Integrate using the trapezoidal rule.
- **trapz** (counted_custom): Alias for trapezoid (deprecated).
- **union1d** (counted_custom): Set union; cost = (n+m)*ceil(log2(n+m)).
- **unique_all** (counted_custom): Sort-based unique; cost = n*ceil(log2(n)).
- **unique_counts** (counted_custom): Sort-based unique; cost = n*ceil(log2(n)).
- **unique_inverse** (counted_custom): Sort-based unique; cost = n*ceil(log2(n)).
- **unique_values** (counted_custom): Sort-based unique; cost = n*ceil(log2(n)).
- **unwrap** (counted_custom): Phase unwrap. Cost: $\text{numel}(\text{input})$ (diff + conditional adjustment).
- **vander** (counted_custom): Vandermonde matrix; cost = len(x)*(N-1).
- **vdot** (counted_custom): Dot product with conjugation; cost = 2*N.
- **vecdot** (counted_binary): Vector dot product along last axis.
