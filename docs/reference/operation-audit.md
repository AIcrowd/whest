# Operation Audit

Complete list of every NumPy operation and its mechestim category.
Generated from the operation registry (`_registry.py`).

## Summary

| Category | Count | Cost |
|----------|-------|------|
| free | 209 | 0 FLOPs |
| counted_unary | 73 | numel(output) |
| counted_binary | 45 | numel(output) |
| counted_reduction | 37 | numel(input) |
| counted_custom | 22 | Per-operation formula |
| blacklisted | 96 | Not available |
| **Total** | **482** | |

## 🟢 Free Operations (0 FLOPs)

No FLOP cost. Tensor creation, reshaping, indexing, random number generation.

**209 operations:**

`allclose`, `append`, `arange`, `argpartition`, `argsort`, `argwhere`
`array`, `array_equal`, `array_equiv`, `array_split`, `asarray`, `asarray_chkfinite`
`astype`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `base_repr`, `binary_repr`
`bincount`, `block`, `bmat`, `broadcast_arrays`, `broadcast_shapes`, `broadcast_to`
`can_cast`, `choose`, `column_stack`, `common_type`, `compress`, `concat`
`concatenate`, `copy`, `copyto`, `delete`, `diag`, `diag_indices`
`diag_indices_from`, `diagflat`, `diagonal`, `digitize`, `dsplit`, `dstack`
`empty`, `empty_like`, `expand_dims`, `extract`, `eye`, `fill_diagonal`
`flatnonzero`, `flip`, `fliplr`, `flipud`, `from_dlpack`, `frombuffer`
`fromfile`, `fromfunction`, `fromiter`, `fromregex`, `fromstring`, `full`
`full_like`, `geomspace`, `histogram`, `histogram2d`, `histogram_bin_edges`, `histogramdd`
`hsplit`, `hstack`, `identity`, `in1d`, `indices`, `insert`
`intersect1d`, `isdtype`, `isfinite`, `isfortran`, `isin`, `isinf`
`isnan`, `isscalar`, `issubdtype`, `iterable`, `ix_`, `lexsort`
`linspace`, `logspace`, `mask_indices`, `matrix_transpose`, `may_share_memory`, `meshgrid`
`min_scalar_type`, `mintypecode`, `moveaxis`, `ndim`, `nonzero`, `ones`
`ones_like`, `packbits`, `pad`, `partition`, `permute_dims`, `place`
`promote_types`, `put`, `put_along_axis`, `putmask`, `random.beta`, `random.binomial`
`random.bytes`, `random.chisquare`, `random.choice`, `random.default_rng`, `random.dirichlet`, `random.exponential`
`random.f`, `random.gamma`, `random.geometric`, `random.get_state`, `random.gumbel`, `random.hypergeometric`
`random.laplace`, `random.logistic`, `random.lognormal`, `random.logseries`, `random.multinomial`, `random.multivariate_normal`
`random.negative_binomial`, `random.noncentral_chisquare`, `random.noncentral_f`, `random.normal`, `random.pareto`, `random.permutation`
`random.poisson`, `random.power`, `random.rand`, `random.randint`, `random.randn`, `random.random`
`random.random_integers`, `random.random_sample`, `random.ranf`, `random.rayleigh`, `random.sample`, `random.seed`
`random.set_state`, `random.shuffle`, `random.standard_cauchy`, `random.standard_exponential`, `random.standard_gamma`, `random.standard_normal`
`random.standard_t`, `random.triangular`, `random.uniform`, `random.vonmises`, `random.wald`, `random.weibull`
`random.zipf`, `ravel`, `ravel_multi_index`, `repeat`, `require`, `reshape`
`resize`, `result_type`, `roll`, `rollaxis`, `rot90`, `row_stack`
`searchsorted`, `select`, `setdiff1d`, `setxor1d`, `shape`, `shares_memory`
`size`, `sort`, `split`, `squeeze`, `stack`, `swapaxes`
`take`, `take_along_axis`, `tile`, `trace`, `transpose`, `tri`
`tril`, `tril_indices`, `tril_indices_from`, `trim_zeros`, `triu`, `triu_indices`
`triu_indices_from`, `typename`, `union1d`, `unique`, `unique_all`, `unique_counts`
`unique_inverse`, `unique_values`, `unpackbits`, `unravel_index`, `unstack`, `vander`
`vsplit`, `vstack`, `where`, `zeros`, `zeros_like`

## 🟡 Counted Unary Operations

Cost: numel(output) per call. Scalar math applied element-wise.

**73 operations:**

`abs`, `absolute`, `acos`, `acosh`, `angle`, `arccos`
`arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`, `around`
`asin`, `asinh`, `atan`, `atanh`, `bitwise_count`, `bitwise_invert`
`bitwise_not`, `cbrt`, `ceil`, `conj`, `conjugate`, `cos`
`cosh`, `deg2rad`, `degrees`, `exp`, `exp2`, `expm1`
`fabs`, `fix`, `floor`, `frexp`, `i0`, `imag`
`invert`, `isclose`, `iscomplex`, `iscomplexobj`, `isnat`, `isneginf`
`isposinf`, `isreal`, `isrealobj`, `log`, `log10`, `log1p`
`log2`, `logical_not`, `modf`, `nan_to_num`, `negative`, `positive`
`rad2deg`, `radians`, `real`, `real_if_close`, `reciprocal`, `rint`
`round`, `sign`, `signbit`, `sin`, `sinc`, `sinh`
`sort_complex`, `spacing`, `sqrt`, `square`, `tan`, `tanh`
`trunc`

## 🟡 Counted Binary Operations

Cost: numel(output) per call. Element-wise operations on two arrays.

**45 operations:**

`add`, `arctan2`, `atan2`, `bitwise_and`, `bitwise_left_shift`, `bitwise_or`
`bitwise_right_shift`, `bitwise_xor`, `copysign`, `divide`, `divmod`, `equal`
`float_power`, `floor_divide`, `fmax`, `fmin`, `fmod`, `gcd`
`greater`, `greater_equal`, `heaviside`, `hypot`, `lcm`, `ldexp`
`left_shift`, `less`, `less_equal`, `logaddexp`, `logaddexp2`, `logical_and`
`logical_or`, `logical_xor`, `maximum`, `minimum`, `mod`, `multiply`
`nextafter`, `not_equal`, `pow`, `power`, `remainder`, `right_shift`
`subtract`, `true_divide`, `vecdot`

## 🟡 Counted Reduction Operations

Cost: numel(input) per call. Reduce array along one or more axes.

**37 operations:**

`all`, `amax`, `amin`, `any`, `argmax`, `argmin`
`average`, `count_nonzero`, `cumprod`, `cumsum`, `cumulative_prod`, `cumulative_sum`
`max`, `mean`, `median`, `min`, `nanargmax`, `nanargmin`
`nancumprod`, `nancumsum`, `nanmax`, `nanmean`, `nanmedian`, `nanmin`
`nanpercentile`, `nanprod`, `nanquantile`, `nanstd`, `nansum`, `nanvar`
`percentile`, `prod`, `ptp`, `quantile`, `std`, `sum`
`var`

## 🟡 Counted Custom Operations

Bespoke cost formulas. See each operation's documentation for details.

**22 operations:**

`clip`, `convolve`, `corrcoef`, `correlate`, `cov`, `cross`
`diff`, `dot`, `ediff1d`, `einsum`, `einsum_path`, `gradient`
`inner`, `interp`, `kron`, `linalg.svd`, `matmul`, `outer`
`tensordot`, `trapezoid`, `trapz`, `vdot`

## 🔴 Unsupported Operations

Calling these raises `AttributeError` with guidance on alternatives.

**96 operations:**

`apply_along_axis`, `apply_over_axes`, `array2string`, `array_repr`, `array_str`, `asmatrix`
`bartlett`, `blackman`, `busday_count`, `busday_offset`, `datetime_as_string`, `datetime_data`
`fft.fft`, `fft.fft2`, `fft.fftfreq`, `fft.fftn`, `fft.fftshift`, `fft.hfft`
`fft.ifft`, `fft.ifft2`, `fft.ifftn`, `fft.ifftshift`, `fft.ihfft`, `fft.irfft`
`fft.irfft2`, `fft.irfftn`, `fft.rfft`, `fft.rfft2`, `fft.rfftfreq`, `fft.rfftn`
`format_float_positional`, `format_float_scientific`, `frompyfunc`, `genfromtxt`, `get_include`, `getbufsize`
`geterr`, `geterrcall`, `hamming`, `hanning`, `is_busday`, `kaiser`
`linalg.cholesky`, `linalg.cond`, `linalg.cross`, `linalg.det`, `linalg.diagonal`, `linalg.eig`
`linalg.eigh`, `linalg.eigvals`, `linalg.eigvalsh`, `linalg.inv`, `linalg.lstsq`, `linalg.matmul`
`linalg.matrix_norm`, `linalg.matrix_power`, `linalg.matrix_rank`, `linalg.matrix_transpose`, `linalg.multi_dot`, `linalg.norm`
`linalg.outer`, `linalg.pinv`, `linalg.qr`, `linalg.slogdet`, `linalg.solve`, `linalg.svdvals`
`linalg.tensordot`, `linalg.tensorinv`, `linalg.tensorsolve`, `linalg.trace`, `linalg.vecdot`, `linalg.vector_norm`
`load`, `loadtxt`, `nested_iters`, `piecewise`, `poly`, `polyadd`
`polyder`, `polydiv`, `polyfit`, `polyint`, `polymul`, `polysub`
`polyval`, `roots`, `save`, `savetxt`, `savez`, `savez_compressed`
`setbufsize`, `seterr`, `seterrcall`, `show_config`, `show_runtime`, `unwrap`

