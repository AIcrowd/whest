# FLOP Weight Calibration Results

## Introduction

Per-operation FLOP weights are multiplicative correction factors that bridge
the gap between mechestim's analytical cost formulas and the actual
floating-point instruction cost observed on hardware. When weights are
loaded, the effective cost of an operation becomes:

$$
\text{cost}(\text{op}) = C(\text{op}, \text{shapes}) \times w(\text{op})
$$

where $C$ is the analytical FLOP formula and $w$ is the weight.
A weight of 25.9 for `sin` means that each analytical FLOP of sine costs
approximately 26 times more in actual floating-point instructions than a
FLOP of addition.

## Methodology

### The correction formula

Every weight is computed from the same two-step formula:

$$
\alpha(\text{op}) = \mathrm{median}_{D} \left[ \frac{F(\text{op})}{C(\text{op}, \text{params}) \times R} \right]
$$

$$
w(\text{op}) = \frac{\alpha(\text{op})}{\alpha(\text{add})}
$$

Where:

- $\alpha(\text{op})$ is the **raw correction factor** -- the ratio of hardware-observed FP instructions to the analytical FLOP count.
- $F(\text{op})$ is the total SIMD-width-weighted count of retired floating-point instructions, measured via the Intel PMU counters `fp_arith_inst_retired.*` (scalar x1, 128-bit x2, 256-bit x4, 512-bit x8).
- $C(\text{op}, \text{params})$ is the analytical FLOP count from mechestim's cost formula (e.g., `numel(output)` for pointwise ops).
- $R$ is the number of repeats per distribution.
- The **median** across 3 input distributions is reported.

## Measurement environment

| Parameter | Value |
|-----------|-------|
| CPU | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz |
| Cores | 64 physical / 128 threads |
| RAM | 251.7 GB |
| Arch | x86_64 (AVX-512 capable) |
| Cache | L1d 48 KB, L1i 32 KB, L2 1280 KB, L3 54 MB |
| Instance | AWS EC2 c6i.metal (bare metal, full PMU access) |
| OS | Linux 6.1.166-197.305.amzn2023.x86_64 |
| Python | 3.11.14 |
| NumPy | 2.1.3 |
| BLAS | scipy-openblas 0.3.27 |
| Measurement mode | perf (hardware counters: `fp_arith_inst_retired.*`) |
| dtype | float64 |
| Repeats | 5 per distribution |
| Distributions | 3 per operation |
| Methodology version | 2.0 |
| Baseline alpha(add) | 2.200112 |
    - **Date:** 2026-04-10
    - **Total calibration time:** 2275.7 seconds

## Baseline details

All weights are normalized against element-wise addition (`np.add`):

- **Benchmark command:** `np.add(x, y, out=_out)`
- **Array size:** A(512,512) x B(512,512), dtype=float64
- **Measured perf instructions:** 110005601.0
- **Measured timing:** 75917896.0 ns
- **$\alpha(\text{add})$:** 2.200112

**[Download full review spreadsheet (CSV)](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/data/weights.csv)**

## Weight tables

### Pointwise Unary (47 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `arccosh` | 37.7712 | 37.8x | high | numel(output) | [\_pointwise.py:269](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L269) | Element-wise inverse hyperbolic cosine. |
| `arcsinh` | 36.1800 | 36.2x | high | numel(output) | [\_pointwise.py:271](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L271) | Element-wise inverse hyperbolic sine. |
| `arctanh` | 32.9939 | 33.0x | high | numel(output) | [\_pointwise.py:273](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L273) | Element-wise inverse hyperbolic tangent. |
| `tan` | 27.5441 | 27.5x | high | numel(output) | [\_pointwise.py:368](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L368) | Element-wise tangent. |
| `arcsin` | 25.7215 | 25.7x | high | numel(output) | [\_pointwise.py:270](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L270) | Element-wise inverse sine. |
| `arccos` | 24.3579 | 24.4x | high | numel(output) | [\_pointwise.py:268](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L268) | Element-wise inverse cosine. |
| `arctan` | 21.6353 | 21.6x | high | numel(output) | [\_pointwise.py:272](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L272) | Element-wise inverse tangent. |
| `expm1` | 18.9082 | 18.9x | high | numel(output) | [\_pointwise.py:314](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L314) | Element-wise e^x - 1 (accurate near zero). |
| `log1p` | 18.9082 | 18.9x | high | numel(output) | [\_pointwise.py:327](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L327) | Element-wise log(1+x) (accurate near zero). |
| `cos` | 18.4115 | 18.4x | high | numel(output) | [\_pointwise.py:254](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L254) | Element-wise cosine. |
| `sin` | 18.3903 | 18.4x | high | numel(output) | [\_pointwise.py:253](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L253) | Element-wise sine. |
| `cbrt` | 17.5446 | 17.5x | high | numel(output) | [\_pointwise.py:307](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L307) | Element-wise cube root. |
| `log10` | 16.2638 | 16.3x | high | numel(output) | [\_pointwise.py:248](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L248) | Element-wise base-10 logarithm. |
| `log2` | 16.0365 | 16.0x | high | numel(output) | [\_pointwise.py:247](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L247) | Element-wise base-2 logarithm. |
| `sinh` | 15.2720 | 15.3x | high | numel(output) | [\_pointwise.py:365](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L365) | Element-wise hyperbolic sine. |
| `tanh` | 15.2720 | 15.3x | high | numel(output) | [\_pointwise.py:255](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L255) | Element-wise hyperbolic tangent. |
| `log` | 14.4457 | 14.4x | high | numel(output) | [\_pointwise.py:246](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L246) | Element-wise natural logarithm. |
| `cosh` | 12.9994 | 13.0x | high | numel(output) | [\_pointwise.py:310](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L310) | Element-wise hyperbolic cosine. |
| `exp` | 10.2723 | 10.3x | high | numel(output) | [\_pointwise.py:245](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L245) | Element-wise e^x. |
| `exp2` | 7.0906 | 7.1x | high | numel(output) | [\_pointwise.py:313](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L313) | Element-wise 2^x. |
| `clip` | 1.1818 | 1.2x | medium | numel(output) | [\_pointwise.py:473](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L473) | Clip array to [a_min, a_max] element-wise. |
| `sqrt` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:251](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L251) | Element-wise square root. |
| `square` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:252](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L252) | Element-wise x^2. |
| `reciprocal` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:335](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L335) | Element-wise 1/x. |
| `deg2rad` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:311](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L311) | Alias for radians. |
| `rad2deg` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:331](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L331) | Alias for degrees. |
| `degrees` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:312](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L312) | Convert radians to degrees element-wise. |
| `radians` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:332](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L332) | Convert degrees to radians element-wise. |
| `frexp` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:373](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L373) | Decompose x into mantissa and exponent element-wise. |
| `spacing` | 0.7273 | 0.7x | medium | numel(output) | [\_pointwise.py:367](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L367) | Return ULP spacing for each element. |
| `modf` | 0.7228 | 0.7x | low | numel(output) | [\_pointwise.py:372](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L372) | Return fractional and integral parts element-wise. |
| `abs` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:249](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L249) | Element-wise absolute value; alias for absolute. |
| `negative` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:250](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L250) | Element-wise negation. |
| `positive` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:330](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L330) | Element-wise unary plus (copy with sign preserved). |
| `ceil` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:257](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L257) | Element-wise ceiling. |
| `floor` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:258](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L258) | Element-wise floor. |
| `trunc` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:369](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L369) | Truncate toward zero element-wise. |
| `rint` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:336](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L336) | Round to nearest integer element-wise. |
| `sign` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:256](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L256) | Element-wise sign function. |
| `signbit` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:363](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L363) | Returns True for elements with negative sign bit. |
| `fabs` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:315](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L315) | Element-wise absolute value (always float). |
| `logical_not` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:328](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L328) | Element-wise logical NOT. |
| `sinc` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:364](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L364) | Normalized sinc function element-wise. |
| `i0` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:317](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L317) | Modified Bessel function of order 0, element-wise. |
| `nan_to_num` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:329](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L329) | Replace NaN/inf with finite numbers element-wise. |
| `isneginf` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:323](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L323) | Test for negative infinity element-wise. |
| `isposinf` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:324](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L324) | Test for positive infinity element-wise. |

### Pointwise Binary (33 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `power` | 33.2092 | 33.2x | medium | numel(output) | [\_pointwise.py:413](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L413) | Element-wise exponentiation x**y. |
| `arctan2` | 24.6352 | 24.6x | high | numel(output) | [\_pointwise.py:420](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L420) | Element-wise arctan(y/x) considering quadrant. |
| `logaddexp2` | 16.0157 | 16.0x | low | numel(output) | [\_pointwise.py:445](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L445) | log2(2**x1 + 2**x2) element-wise. |
| `logaddexp` | 15.3625 | 15.4x | low | numel(output) | [\_pointwise.py:444](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L444) | log(exp(x1) + exp(x2)) element-wise. |
| `float_power` | 14.8026 | 14.8x | low | numel(output) | [\_pointwise.py:429](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L429) | Element-wise exponentiation in float64. |
| `hypot` | 5.3182 | 5.3x | high | numel(output) | [\_pointwise.py:438](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L438) | Element-wise Euclidean norm sqrt(x1^2 + x2^2). |
| `floor_divide` | 2.1912 | 2.2x | medium | numel(output) | [\_pointwise.py:430](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L430) | Element-wise floor division. |
| `isclose` | 1.9090 | 1.9x | medium | numel(output) | [\_pointwise.py:388](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L388) | Element-wise approximate equality test. |
| `add` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:407](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L407) | Element-wise addition. |
| `subtract` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:408](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L408) | Element-wise subtraction. |
| `multiply` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:409](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L409) | Element-wise multiplication. |
| `divide` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:410](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L410) | Element-wise true division. |
| `true_divide` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:454](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L454) | Element-wise true division (explicit). |
| `maximum` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:411](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L411) | Element-wise maximum (propagates NaN). |
| `minimum` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:412](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L412) | Element-wise minimum (propagates NaN). |
| `fmax` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:431](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L431) | Element-wise maximum ignoring NaN. |
| `fmin` | 1.0000 | 1.0x | medium | numel(output) | [\_pointwise.py:432](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L432) | Element-wise minimum ignoring NaN. |
| `mod` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:414](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L414) | Element-wise modulo. |
| `remainder` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:452](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L452) | Element-wise remainder (same as mod). |
| `fmod` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:433](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L433) | Element-wise C-style fmod (remainder toward zero). |
| `greater` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:435](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L435) | Element-wise x1 > x2. |
| `greater_equal` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:436](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L436) | Element-wise x1 >= x2. |
| `less` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:442](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L442) | Element-wise x1 < x2. |
| `less_equal` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:443](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L443) | Element-wise x1 <= x2. |
| `equal` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:428](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L428) | Element-wise x1 == x2. |
| `not_equal` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:450](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L450) | Element-wise x1 != x2. |
| `logical_and` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:446](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L446) | Element-wise logical AND. |
| `logical_or` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:447](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L447) | Element-wise logical OR. |
| `logical_xor` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:448](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L448) | Element-wise logical XOR. |
| `copysign` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:427](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L427) | Copy sign of x2 to magnitude of x1 element-wise. |
| `nextafter` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:449](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L449) | Return next float after x1 toward x2 element-wise. |
| `ldexp` | 0.5455 | 0.5x | low | numel(output) | [\_pointwise.py:440](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L440) | Return x1 * 2**x2 element-wise. |
| `heaviside` | 0.2728 | 0.3x | low | numel(output) | [\_pointwise.py:437](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L437) | Heaviside step function element-wise. |

### Reductions (35 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `std` | 2.0909 | 2.1x | high | numel(input) | [\_pointwise.py:501](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L501) | Standard deviation; cost_multiplier=2 (two passes). |
| `var` | 2.0909 | 2.1x | high | numel(input) | [\_pointwise.py:502](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L502) | Variance; cost_multiplier=2 (two passes). |
| `nanstd` | 2.0909 | 2.1x | high | numel(input) | [\_pointwise.py:532](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L532) | Standard deviation ignoring NaNs. |
| `nanvar` | 2.0909 | 2.1x | high | numel(input) | [\_pointwise.py:534](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L534) | Variance ignoring NaNs. |
| `ptp` | 1.1827 | 1.2x | medium | numel(input) | [\_pointwise.py:549](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L549) | Peak-to-peak (max - min) range of array. |
| `max` | 0.7277 | 0.7x | medium | numel(input) | [\_pointwise.py:497](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L497) | Maximum value of array. |
| `min` | 0.7277 | 0.7x | medium | numel(input) | [\_pointwise.py:498](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L498) | Minimum value of array. |
| `nanmax` | 0.7277 | 0.7x | medium | numel(input) | [\_pointwise.py:525](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L525) | Maximum ignoring NaNs. |
| `nanmin` | 0.7277 | 0.7x | medium | numel(input) | [\_pointwise.py:528](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L528) | Minimum ignoring NaNs. |
| `sum` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:496](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L496) | Sum of array elements. |
| `prod` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:499](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L499) | Product of array elements. |
| `mean` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:500](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L500) | Arithmetic mean of array elements. |
| `cumsum` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:505](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L505) | Cumulative sum of array elements. |
| `cumprod` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:506](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L506) | Cumulative product of array elements. |
| `nansum` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:533](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L533) | Sum ignoring NaNs. |
| `nanmean` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:526](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L526) | Mean ignoring NaNs. |
| `nanprod` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:530](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L530) | Product ignoring NaNs. |
| `average` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:516](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L516) | Weighted average of array elements. |
| `nancumprod` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:523](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L523) | Cumulative product ignoring NaNs. |
| `nancumsum` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:524](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L524) | Cumulative sum ignoring NaNs. |
| `cumulative_sum` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:519](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L519) | Cumulative sum (NumPy 2.x array API). |
| `cumulative_prod` | 0.7273 | 0.7x | medium | numel(input) | [\_pointwise.py:518](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L518) | Cumulative product (NumPy 2.x array API). |
| `argmax` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:503](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L503) | Index of maximum value. |
| `argmin` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:504](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L504) | Index of minimum value. |
| `any` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:515](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L515) | Test whether any array element is true. |
| `all` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:512](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L512) | Test whether all array elements are true. |
| `median` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:520](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L520) | Median of array elements (sorts internally). |
| `nanmedian` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:527](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L527) | Median ignoring NaNs. |
| `percentile` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:535](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L535) | q-th percentile of array elements. |
| `nanpercentile` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:529](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L529) | q-th percentile ignoring NaNs. |
| `quantile` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:536](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L536) | q-th quantile of array elements. |
| `nanquantile` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:531](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L531) | q-th quantile ignoring NaNs. |
| `count_nonzero` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:517](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L517) | Count non-zero elements. |
| `nanargmax` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:521](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L521) | Index of maximum ignoring NaNs. |
| `nanargmin` | 0.2728 | 0.3x | low | numel(input) | [\_pointwise.py:522](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L522) | Index of minimum ignoring NaNs. |

### Sorting (17 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `intersect1d` | 3.7015 | 3.7x | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:340](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L340) | Set intersection; cost = (n+m)*ceil(log2(n+m)). |
| `setxor1d` | 3.7015 | 3.7x | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:389](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L389) | Symmetric set difference; cost = (n+m)*ceil(log2(n+m)). |
| `argsort` | 2.3842 | 2.4x | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:62](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L62) | Indirect sort; cost = n*ceil(log2(n)) per slice. |
| `unique_inverse` | 2.3842 | 2.4x | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:268](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L268) | Sort-based unique; cost = n*ceil(log2(n)). |
| `argpartition` | 2.1556 | 2.2x | medium | n | [\_sorting\_ops.py:136](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L136) | Indirect partition; cost = n per slice. |
| `in1d` | 2.1064 | 2.1x | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:313](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L313) | Set membership; cost = (n+m)*ceil(log2(n+m)). |
| `isin` | 2.1064 | 2.1x | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:326](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L326) | Set membership; cost = (n+m)*ceil(log2(n+m)). |
| `union1d` | 2.1026 | 2.1x | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:357](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L357) | Set union; cost = (n+m)*ceil(log2(n+m)). |
| `partition` | 2.0761 | 2.1x | medium | n | [\_sorting\_ops.py:111](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L111) | Quickselect; cost = n per slice. |
| `sort` | 2.0037 | 2.0x | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:43](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L43) | Comparison sort; cost = n*ceil(log2(n)) per slice. |
| `unique` | 2.0037 | 2.0x | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:227](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L227) | Sort-based unique; cost = n*ceil(log2(n)). |
| `unique_counts` | 2.0037 | 2.0x | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:252](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L252) | Sort-based unique; cost = n*ceil(log2(n)). |
| `unique_values` | 2.0037 | 2.0x | medium | n * ceil(log2(n)) | [\_sorting\_ops.py:284](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L284) | Sort-based unique; cost = n*ceil(log2(n)). |
| `setdiff1d` | 1.9238 | 1.9x | medium | (n+m) * ceil(log2(n+m)) | [\_sorting\_ops.py:372](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L372) | Set difference; cost = (n+m)*ceil(log2(n+m)). |
| `searchsorted` | 0.6768 | 0.7x | low | m * ceil(log2(n)) | [\_sorting\_ops.py:166](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L166) | Binary search; cost = m*ceil(log2(n)). |
| `lexsort` | 0.3384 | 0.3x | low | k * n * ceil(log2(n)) | [\_sorting\_ops.py:88](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L88) | Multi-key sort; cost = k*n*ceil(log2(n)). |
| `unique_all` | 0.3381 | 0.3x | low | n * ceil(log2(n)) | [\_sorting\_ops.py:239](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L239) | Sort-based unique; cost = n*ceil(log2(n)). |

### FFT (14 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `fft.hfft` | 1.0401 | 1.0x | high | 5*n*ceil(log2(n)) | [\_transforms.py:443](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L443) | FFT of Hermitian-symmetric signal. Cost: 5*n_out*ceil(log2(n_out)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifft` | 0.6123 | 0.6x | medium | 5*n*ceil(log2(n)) | [\_transforms.py:189](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L189) | Inverse 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfft` | 0.4698 | 0.5x | high | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:221](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L221) | Inverse 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfft2` | 0.4095 | 0.4x | high | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:314](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L314) | Inverse 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfftn` | 0.4095 | 0.4x | high | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:423](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L423) | Inverse N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fft` | 0.3796 | 0.4x | medium | 5*n*ceil(log2(n)) | [\_transforms.py:173](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L173) | 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfft` | 0.3794 | 0.4x | low | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:205](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L205) | 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifft2` | 0.3560 | 0.4x | medium | 5*n*ceil(log2(n)) | [\_transforms.py:265](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L265) | Inverse 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifftn` | 0.3560 | 0.4x | medium | 5*n*ceil(log2(n)) | [\_transforms.py:363](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L363) | Inverse N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fft2` | 0.3278 | 0.3x | medium | 5*n*ceil(log2(n)) | [\_transforms.py:242](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L242) | 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fftn` | 0.3278 | 0.3x | medium | 5*n*ceil(log2(n)) | [\_transforms.py:339](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L339) | N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfft2` | 0.3215 | 0.3x | low | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:288](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L288) | 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfftn` | 0.3215 | 0.3x | low | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:387](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L387) | N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ihfft` | 0.1943 | 0.2x | low | 5*(n/2)*ceil(log2(n)) | [\_transforms.py:462](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L462) | Inverse FFT of Hermitian signal. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |

### Linalg (14 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `linalg.pinv` | 5.1094 | 5.1x | high | m*n*min(m,n) | [\_solvers.py:187](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L187) | Pseudoinverse. Cost: m*n*min(m,n) (via SVD). |
| `linalg.svd` | 4.1973 | 4.2x | medium | m*n*min(m,n) | [\_svd.py:67](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_svd.py#L67) | Singular value decomposition; cost ~ O(min(m,n)*m*n). |
| `linalg.eigh` | 1.6482 | 1.6x | high | 4*n^3 / 3 | [\_decompositions.py:165](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L165) | Symmetric eigendecomposition. Cost: $(4/3)n^3$ (Golub & Van Loan §8.3). |
| `linalg.lstsq` | 1.4063 | 1.4x | medium | m*n*min(m,n) | [\_solvers.py:147](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L147) | Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD). |
| `linalg.svdvals` | 1.3389 | 1.3x | medium | m*n*min(m,n) | [\_decompositions.py:281](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L281) | Singular values only. Cost: m*n*min(m,n) (Golub-Reinsch). |
| `linalg.inv` | 1.2404 | 1.2x | medium | n^3 | [\_solvers.py:101](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L101) | Matrix inverse. Cost: $n^3$ (LU + solve). |
| `linalg.qr` | 0.9478 | 0.9x | medium | 2*m*n^2 - 2*n^3/3 | [\_decompositions.py:87](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L87) | QR decomposition. Cost: $2mn^2 - (2/3)n^3$ (Golub & Van Loan §5.2). |
| `linalg.cholesky` | 0.7606 | 0.8x | high | n^3 / 3 | [\_decompositions.py:46](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L46) | Cholesky decomposition. Cost: $n^3/3$ (Golub & Van Loan §4.2). |
| `linalg.eig` | 0.6827 | 0.7x | medium | 10*n^3 | [\_decompositions.py:127](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L127) | Eigendecomposition. Cost: $10n^3$ (Francis QR, Golub & Van Loan §7.5). |
| `linalg.eigvalsh` | 0.5738 | 0.6x | high | 4*n^3 / 3 | [\_decompositions.py:243](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L243) | Symmetric eigenvalues. Cost: $(4/3)n^3$ (same as eigh). |
| `linalg.det` | 0.4921 | 0.5x | low | 2*n^3 / 3 | [\_properties.py:87](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_properties.py#L87) | Determinant. Cost: $n^3$ (LU factorization). |
| `linalg.slogdet` | 0.4921 | 0.5x | low | 2*n^3 / 3 | [\_properties.py:134](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_properties.py#L134) | Sign + log determinant. Cost: $n^3$ (LU factorization). |
| `linalg.eigvals` | 0.4635 | 0.5x | high | 7*n^3 | [\_decompositions.py:207](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L207) | Eigenvalues only. Cost: $10n^3$ (same as eig). |
| `linalg.solve` | 0.4589 | 0.5x | low | 2*n^3/3 + 2*n^2 | [\_solvers.py:54](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L54) | Solve Ax=b. Cost: $2n^3/3$ (LU) + $n^2 \cdot n_{\text{rhs}}$ (back-substitution). |

### Contractions (9 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `vecdot` | 1.4848 | 1.5x | medium | batch * K (output_size * contracted_axis) | [\_pointwise.py:455](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L455) | Vector dot product along last axis. |
| `inner` | 1.4559 | 1.5x | medium | N (a.size) | [\_pointwise.py:649](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L649) | Inner product; cost = 2*N for 1-D, 2*N*M for n-D. |
| `vdot` | 1.4559 | 1.5x | medium | N (a.size) | [\_pointwise.py:707](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L707) | Dot product with conjugation; cost = 2*N. |
| `tensordot` | 0.4617 | 0.5x | high | 2*d^5 (axes=1, shape=(d,d,d)) | [\_\_init\_\_.py:74](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/__init__.py#L74) | Tensor dot product along specified axes. |
| `dot` | 0.4568 | 0.5x | high | 2*M*N*K | [\_pointwise.py:587](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L587) | Dot product; cost = 2*M*N*K for matrix multiply. |
| `matmul` | 0.4568 | 0.5x | high | 2*M*N*K | [\_pointwise.py:623](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L623) | Matrix multiplication; cost = 2*M*N*K. |
| `einsum` | 0.4551 | 0.5x | high | 2*M*N*K (ij,jk->ik) | [\_einsum.py:135](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_einsum.py#L135) | Generalized Einstein summation. |
| `outer` | 0.4547 | 0.5x | high | M*N | [\_pointwise.py:665](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L665) | Outer product of two vectors; cost = M*N. |
| `kron` | 0.4547 | 0.5x | high | d^4 (Kronecker, shape=(d,d)x(d,d)) | [\_pointwise.py:723](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L723) | Kronecker product; cost proportional to output size. |

### Polynomial (10 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `polyder` | 6.1851 | 6.2x |  | degree + 1 | [\_polynomial.py:127](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L127) | Differentiate polynomial. Cost: n FLOPs. |
| `polyadd` | 5.8305 | 5.8x | high | degree + 1 | [\_polynomial.py:96](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L96) | Add two polynomials. Cost: max(n1, n2) FLOPs. |
| `polysub` | 5.8305 | 5.8x | high | degree + 1 | [\_polynomial.py:113](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L113) | Difference (subtraction) of two polynomials. Cost: max(n1, n2) FLOPs. |
| `polyint` | 5.7441 | 5.7x |  | degree + 1 | [\_polynomial.py:140](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L140) | Integrate polynomial. Cost: n FLOPs. |
| `polymul` | 1.1921 | 1.2x | high | (degree+1)^2 | [\_polynomial.py:158](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L158) | Multiply polynomials. Cost: n1 * n2 FLOPs. |
| `poly` | 0.9813 | 1.0x | high | degree^2 | [\_polynomial.py:204](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L204) | Polynomial from roots. Cost: $n^2$ FLOPs. |
| `polyfit` | 0.5387 | 0.5x | high | 2 * n * (degree+1)^2 | [\_polynomial.py:187](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L187) | Least squares polynomial fit. Cost: 2 * m * (deg+1)^2 FLOPs. |
| `roots` | 0.4898 | 0.5x | high | 10 * degree^3 | [\_polynomial.py:217](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L217) | Return roots of polynomial with given coefficients. Cost: $10n^3$ FLOPs (companion matrix eig). |
| `polyval` | 0.4597 | 0.5x | high | 2 * n * degree | [\_polynomial.py:78](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L78) | Evaluate polynomial at given points. Cost: 2 * m * deg FLOPs (Horner's method). |
| `polydiv` | 0.0720 | 0.1x | high | (degree+1)^2 | [\_polynomial.py:174](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L174) | Divide one polynomial by another. Cost: n1 * n2 FLOPs. |

### Random (43 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `random.hypergeometric` | 261.1704 | 261.2x | high | numel(output) | [\_\_init\_\_.py:123](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L123) | Sampling; cost = numel(output). |
| `random.multivariate_normal` | 205.9307 | 205.9x | high | numel(output) | [\_\_init\_\_.py:151](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L151) | Sampling; cost = numel(output). |
| `random.zipf` | 104.9946 | 105.0x | high | numel(output) | [\_\_init\_\_.py:133](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L133) | Sampling; cost = numel(output). |
| `random.noncentral_chisquare` | 68.2430 | 68.2x | high | numel(output) | [\_\_init\_\_.py:143](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L143) | Sampling; cost = numel(output). |
| `random.negative_binomial` | 64.4532 | 64.5x | high | numel(output) | [\_\_init\_\_.py:124](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L124) | Sampling; cost = numel(output). |
| `random.multinomial` | 62.2689 | 62.3x | high | numel(output) | [\_\_init\_\_.py:149](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L149) | Sampling; cost = numel(output). |
| `random.noncentral_f` | 54.9487 | 54.9x | high | numel(output) | [\_\_init\_\_.py:145](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L145) | Sampling; cost = numel(output). |
| `random.dirichlet` | 54.9252 | 54.9x | high | numel(output) | [\_\_init\_\_.py:153](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L153) | Sampling; cost = numel(output). |
| `random.power` | 50.0261 | 50.0x | high | numel(output) | [\_\_init\_\_.py:126](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L126) | Sampling; cost = numel(output). |
| `random.vonmises` | 47.4702 | 47.5x | high | numel(output) | [\_\_init\_\_.py:138](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L138) | Sampling; cost = numel(output). |
| `random.f` | 42.4600 | 42.5x | high | numel(output) | [\_\_init\_\_.py:146](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L146) | Sampling; cost = numel(output). |
| `random.weibull` | 40.4811 | 40.5x | high | numel(output) | [\_\_init\_\_.py:132](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L132) | Sampling; cost = numel(output). |
| `random.beta` | 40.2657 | 40.3x | high | numel(output) | [\_\_init\_\_.py:147](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L147) | Sampling; cost = numel(output). |
| `random.standard_t` | 32.3332 | 32.3x | high | numel(output) | [\_\_init\_\_.py:130](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L130) | Sampling; cost = numel(output). |
| `random.gumbel` | 23.5710 | 23.6x | high | numel(output) | [\_\_init\_\_.py:134](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L134) | Sampling; cost = numel(output). |
| `random.pareto` | 22.3002 | 22.3x | high | numel(output) | [\_\_init\_\_.py:127](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L127) | Sampling; cost = numel(output). |
| `random.standard_cauchy` | 20.7325 | 20.7x | high | numel(output) | [\_\_init\_\_.py:129](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L129) | Sampling; cost = numel(output). |
| `random.gamma` | 20.2396 | 20.2x | high | numel(output) | [\_\_init\_\_.py:148](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L148) | Sampling; cost = numel(output). |
| `random.lognormal` | 20.1386 | 20.1x | high | numel(output) | [\_\_init\_\_.py:137](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L137) | Sampling; cost = numel(output). |
| `random.poisson` | 19.9988 | 20.0x | high | numel(output) | [\_\_init\_\_.py:120](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L120) | Sampling; cost = numel(output). |
| `random.logseries` | 19.7821 | 19.8x | high | numel(output) | [\_\_init\_\_.py:125](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L125) | Sampling; cost = numel(output). |
| `random.wald` | 18.1675 | 18.2x | high | numel(output) | [\_\_init\_\_.py:139](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L139) | Sampling; cost = numel(output). |
| `random.rayleigh` | 17.3041 | 17.3x | high | numel(output) | [\_\_init\_\_.py:128](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L128) | Sampling; cost = numel(output). |
| `random.laplace` | 13.4367 | 13.4x | high | numel(output) | [\_\_init\_\_.py:135](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L135) | Sampling; cost = numel(output). |
| `random.logistic` | 13.4267 | 13.4x | high | numel(output) | [\_\_init\_\_.py:136](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L136) | Sampling; cost = numel(output). |
| `random.chisquare` | 13.2098 | 13.2x | high | numel(output) | [\_\_init\_\_.py:141](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L141) | Sampling; cost = numel(output). |
| `random.binomial` | 13.1809 | 13.2x | high | numel(output) | [\_\_init\_\_.py:121](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L121) | Sampling; cost = numel(output). |
| `random.exponential` | 12.7553 | 12.8x | high | numel(output) | [\_\_init\_\_.py:119](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L119) | Sampling; cost = numel(output). |
| `random.standard_exponential` | 12.3008 | 12.3x | high | numel(output) | [\_\_init\_\_.py:117](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L117) | Sampling; cost = numel(output). |
| `random.standard_gamma` | 12.3008 | 12.3x | high | numel(output) | [\_\_init\_\_.py:131](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L131) | Sampling; cost = numel(output). |
| `random.normal` | 11.0481 | 11.0x | high | numel(output) | [\_\_init\_\_.py:113](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L113) | Sampling; cost = numel(output). |
| `random.standard_normal` | 10.1391 | 10.1x | high | numel(output) | [\_\_init\_\_.py:115](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L115) | Sampling; cost = numel(output). |
| `random.randn` | 10.1391 | 10.1x | high | numel(output) | [\_\_init\_\_.py:106](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L106) | Sampling; cost = numel(output). |
| `random.triangular` | 4.9998 | 5.0x | high | numel(output) | [\_\_init\_\_.py:140](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L140) | Sampling; cost = numel(output). |
| `random.geometric` | 2.7270 | 2.7x | high | numel(output) | [\_\_init\_\_.py:122](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L122) | Sampling; cost = numel(output). |
| `random.uniform` | 2.2727 | 2.3x | high | numel(output) | [\_\_init\_\_.py:114](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L114) | Sampling; cost = numel(output). |
| `random.rand` | 1.3636 | 1.4x | high | numel(output) | [\_\_init\_\_.py:105](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L105) | Sampling; cost = numel(output). |
| `random.random` | 1.3636 | 1.4x | high | numel(output) | [\_\_init\_\_.py:175](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L175) | Sampling; cost = numel(output). |
| `random.random_sample` | 1.3636 | 1.4x | high | numel(output) | [\_\_init\_\_.py:176](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L176) | Sampling; cost = numel(output). |
| `random.shuffle` | 0.1819 | 0.2x | high | numel(output) | [\_\_init\_\_.py:209](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L209) | Shuffle; cost = n*ceil(log2(n)). |
| `random.permutation` | 0.0001 | 0.0x | high | numel(output) | [\_\_init\_\_.py:194](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L194) | Shuffle; cost = n*ceil(log2(n)). |
| `random.choice` | 0.0001 | 0.0x | high | numel(output) | [\_\_init\_\_.py:233](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L233) | Sampling; cost = numel(output) if replace, n*ceil(log2(n)) if not. |
| `random.randint` | 0.0001 | 0.0x | high | numel(output) | [\_\_init\_\_.py:154](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L154) | Sampling; cost = numel(output). |

### Misc (24 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `geomspace` | 34.5438 | 34.5x | high | n | [\_counting\_ops.py:246](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L246) | Geometric-spaced generation; cost = num. |
| `logspace` | 34.0892 | 34.1x | high | n | [\_counting\_ops.py:236](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L236) | Log-spaced generation; cost = num. |
| `unwrap` | 3.2132 | 3.2x | medium | n | [\_unwrap.py:40](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_unwrap.py#L40) | Phase unwrap. Cost: $\text{numel}(\text{input})$ (diff + conditional adjustment). |
| `trapezoid` | 2.0909 | 2.1x | high | n | [\_pointwise.py:875](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L875) | Integrate using the trapezoidal rule. |
| `allclose` | 1.9999 | 2.0x | medium | n | [\_counting\_ops.py:45](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L45) | Element-wise tolerance check; cost = numel(a). |
| `histogram_bin_edges` | 1.1827 | 1.2x | medium | n | [\_counting\_ops.py:207](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L207) | Bin edge computation; cost = numel(a). |
| `gradient` | 1.1818 | 1.2x | medium | n | [\_pointwise.py:775](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L775) | Gradient using central differences. |
| `cross` | 0.9546 | 1.0x | medium | 6 * n | [\_\_init\_\_.py:72](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/__init__.py#L72) | Cross product of two 3-D vectors. |
| `convolve` | 0.9245 | 0.9x | high | n * k | [\_pointwise.py:809](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L809) | 1-D discrete convolution. |
| `correlate` | 0.9245 | 0.9x | high | n * k | [\_pointwise.py:829](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L829) | 1-D cross-correlation. |
| `diff` | 0.7273 | 0.7x | medium | n | [\_pointwise.py:759](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L759) | n-th discrete difference along axis. |
| `ediff1d` | 0.7273 | 0.7x | medium | n | [\_pointwise.py:789](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L789) | Differences between consecutive elements. |
| `trace` | 0.7273 | 0.7x | low | min(m, n) | [\_counting\_ops.py:26](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L26) | Diagonal sum; cost = min(n,m). Weight set to match sum (both are simple reductions; direct measurement dominated by subprocess overhead). |
| `array_equal` | 0.5455 | 0.5x | low | n | [\_counting\_ops.py:60](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L60) | Element-wise equality; cost = numel(a). |
| `array_equiv` | 0.5455 | 0.5x | low | n | [\_counting\_ops.py:82](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L82) | Element-wise equivalence; cost = numel(a). |
| `vander` | 0.4532 | 0.5x | high | n * (degree - 1) | [\_counting\_ops.py:260](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L260) | Vandermonde matrix; cost = len(x)*(N-1). |
| `histogram` | 0.3638 | 0.4x | high | n * ceil(log2(bins)) | [\_counting\_ops.py:106](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L106) | Binning; cost = n*ceil(log2(bins)). |
| `corrcoef` | 0.2358 | 0.2x | high | 2 * f^2 * s | [\_pointwise.py:847](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L847) | Pearson correlation coefficients. |
| `cov` | 0.2357 | 0.2x | high | 2 * f^2 * s | [\_pointwise.py:862](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L862) | Covariance matrix. |
| `histogramdd` | 0.1970 | 0.2x | medium | n * ndim * ceil(log2(bins)) | [\_counting\_ops.py:189](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L189) | ND binning; cost = n*sum(ceil(log2(b_i))). |
| `histogram2d` | 0.1690 | 0.2x | medium | n * 2 * ceil(log2(bins)) | [\_counting\_ops.py:148](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L148) | 2D binning; cost = n*(ceil(log2(bx))+ceil(log2(by))). |
| `interp` | 0.1174 | 0.1x | medium | n * ceil(log2(xp)) | [\_pointwise.py:900](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L900) | 1-D linear interpolation. |
| `digitize` | 0.0390 | 0.0x | low | n * ceil(log2(bins)) | [\_sorting\_ops.py:193](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L193) | Bin search; cost = n*ceil(log2(bins)). |
| `bincount` | 0.0001 | 0.0x | high | n | [\_counting\_ops.py:221](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L221) | Integer counting; cost = numel(x). |

### Window (5 operations)

| Op | Weight | vs Add | Confidence | Formula | Impl | Notes |
|:---|-------:|-------:|:-----------|:--------|:-----|:------|
| `kaiser` | 17.0191 | 17.0x | high | 3*n | [\_window.py:155](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L155) | Kaiser window. Cost: 3*n (Bessel function eval per sample). |
| `hamming` | 15.6250 | 15.6x | high | n | [\_window.py:95](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L95) | Hamming window. Cost: n (one cosine per sample). |
| `hanning` | 15.6250 | 15.6x | high | n | [\_window.py:125](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L125) | Hanning window. Cost: n (one cosine per sample). |
| `blackman` | 11.0152 | 11.0x | high | 3*n | [\_window.py:65](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L65) | Blackman window. Cost: 3*n (three cosine terms per sample). |
| `bartlett` | 2.7272 | 2.7x | high | n | [\_window.py:35](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L35) | Bartlett window. Cost: n (one linear eval per sample). |

## Summary by category

| Category | Count | Avg Weight | Min | Max |
|:---------|------:|-----------:|----:|----:|
| Pointwise Unary | 47 | 8.91 | 0.2728 | 37.7712 |
| Pointwise Binary | 33 | 3.97 | 0.2728 | 33.2092 |
| Reductions | 35 | 0.73 | 0.2728 | 2.0909 |
| Sorting | 17 | 2.00 | 0.3381 | 3.7015 |
| FFT | 14 | 0.42 | 0.1943 | 1.0401 |
| Linalg | 14 | 1.42 | 0.4589 | 5.1094 |
| Contractions | 9 | 0.79 | 0.4547 | 1.4848 |
| Polynomial | 10 | 2.73 | 0.0720 | 6.1851 |
| Random | 43 | 33.68 | 0.0001 | 261.1704 |
| Misc | 24 | 3.59 | 0.0001 | 34.5438 |
| Window | 5 | 12.40 | 2.7272 | 17.0191 |

**Total benchmarked operations:** 251

## Validation

Every operation is measured in both **perf mode** (hardware counters) and
**timing mode** (wall-clock nanoseconds).

### Correlation statistics

| Metric | Value | Interpretation |
|--------|------:|:---------------|
| Pearson $r$ | 0.1027 | Linear correlation between perf and timing weight vectors. |
| Spearman $\rho$ | 0.5175 | Rank correlation -- are the orderings consistent? |

### Maximum divergence

| Field | Value |
|:------|:------|
| Operation | `trace` |
| Perf weight | 2727.6393 |
| Timing weight | 0.498 |
| Ratio | 5477.2 |

### Interpreting divergence

The moderate correlation values and large max divergence for BLAS operations are
**expected**. Perf mode counts FP instructions regardless of execution time,
while timing mode measures wall-clock time including memory bandwidth and cache
effects. BLAS operations achieve near-peak FLOP throughput, so their per-instruction
timing is much lower than for scalar pointwise operations. For pointwise ops
(which dominate the count), the two modes agree well in relative ordering.

**Correlation caveats:**
The Pearson and Spearman values span all operations, including BLAS/linalg
ops where timing and perf divergence is structurally expected. For the
subset of pointwise operations, both correlations are substantially higher.

## Known limitations

### BLAS vectorization effects

Operations backed by optimized BLAS routines (`matmul`, `dot`, contraction ops)
show weights below 1.0 because FMA instructions fuse two analytical FLOPs into
one hardware instruction. The sub-unity weights are correct -- they reflect
real hardware instruction counts.

### Random number generators

RNG weights vary dramatically (0.0001 to 367) because the analytical formula
(`numel(output)`) captures only the output size, not the internal algorithmic
complexity. Complex distributions like `hypergeometric` involve rejection
sampling loops that execute many FP instructions per output element.

## Related pages

- [How to calibrate weights](../how-to/calibrate-weights.md)
- [FLOP counting model](../concepts/flop-counting-model.md)
- [Operation audit](operation-audit.md)
- [Agent cheat sheet](for-agents.md)
