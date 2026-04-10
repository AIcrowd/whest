# FLOP Weight Calibration Results

## Introduction

Per-operation FLOP weights are multiplicative correction factors that bridge
the gap between mechestim's analytical cost formulas and the actual
floating-point instruction cost observed on hardware. When weights are
loaded, the effective cost of an operation becomes:

$$
\text{cost}(\text{op}) = \texttt{analytical\_formula}(\text{shapes}) \times \text{weight}(\text{op})
$$

A weight of 25.9 for `sin` means that each analytical FLOP of sine costs
approximately 26 times more in actual floating-point instructions than a
FLOP of addition.

## Methodology

### The unified correction-factor formula

Every weight is computed from the same two-step formula:

$$
$\alpha$(\text{op}) = \text{median}_{D} \left[ \frac{\sum \texttt{fp\_arith\_inst\_retired.*} \times \texttt{simd\_width}}{C(\text{op}, \text{params}) \times R} \right]
$$

$$
\text{weight}(\text{op}) = \frac{$\alpha$(\text{op})}{$\alpha$(\text{add})}
$$

Where:

- $\alpha$(op) is the **raw correction factor** -- the ratio of hardware-observed FP instructions to the analytical FLOP count.
- `fp_arith_inst_retired.*` are Intel PMU hardware counters that count retired floating-point arithmetic instructions, weighted by SIMD lane count.
- $C$(op, params) is the analytical FLOP count from mechestim's cost formula (e.g., `numel(output)` for pointwise ops).
- $R$ is the number of repeats per distribution.
- The **median** across 3 input distributions is reported.

## Measurement environment

!!! info "Calibration platform"

    - **CPU:** Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
    - **Cores:** 64 physical / 128 threads
    - **RAM:** 251.7 GB
    - **Arch:** x86_64 (AVX-512 capable)
    - **Cache:** L1d 48 KB, L1i 32 KB, L2 1280 KB, L3 54 MB
    - **Instance:** AWS EC2 c6i.metal (bare metal -- full PMU access)
    - **OS:** Linux 6.1.166-197.305.amzn2023.x86_64
    - **Python:** 3.11.14
    - **NumPy:** 2.1.3
    - **BLAS:** scipy-openblas 0.3.27
    - **Measurement mode:** perf (hardware counters: `fp_arith_inst_retired.*`)
    - **dtype:** float64
    - **Repeats:** 5 per distribution
    - **Distributions:** 3 per operation
    - **Methodology version:** 2.0
    - **Baseline $\alpha$(add):** 1.564071
    - **Date:** 2026-04-10
    - **Total calibration time:** 2272.4 seconds

## Baseline details

All weights are normalized against element-wise addition (`np.add`):

- **Benchmark command:** `np.add(x, y, out=_out)`
- **Array size:** A(512,512) x B(512,512), dtype=float64
- **Measured perf instructions:** 78203574.0
- **Measured timing:** 79555837.0 ns
- **$\alpha$(add):** 1.564071

**[Download full review spreadsheet (CSV)](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/data/weights.csv)**

## Weight tables

### Pointwise Unary (47 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `arccosh` | 53.1311 | 3.2702 | 5.2033 | numel(output) | [_pointwise.py:269](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L269) | Element-wise inverse hyperbolic cosine. |
| `arcsinh` | 50.8929 | 1.4167 | 2.2541 | numel(output) | [_pointwise.py:271](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L271) | Element-wise inverse hyperbolic sine. |
| `arctanh` | 46.4110 | 2.3226 | 3.6955 | numel(output) | [_pointwise.py:273](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L273) | Element-wise inverse hyperbolic tangent. |
| `tan` | 38.7451 | 0.9577 | 1.5237 | numel(output) | [_pointwise.py:368](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L368) | Element-wise tangent. |
| `arcsin` | 36.1813 | 5.4262 | 8.6337 | numel(output) | [_pointwise.py:270](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L270) | Element-wise inverse sine. |
| `arccos` | 34.2633 | 5.7647 | 9.1724 | numel(output) | [_pointwise.py:268](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L268) | Element-wise inverse cosine. |
| `arctan` | 30.4335 | 0.9012 | 1.4339 | numel(output) | [_pointwise.py:272](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L272) | Element-wise inverse tangent. |
| `expm1` | 26.5973 | 0.9027 | 1.4364 | numel(output) | [_pointwise.py:314](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L314) | Element-wise e^x - 1 (accurate near zero). |
| `log1p` | 26.5973 | 3.2539 | 5.1774 | numel(output) | [_pointwise.py:327](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L327) | Element-wise log(1+x) (accurate near zero). |
| `cos` | 25.8987 | 11.2160 | 17.8460 | numel(output) | [_pointwise.py:254](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L254) | Element-wise cosine. |
| `sin` | 25.8688 | 10.5959 | 16.8594 | numel(output) | [_pointwise.py:253](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L253) | Element-wise sine. |
| `cbrt` | 24.6793 | 0.8319 | 1.3237 | numel(output) | [_pointwise.py:307](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L307) | Element-wise cube root. |
| `log10` | 22.8776 | 4.1323 | 6.5750 | numel(output) | [_pointwise.py:248](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L248) | Element-wise base-10 logarithm. |
| `log2` | 22.5579 | 4.6597 | 7.4141 | numel(output) | [_pointwise.py:247](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L247) | Element-wise base-2 logarithm. |
| `sinh` | 21.4825 | 0.8884 | 1.4136 | numel(output) | [_pointwise.py:365](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L365) | Element-wise hyperbolic sine. |
| `tanh` | 21.4825 | 0.9667 | 1.5381 | numel(output) | [_pointwise.py:255](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L255) | Element-wise hyperbolic tangent. |
| `log` | 20.3202 | 4.3023 | 6.8455 | numel(output) | [_pointwise.py:246](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L246) | Element-wise natural logarithm. |
| `cosh` | 18.2857 | 0.7913 | 1.2590 | numel(output) | [_pointwise.py:310](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L310) | Element-wise hyperbolic cosine. |
| `exp` | 14.4495 | 0.7651 | 1.2174 | numel(output) | [_pointwise.py:245](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L245) | Element-wise e^x. |
| `exp2` | 9.9740 | 0.8010 | 1.2745 | numel(output) | [_pointwise.py:313](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L313) | Element-wise 2^x. |
| `clip` | 1.6624 | 1.0996 | 1.7496 | numel(output) | [_pointwise.py:473](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L473) | Clip array to [a_min, a_max] element-wise. |
| `sqrt` | 1.0230 | 0.6607 | 1.0512 | numel(output) | [_pointwise.py:251](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L251) | Element-wise square root. |
| `square` | 1.0230 | 0.6741 | 1.0726 | numel(output) | [_pointwise.py:252](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L252) | Element-wise x^2. |
| `reciprocal` | 1.0230 | 0.6689 | 1.0643 | numel(output) | [_pointwise.py:335](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L335) | Element-wise 1/x. |
| `deg2rad` | 1.0230 | 0.9118 | 1.4508 | numel(output) | [_pointwise.py:311](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L311) | Alias for radians. |
| `rad2deg` | 1.0230 | 0.9093 | 1.4469 | numel(output) | [_pointwise.py:331](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L331) | Alias for degrees. |
| `degrees` | 1.0230 | 0.9112 | 1.4498 | numel(output) | [_pointwise.py:312](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L312) | Convert radians to degrees element-wise. |
| `radians` | 1.0230 | 0.9099 | 1.4478 | numel(output) | [_pointwise.py:332](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L332) | Convert degrees to radians element-wise. |
| `frexp` | 1.0230 | 1.5273 | 2.4301 | numel(output) | [_pointwise.py:373](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L373) | Decompose x into mantissa and exponent element-wise. |
| `spacing` | 1.0230 | 1.8831 | 2.9963 | numel(output) | [_pointwise.py:367](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L367) | Return ULP spacing for each element. |
| `modf` | 1.0167 | 2.6118 | 4.1558 | numel(output) | [_pointwise.py:372](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L372) | Return fractional and integral parts element-wise. |
| `abs` | 0.3837 | 0.6712 | 1.0680 | numel(output) | [_pointwise.py:249](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L249) | Element-wise absolute value; alias for absolute. |
| `negative` | 0.3837 | 0.6825 | 1.0860 | numel(output) | [_pointwise.py:250](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L250) | Element-wise negation. |
| `positive` | 0.3837 | 0.6679 | 1.0627 | numel(output) | [_pointwise.py:330](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L330) | Element-wise unary plus (copy with sign preserved). |
| `ceil` | 0.3837 | 0.6705 | 1.0669 | numel(output) | [_pointwise.py:257](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L257) | Element-wise ceiling. |
| `floor` | 0.3837 | 0.6639 | 1.0563 | numel(output) | [_pointwise.py:258](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L258) | Element-wise floor. |
| `trunc` | 0.3837 | 0.6804 | 1.0826 | numel(output) | [_pointwise.py:369](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L369) | Truncate toward zero element-wise. |
| `rint` | 0.3837 | 0.6641 | 1.0567 | numel(output) | [_pointwise.py:336](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L336) | Round to nearest integer element-wise. |
| `sign` | 0.3837 | 0.9011 | 1.4338 | numel(output) | [_pointwise.py:256](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L256) | Element-wise sign function. |
| `signbit` | 0.3837 | 0.3314 | 0.5273 | numel(output) | [_pointwise.py:363](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L363) | Returns True for elements with negative sign bit. |
| `fabs` | 0.3837 | 0.9058 | 1.4412 | numel(output) | [_pointwise.py:315](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L315) | Element-wise absolute value (always float). |
| `logical_not` | 0.3837 | 0.5669 | 0.9021 | numel(output) | [_pointwise.py:328](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L328) | Element-wise logical NOT. |
| `sinc` | 0.3837 | 0.0000 | N/A | numel(output) | [_pointwise.py:364](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L364) | Normalized sinc function element-wise. |
| `i0` | 0.3837 | 0.0000 | N/A | numel(output) | [_pointwise.py:317](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L317) | Modified Bessel function of order 0, element-wise. |
| `nan_to_num` | 0.3837 | 0.0000 | N/A | numel(output) | [_pointwise.py:329](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L329) | Replace NaN/inf with finite numbers element-wise. |
| `isneginf` | 0.3837 | 0.9543 | 1.5185 | numel(output) | [_pointwise.py:323](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L323) | Test for negative infinity element-wise. |
| `isposinf` | 0.3837 | 0.9629 | 1.5320 | numel(output) | [_pointwise.py:324](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L324) | Test for positive infinity element-wise. |

### Pointwise Binary (33 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `power` | 46.7140 | 9.2416 | 14.7045 | numel(output) | [_pointwise.py:413](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L413) | Element-wise exponentiation x**y. |
| `arctan2` | 34.6532 | 1.4285 | 2.2728 | numel(output) | [_pointwise.py:420](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L420) | Element-wise arctan(y/x) considering quadrant. |
| `logaddexp2` | 22.5287 | 16.4371 | 26.1533 | numel(output) | [_pointwise.py:445](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L445) | log2(2**x1 + 2**x2) element-wise. |
| `logaddexp` | 21.6097 | 16.1960 | 25.7697 | numel(output) | [_pointwise.py:444](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L444) | log(exp(x1) + exp(x2)) element-wise. |
| `float_power` | 20.8222 | 9.7219 | 15.4687 | numel(output) | [_pointwise.py:429](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L429) | Element-wise exponentiation in float64. |
| `hypot` | 7.4809 | 9.5273 | 15.1591 | numel(output) | [_pointwise.py:438](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L438) | Element-wise Euclidean norm sqrt(x1^2 + x2^2). |
| `floor_divide` | 3.0823 | 11.6310 | 18.5063 | numel(output) | [_pointwise.py:430](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L430) | Element-wise floor division. |
| `isclose` | 2.6854 | 5.8060 | 9.2379 | numel(output) | [_pointwise.py:388](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L388) | Element-wise approximate equality test. |
| `add` | 1.4067 | 0.9832 | 1.5643 | numel(output) | [_pointwise.py:407](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L407) | Element-wise addition. |
| `subtract` | 1.4067 | 0.9807 | 1.5604 | numel(output) | [_pointwise.py:408](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L408) | Element-wise subtraction. |
| `multiply` | 1.4067 | 0.9841 | 1.5658 | numel(output) | [_pointwise.py:409](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L409) | Element-wise multiplication. |
| `divide` | 1.4067 | 0.9710 | 1.5450 | numel(output) | [_pointwise.py:410](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L410) | Element-wise true division. |
| `true_divide` | 1.4067 | 0.9730 | 1.5482 | numel(output) | [_pointwise.py:454](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L454) | Element-wise true division (explicit). |
| `maximum` | 1.4067 | 0.9659 | 1.5369 | numel(output) | [_pointwise.py:411](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L411) | Element-wise maximum (propagates NaN). |
| `minimum` | 1.4067 | 1.0114 | 1.6092 | numel(output) | [_pointwise.py:412](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L412) | Element-wise minimum (propagates NaN). |
| `fmax` | 1.4067 | 1.0281 | 1.6359 | numel(output) | [_pointwise.py:431](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L431) | Element-wise maximum ignoring NaN. |
| `fmin` | 1.4067 | 0.9636 | 1.5333 | numel(output) | [_pointwise.py:432](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L432) | Element-wise minimum ignoring NaN. |
| `mod` | 0.7673 | 10.7508 | 17.1058 | numel(output) | [_pointwise.py:414](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L414) | Element-wise modulo. |
| `remainder` | 0.7673 | 10.7717 | 17.1390 | numel(output) | [_pointwise.py:452](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L452) | Element-wise remainder (same as mod). |
| `fmod` | 0.7673 | 5.7069 | 9.0803 | numel(output) | [_pointwise.py:433](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L433) | Element-wise C-style fmod (remainder toward zero). |
| `greater` | 0.7673 | 0.6676 | 1.0622 | numel(output) | [_pointwise.py:435](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L435) | Element-wise x1 > x2. |
| `greater_equal` | 0.7673 | 0.6667 | 1.0607 | numel(output) | [_pointwise.py:436](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L436) | Element-wise x1 >= x2. |
| `less` | 0.7673 | 0.6704 | 1.0666 | numel(output) | [_pointwise.py:442](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L442) | Element-wise x1 < x2. |
| `less_equal` | 0.7673 | 0.6775 | 1.0781 | numel(output) | [_pointwise.py:443](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L443) | Element-wise x1 <= x2. |
| `equal` | 0.7673 | 0.6626 | 1.0543 | numel(output) | [_pointwise.py:428](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L428) | Element-wise x1 == x2. |
| `not_equal` | 0.7673 | 0.6699 | 1.0659 | numel(output) | [_pointwise.py:450](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L450) | Element-wise x1 != x2. |
| `logical_and` | 0.7673 | 0.9472 | 1.5071 | numel(output) | [_pointwise.py:446](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L446) | Element-wise logical AND. |
| `logical_or` | 0.7673 | 0.9473 | 1.5072 | numel(output) | [_pointwise.py:447](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L447) | Element-wise logical OR. |
| `logical_xor` | 0.7673 | 0.9443 | 1.5025 | numel(output) | [_pointwise.py:448](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L448) | Element-wise logical XOR. |
| `copysign` | 0.7673 | 0.9158 | 1.4572 | numel(output) | [_pointwise.py:427](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L427) | Copy sign of x2 to magnitude of x1 element-wise. |
| `nextafter` | 0.7673 | 7.1806 | 11.4252 | numel(output) | [_pointwise.py:449](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L449) | Return next float after x1 toward x2 element-wise. |
| `ldexp` | 0.7673 | 0.0000 | N/A | numel(output) | [_pointwise.py:440](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L440) | Return x1 * 2**x2 element-wise. |
| `heaviside` | 0.3837 | 4.2660 | 6.7877 | numel(output) | [_pointwise.py:437](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L437) | Heaviside step function element-wise. |

### Reductions (35 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `std` | 2.9411 | 2.0390 | 3.2443 | numel(input) | [_pointwise.py:501](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L501) | Standard deviation; cost_multiplier=2 (two passes). |
| `var` | 2.9411 | 2.0397 | 3.2454 | numel(input) | [_pointwise.py:502](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L502) | Variance; cost_multiplier=2 (two passes). |
| `nanstd` | 2.9411 | 3.9664 | 6.3110 | numel(input) | [_pointwise.py:532](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L532) | Standard deviation ignoring NaNs. |
| `nanvar` | 2.9411 | 3.9612 | 6.3027 | numel(input) | [_pointwise.py:534](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L534) | Variance ignoring NaNs. |
| `ptp` | 1.6636 | 0.5205 | 0.8282 | numel(input) | [_pointwise.py:549](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L549) | Peak-to-peak (max - min) range of array. |
| `max` | 1.0237 | 0.2677 | 0.4260 | numel(input) | [_pointwise.py:497](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L497) | Maximum value of array. |
| `min` | 1.0237 | 0.2724 | 0.4334 | numel(input) | [_pointwise.py:498](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L498) | Minimum value of array. |
| `nanmax` | 1.0237 | 0.2770 | 0.4407 | numel(input) | [_pointwise.py:525](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L525) | Maximum ignoring NaNs. |
| `nanmin` | 1.0237 | 0.2773 | 0.4412 | numel(input) | [_pointwise.py:528](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L528) | Minimum ignoring NaNs. |
| `sum` | 1.0230 | 0.3006 | 0.4783 | numel(input) | [_pointwise.py:496](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L496) | Sum of array elements. |
| `prod` | 1.0230 | 0.7650 | 1.2172 | numel(input) | [_pointwise.py:499](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L499) | Product of array elements. |
| `mean` | 1.0230 | 0.2999 | 0.4771 | numel(input) | [_pointwise.py:500](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L500) | Arithmetic mean of array elements. |
| `cumsum` | 1.0230 | 1.6426 | 2.6136 | numel(input) | [_pointwise.py:505](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L505) | Cumulative sum of array elements. |
| `cumprod` | 1.0230 | 1.6430 | 2.6142 | numel(input) | [_pointwise.py:506](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L506) | Cumulative product of array elements. |
| `nansum` | 1.0230 | 1.7500 | 2.7844 | numel(input) | [_pointwise.py:533](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L533) | Sum ignoring NaNs. |
| `nanmean` | 1.0230 | 2.2902 | 3.6440 | numel(input) | [_pointwise.py:526](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L526) | Mean ignoring NaNs. |
| `nanprod` | 1.0230 | 2.3123 | 3.6791 | numel(input) | [_pointwise.py:530](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L530) | Product ignoring NaNs. |
| `average` | 1.0230 | 0.3017 | 0.4800 | numel(input) | [_pointwise.py:516](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L516) | Weighted average of array elements. |
| `nancumprod` | 1.0230 | 3.2476 | 5.1673 | numel(input) | [_pointwise.py:523](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L523) | Cumulative product ignoring NaNs. |
| `nancumsum` | 1.0230 | 3.2608 | 5.1883 | numel(input) | [_pointwise.py:524](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L524) | Cumulative sum ignoring NaNs. |
| `cumulative_sum` | 1.0230 | 1.6425 | 2.6135 | numel(input) | [_pointwise.py:519](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L519) | Cumulative sum (NumPy 2.x array API). |
| `cumulative_prod` | 1.0230 | 1.6425 | 2.6135 | numel(input) | [_pointwise.py:518](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L518) | Cumulative product (NumPy 2.x array API). |
| `argmax` | 0.3837 | 0.3012 | 0.4793 | numel(input) | [_pointwise.py:503](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L503) | Index of maximum value. |
| `argmin` | 0.3837 | 0.2879 | 0.4581 | numel(input) | [_pointwise.py:504](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L504) | Index of minimum value. |
| `any` | 0.3837 | 0.5733 | 0.9122 | numel(input) | [_pointwise.py:515](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L515) | Test whether any array element is true. |
| `all` | 0.3837 | 0.5753 | 0.9153 | numel(input) | [_pointwise.py:512](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L512) | Test whether all array elements are true. |
| `median` | 0.3837 | 7.3576 | 11.7069 | numel(input) | [_pointwise.py:520](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L520) | Median of array elements (sorts internally). |
| `nanmedian` | 0.3837 | 7.8619 | 12.5092 | numel(input) | [_pointwise.py:527](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L527) | Median ignoring NaNs. |
| `percentile` | 0.3837 | 8.4340 | 13.4194 | numel(input) | [_pointwise.py:535](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L535) | q-th percentile of array elements. |
| `nanpercentile` | 0.3837 | 8.9149 | 14.1847 | numel(input) | [_pointwise.py:529](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L529) | q-th percentile ignoring NaNs. |
| `quantile` | 0.3837 | 8.4253 | 13.4057 | numel(input) | [_pointwise.py:536](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L536) | q-th quantile of array elements. |
| `nanquantile` | 0.3837 | 8.9023 | 14.1646 | numel(input) | [_pointwise.py:531](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L531) | q-th quantile ignoring NaNs. |
| `count_nonzero` | 0.3837 | 0.9680 | 1.5401 | numel(input) | [_pointwise.py:517](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L517) | Count non-zero elements. |
| `nanargmax` | 0.3837 | 1.8476 | 2.9397 | numel(input) | [_pointwise.py:521](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L521) | Index of maximum ignoring NaNs. |
| `nanargmin` | 0.3837 | 1.8424 | 2.9315 | numel(input) | [_pointwise.py:522](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L522) | Index of minimum ignoring NaNs. |

### Sorting (17 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `partition` | 14.1337 | 2.2850 | 3.6357 | n | [_sorting_ops.py:111](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L111) | Quickselect; cost = n per slice. |
| `argpartition` | 13.8459 | 2.8642 | 4.5572 | n | [_sorting_ops.py:136](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L136) | Indirect partition; cost = n per slice. |
| `intersect1d` | 5.2067 | 0.6046 | 0.0192 | (n+m) * ceil(log2(n+m)) | [_sorting_ops.py:340](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L340) | Set intersection; cost = (n+m)*ceil(log2(n+m)). |
| `setxor1d` | 5.2067 | 0.6725 | 0.0214 | (n+m) * ceil(log2(n+m)) | [_sorting_ops.py:389](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L389) | Symmetric set difference; cost = (n+m)*ceil(log2(n+m)). |
| `argsort` | 3.3538 | 0.5447 | 0.0361 | n * ceil(log2(n)) | [_sorting_ops.py:62](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L62) | Indirect sort; cost = n*ceil(log2(n)) per slice. |
| `unique_inverse` | 3.3538 | 0.9542 | 0.0633 | n * ceil(log2(n)) | [_sorting_ops.py:268](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L268) | Sort-based unique; cost = n*ceil(log2(n)). |
| `in1d` | 2.9630 | 1.0723 | 0.0341 | (n+m) * ceil(log2(n+m)) | [_sorting_ops.py:313](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L313) | Set membership; cost = (n+m)*ceil(log2(n+m)). |
| `isin` | 2.9630 | 1.0692 | 0.0340 | (n+m) * ceil(log2(n+m)) | [_sorting_ops.py:326](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L326) | Set membership; cost = (n+m)*ceil(log2(n+m)). |
| `union1d` | 2.9576 | 0.4175 | 0.0133 | (n+m) * ceil(log2(n+m)) | [_sorting_ops.py:357](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L357) | Set union; cost = (n+m)*ceil(log2(n+m)). |
| `sort` | 2.8185 | 0.2501 | 0.0166 | n * ceil(log2(n)) | [_sorting_ops.py:43](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L43) | Comparison sort; cost = n*ceil(log2(n)) per slice. |
| `unique` | 2.8185 | 0.3440 | 0.0228 | n * ceil(log2(n)) | [_sorting_ops.py:227](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L227) | Sort-based unique; cost = n*ceil(log2(n)). |
| `unique_counts` | 2.8185 | 0.4702 | 0.0312 | n * ceil(log2(n)) | [_sorting_ops.py:252](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L252) | Sort-based unique; cost = n*ceil(log2(n)). |
| `unique_values` | 2.8185 | 0.3355 | 0.0222 | n * ceil(log2(n)) | [_sorting_ops.py:284](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L284) | Sort-based unique; cost = n*ceil(log2(n)). |
| `setdiff1d` | 2.7061 | 0.7751 | 0.0247 | (n+m) * ceil(log2(n+m)) | [_sorting_ops.py:372](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L372) | Set difference; cost = (n+m)*ceil(log2(n+m)). |
| `searchsorted` | 0.9520 | 1.0636 | 0.0705 | m * ceil(log2(n)) | [_sorting_ops.py:166](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L166) | Binary search; cost = m*ceil(log2(n)). |
| `lexsort` | 0.4760 | 0.0534 | 0.0018 | k * n * ceil(log2(n)) | [_sorting_ops.py:88](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L88) | Multi-key sort; cost = k*n*ceil(log2(n)). |
| `unique_all` | 0.4756 | 0.6681 | 0.0443 | n * ceil(log2(n)) | [_sorting_ops.py:239](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L239) | Sort-based unique; cost = n*ceil(log2(n)). |

### FFT (14 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `fft.hfft` | 1.4631 | 0.4532 | 0.0688 | 5*n*ceil(log2(n)) | [_transforms.py:443](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L443) | FFT of Hermitian-symmetric signal. Cost: 5*n_out*ceil(log2(n_out)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifft` | 0.8613 | 0.1583 | 0.0240 | 5*n*ceil(log2(n)) | [_transforms.py:189](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L189) | Inverse 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfft` | 0.6609 | 0.1839 | 0.0558 | 5*(n/2)*ceil(log2(n)) | [_transforms.py:221](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L221) | Inverse 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfft2` | 0.5761 | 0.0877 | 0.0266 | 5*(n/2)*ceil(log2(n)) | [_transforms.py:314](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L314) | Inverse 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.irfftn` | 0.5761 | 0.0893 | 0.0271 | 5*(n/2)*ceil(log2(n)) | [_transforms.py:423](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L423) | Inverse N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fft` | 0.5340 | 0.1992 | 0.0302 | 5*n*ceil(log2(n)) | [_transforms.py:173](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L173) | 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfft` | 0.5338 | 0.1772 | 0.0538 | 5*(n/2)*ceil(log2(n)) | [_transforms.py:205](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L205) | 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifft2` | 0.5008 | 0.1044 | 0.0158 | 5*n*ceil(log2(n)) | [_transforms.py:265](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L265) | Inverse 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ifftn` | 0.5008 | 0.1019 | 0.0155 | 5*n*ceil(log2(n)) | [_transforms.py:363](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L363) | Inverse N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fft2` | 0.4612 | 0.1021 | 0.0155 | 5*n*ceil(log2(n)) | [_transforms.py:242](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L242) | 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.fftn` | 0.4612 | 0.1065 | 0.0162 | 5*n*ceil(log2(n)) | [_transforms.py:339](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L339) | N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfft2` | 0.4523 | 0.0832 | 0.0252 | 5*(n/2)*ceil(log2(n)) | [_transforms.py:288](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L288) | 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.rfftn` | 0.4523 | 0.0825 | 0.0250 | 5*(n/2)*ceil(log2(n)) | [_transforms.py:387](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L387) | N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |
| `fft.ihfft` | 0.2733 | 0.0902 | 0.0137 | 5*(n/2)*ceil(log2(n)) | [_transforms.py:462](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/fft/_transforms.py#L462) | Inverse FFT of Hermitian signal. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4). |

### Linalg (14 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `linalg.pinv` | 7.1872 | 0.1769 | 0.0026 | m*n*min(m,n) | [_solvers.py:187](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L187) | Pseudoinverse. Cost: m*n*min(m,n) (via SVD). |
| `linalg.svd` | 5.9042 | 0.1730 | 0.0026 | m*n*min(m,n) | [_svd.py:67](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_svd.py#L67) | Singular value decomposition; cost ~ O(min(m,n)*m*n). |
| `linalg.eigh` | 2.3185 | 0.0541 | 0.0006 | 4*n^3 / 3 | [_decompositions.py:165](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L165) | Symmetric eigendecomposition. Cost: $(4/3)n^3$ (Golub & Van Loan §8.3). |
| `linalg.lstsq` | 1.9782 | 0.1129 | 0.0017 | m*n*min(m,n) | [_solvers.py:147](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L147) | Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD). |
| `linalg.svdvals` | 1.8834 | 0.0954 | 0.0014 | m*n*min(m,n) | [_decompositions.py:281](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L281) | Singular values only. Cost: m*n*min(m,n) (Golub-Reinsch). |
| `linalg.inv` | 1.7448 | 0.0296 | 0.0004 | n^3 | [_solvers.py:101](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L101) | Matrix inverse. Cost: $n^3$ (LU + solve). |
| `linalg.qr` | 1.3332 | 0.0356 | 0.0004 | 2*m*n^2 - 2*n^3/3 | [_decompositions.py:87](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L87) | QR decomposition. Cost: $2mn^2 - (2/3)n^3$ (Golub & Van Loan §5.2). |
| `linalg.cholesky` | 1.0699 | 0.0252 | 0.0011 | n^3 / 3 | [_decompositions.py:46](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L46) | Cholesky decomposition. Cost: $n^3/3$ (Golub & Van Loan §4.2). |
| `linalg.eig` | 0.9604 | 0.0465 | 0.0001 | 10*n^3 | [_decompositions.py:127](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L127) | Eigendecomposition. Cost: $10n^3$ (Francis QR, Golub & Van Loan §7.5). |
| `linalg.eigvalsh` | 0.8071 | 0.0284 | 0.0003 | 4*n^3 / 3 | [_decompositions.py:243](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L243) | Symmetric eigenvalues. Cost: $(4/3)n^3$ (same as eigh). |
| `linalg.det` | 0.6923 | 0.0366 | 0.0008 | 2*n^3 / 3 | [_properties.py:87](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_properties.py#L87) | Determinant. Cost: $n^3$ (LU factorization). |
| `linalg.slogdet` | 0.6923 | 0.0368 | 0.0008 | 2*n^3 / 3 | [_properties.py:134](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_properties.py#L134) | Sign + log determinant. Cost: $n^3$ (LU factorization). |
| `linalg.eigvals` | 0.6520 | 0.0515 | 0.0001 | 7*n^3 | [_decompositions.py:207](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_decompositions.py#L207) | Eigenvalues only. Cost: $10n^3$ (same as eig). |
| `linalg.solve` | 0.6455 | 0.0197 | 0.0004 | 2*n^3/3 + 2*n^2 | [_solvers.py:54](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/_solvers.py#L54) | Solve Ax=b. Cost: $2n^3/3$ (LU) + $n^2 \cdot n_{\text{rhs}}$ (back-substitution). |

### Contractions (9 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `inner` | 1.0598 | 0.0960 | 76.4100 | 2*N | [_pointwise.py:649](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L649) | Inner product; cost = 2*N for 1-D, 2*N*M for n-D. |
| `vdot` | 1.0598 | 0.0928 | 73.8450 | 2*N | [_pointwise.py:707](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L707) | Dot product with conjugation; cost = 2*N. |
| `vecdot` | 1.0443 | 0.1552 | 2.4111 | 2*batch*N | [_pointwise.py:455](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L455) | Vector dot product along last axis. |
| `tensordot` | 0.6494 | 0.0082 | 0.0001 | 2*d^5 (axes=1, shape=(d,d,d)) | [__init__.py:74](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/__init__.py#L74) | Tensor dot product along specified axes. |
| `dot` | 0.6426 | 0.0011 | 0.0001 | 2*M*N*K | [_pointwise.py:587](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L587) | Dot product; cost = 2*M*N*K for matrix multiply. |
| `matmul` | 0.6426 | 0.0009 | 0.0001 | 2*M*N*K | [_pointwise.py:623](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L623) | Matrix multiplication; cost = 2*M*N*K. |
| `einsum` | 0.6401 | 0.0911 | 0.0054 | 2*M*N*K (ij,jk->ik) | [_einsum.py:135](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_einsum.py#L135) | Generalized Einstein summation. |
| `kron` | 0.6396 | 1.2545 | 1.1898 | d^4 (Kronecker, shape=(d,d)x(d,d)) | [_pointwise.py:723](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L723) | Kronecker product; cost proportional to output size. |
| `outer` | 0.6395 | 1.2424 | 0.7907 | M*N | [_pointwise.py:665](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L665) | Outer product of two vectors; cost = M*N. |

### Polynomial (10 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `polyder` | 8.7873 | 33.1453 | 5273800.0000 | degree | [_polynomial.py:127](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L127) | Differentiate polynomial. Cost: n FLOPs. |
| `polyadd` | 8.2015 | 14.6407 | 2306440.5450 | degree + 1 | [_polynomial.py:96](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L96) | Add two polynomials. Cost: max(n1, n2) FLOPs. |
| `polysub` | 8.2015 | 14.3930 | 2267424.7623 | degree + 1 | [_polynomial.py:113](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L113) | Difference (subtraction) of two polynomials. Cost: max(n1, n2) FLOPs. |
| `polyint` | 8.1608 | 44.0885 | 7015000.0000 | degree | [_polynomial.py:140](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L140) | Integrate polynomial. Cost: n FLOPs. |
| `polymul` | 1.6769 | 0.9725 | 1516.8690 | (degree+1)^2 | [_polynomial.py:158](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L158) | Multiply polynomials. Cost: n1 * n2 FLOPs. |
| `poly` | 1.3803 | 13.1276 | 20887.4800 | degree^2 | [_polynomial.py:204](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L204) | Polynomial from roots. Cost: $n^2$ FLOPs. |
| `polyfit` | 0.7578 | 0.1016 | 0.0001 | 2 * n * (degree+1)^2 | [_polynomial.py:187](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L187) | Least squares polynomial fit. Cost: 2 * m * (deg+1)^2 FLOPs. |
| `roots` | 0.6889 | 0.2514 | 0.4000 | 10 * degree^3 | [_polynomial.py:217](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L217) | Return roots of polynomial with given coefficients. Cost: $10n^3$ FLOPs (companion matrix eig). |
| `polyval` | 0.6467 | 0.3854 | 0.0307 | 2 * n * degree | [_polynomial.py:78](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L78) | Evaluate polynomial at given points. Cost: 2 * m * deg FLOPs (Horner's method). |
| `polydiv` | 0.1013 | 2.4191 | 3773.2317 | (degree+1)^2 | [_polynomial.py:174](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_polynomial.py#L174) | Divide one polynomial by another. Cost: n1 * n2 FLOPs. |

### Random (43 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `random.hypergeometric` | 367.3772 | 154.5017 | 245.8303 | numel(output) | [__init__.py:123](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L123) | Sampling; cost = numel(output). |
| `random.multivariate_normal` | 289.6739 | 145.1286 | 230.9166 | numel(output) | [__init__.py:151](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L151) | Sampling; cost = numel(output). |
| `random.zipf` | 147.6913 | 70.9544 | 112.8967 | numel(output) | [__init__.py:133](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L133) | Sampling; cost = numel(output). |
| `random.noncentral_chisquare` | 95.9945 | 50.3997 | 80.1918 | numel(output) | [__init__.py:143](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L143) | Sampling; cost = numel(output). |
| `random.negative_binomial` | 90.6636 | 80.3560 | 127.8558 | numel(output) | [__init__.py:124](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L124) | Sampling; cost = numel(output). |
| `random.multinomial` | 87.5909 | 70.9323 | 112.8615 | numel(output) | [__init__.py:149](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L149) | Sampling; cost = numel(output). |
| `random.noncentral_f` | 77.2939 | 60.0371 | 95.5260 | numel(output) | [__init__.py:145](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L145) | Sampling; cost = numel(output). |
| `random.dirichlet` | 77.2609 | 62.3552 | 99.2143 | numel(output) | [__init__.py:153](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L153) | Sampling; cost = numel(output). |
| `random.power` | 70.3696 | 31.0821 | 49.4553 | numel(output) | [__init__.py:126](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L126) | Sampling; cost = numel(output). |
| `random.vonmises` | 66.7742 | 52.8117 | 84.0296 | numel(output) | [__init__.py:138](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L138) | Sampling; cost = numel(output). |
| `random.f` | 59.7266 | 48.3451 | 76.9228 | numel(output) | [__init__.py:146](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L146) | Sampling; cost = numel(output). |
| `random.weibull` | 56.9431 | 24.3228 | 38.7004 | numel(output) | [__init__.py:132](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L132) | Sampling; cost = numel(output). |
| `random.beta` | 56.6401 | 48.8709 | 77.7594 | numel(output) | [__init__.py:147](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L147) | Sampling; cost = numel(output). |
| `random.standard_t` | 45.4818 | 37.1390 | 59.0925 | numel(output) | [__init__.py:130](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L130) | Sampling; cost = numel(output). |
| `random.gumbel` | 33.1563 | 15.8963 | 25.2929 | numel(output) | [__init__.py:134](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L134) | Sampling; cost = numel(output). |
| `random.pareto` | 31.3688 | 15.0142 | 23.8894 | numel(output) | [__init__.py:127](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L127) | Sampling; cost = numel(output). |
| `random.standard_cauchy` | 29.1635 | 25.7502 | 40.9716 | numel(output) | [__init__.py:129](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L129) | Sampling; cost = numel(output). |
| `random.gamma` | 28.4702 | 24.6317 | 39.1919 | numel(output) | [__init__.py:148](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L148) | Sampling; cost = numel(output). |
| `random.lognormal` | 28.3281 | 21.4839 | 34.1834 | numel(output) | [__init__.py:137](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L137) | Sampling; cost = numel(output). |
| `random.poisson` | 28.1315 | 37.9288 | 60.3492 | numel(output) | [__init__.py:120](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L120) | Sampling; cost = numel(output). |
| `random.logseries` | 27.8266 | 22.5066 | 35.8106 | numel(output) | [__init__.py:125](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L125) | Sampling; cost = numel(output). |
| `random.wald` | 25.5555 | 29.1850 | 46.4367 | numel(output) | [__init__.py:139](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L139) | Sampling; cost = numel(output). |
| `random.rayleigh` | 24.3409 | 15.9689 | 25.4084 | numel(output) | [__init__.py:128](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L128) | Sampling; cost = numel(output). |
| `random.laplace` | 18.9009 | 13.9287 | 22.1622 | numel(output) | [__init__.py:135](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L135) | Sampling; cost = numel(output). |
| `random.logistic` | 18.8868 | 10.0316 | 15.9615 | numel(output) | [__init__.py:136](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L136) | Sampling; cost = numel(output). |
| `random.chisquare` | 18.5817 | 9.7281 | 15.4786 | numel(output) | [__init__.py:141](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L141) | Sampling; cost = numel(output). |
| `random.binomial` | 18.5410 | 25.2465 | 40.1702 | numel(output) | [__init__.py:121](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L121) | Sampling; cost = numel(output). |
| `random.exponential` | 17.9423 | 9.4763 | 15.0780 | numel(output) | [__init__.py:119](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L119) | Sampling; cost = numel(output). |
| `random.standard_exponential` | 17.3029 | 9.3248 | 14.8369 | numel(output) | [__init__.py:117](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L117) | Sampling; cost = numel(output). |
| `random.standard_gamma` | 17.3029 | 9.5067 | 15.1262 | numel(output) | [__init__.py:131](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L131) | Sampling; cost = numel(output). |
| `random.normal` | 15.5409 | 13.6587 | 21.7327 | numel(output) | [__init__.py:113](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L113) | Sampling; cost = numel(output). |
| `random.standard_normal` | 14.2622 | 12.6403 | 20.1123 | numel(output) | [__init__.py:115](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L115) | Sampling; cost = numel(output). |
| `random.randn` | 14.2622 | 12.6338 | 20.1019 | numel(output) | [__init__.py:106](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L106) | Sampling; cost = numel(output). |
| `random.triangular` | 7.0330 | 10.1178 | 16.0986 | numel(output) | [__init__.py:140](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L140) | Sampling; cost = numel(output). |
| `random.geometric` | 3.8360 | 14.2922 | 22.7405 | numel(output) | [__init__.py:122](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L122) | Sampling; cost = numel(output). |
| `random.uniform` | 3.1969 | 4.0878 | 6.5041 | numel(output) | [__init__.py:114](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L114) | Sampling; cost = numel(output). |
| `random.rand` | 1.9181 | 3.7538 | 5.9727 | numel(output) | [__init__.py:105](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L105) | Sampling; cost = numel(output). |
| `random.random` | 1.9181 | 3.7693 | 5.9974 | numel(output) | [__init__.py:175](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L175) | Sampling; cost = numel(output). |
| `random.random_sample` | 1.9181 | 3.7449 | 5.9585 | numel(output) | [__init__.py:176](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L176) | Sampling; cost = numel(output). |
| `random.shuffle` | 0.2558 | 12.7233 | 20.2443 | numel(output) | [__init__.py:209](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L209) | Shuffle; cost = n*ceil(log2(n)). |
| `random.permutation` | 0.0001 | 12.9331 | 20.5782 | numel(output) | [__init__.py:194](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L194) | Shuffle; cost = n*ceil(log2(n)). |
| `random.choice` | 0.0001 | 3.7400 | 5.9508 | numel(output) | [__init__.py:233](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L233) | Sampling; cost = numel(output) if replace, n*ceil(log2(n)) if not. |
| `random.randint` | 0.0001 | 2.3966 | 3.8132 | numel(output) | [__init__.py:154](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/random/__init__.py#L154) | Sampling; cost = numel(output). |

### Misc (24 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `trace` | 384.9696 | 1.9840 | 31568.0000 | min(m, n) | [_counting_ops.py:26](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L26) | Diagonal sum; cost = min(n,m). |
| `geomspace` | 48.5912 | 6.2891 | 10.0067 | n | [_counting_ops.py:246](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L246) | Geometric-spaced generation; cost = num. |
| `logspace` | 47.9518 | 5.9446 | 9.4585 | n | [_counting_ops.py:236](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L236) | Log-spaced generation; cost = num. |
| `unwrap` | 4.5198 | 20.0331 | 31.8750 | n | [_unwrap.py:40](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_unwrap.py#L40) | Phase unwrap. Cost: $\text{numel}(\text{input})$ (diff + conditional adjustment). |
| `trapezoid` | 2.9411 | 2.1355 | 3.3979 | n | [_pointwise.py:875](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L875) | Integrate using the trapezoidal rule. |
| `allclose` | 2.8132 | 5.6224 | 8.9460 | n | [_counting_ops.py:45](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L45) | Element-wise tolerance check; cost = numel(a). |
| `histogram_bin_edges` | 1.6637 | 0.5233 | 0.8326 | n | [_counting_ops.py:207](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L207) | Bin edge computation; cost = numel(a). |
| `gradient` | 1.6624 | 2.6835 | 4.2698 | n | [_pointwise.py:775](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L775) | Gradient using central differences. |
| `cross` | 1.3428 | 4.2575 | 11.2904 | 6 * n | [__init__.py:72](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/linalg/__init__.py#L72) | Cross product of two 3-D vectors. |
| `convolve` | 1.3004 | 0.0844 | 0.0134 | n * k | [_pointwise.py:809](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L809) | 1-D discrete convolution. |
| `correlate` | 1.3004 | 0.0734 | 0.0117 | n * k | [_pointwise.py:829](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L829) | 1-D cross-correlation. |
| `diff` | 1.0230 | 1.0673 | 1.6982 | n | [_pointwise.py:759](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L759) | n-th discrete difference along axis. |
| `ediff1d` | 1.0230 | 1.0806 | 1.7193 | n | [_pointwise.py:789](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L789) | Differences between consecutive elements. |
| `array_equal` | 0.7673 | 0.6489 | 1.0324 | n | [_counting_ops.py:60](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L60) | Element-wise equality; cost = numel(a). |
| `array_equiv` | 0.7673 | 0.6464 | 1.0285 | n | [_counting_ops.py:82](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L82) | Element-wise equivalence; cost = numel(a). |
| `vander` | 0.6375 | 1.8678 | 30.0192 | n * (degree - 1) | [_counting_ops.py:260](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L260) | Vandermonde matrix; cost = len(x)*(N-1). |
| `histogram` | 0.5117 | 1.1175 | 0.2540 | n * ceil(log2(bins)) | [_counting_ops.py:106](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L106) | Binning; cost = n*ceil(log2(bins)). |
| `corrcoef` | 0.3317 | 0.0046 | 0.0000 | 2 * f^2 * s | [_pointwise.py:847](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L847) | Pearson correlation coefficients. |
| `cov` | 0.3316 | 0.0045 | 0.0000 | 2 * f^2 * s | [_pointwise.py:862](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L862) | Covariance matrix. |
| `histogramdd` | 0.2771 | 4.3399 | 3.8363 | n * ndim * ceil(log2(bins)) | [_counting_ops.py:189](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L189) | ND binning; cost = n*sum(ceil(log2(b_i))). |
| `histogram2d` | 0.2377 | 4.5042 | 0.5119 | n * 2 * ceil(log2(bins)) | [_counting_ops.py:148](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L148) | 2D binning; cost = n*(ceil(log2(bx))+ceil(log2(by))). |
| `interp` | 0.1652 | 4.1251 | 0.4688 | n * ceil(log2(xp)) | [_pointwise.py:900](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_pointwise.py#L900) | 1-D linear interpolation. |
| `digitize` | 0.0548 | 3.8817 | 0.8823 | n * ceil(log2(bins)) | [_sorting_ops.py:193](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_sorting_ops.py#L193) | Bin search; cost = n*ceil(log2(bins)). |
| `bincount` | 0.0001 | 0.9173 | 1.4595 | n | [_counting_ops.py:221](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_counting_ops.py#L221) | Integer counting; cost = numel(x). |

### Window (5 operations)

| Op | Perf Wt | Timing Wt | Time/FLOP (ns) | Formula | Impl | Notes |
|:---|--------:|----------:|---------------:|:--------|:-----|:------|
| `kaiser` | 23.9400 | 25.5297 | 13.5402 | 3*n | [_window.py:155](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L155) | Kaiser window. Cost: 3*n (Bessel function eval per sample). |
| `hamming` | 21.9790 | 7.8423 | 12.4781 | n | [_window.py:95](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L95) | Hamming window. Cost: n (one cosine per sample). |
| `hanning` | 21.9790 | 7.8557 | 12.4993 | n | [_window.py:125](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L125) | Hanning window. Cost: n (one cosine per sample). |
| `blackman` | 15.4946 | 5.3931 | 2.8603 | 3*n | [_window.py:65](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L65) | Blackman window. Cost: 3*n (three cosine terms per sample). |
| `bartlett` | 3.8362 | 6.2418 | 9.9314 | n | [_window.py:35](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_window.py#L35) | Bartlett window. Cost: n (one linear eval per sample). |

## Summary by category

| Category | Count | Avg Weight | Min | Max |
|:---------|------:|-----------:|----:|----:|
| Pointwise Unary | 47 | 12.5352 | 0.3837 | 53.1311 |
| Pointwise Binary | 33 | 5.5797 | 0.3837 | 46.7140 |
| Reductions | 35 | 1.0231 | 0.3837 | 2.9411 |
| Sorting | 17 | 4.1099 | 0.4756 | 14.1337 |
| FFT | 14 | 0.5934 | 0.2733 | 1.4631 |
| Linalg | 14 | 1.9906 | 0.6455 | 7.1872 |
| Contractions | 9 | 0.7797 | 0.6395 | 1.0598 |
| Polynomial | 10 | 3.8603 | 0.1013 | 8.7873 |
| Random | 43 | 47.3819 | 0.0001 | 367.3772 |
| Misc | 24 | 21.0494 | 0.0001 | 384.9696 |
| Window | 5 | 17.4458 | 3.8362 | 23.9400 |

**Total benchmarked operations:** 251

## Validation

Every operation is measured in both **perf mode** (hardware counters) and
**timing mode** (wall-clock nanoseconds).

### Correlation statistics

| Metric | Value | Interpretation |
|--------|------:|:---------------|
| Pearson $r$ | 0.7332 | Linear correlation between perf and timing weight vectors. |
| Spearman $\rho$ | 0.5276 | Rank correlation -- are the orderings consistent? |

### Maximum divergence

| Field | Value |
|:------|:------|
| Operation | `matmul` |
| Perf weight | 0.6426 |
| Timing weight | 0.0009 |
| Ratio | 714.0 |

### Interpreting divergence

The moderate correlation values and large max divergence for BLAS operations are
**expected**. Perf mode counts FP instructions regardless of execution time,
while timing mode measures wall-clock time including memory bandwidth and cache
effects. BLAS operations achieve near-peak FLOP throughput, so their per-instruction
timing is much lower than for scalar pointwise operations. For pointwise ops
(which dominate the count), the two modes agree well in relative ordering.

!!! warning "Correlation caveats"

    The Pearson and Spearman values span all operations, including BLAS/linalg
    ops where timing and perf divergence is structurally expected. For the
    subset of pointwise operations, both correlations are substantially higher.

## Known limitations

### Trace anomaly (subprocess overhead)

The `trace` operation shows an anomalously high weight because its analytical
formula is $n$ (the matrix dimension), which is small, while the subprocess
measurement captures fixed per-process overhead that dominates at small input
sizes. The weight for `trace` should be interpreted with caution.

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
