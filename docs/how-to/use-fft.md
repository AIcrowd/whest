# Use FFT

## When to use this page

Use this page to learn how to use `me.fft` operations and their FLOP costs.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Available operations

### Transforms (counted)

| Operation | Cost | Notes |
|-----------|------|-------|
| `me.fft.fft(a)` | $5n \lceil\log_2 n\rceil$ | 1-D complex FFT |
| `me.fft.ifft(a)` | $5n \lceil\log_2 n\rceil$ | 1-D inverse complex FFT |
| `me.fft.rfft(a)` | $5(n/2) \lceil\log_2 n\rceil$ | 1-D real FFT — ~2x cheaper |
| `me.fft.irfft(a)` | $5(n/2) \lceil\log_2 n\rceil$ | 1-D inverse real FFT |
| `me.fft.fft2(a)` | $5N \lceil\log_2 N\rceil$ | 2-D complex FFT ($N = \prod n_i$) |
| `me.fft.ifft2(a)` | $5N \lceil\log_2 N\rceil$ | 2-D inverse complex FFT |
| `me.fft.rfft2(a)` | $5(N/2) \lceil\log_2 N\rceil$ | 2-D real FFT |
| `me.fft.irfft2(a)` | $5(N/2) \lceil\log_2 N\rceil$ | 2-D inverse real FFT |
| `me.fft.fftn(a)` | $5N \lceil\log_2 N\rceil$ | N-D complex FFT |
| `me.fft.ifftn(a)` | $5N \lceil\log_2 N\rceil$ | N-D inverse complex FFT |
| `me.fft.rfftn(a)` | $5(N/2) \lceil\log_2 N\rceil$ | N-D real FFT |
| `me.fft.irfftn(a)` | $5(N/2) \lceil\log_2 N\rceil$ | N-D inverse real FFT |
| `me.fft.hfft(a)` | $5n_{\text{out}} \lceil\log_2 n_{\text{out}}\rceil$ | Hermitian FFT |
| `me.fft.ihfft(a)` | $5n_{\text{out}} \lceil\log_2 n_{\text{out}}\rceil$ | Inverse Hermitian FFT |

### Utilities (free — 0 FLOPs)

| Operation | Notes |
|-----------|-------|
| `me.fft.fftfreq(n)` | DFT sample frequencies |
| `me.fft.rfftfreq(n)` | Real FFT sample frequencies |
| `me.fft.fftshift(x)` | Shift zero-frequency to center |
| `me.fft.ifftshift(x)` | Inverse of `fftshift` |

## Use `rfft` for real-valued data

When your input is real-valued, always prefer the `r`-prefixed variants. They exploit conjugate symmetry and cost roughly half:

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**7) as budget:
    signal = me.random.randn(1024)       # free

    # Expensive: full complex FFT
    spec_full = me.fft.fft(signal)       # 5 * 1024 * 10 = 51,200 FLOPs

with me.BudgetContext(flop_budget=10**7) as budget:
    signal = me.random.randn(1024)       # free

    # Cheaper: real FFT
    spec_real = me.fft.rfft(signal)      # 5 * 512 * 10 = 25,600 FLOPs
```

## Query cost before running

```python
from mechestim.fft import fft_cost, rfft_cost, fftn_cost, rfftn_cost

print(f"fft(1024):   {fft_cost(1024):,}")    # 51,200
print(f"rfft(1024):  {rfft_cost(1024):,}")   # 25,600
print(f"fftn(64x64): {fftn_cost((64, 64)):,}")  # 245,760
```

## Multi-dimensional transforms

For 2-D and N-D data, the same `r`-variant savings apply:

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    image = me.random.randn(256, 256)    # free

    # 2-D real FFT — cheaper than fft2
    spectrum = me.fft.rfft2(image)

    # Frequency grids are free
    freqs_x = me.fft.fftfreq(256)
    freqs_y = me.fft.rfftfreq(256)

    print(f"Cost: {budget.flops_used:,}")
```

## Common pitfalls

**Symptom:** Using `numpy.fft.fft` instead of `me.fft.fft`

**Fix:** Operations called through `numpy` directly bypass FLOP counting. Always use `me.fft.*`.

**Symptom:** Using `fft` on real-valued data

**Fix:** Switch to `rfft` / `rfft2` / `rfftn` for ~2x savings. The output is shorter (only non-redundant frequencies) but contains the same information for real inputs.

## Related pages

- [Plan Your Budget](./plan-your-budget.md) — query costs before running
- [Use Linear Algebra](./use-linalg.md) — linalg operations and costs
- [Operation Audit](../reference/operation-audit.md) — full list of supported operations
