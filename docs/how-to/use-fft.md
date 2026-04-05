# Use FFT

## When to use this page

Use this page to learn how to use `me.fft` operations and understand their FLOP costs.

## Prerequisites

- [Your First Budget](../getting-started/first-budget.md)

## Cost model

FFT costs are based on the Cooley-Tukey radix-2 algorithm:

| Transform | Cost Formula | Example (n=1024) |
|-----------|-------------|------------------|
| `fft`, `ifft` | $5n \cdot \lceil\log_2 n\rceil$ | 51,200 |
| `rfft`, `irfft` | $5(n/2) \cdot \lceil\log_2 n\rceil$ | 25,600 |
| `fft2`, `ifft2` | $5N \cdot \lceil\log_2 N\rceil$ where $N = n_1 \cdot n_2$ | varies |
| `fftn`, `ifftn` | $5N \cdot \lceil\log_2 N\rceil$ where $N = \prod_i n_i$ | varies |
| `fftfreq`, `rfftfreq` | 0 (free) | 0 |
| `fftshift`, `ifftshift` | 0 (free) | 0 |

Real-valued transforms (`rfft`, `irfft`, `rfftn`, `irfftn`) cost roughly half of their complex counterparts because they exploit conjugate symmetry.

## Basic usage

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    # Generate a signal (free)
    signal = me.random.randn(1024)

    # Forward FFT: 5 * 1024 * 10 = 51,200 FLOPs
    spectrum = me.fft.fft(signal)

    # Inverse FFT: same cost
    reconstructed = me.fft.ifft(spectrum)

    # Frequency bins (free)
    freqs = me.fft.fftfreq(1024)

    print(f"Total FFT cost: {budget.flops_used:,}")  # 102,400
```

## Real vs complex transforms

When your input is real-valued (which is common in signal processing), prefer `rfft` over `fft` — it costs half as much:

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    signal = me.random.randn(1024)

    # Complex FFT: 51,200 FLOPs
    spec_complex = me.fft.fft(signal)

    budget_after_fft = budget.flops_used

    # Real FFT: 25,600 FLOPs
    spec_real = me.fft.rfft(signal)

    rfft_cost = budget.flops_used - budget_after_fft

    print(f"fft cost:  {budget_after_fft:,}")   # 51,200
    print(f"rfft cost: {rfft_cost:,}")           # 25,600
```

The output of `rfft` has shape `(n//2 + 1,)` instead of `(n,)`, since the negative frequencies are redundant for real inputs.

## Multi-dimensional FFT

Use `fft2` for 2-D transforms (e.g., images) and `fftn` for arbitrary dimensions:

```python
import mechestim as me

with me.BudgetContext(flop_budget=10**8) as budget:
    # 2-D image (free to create)
    image = me.random.randn(256, 256)

    # 2-D FFT
    spectrum_2d = me.fft.fft2(image)
    print(f"2D FFT cost: {budget.flops_used:,}")

    # N-D FFT with explicit shape
    volume = me.random.randn(32, 32, 32)
    spectrum_3d = me.fft.fftn(volume)
```

## Windowed FFT pattern

A common signal processing pattern — window the signal before FFT to reduce spectral leakage:

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    signal = me.random.randn(1024)

    # Window function (counted — hamming costs n FLOPs)
    window = me.hamming(1024)

    # Apply window (counted — multiply costs n FLOPs)
    windowed = me.multiply(signal, window)

    # FFT (counted)
    spectrum = me.fft.rfft(windowed)

    print(budget.summary())
```

## Query costs before running

```python
from mechestim.flops import fft_cost, rfft_cost

# Check cost of a large FFT before committing budget
n = 2**20  # ~1 million points
print(f"Complex FFT: {fft_cost(n):,} FLOPs")   # 104,857,600
print(f"Real FFT:    {rfft_cost(n):,} FLOPs")   # 52,428,800
```

## ⚠️ Common pitfalls

**Symptom:** Using `me.fft.fft` on real data when `me.fft.rfft` would suffice

**Fix:** `rfft` costs half as much. If your input is real-valued, always prefer `rfft`/`irfft` over `fft`/`ifft`.

**Symptom:** Unexpectedly high cost for multi-dimensional FFT

**Fix:** The cost scales as $5 \cdot \prod n_i \cdot \lceil\log_2(\prod n_i)\rceil$. A 256x256 2-D FFT processes 65,536 elements, not 256. Use `fft_cost` to estimate before running.

## 📎 Related pages

- [FFT API Reference](../api/fft.md) — full function signatures and docstrings
- [Plan Your Budget](./plan-your-budget.md) — general cost estimation workflow
- [FLOP Counting Model](../concepts/flop-counting-model.md) — how all costs are computed
