"""Statistical distributions with FLOP counting (scipy.stats compatible).

``flops.stats`` provides continuous distributions that match scipy.stats
exactly (same signatures, same numerical results) while tracking FLOP
costs.  Each distribution has ``.pdf()``, ``.cdf()``, and ``.ppf()``
methods.

Run: uv run python examples/11_stats_distributions.py
"""

import flopscope as flops
import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
# ---------------------------------------------------------------------------
# 1. Basic usage — normal distribution
# ---------------------------------------------------------------------------
print("=== Normal distribution ===\n")

with flops.BudgetContext(flop_budget=10**6) as budget:
    x = fnp.random.randn(1000)

    pdf_vals = flops.stats.norm.pdf(x, loc=0, scale=1)
    cdf_vals = flops.stats.norm.cdf(x, loc=0, scale=1)

    # Inverse CDF (percent-point function)
    q = fnp.array([0.025, 0.5, 0.975])
    quantiles = flops.stats.norm.ppf(q)

print(f"PDF at x=0:        {flops.stats.norm.pdf(0.0):.6f}")
print(f"CDF at x=1.96:     {flops.stats.norm.cdf(1.96):.6f}")
print(f"Quantiles (2.5%, 50%, 97.5%): {quantiles}")
print(f"FLOPs used:        {budget.flops_used:,}")

# ---------------------------------------------------------------------------
# 2. Cost comparison across distributions
# ---------------------------------------------------------------------------
print("\n=== FLOP cost per 10,000 elements ===\n")

n = 10_000
x = fnp.random.randn(n)

distributions = [
    ("norm", flops.stats.norm),
    ("uniform", flops.stats.uniform),
    ("expon", flops.stats.expon),
    ("cauchy", flops.stats.cauchy),
    ("logistic", flops.stats.logistic),
    ("laplace", flops.stats.laplace),
    ("lognorm", flops.stats.lognorm),
    ("truncnorm", flops.stats.truncnorm),
]

print(f"{'Distribution':<14} {'PDF cost':>10} {'CDF cost':>10} {'PPF cost':>10}")
print("-" * 48)

for name, dist in distributions:
    # Each distribution needs different args
    if name == "lognorm":
        args = (x, 1.0)  # shape s=1.0
        q_args = (0.5, 1.0)
    elif name == "truncnorm":
        args = (x, -2, 2)  # bounds a=-2, b=2
        q_args = (0.5, -2, 2)
    else:
        args = (x,)
        q_args = (0.5,)

    for method_name in ("pdf", "cdf", "ppf"):
        with flops.BudgetContext(flop_budget=10**8, quiet=True) as b:
            method = getattr(dist, method_name)
            if method_name == "ppf":
                # ppf takes quantiles in [0,1], use uniform random
                u = fnp.random.randn(n)
                u = 1 / (1 + fnp.exp(-u))  # sigmoid to [0,1]
                method(u, *q_args[1:]) if len(q_args) > 1 else method(u)
            else:
                method(*args)
        cost = b.flops_used
        if method_name == "pdf":
            pdf_cost = cost
        elif method_name == "cdf":
            cdf_cost = cost
        else:
            ppf_cost = cost

    print(f"{name:<14} {pdf_cost:>10,} {cdf_cost:>10,} {ppf_cost:>10,}")

# ---------------------------------------------------------------------------
# 3. Practical example — budget-aware sampling + CDF
# ---------------------------------------------------------------------------
print("\n=== Practical: Gaussian quantile computation under budget ===\n")

flop_limit = 500_000

with flops.BudgetContext(flop_budget=flop_limit) as budget:
    # Generate 10k samples
    samples = fnp.random.randn(10_000)

    # Compute CDF values (how likely is each sample?)
    probabilities = flops.stats.norm.cdf(samples)

    # What fraction of budget did fnp use?
    remaining = flop_limit - budget.flops_used

print(f"Samples:     {samples.shape[0]:,}")
print(f"Budget used: {budget.flops_used:,} / {flop_limit:,} FLOPs")
print(f"Remaining:   {remaining:,} FLOPs")
print(f"Mean CDF:    {probabilities.mean():.4f} (expected ~0.5)")
