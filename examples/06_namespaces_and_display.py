"""Namespaces and Rich display — label phases and inspect session-wide data.

Run: uv run python examples/06_namespaces_and_display.py
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
# ---------------------------------------------------------------------------
# 1. Operations without an explicit BudgetContext (global default)
# ---------------------------------------------------------------------------
# flopscope activates a global default context automatically when no explicit
# BudgetContext is in scope.  These FLOPs are recorded under namespace=None.

x = fnp.array(np.random.randn(64).astype(np.float32))
W = fnp.array(np.random.randn(64, 64).astype(np.float32))

# A quick dot product — charged to the global default budget
_ = fnp.einsum("ij,j->i", W, x)

# ---------------------------------------------------------------------------
# 2. Named context: "training" phase
# ---------------------------------------------------------------------------
# Use namespace= to label a block of work.  quiet=True suppresses the
# per-context banner so only the final summary is visible.

layers = [
    (
        fnp.array(np.random.randn(64, 64).astype(np.float32)),  # W1
        fnp.array(np.random.randn(64).astype(np.float32)),
    ),  # b1
    (
        fnp.array(np.random.randn(64, 64).astype(np.float32)),  # W2
        fnp.array(np.random.randn(64).astype(np.float32)),
    ),  # b2
    (
        fnp.array(np.random.randn(64, 64).astype(np.float32)),  # W3
        fnp.array(np.random.randn(64).astype(np.float32)),
    ),  # b3
]

batch_size = 16
flop_budget_train = 50_000_000  # 50 M FLOPs for 5 training forward passes

with flops.BudgetContext(
    flop_budget=flop_budget_train,
    namespace="training",
    quiet=True,
) as train_ctx:
    for _step in range(5):  # 5 forward passes
        h = fnp.array(np.random.randn(batch_size, 64).astype(np.float32))
        for W_layer, b in layers:
            # Linear: h @ W.T  (batch_size x 64) @ (64 x 64) -> (batch_size x 64)
            h = fnp.einsum("bi,ji->bj", h, W_layer)
            # Bias add and ReLU activation
            h = fnp.maximum(h + b, 0)

    print(f"Training used: {train_ctx.flops_used:,} / {flop_budget_train:,} FLOPs")

# ---------------------------------------------------------------------------
# 3. Decorator form: "inference" phase
# ---------------------------------------------------------------------------
# BudgetContext can also wrap a function directly.  Here fnp define a single
# forward pass and label it with namespace="inference".


@flops.BudgetContext(flop_budget=5_000_000, namespace="inference", quiet=True)
def run_inference():
    h = fnp.array(np.random.randn(batch_size, 64).astype(np.float32))
    for W_layer, b in layers:
        h = fnp.einsum("bi,ji->bj", h, W_layer)
        h = fnp.maximum(h + b, 0)
    return h


output = run_inference()
print(f"Inference output shape: {output.shape}")

# ---------------------------------------------------------------------------
# 4. Session-wide Rich summary across all phases
# ---------------------------------------------------------------------------
# flops.budget_summary() collects every BudgetContext that has exited (plus the
# global default) and renders a unified table via Rich (or plain text).

print("\n--- Session budget summary ---")
flops.budget_summary()

# ---------------------------------------------------------------------------
# 5. Programmatic access with by_namespace=True
# ---------------------------------------------------------------------------
# flops.budget_summary_dict(by_namespace=True) returns a plain dict, useful for logging,
# assertions, or downstream analysis.

data = flops.budget_summary_dict(by_namespace=True)

print("\n--- Per-namespace breakdown ---")
for ns, stats in data.get("by_namespace", {}).items():
    label = ns if ns is not None else "(global default)"
    print(f"  {label}: {stats['flops_used']:,} FLOPs used")
