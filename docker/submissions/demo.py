"""
Demo: A participant's submission that uses whest exactly like they would locally.
This runs inside a container with NO numpy installed.
"""

import whest as we

print("=" * 60)
print("  Whest Client-Server Demo")
print("  (this container has NO numpy installed)")
print("=" * 60)

# Verify numpy is not available
try:
    import numpy

    print("\nWARNING: numpy is installed (not expected)")
except ImportError:
    print("\nConfirmed: numpy is NOT installed in this container")

print(f"whest version: {we.__version__}")
print()

with we.BudgetContext(flop_budget=1_000_000) as budget:
    # ---- 1. Basic array creation ----
    print("--- 1. Array Creation ---")
    x = we.zeros((4, 4))
    print(f"zeros(4,4) shape={x.shape} dtype={x.dtype}")

    W = we.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    print(f"W = diag(1,2,3,4):\n{W}")

    # ---- 2. Computation with budget tracking ----
    print("\n--- 2. Neural Network Forward Pass ---")
    # Simulated single-layer neural network
    batch_size = 32
    input_dim = 64
    output_dim = 16

    # Create random input and weights
    x = we.random.randn(batch_size, input_dim)
    W = we.random.randn(output_dim, input_dim)
    b = we.zeros((output_dim,))

    print(f"Input:   x.shape={x.shape}")
    print(f"Weights: W.shape={W.shape}")
    print(f"Bias:    b.shape={b.shape}")

    # Forward pass: y = ReLU(Wx + b)
    h = we.einsum("oi,bi->bo", W, x)  # matrix multiply
    h = we.add(h, b)  # add bias
    y = we.maximum(h, we.zeros_like(h))  # ReLU

    print(f"Output:  y.shape={y.shape}")
    print(f"FLOPs used: {budget.flops_used:,}")
    print(f"FLOPs remaining: {budget.flops_remaining:,}")

    # ---- 3. Python operators work! ----
    print("\n--- 3. Python Operators ---")
    a = we.array([1.0, 2.0, 3.0, 4.0])
    b = we.array([10.0, 20.0, 30.0, 40.0])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * 2 = {a * 2.0}")
    print(f"2 ** a = {2.0**a}")
    print(f"-a = {-a}")

    # ---- 4. Comparisons ----
    print("\n--- 4. Comparisons & Reductions ---")
    mask = a > 2.0
    print(f"a > 2.0 = {mask}")
    print(f"sum(a) = {float(we.sum(a))}")
    print(f"mean(a) = {float(we.mean(a))}")
    print(f"max(a) = {float(a.max())}")
    print(f"min(a) = {float(a.min())}")

    # ---- 5. Methods on arrays ----
    print("\n--- 5. Array Methods ---")
    M = we.array([[1.0, 2.0], [3.0, 4.0]])
    print(f"M = {M}")
    print(f"M.T = {M.T}")
    print(f"M.reshape(4) = {M.reshape(4)}")
    print(f"M.sum(axis=0) = {M.sum(axis=0)}")
    print(f"M.flatten() = {M.flatten()}")

    # ---- 6. Indexing ----
    print("\n--- 6. Indexing ---")
    v = we.array([50.0, 40.0, 30.0, 20.0, 10.0])
    print(f"v = {v}")
    print(f"v[0] = {v[0]}")
    print(f"v[-1] = {v[-1]}")
    print(f"v[1:3] = {v[1:3]}")

    # ---- 7. Iteration ----
    print("\n--- 7. Iteration ---")
    for i, val in enumerate(a):
        print(f"  a[{i}] = {val}")

    # ---- 8. SVD ----
    print("\n--- 8. Linear Algebra (SVD) ---")
    A = we.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    U, S, Vt = we.linalg.svd(A)
    print(f"A.shape = {A.shape}")
    print(f"U.shape = {U.shape}")
    print(f"S = {S}")
    print(f"Vt.shape = {Vt.shape}")

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print(f"  Total FLOPs used: {budget.flops_used:,}")
    print(f"  Budget remaining: {budget.flops_remaining:,}")
    print(budget.summary())
    print("=" * 60)
    print("\n  SUCCESS: Everything works without numpy!")
    print("  The participant never knew they were using a proxy.")
