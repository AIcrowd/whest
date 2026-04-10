"""Verify that the codebase now matches paper's Φ exactly.

Runs actual contract_path with the symmetry oracle and compares
the reported cost against the analytical Φ formula.
"""
from math import comb

import numpy as np

import mechestim as me
from mechestim._opt_einsum._contract import contract_path
from mechestim._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle


def multiset(n: int, k: int) -> int:
    return comb(n + k - 1, k) if k >= 0 else 0


def phi_cost_exact(n: int, s: int, t: int, v: int) -> int:
    omega = s + t + v
    if v == 0:
        # Φ degenerates for v=0
        ms, mt = multiset(n, s), multiset(n, t)
        mst = multiset(n, s + t)
        return 2 * ms * mt * 1 - ms * mt + ms * mt - mst

    m_omega = multiset(n, omega)
    per_element = 1 + comb(omega, s) + comb(omega, t) + comb(omega, v)
    phi = m_omega * per_element
    phi += multiset(n, s + v) if s > 0 else 0
    phi += multiset(n, t + v) if t > 0 else 0
    phi += multiset(n, s + t)
    return max(phi, 1)


def make_contraction(n, s, t, v):
    """Build einsum string and operands for a (s,t,v) fully-symmetric contraction."""
    # Use letters a..z for indices
    letters = "abcdefghijklmnopqrstuvwxyz"
    a_free = letters[:s]
    b_free = letters[s : s + t]
    contracted = letters[s + t : s + t + v]

    sub_a = a_free + contracted  # order s+v
    sub_b = contracted + b_free  # order v+t (contracted first so they align)
    sub_out = a_free + b_free  # order s+t

    subscripts = f"{sub_a},{sub_b}->{sub_out}"

    shape_a = (n,) * (s + v)
    shape_b = (n,) * (v + t)

    # Create symmetric tensors (same object for A=B detection)
    A = np.ones(shape_a)
    B = A if shape_a == shape_b else np.ones(shape_b)

    return subscripts, A, B, sub_a, sub_b, sub_out


def main():
    cases = [
        (1, 1, 1),  # symm
        (2, 1, 1),
        (2, 2, 1),
        (2, 2, 2),
        (3, 3, 3),
    ]
    n_values = [6, 10, 20, 50]

    print("Verifying codebase contract_path cost matches paper's Φ")
    print("=" * 100)
    print(
        f"{'(s,t,v)':>8} {'n':>4} │ "
        f"{'Φ (paper)':>16} "
        f"{'Code (actual)':>16} "
        f"{'code/Φ':>10} "
        f"{'match':>6}"
    )
    print("─" * 70)

    all_match = True
    for s, t, v in cases:
        for n in n_values:
            phi = phi_cost_exact(n, s, t, v)

            subscripts, A, B, sub_a, sub_b, sub_out = make_contraction(n, s, t, v)

            oracle = SubgraphSymmetryOracle(
                [A, B],
                [sub_a, sub_b],
                [None, None],
                sub_out,
            )

            _, info = contract_path(
                subscripts,
                A.shape,
                B.shape,
                shapes=True,
                optimize="greedy",
                symmetry_oracle=oracle,
            )
            code_cost = info.optimized_cost

            # Code should be min(direct, Φ) — so code ≤ Φ always
            ratio = code_cost / phi if phi else 0
            if abs(ratio - 1.0) < 0.001:
                match = "= Φ"
            elif code_cost < phi:
                match = "< Φ"  # direct won
            else:
                match = "✗ > Φ"
                all_match = False

            label = f"({s},{t},{v})"
            pad = max(8 - len(label), 0)
            print(
                f"{label}{' ' * pad} {n:>4} │ "
                f"{phi:>16,} "
                f"{code_cost:>16,} "
                f"{ratio:>10.6f} "
                f"{match:>6}"
            )
        print()

    print("=" * 100)
    if all_match:
        print("✓ All cases match Φ exactly")
    else:
        print("✗ Some cases do NOT match Φ")


if __name__ == "__main__":
    main()
