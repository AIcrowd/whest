"""Compare FLOP cost models for symmetric tensor contractions.

Four models:
  Naive    — no symmetry, 2n^ω
  Ψ (paper Thm 4.3) — direct evaluation, per-group symmetry
  Φ (paper Thm 5.4) — symmetry-preserving algorithm
  Code     — your codebase's symmetric_flop_count (output ± inner symmetry)

All models use μ = ν = 1 (MUL = ADD = 1 FLOP).
"""

from math import comb, factorial


def multiset(n: int, k: int) -> int:
    """Multiset coefficient ((n,k)) = C(n+k-1, k)."""
    if k == 0:
        return 1
    return comb(n + k - 1, k)


def F_naive(n: int, s: int, t: int, v: int) -> int:
    """Naive cost: treat as dense matrix multiply, no symmetry."""
    omega = s + t + v
    op_factor = 2 if v > 0 else 1
    return op_factor * n**omega


def F_psi(n: int, s: int, t: int, v: int) -> int:
    """Paper's Ψ (Theorem 4.3): direct evaluation, per-group symmetry.

    F^Ψ = F^MM(ρ, ((n,s)), ((n,t)), ((n,v))) + ν^C·[((n,s))·((n,t)) - ((n,s+t))]

    With μ=ν=1: F^MM(m,n,k) = 2mnk - mn  (the -mn accounts for first
    additions being folded into output).
    """
    ms = multiset(n, s)
    mt = multiset(n, t)
    mv = multiset(n, v)
    mst = multiset(n, s + t)

    # Matrix multiplication cost of ms×mv by mv×mt
    matmul_cost = 2 * ms * mt * mv - ms * mt
    # Symmetrization of output
    sym_cost = ms * mt - mst
    return max(matmul_cost + sym_cost, 1)


def F_phi(n: int, s: int, t: int, v: int) -> int:
    """Paper's Φ (Theorem 5.4): symmetry-preserving algorithm.

    Leading-order terms for v > 0:
      mults: n^ω/ω! × μ
      adds for Ẑ: n^ω/ω! × [C(ω,t)·ν^A + C(ω,s)·ν^B + C(ω,v)·ν^C]
      adds for A^(p), B^(q): n^(s+v)/(s+v)!·ν^A + n^(t+v)/(t+v)!·ν^B
      adds for output sym: n^(s+t)/(s+t)!·ν^C

    With μ=ν=1, the leading multiplication + Ẑ addition cost dominates.
    """
    omega = s + t + v
    m_omega = multiset(n, omega)  # n^ω/ω! to leading order

    if v == 0:
        # When v=0, Φ degenerates: no inner sum, just outer product.
        # The symmetry-preserving algorithm for v=0 reduces to computing
        # unique output elements: ((n, s+t)).
        # Each needs s!·t! multiplied terms (permutations of partition).
        # Total mults ≈ ((n,s+t)) and adds ≈ ((n,s+t))·(s!·t! - 1)
        mst = multiset(n, s + t)
        return max(mst * factorial(s) * factorial(t), 1)

    # Multiplications: one per unique element of order-ω symmetric tensor Ẑ
    mults = m_omega

    # Additions for computing Ẑ (accumulating sums of A and B elements):
    # Each Ẑ element is a product of (sum of C(ω,t) A-entries) × (sum of C(ω,s) B-entries)
    z_add_A = comb(omega, t) if t > 0 else 0  # additions in A-sum per Ẑ element
    z_add_B = comb(omega, s) if s > 0 else 0  # additions in B-sum per Ẑ element
    z_add_C = comb(omega, v)  # additions for accumulating Ẑ into Z
    z_adds = m_omega * (z_add_A + z_add_B + z_add_C)

    # Additions for A^(p), B^(q) intermediates (lower order: n^(s+v), n^(t+v))
    a_intermediate = multiset(n, s + v) if s > 0 else 0
    b_intermediate = multiset(n, t + v) if t > 0 else 0

    # Additions for V, W correction terms (lower order)
    correction = multiset(n, s + t)

    return max(mults + z_adds + a_intermediate + b_intermediate + correction, 1)


def F_code(
    n: int,
    s: int,
    t: int,
    v: int,
    *,
    same_object: bool = False,
    use_inner_symmetry: bool = False,
) -> int:
    """Your codebase's symmetric_flop_count logic.

    base = op_factor × n^ω
    × (unique_output / total_output)
    × (unique_inner / total_inner)   [if use_inner_symmetry]
    """
    omega = s + t + v
    op_factor = 2 if v > 0 else 1
    base = op_factor * n**omega

    # Output symmetry reduction
    if same_object and (s + t) >= 2:
        # Oracle merges s+t free indices into one symmetric group
        unique_out = multiset(n, s + t)
        total_out = n ** (s + t)
    else:
        # Separate s-group and t-group
        unique_out = multiset(n, s) * multiset(n, t)
        total_out = n ** (s + t)

    if total_out > 0:
        base = base * unique_out // total_out

    # Inner symmetry reduction
    if use_inner_symmetry and v >= 2:
        unique_inner = multiset(n, v)
        total_inner = n**v
        if total_inner > 0:
            base = base * unique_inner // total_inner

    return max(base, 1)


def print_comparison(cases, n_values):
    """Print comparison table."""
    header = (
        f"{'(s,t,v)':>8} {'ω':>3} {'n':>4} │ "
        f"{'Naive':>12} {'Ψ (paper)':>12} {'Φ (paper)':>12} │ "
        f"{'Code A≠B':>12} {'Code A=B':>12} │ "
        f"{'Code/Ψ':>8} {'Code/Φ':>8}"
    )
    print(header)
    print("─" * len(header))

    for s, t, v in cases:
        for n in n_values:
            naive = F_naive(n, s, t, v)
            psi = F_psi(n, s, t, v)
            phi = F_phi(n, s, t, v)
            code_neq = F_code(
                n, s, t, v, same_object=False, use_inner_symmetry=True
            )
            code_eq = F_code(
                n, s, t, v, same_object=True, use_inner_symmetry=True
            )

            # Best code estimate (A=B when possible)
            code_best = code_eq
            ratio_psi = f"{code_best / psi:.3f}" if psi > 0 else "—"
            ratio_phi = f"{code_best / phi:.3f}" if phi > 0 else "—"

            label = f"({s},{t},{v})"
            pad = max(8 - len(label), 0)
            print(
                f"{label}{' ' * pad} {s+t+v:>3} {n:>4} │ "
                f"{naive:>12,} {psi:>12,} {phi:>12,} │ "
                f"{code_neq:>12,} {code_eq:>12,} │ "
                f"{ratio_psi:>8} {ratio_phi:>8}"
            )
        print()


def print_asymptotic_table(cases):
    """Print asymptotic (n→∞) formulas."""
    print("\nAsymptotic leading-order costs (n → ∞, μ=ν=1):")
    print(f"{'(s,t,v)':>8} {'ω':>3} │ {'Naive':>16} {'Ψ':>16} {'Φ (approx)':>16} │ {'Code A≠B':>16} {'Code A=B':>16}")
    print("─" * 100)

    for s, t, v in cases:
        omega = s + t + v
        op = 2 if v > 0 else 1

        naive_str = f"{op}n^{omega}"

        # Ψ denominator: s!·t!·v! (the -mn term is lower order)
        psi_denom = factorial(s) * factorial(t) * factorial(v)
        psi_str = f"{op}n^{omega}/{psi_denom}" if psi_denom > 1 else f"{op}n^{omega}"

        # Φ: roughly (1 + C(ω,t) + C(ω,s) + C(ω,v)) × n^ω/ω!
        phi_coeff = 1 + comb(omega, t) + comb(omega, s) + comb(omega, v)
        phi_denom = factorial(omega)
        phi_str = f"{phi_coeff}n^{omega}/{phi_denom}"

        # Code A≠B with inner: op × n^ω / (s!·t!·v!)
        code_neq_denom = factorial(s) * factorial(t) * (factorial(v) if v >= 2 else 1)
        code_neq_str = f"{op}n^{omega}/{code_neq_denom}" if code_neq_denom > 1 else f"{op}n^{omega}"

        # Code A=B with inner: op × n^ω / ((s+t)!·v!)
        code_eq_denom = factorial(s + t) * (factorial(v) if v >= 2 else 1)
        code_eq_str = f"{op}n^{omega}/{code_eq_denom}" if code_eq_denom > 1 else f"{op}n^{omega}"

        label = f"({s},{t},{v})"
        pad = max(8 - len(label), 0)
        print(f"{label}{' ' * pad} {omega:>3} │ {naive_str:>16} {psi_str:>16} {phi_str:>16} │ {code_neq_str:>16} {code_eq_str:>16}")


if __name__ == "__main__":
    cases = [
        (1, 0, 1),  # symv: symmetric matrix × vector
        (1, 1, 0),  # syr2: rank-2 outer product
        (1, 1, 1),  # symm: Jordan ring A⊗₁B
        (2, 1, 1),  # order-4 contraction
        (2, 2, 1),  # order-5 contraction
        (2, 2, 2),  # order-6 contraction
        (3, 3, 3),  # order-9 contraction
    ]

    n_values = [10, 50, 100]

    print("=" * 80)
    print("EXACT FLOP COUNTS (use_inner_symmetry=True)")
    print("Code A=B column: same-object detection + inner symmetry")
    print("=" * 80)
    print()
    print_comparison(cases, n_values)
    print_asymptotic_table(cases)

    # Also show the ratios at large n
    print("\n\nRatios at n=100 (Code A=B with inner / other models):")
    print(f"{'(s,t,v)':>8} │ {'Code/Naive':>12} {'Code/Ψ':>12} {'Code/Φ':>12} {'Ψ/Φ':>12}")
    print("─" * 65)
    for s, t, v in cases:
        n = 100
        naive = F_naive(n, s, t, v)
        psi = F_psi(n, s, t, v)
        phi = F_phi(n, s, t, v)
        code = F_code(n, s, t, v, same_object=True, use_inner_symmetry=True)
        label = f"({s},{t},{v})"
        pad = max(8 - len(label), 0)
        print(
            f"{label}{' ' * pad} │ "
            f"{code/naive:>12.4f} {code/psi:>12.4f} {code/phi:>12.4f} {psi/phi:>12.4f}"
        )
