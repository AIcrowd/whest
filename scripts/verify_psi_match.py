"""Verify codebase symmetric_flop_count against paper's Ψ and Φ formulas.

Tests the actual codebase functions against Theorem 4.3 (Ψ) and
Theorem 5.4 (Φ) from Solomonik & Demmel (2015) for a range of
(s, t, v, n) combinations. Shows exact values and ratios.
"""

from math import comb

from mechestim._opt_einsum._helpers import compute_size_by_dict, flop_count
from mechestim._opt_einsum._symmetry import (
    IndexSymmetry,
    symmetric_flop_count,
    unique_elements,
)


def multiset(n: int, k: int) -> int:
    """((n, k)) = C(n+k-1, k)."""
    return comb(n + k - 1, k) if k >= 0 else 0


def psi_cost(n: int, s: int, t: int, v: int) -> int:
    """Paper's Ψ (Theorem 4.3) with μ=ν=1.

    F^Ψ = F^MM(((n,s)), ((n,t)), ((n,v))) + [((n,s))·((n,t)) - ((n,s+t))]
    where F^MM(m,p,k) = 2mpk - mp
    """
    ms = multiset(n, s)
    mt = multiset(n, t)
    mv = multiset(n, v)
    mst = multiset(n, s + t)

    matmul = 2 * ms * mt * mv - ms * mt
    symmetrization = ms * mt - mst
    return max(matmul + symmetrization, 1)


def build_einsum_args(n: int, s: int, t: int, v: int):
    """Build index labels, size_dict, and symmetry info for a (s,t,v) contraction.

    Operand A has order s+v (symmetric in all indices).
    Operand B has order t+v (symmetric in all indices).
    Output C has order s+t (symmetric in all indices).
    v indices are contracted.

    Returns (idx_contraction, size_dict, output_sym, output_inds,
             inner_sym, inner_inds, same_object_output_sym).
    """
    # Label indices: a0..a_{s-1} for A's free, b0..b_{t-1} for B's free,
    # c0..c_{v-1} for contracted
    a_labels = [f"a{i}" for i in range(s)]
    b_labels = [f"b{i}" for i in range(t)]
    c_labels = [f"c{i}" for i in range(v)]

    all_labels = a_labels + b_labels + c_labels
    output_labels = a_labels + b_labels
    inner_labels = c_labels

    size_dict = {lbl: n for lbl in all_labels}

    # A is symmetric in all its indices (a_labels + c_labels)
    # B is symmetric in all its indices (b_labels + c_labels)
    # For A≠B case:
    #   output symmetry: separate groups for a-indices and b-indices
    #   inner symmetry: one group for c-indices (if v >= 2)
    output_sym_separate: IndexSymmetry = []
    if s >= 2:
        output_sym_separate.append(frozenset((lbl,) for lbl in a_labels))
    if t >= 2:
        output_sym_separate.append(frozenset((lbl,) for lbl in b_labels))

    inner_sym: IndexSymmetry = []
    if v >= 2:
        inner_sym.append(frozenset((lbl,) for lbl in c_labels))

    # For A=B case: output has full (s+t)-symmetry
    output_sym_merged: IndexSymmetry = []
    if s + t >= 2:
        output_sym_merged.append(frozenset((lbl,) for lbl in output_labels))

    return {
        "idx_contraction": frozenset(all_labels),
        "size_dict": size_dict,
        "output_sym_separate": output_sym_separate or None,
        "output_sym_merged": output_sym_merged or None,
        "output_indices": frozenset(output_labels),
        "inner_sym": inner_sym or None,
        "inner_indices": frozenset(inner_labels) if inner_labels else None,
        "has_inner": v > 0,
        "num_terms": 2,
    }


def code_cost(n, s, t, v, *, same_object=False, use_inner=False):
    """Compute cost using actual codebase symmetric_flop_count."""
    args = build_einsum_args(n, s, t, v)
    out_sym = args["output_sym_merged"] if same_object else args["output_sym_separate"]
    return symmetric_flop_count(
        args["idx_contraction"],
        args["has_inner"],
        args["num_terms"],
        args["size_dict"],
        output_symmetry=out_sym,
        output_indices=args["output_indices"],
        inner_symmetry=args["inner_sym"],
        inner_indices=args["inner_indices"],
        use_inner_symmetry=use_inner,
    )


def phi_cost_exact(n: int, s: int, t: int, v: int) -> int:
    """Paper's Φ (Theorem 5.4) exact cost with μ=ν=1.

    F^Φ = ((n,ω)) × [1 + C(ω,s) + C(ω,t) + C(ω,v)]   ← Ẑ mults + adds
        + ((n, s+v))                                      ← A^(p) intermediates
        + ((n, t+v))                                      ← B^(q) intermediates
        + ((n, s+t))                                      ← output symmetrization

    For v=0: degenerates to outer product, use Ψ cost.
    """
    omega = s + t + v

    if v == 0:
        # Φ degenerates for v=0; fall back to Ψ
        return psi_cost(n, s, t, v)

    m_omega = multiset(n, omega)  # ((n, ω)) = unique Ẑ elements

    # Per-element cost: 1 mult + additions for partial sums + accumulation
    per_element = 1 + comb(omega, s) + comb(omega, t) + comb(omega, v)
    z_cost = m_omega * per_element

    # Lower-order terms: A^(p), B^(q) intermediates and output symmetrization
    a_cost = multiset(n, s + v) if s > 0 else 0
    b_cost = multiset(n, t + v) if t > 0 else 0
    out_cost = multiset(n, s + t)

    return max(z_cost + a_cost + b_cost + out_cost, 1)


def phi_cost_breakdown(n: int, s: int, t: int, v: int) -> dict:
    """Return a breakdown of each Φ cost component."""
    omega = s + t + v
    if v == 0:
        return {"total": psi_cost(n, s, t, v), "note": "v=0, using Ψ"}

    m_omega = multiset(n, omega)
    mults = m_omega
    z_adds_A = m_omega * comb(omega, s)
    z_adds_B = m_omega * comb(omega, t)
    z_accum = m_omega * comb(omega, v)
    a_inter = multiset(n, s + v) if s > 0 else 0
    b_inter = multiset(n, t + v) if t > 0 else 0
    out_sym = multiset(n, s + t)

    return {
        "((n,ω))": m_omega,
        "mults (Ẑ)": mults,
        "adds A-sums": z_adds_A,
        "adds B-sums": z_adds_B,
        "adds Ẑ→Z": z_accum,
        "A^(p) inter": a_inter,
        "B^(q) inter": b_inter,
        "out sym": out_sym,
        "total": mults + z_adds_A + z_adds_B + z_accum + a_inter + b_inter + out_sym,
    }


def main():
    cases = [
        (1, 0, 1),  # symv
        (1, 1, 0),  # syr2
        (1, 1, 1),  # symm (Jordan ring)
        (2, 1, 1),
        (2, 2, 1),
        (2, 2, 2),
        (3, 3, 3),
    ]
    n_values = [10, 50, 100]

    # ─── Table 1: Code vs Ψ vs Φ ───
    print("=" * 140)
    print("EXACT FLOP COUNTS: Code (current & +inner) vs Paper Ψ vs Paper Φ")
    print("=" * 140)
    print(
        f"{'(s,t,v)':>8} {'n':>4} │ "
        f"{'Ψ (paper)':>16} "
        f"{'Φ (paper)':>16} "
        f"{'Code curr':>16} "
        f"{'Code+inner':>16} │ "
        f"{'inner/Ψ':>8} "
        f"{'inner/Φ':>8} "
        f"{'Ψ/Φ':>8}"
    )
    print("─" * 140)

    for s, t, v in cases:
        for n in n_values:
            psi = psi_cost(n, s, t, v)
            phi = phi_cost_exact(n, s, t, v)
            curr = code_cost(n, s, t, v, same_object=False, use_inner=False)
            with_inner = code_cost(n, s, t, v, same_object=False, use_inner=True)

            r_inner_psi = with_inner / psi if psi else 0
            r_inner_phi = with_inner / phi if phi else 0
            r_psi_phi = psi / phi if phi else 0

            label = f"({s},{t},{v})"
            pad = max(8 - len(label), 0)
            print(
                f"{label}{' ' * pad} {n:>4} │ "
                f"{psi:>16,} "
                f"{phi:>16,} "
                f"{curr:>16,} "
                f"{with_inner:>16,} │ "
                f"{r_inner_psi:>8.4f} "
                f"{r_inner_phi:>8.4f} "
                f"{r_psi_phi:>8.4f}"
            )
        print()

    # ─── Table 2: Φ cost breakdown for n=100 ───
    print("\n" + "=" * 110)
    print("Φ COST BREAKDOWN at n=100 (each component)")
    print("=" * 110)
    for s, t, v in cases:
        n = 100
        omega = s + t + v
        bd = phi_cost_breakdown(n, s, t, v)
        label = f"({s},{t},{v}) ω={omega}"
        print(f"\n  {label}")
        for k, val in bd.items():
            if isinstance(val, int):
                print(f"    {k:>16}: {val:>20,}")
            else:
                print(f"    {k:>16}: {val!s:>20}")

    # ─── Table 3: What code would need to produce to match Φ ───
    print("\n\n" + "=" * 130)
    print("GAP ANALYSIS: What symmetric_flop_count produces vs Φ target")
    print("=" * 130)
    print(
        f"{'(s,t,v)':>8} {'n':>4} │ "
        f"{'Φ target':>16} "
        f"{'Code+inner':>16} "
        f"{'Code A=B+inn':>16} "
        f"{'min(Ψ,Φ)':>16} │ "
        f"{'code/Φ':>8} "
        f"{'eq/Φ':>8} "
        f"{'best':>6}"
    )
    print("─" * 130)

    for s, t, v in cases:
        for n in n_values:
            psi = psi_cost(n, s, t, v)
            phi = phi_cost_exact(n, s, t, v)
            best_paper = min(psi, phi)
            with_inner = code_cost(n, s, t, v, same_object=False, use_inner=True)
            eq_inner = code_cost(n, s, t, v, same_object=True, use_inner=True)

            r_code = with_inner / phi if phi else 0
            r_eq = eq_inner / phi if phi else 0
            best = "Ψ" if psi <= phi else "Φ"

            label = f"({s},{t},{v})"
            pad = max(8 - len(label), 0)
            print(
                f"{label}{' ' * pad} {n:>4} │ "
                f"{phi:>16,} "
                f"{with_inner:>16,} "
                f"{eq_inner:>16,} "
                f"{best_paper:>16,} │ "
                f"{r_code:>8.4f} "
                f"{r_eq:>8.4f} "
                f"{best:>6}"
            )
        print()

    # ─── Table 4: Asymptotic formulas ───
    print("\n" + "=" * 100)
    print("ASYMPTOTIC FORMULAS (n → ∞, μ=ν=1)")
    print("=" * 100)
    from math import factorial
    print(f"{'(s,t,v)':>8} {'ω':>3} │ {'Ψ leading':>20} {'Φ leading':>20} {'Φ coeff':>8} {'best':>6}")
    print("─" * 75)
    for s, t, v in cases:
        omega = s + t + v
        psi_denom = factorial(s) * factorial(t) * factorial(v)
        op = 2 if v > 0 else 1

        phi_coeff = 1 + comb(omega, s) + comb(omega, t) + comb(omega, v)
        phi_denom = factorial(omega)

        psi_leading = op / psi_denom  # coefficient of n^ω
        phi_leading = phi_coeff / phi_denom

        best = "Ψ" if psi_leading <= phi_leading else "Φ"

        psi_str = f"{op}n^{omega}/{psi_denom}" if psi_denom > 1 else f"{op}n^{omega}"
        phi_str = f"{phi_coeff}n^{omega}/{phi_denom}"

        label = f"({s},{t},{v})"
        pad = max(8 - len(label), 0)
        print(f"{label}{' ' * pad} {omega:>3} │ {psi_str:>20} {phi_str:>20} {phi_coeff:>8} {best:>6}")


if __name__ == "__main__":
    main()
