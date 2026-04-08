#!/usr/bin/env python3
"""Extended fuzz test: 100k random einsums with broader scope.

Covers:
- 2-5 operands (up from 2-4)
- 2-8 indices (up from 2-6)
- Index sizes 2-30 (up from 3-15)
- Multiple symmetry groups per operand (up from 1)
- Higher-order symmetry groups (S2, S3, S4)
- All-contracted outputs (scalars)
- Single-index operands
- Repeated index patterns
- Mixed symmetric and dense operands
- Repeated operands (same ndarray object at multiple positions) → induced output symmetry

Usage:
    uv run python .claude/visual/verify_fuzz_100k.py
"""

import random
import string
import sys
import time
from itertools import combinations
from math import comb

import numpy as np

from mechestim._einsum import _detect_induced_output_symmetry
from mechestim._opt_einsum._contract import contract_path
from mechestim._opt_einsum._symmetry import propagate_symmetry, symmetric_flop_count


# ─── JS logic (must match explorer.html + all fixes) ───

def js_propagate_and_merge(sym_list, out_idx):
    """Collect surviving groups, merge overlapping ones."""
    candidates = []
    seen = set()
    for sym in sym_list:
        if sym is None:
            continue
        for group in sym:
            surviving = group & out_idx
            key = frozenset(surviving)
            if len(surviving) >= 2 and key not in seen:
                seen.add(key)
                candidates.append(surviving)
    # Merge overlapping groups (connected components)
    if len(candidates) > 1:
        merged = []
        for g in candidates:
            overlapping = [i for i, m in enumerate(merged) if m & g]
            if not overlapping:
                merged.append(g)
            else:
                combined = g
                for i in sorted(overlapping, reverse=True):
                    combined = combined | merged.pop(i)
                merged.append(combined)
        candidates = merged
    return candidates if candidates else None


def js_contract_pair(op1, op2, remaining_ops, final_output, index_sizes):
    all_idx = op1["indices"] | op2["indices"]
    needed = set(final_output)
    for op in remaining_ops:
        needed |= op["indices"]
    out_idx = all_idx & needed
    contracted = all_idx - out_idx
    is_inner = len(contracted) > 0

    all_size = 1
    for idx in all_idx:
        all_size *= index_sizes[idx]
    op_factor = max(1, 2 - 1) + (1 if is_inner else 0)
    dense_cost = all_size * op_factor

    out_symmetry = js_propagate_and_merge(
        [op1.get("symmetry"), op2.get("symmetry")], out_idx
    )

    sym_cost = dense_cost
    if out_symmetry and out_idx:
        total_out = 1
        for idx in out_idx:
            total_out *= index_sizes[idx]
        unique_out = 1
        accounted = set()
        for group in out_symmetry:
            active = group & out_idx
            if active <= accounted or len(active) < 2:
                continue
            n = index_sizes[next(iter(active))]
            k = len(active)
            unique_out *= comb(n + k - 1, k)
            accounted |= active
        for idx in out_idx:
            if idx not in accounted:
                unique_out *= index_sizes[idx]
        sym_cost = dense_cost * unique_out // total_out

    return {
        "sym_cost": sym_cost,
        "result_op": {
            "id": "R",
            "indices": out_idx,
            "symmetry": out_symmetry,
        },
    }


def js_best_cost(operands, final_output, index_sizes):
    """Find minimum cost across all contraction orderings (brute force)."""
    if len(operands) == 1:
        return 0
    best = float("inf")
    for i, j in combinations(range(len(operands)), 2):
        remaining = [op for k, op in enumerate(operands) if k != i and k != j]
        step = js_contract_pair(operands[i], operands[j], remaining, final_output, index_sizes)
        rest = js_best_cost([step["result_op"]] + remaining, final_output, index_sizes)
        total = step["sym_cost"] + rest
        if total < best:
            best = total
    return best


# ─── Python brute-force using propagate_symmetry (for induced symmetry) ───

def py_contract_pair(op1, op2, remaining_ops, final_output, index_sizes, induced_output_symmetry):
    """Pairwise contraction cost using propagate_symmetry + symmetric_flop_count."""
    all_idx = frozenset(op1["indices"]) | frozenset(op2["indices"])
    needed = frozenset(final_output)
    for op in remaining_ops:
        needed |= frozenset(op["indices"])
    out_idx = all_idx & needed
    contracted = all_idx - out_idx
    is_inner = len(contracted) > 0

    sym1 = op1.get("py_symmetry")
    sym2 = op2.get("py_symmetry")

    out_sym = propagate_symmetry(
        sym1,
        frozenset(op1["indices"]),
        sym2,
        frozenset(op2["indices"]),
        out_idx,
        induced_output_symmetry=induced_output_symmetry,
    )

    cost = symmetric_flop_count(
        all_idx,
        is_inner,
        2,
        index_sizes,
        input_symmetries=[sym1, sym2],
        output_symmetry=out_sym,
        output_indices=out_idx,
    )

    return {
        "cost": cost,
        "result_op": {
            "indices": out_idx,
            "py_symmetry": out_sym,
        },
    }


def py_best_cost(operands, final_output, index_sizes, induced_output_symmetry):
    """Find minimum cost across all contraction orderings using Python symmetry primitives."""
    if len(operands) == 1:
        return 0
    best = float("inf")
    for i, j in combinations(range(len(operands)), 2):
        remaining = [op for k, op in enumerate(operands) if k != i and k != j]
        step = py_contract_pair(
            operands[i], operands[j], remaining, final_output, index_sizes,
            induced_output_symmetry,
        )
        rest = py_best_cost(
            [step["result_op"]] + remaining, final_output, index_sizes,
            induced_output_symmetry,
        )
        total = step["cost"] + rest
        if total < best:
            best = total
    return best


# ─── Shared-operand helper ───

def _maybe_share_operands(shapes, rng, share_prob=0.3):
    """With probability share_prob, pick a subset of same-shaped positions
    and make them all reference the same ndarray object."""
    if rng.random() > share_prob:
        return [np.ones(s) for s in shapes]

    # Group positions by shape
    shape_groups: dict = {}
    for idx, shape in enumerate(shapes):
        shape_groups.setdefault(shape, []).append(idx)

    # Find eligible groups (≥ 2 positions with the same shape)
    eligible = [positions for positions in shape_groups.values() if len(positions) >= 2]
    if not eligible:
        return [np.ones(s) for s in shapes]

    # Pick one eligible group and share a single ndarray across those positions
    group = rng.choice(eligible)
    shared_arr = np.ones(shapes[group[0]])
    return [
        shared_arr if idx in group else np.ones(shapes[idx])
        for idx in range(len(shapes))
    ]


# ─── Random einsum generator (extended scope) ───

def generate_random_einsum(rng, difficulty="mixed"):
    if difficulty == "easy":
        num_ops = rng.randint(2, 3)
        num_indices = rng.randint(2, 4)
        max_size = 15
        sym_prob = 0.2
        max_sym_order = 3
        multi_sym_prob = 0.0
    elif difficulty == "medium":
        num_ops = rng.randint(2, 4)
        num_indices = rng.randint(3, 6)
        max_size = 20
        sym_prob = 0.3
        max_sym_order = 3
        multi_sym_prob = 0.1
    elif difficulty == "hard":
        num_ops = rng.randint(3, 5)
        num_indices = rng.randint(4, 8)
        max_size = 30
        sym_prob = 0.4
        max_sym_order = 4
        multi_sym_prob = 0.2
    else:  # mixed
        d = rng.choice(["easy", "medium", "hard"])
        return generate_random_einsum(rng, d)

    all_indices = list(string.ascii_lowercase[:num_indices])
    sizes = {idx: rng.randint(2, max_size) for idx in all_indices}

    # Generate operand index sets
    op_indices = []
    for _ in range(num_ops):
        n_idx = rng.randint(1, min(5, num_indices))
        indices = set(rng.sample(all_indices, n_idx))
        op_indices.append(indices)

    # Ensure every index appears in at least one operand
    used = set()
    for oi in op_indices:
        used |= oi
    for idx in all_indices:
        if idx not in used:
            op_indices[rng.randint(0, num_ops - 1)].add(idx)

    # Determine output
    all_used = set()
    for oi in op_indices:
        all_used |= oi
    idx_counts = {}
    for oi in op_indices:
        for idx in oi:
            idx_counts[idx] = idx_counts.get(idx, 0) + 1

    output_style = rng.choice(["implicit", "explicit_subset", "scalar", "full"])
    if output_style == "implicit":
        out_indices = {idx for idx, c in idx_counts.items() if c == 1}
    elif output_style == "scalar":
        out_indices = set()
    elif output_style == "full":
        out_indices = set(all_used)
    else:
        out_indices = set(rng.sample(list(all_used), rng.randint(0, len(all_used))))

    # Build einsum string
    op_strings = ["".join(sorted(oi)) for oi in op_indices]
    einsum_str = ",".join(op_strings) + "->" + "".join(sorted(out_indices))

    # Random symmetry
    symmetries_py = [None] * num_ops
    sym_inputs = {}

    for i in range(num_ops):
        if len(op_indices[i]) < 2:
            continue

        if rng.random() < sym_prob:
            candidates = sorted(op_indices[i])
            sym_order = rng.randint(2, min(max_sym_order, len(candidates)))
            sym_group = set(rng.sample(candidates, sym_order))

            # All indices in group must have same size
            common_size = sizes[next(iter(sym_group))]
            for idx in sym_group:
                sizes[idx] = common_size

            groups = [frozenset(sym_group)]

            # Possibly add a second non-overlapping group
            if rng.random() < multi_sym_prob:
                remaining = sorted(op_indices[i] - sym_group)
                if len(remaining) >= 2:
                    sym_order2 = rng.randint(2, min(3, len(remaining)))
                    sym_group2 = set(rng.sample(remaining, sym_order2))
                    common_size2 = sizes[next(iter(sym_group2))]
                    for idx in sym_group2:
                        sizes[idx] = common_size2
                    groups.append(frozenset(sym_group2))

            symmetries_py[i] = groups
            sym_inputs[i] = groups

    # Convert per-operand symmetries to IndexSymmetry format (frozenset of tuples)
    index_symmetries = []
    for i, op_str in enumerate(op_strings):
        if i not in sym_inputs:
            index_symmetries.append(None)
        else:
            # Convert frozenset of chars -> frozenset of 1-tuples (IndexSymmetry format)
            groups_tuple = [frozenset({(c,) for c in g}) for g in sym_inputs[i]]
            index_symmetries.append(groups_tuple)

    # Build JS operands (using set-of-chars format for JS brute-force)
    js_operands = []
    for i, op_str in enumerate(op_strings):
        sym = None
        if i in sym_inputs:
            sym = [set(g) for g in sym_inputs[i]]
        js_operands.append({
            "id": f"Op{i}({op_str})",
            "indices": op_indices[i],
            "symmetry": sym,
        })

    # Build Python brute-force operands (using IndexSymmetry format)
    py_operands = []
    for i, op_str in enumerate(op_strings):
        py_operands.append({
            "id": f"Op{i}({op_str})",
            "indices": op_indices[i],
            "py_symmetry": index_symmetries[i],
        })

    shapes = [tuple(sizes[c] for c in op_str) for op_str in op_strings]

    return {
        "einsum": einsum_str,
        "sizes": sizes,
        "shapes": shapes,
        "symmetries_py": symmetries_py,
        "index_symmetries": index_symmetries,
        "js_operands": js_operands,
        "py_operands": py_operands,
        "op_strings": op_strings,
        "out_indices": out_indices,
        "num_ops": num_ops,
        "num_indices": num_indices,
    }


def run_fuzz(num_tests=100_000, seed=42):
    rng = random.Random(seed)
    np_rng = random.Random(seed + 1)
    passed = 0
    failed = 0
    errors = 0
    js_wins = 0
    py_wins = 0
    skipped = 0
    mismatches = []
    shared_count = 0

    t0 = time.time()
    print(f"Running {num_tests:,} random einsum verifications (extended scope)...")
    print(f"  Operands: 2-5, Indices: 2-8, Sizes: 2-30")
    print(f"  Symmetry: S2-S4, multi-group, overlapping")
    print(f"  Outputs: implicit, explicit, scalar, full")
    print(f"  Shared operands: ~30% of cases")
    print()

    for test_num in range(num_tests):
        tc = generate_random_einsum(rng)

        # Skip if brute force would be too slow (>5 operands = 945+ paths)
        path_count = 1
        for k in range(tc["num_ops"], 1, -1):
            path_count *= k * (k - 1) // 2
        if path_count > 1000:
            skipped += 1
            continue

        try:
            # Generate operands (possibly sharing array objects)
            operand_list = _maybe_share_operands(tc["shapes"], np_rng, share_prob=0.3)

            # Detect induced output symmetry from equal-operand positions
            einsum_str = tc["einsum"]
            subscript_parts = einsum_str.split("->")[0].split(",")
            output_chars = einsum_str.split("->")[1] if "->" in einsum_str else ""

            induced = _detect_induced_output_symmetry(
                operands=operand_list,
                subscript_parts=subscript_parts,
                output_chars=output_chars,
                per_op_syms=tc["index_symmetries"],
            )
            has_induced = induced is not None
            if has_induced:
                shared_count += 1

            # Python brute-force: exhaustive search with propagate_symmetry + induced symmetry
            # Build py_operands augmented with induced symmetry contribution at root
            # (induced is threaded through py_best_cost at every step)
            ref_best = py_best_cost(
                tc["py_operands"],
                tc["out_indices"],
                tc["sizes"],
                induced,
            )

            # contract_path with induced_output_symmetry passed explicitly.
            # Always pass input_symmetries as a list (even all-None) so the
            # optimal path optimizer disables its index-set cache, which can
            # give incorrect results when identical index-sets appear at
            # different SSA positions in the search tree.
            py_costs = {}
            for algo in ["optimal", "greedy"]:
                _, inf = contract_path(
                    tc["einsum"], *tc["shapes"],
                    shapes=True, optimize=algo,
                    input_symmetries=tc["index_symmetries"],
                    induced_output_symmetry=induced,
                )
                py_costs[algo] = inf.optimized_cost
            py_best = min(py_costs.values())

            if ref_best == py_best:
                passed += 1
            else:
                failed += 1
                if ref_best < py_best:
                    js_wins += 1
                else:
                    py_wins += 1
                mismatches.append({
                    "test": test_num,
                    "einsum": tc["einsum"],
                    "sizes": tc["sizes"],
                    "ref": ref_best,
                    "py": py_best,
                    "induced": induced,
                    "sym": {i: [set(g) for g in s] for i, s in enumerate(tc["symmetries_py"]) if s},
                    "ops": tc["num_ops"],
                    "idx": tc["num_indices"],
                    "shared": has_induced,
                })
                if failed <= 5:
                    direction = "Ref<Py" if ref_best < py_best else "Ref>Py"
                    print(f"  MISMATCH #{failed}: {tc['einsum']}  Ref={ref_best:,} Py={py_best:,} [{direction}] induced={induced}")
                    print(f"    sizes={tc['sizes']} sym={mismatches[-1]['sym']}")

        except Exception as e:
            errors += 1
            if errors <= 3:
                import traceback
                print(f"  ERROR: {tc['einsum']} - {type(e).__name__}: {e}")
                traceback.print_exc()

        if (test_num + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (test_num + 1) / elapsed
            eta = (num_tests - test_num - 1) / rate
            print(
                f"  [{test_num+1:>6,}/{num_tests:,}]  "
                f"pass={passed:,} fail={failed} err={errors} skip={skipped} shared={shared_count}  "
                f"({rate:.0f}/s, ETA {eta:.0f}s)"
            )

    elapsed = time.time() - t0
    tested = passed + failed
    induced_mismatches = sum(1 for m in mismatches if m["shared"])
    print()
    print("=" * 70)
    print(f"  Total generated:  {num_tests:>10,}")
    print(f"  Skipped (>1000 paths): {skipped:>5,}")
    print(f"  Tested:           {tested:>10,}")
    print(f"  Passed:           {passed:>10,}  ({passed/tested*100:.3f}%)" if tested else "")
    print(f"  Failed:           {failed:>10,}")
    print(f"    Ref cheaper:    {js_wins:>10,}")
    print(f"    Py cheaper:     {py_wins:>10,}")
    print(f"    w/ induced sym: {induced_mismatches:>10,}")
    print(f"  Errors:           {errors:>10,}")
    print(f"  With induced sym: {shared_count:>10,}")
    print(f"  Time:             {elapsed:>10.1f}s ({tested/elapsed:.0f} tests/s)")
    print("=" * 70)

    if mismatches:
        print(f"\nFirst 10 mismatches:")
        for m in mismatches[:10]:
            d = "Ref<Py" if m["ref"] < m["py"] else "Ref>Py"
            print(f"  {m['einsum']:35s} Ref={m['ref']:>10,} Py={m['py']:>10,} [{d}] ops={m['ops']} idx={m['idx']} shared={m['shared']}")

    return failed == 0 and errors == 0


if __name__ == "__main__":
    success = run_fuzz(100_000)
    sys.exit(0 if success else 1)
