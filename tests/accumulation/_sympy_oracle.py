# tests/accumulation/_sympy_oracle.py

from __future__ import annotations

import itertools
from collections.abc import Sequence

MAX_PAIR_TOUCHES = 100_000  # |X| · |G| budget


def _restrict_stabilizer(elements, visible_positions):
    """Restrict elements that preserve V to local-V coords."""
    visible_set = set(visible_positions)
    by_key: dict[tuple[int, ...], tuple[int, ...]] = {}
    for g in elements:
        preserves = all(g.array_form[p] in visible_set for p in visible_positions)
        if not preserves:
            continue
        local_index = {gp: lp for lp, gp in enumerate(visible_positions)}
        local_arr = tuple(local_index[g.array_form[p]] for p in visible_positions)
        by_key[local_arr] = local_arr
    return tuple(by_key.values())


def _apply_perm_to_tuple(tup, perm_array_form):
    out = [0] * len(tup)
    for src in range(len(tup)):
        out[perm_array_form[src]] = tup[src]
    return tuple(out)


def _canonical_under(tup, h_local_arrays):
    if not h_local_arrays:
        return tup
    best = None
    for h in h_local_arrays:
        moved = _apply_perm_to_tuple(tup, h)
        if best is None or moved < best:
            best = moved
    return best


def sympy_brute_force_alpha(*, elements, sizes, visible_positions):
    """Brute-force α via explicit orbit enumeration. Bounded to |X|·|G| ≤ 100k."""
    x_size = 1
    for s in sizes:
        x_size *= s
    pair_touches = x_size * len(elements)
    if pair_touches > MAX_PAIR_TOUCHES:
        raise ValueError(
            f'sympy_brute_force_alpha: input too large '
            f'(|X|·|G| = {pair_touches} > budget {MAX_PAIR_TOUCHES})'
        )

    h_local = _restrict_stabilizer(elements, tuple(visible_positions))
    all_assignments = list(itertools.product(*[range(s) for s in sizes]))

    remaining = set(all_assignments)
    total = 0
    while remaining:
        rep = next(iter(remaining))
        orbit = set()
        for g in elements:
            moved = _apply_perm_to_tuple(rep, g.array_form)
            orbit.add(moved)
        for tup in orbit:
            remaining.discard(tup)
        projected_canonical = set()
        for tup in orbit:
            visible = tuple(tup[p] for p in visible_positions)
            projected_canonical.add(_canonical_under(visible, h_local))
        total += len(projected_canonical)

    return total
