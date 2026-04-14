# Stabilizer-Based Symmetry Propagation for Slices and Reductions

## Problem

`propagate_symmetry_slice` and `propagate_symmetry_reduce` operate on
`list[tuple[int, ...]]` — bare axis tuples that implicitly assume the full
symmetric group S_k. Now that the codebase supports general permutation groups
(C_k, D_k, arbitrary generators), these functions produce incorrect results
for non-S_k groups.

**Example of the bug:** A tensor with C_3 on axes {0,1,2}. Slicing axis 2
with an integer leaves axes {0,1}. The current code reports S_2 on {0,1}
(swap allowed), but C_3 never contained a (0↔1) transposition — no symmetry
should survive.

## Core Insight

When axes are removed from a tensor (by slicing or reduction), the output
symmetry is the **stabilizer subgroup** of the input group — the subgroup of
elements that don't mix removed and kept axes. The type of stabilizer depends
on the operation:

- **Slicing** (integer index pins an axis to a specific value):
  **pointwise stabilizer** — each removed axis must map to itself individually.
- **Reduction** (e.g. `sum(axis=k)` eliminates an axis by aggregation):
  **setwise stabilizer** — removed axes must map among themselves as a set
  (since summation treats all positions equivalently, swapping two summed
  axes is valid).

After computing the stabilizer, the surviving permutations are **projected**
(re-indexed) to the new, smaller axis numbering.

### Worked Examples

**C_4 on {0,1,2,3}, slice axes {1,3} (integer indexing):**

| Element | 0→ | 1→ | 2→ | 3→ | Fixes 1 AND 3? |
|---------|----|----|----|----|-----------------|
| id      | 0  | 1  | 2  | 3  | yes             |
| rot     | 1  | 2  | 3  | 0  | no              |
| rot²    | 2  | 3  | 0  | 1  | no              |
| rot³    | 3  | 0  | 1  | 2  | no              |

Pointwise stabilizer = {id}. No symmetry survives.

**C_4 on {0,1,2,3}, reduce axes {1,3} (sum):**

| Element | {1,3}→{1,3}? | {0,2}→{0,2}? |
|---------|--------------|--------------|
| id      | yes          | yes          |
| rot     | no           | no           |
| rot²    | yes (1→3,3→1)| yes (0→2,2→0)|
| rot³    | no           | no           |

Setwise stabilizer = {id, rot²}. After projecting to kept axes {0,2} →
re-index to {0,1}: output has C_2 (the swap 0↔1).

**S_3 on {0,1,2}, slice axis 2:**

Pointwise stabilizer of {2} = elements where 2→2:
{id, (0↔1)}. Output has S_2 on {0,1}. Matches current behavior.

## Design

### 1. New Methods on `PermutationGroup`

**`pointwise_stabilizer(fixed: set[int]) -> PermutationGroup`**
- Returns subgroup where every point in `fixed` maps to itself
- Algorithm: filter `self.elements()`, build new group from surviving elements
- Used by slice propagation

**`setwise_stabilizer(subset: set[int]) -> PermutationGroup`**
- Returns subgroup where `subset` maps to `subset` as a set
- Algorithm: filter `self.elements()` by `{g(x) for x in subset} == subset`
- Used by reduce propagation

**`restrict(kept: tuple[int, ...]) -> PermutationGroup`**
- Precondition: group already stabilizes `kept` (setwise)
- Projects each permutation to only the `kept` positions, re-indexes to 0..len(kept)-1
- Updates `axes`: selects the corresponding entries from the original `axes` tuple.
  E.g. original `axes=(2,5,7)`, `kept=(0,2)` (group-local) → new `axes=(2,7)`.
  The caller (propagation function) then remaps these tensor-level axes through
  the old→new axis mapping to get final output axis numbers.
- Returns group of smaller degree

All three use the existing Dimino-based enumeration (groups are small).

### 2. Updated `propagate_symmetry_slice`

```
propagate_symmetry_slice(groups: list[PermutationGroup], shape, key)
    -> list[PermutationGroup] | None
```

Algorithm:
1. Expand/normalize key (same as today)
2. Classify each original axis: "removed" (int), "resized" (sub-range), "untouched" (full slice)
3. Build old→new axis mapping (removed dims get None, account for newaxis)
4. For each group G:
   a. Map group's `axes` to the dim_actions classification
   b. Compute pointwise stabilizer of removed axes (group-local indices)
   c. Among surviving axes, if sizes differ (some resized, some not),
      compute setwise stabilizer of equal-size partitions
   d. `restrict()` to kept axes, remap via old→new
   e. Keep if degree >= 2
5. Return surviving groups or None

### 3. Updated `propagate_symmetry_reduce`

```
propagate_symmetry_reduce(groups: list[PermutationGroup], ndim, axis, keepdims)
    -> list[PermutationGroup] | None
```

Algorithm:
1. Normalize `axis` to absolute indices. If None, return None.
2. For each group G:
   a. Let `reduced` = group-local indices of axes being reduced
   b. Compute setwise stabilizer of `reduced`
   c. `restrict()` to kept (non-reduced) axes
   d. If keepdims: reduced axes stay at size 1 — skip `restrict()` (no axes
      removed), but the setwise stabilizer from step (b) already ensures
      only valid permutations survive. No re-indexing needed since axis
      positions are unchanged.
   e. If not keepdims: remap group axes through old→new axis mapping
   f. Keep if degree >= 2
3. Return surviving groups or None

### 4. Updated `intersect_symmetry`

```
intersect_symmetry(groups_a, groups_b, shape_a, shape_b, output_shape)
    -> list[PermutationGroup] | None
```

For binary ops, the output group is the intersection of both operands' groups:
- Align each operand's groups to output shape (broadcasting)
- For groups acting on the same axis set: intersect element sets, build new group
- Drop groups that don't appear in both operands

### 5. Callers

`SymmetricTensor.__getitem__` and ufunc hooks pass `self._symmetry_groups`
(list of `PermutationGroup`) to the updated propagation functions. Results
are stored directly as `_symmetry_groups` on the output tensor.

`SymmetryInfo` continues to accept `symmetric_axes` tuples as a convenience
and auto-builds S_k groups in `__post_init__`.

## Files to Change

| File | Changes |
|------|---------|
| `src/whest/_perm_group.py` | Add `pointwise_stabilizer`, `setwise_stabilizer`, `restrict` |
| `src/whest/_symmetric.py` | Rewrite `propagate_symmetry_slice`, `propagate_symmetry_reduce`, `intersect_symmetry` to take groups; update `SymmetricTensor.__getitem__` and ufunc callers |
| `tests/test_symmetric_coverage.py` | Add tests for C_k/D_k stabilizer cases (the worked examples above) |
| `tests/test_perm_group.py` | Unit tests for new PermutationGroup methods |

## Non-Goals

- Schreier-Sims or other scalable algorithms (Dimino is sufficient for our group sizes)
- Changes to the einsum σ-loop (already handles general groups correctly)
- Changes to `SymmetryInfo.__post_init__` (already builds groups from tuples)
