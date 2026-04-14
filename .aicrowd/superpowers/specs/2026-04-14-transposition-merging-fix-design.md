# Fix: Remove Axis Merging & Expand σ-Loop with Per-Operand Generators

## Problem

The bipartite graph construction in the subgraph symmetry algorithm uses **orbit-based merging** to collapse per-operand symmetric axes into a single U-vertex. This is only correct when the group is the full symmetric group S_k (where all transpositions are present). For non-S_k groups (C_k, D_k, custom), orbit-based merging is too aggressive and produces false symmetry detections.

### Concrete Bug

`einsum('ijk,jki->ik', T, T)` where T has C₃ symmetry on axes (0,1,2):

- **Current code reports:** S₂{i,k} with speedup 1.6x
- **Reality:** The result tensor is NOT symmetric — `Result[0,1] ≠ Result[1,0]`
- **Without declared C₃:** correctly reports trivial (no symmetry)

C₃ = {e, (0 1 2), (0 2 1)} has orbit {0,1,2} but contains no transpositions. The orbit-based merge collapses {i,j,k} into one U-vertex, causing the fingerprint fast path to falsely detect S₂ on output labels {i,k}.

### Root Cause

Two compounding issues:

1. **Axis merging criterion is wrong.** Axes i,j should only be merged when transposition (i j) ∈ G. Orbit membership is necessary but not sufficient.

2. **σ-loop is incomplete.** It only enumerates identical-operand swap permutations. Per-operand internal symmetry permutations are not iterated, so the σ-loop finds nothing for non-identical operands with declared symmetry. The fingerprint fast path was added as a workaround, but it bypasses the σ-loop's correctness checks.

## Design

### Change 1: No Axis Merging in Bipartite Graph

Remove all orbit-based and symmetry-based axis merging from `_build_bipartite`. Every axis gets its own U-vertex unconditionally.

A rank-4 operand with subscript "aijk" always produces 4 U-vertices: {a}, {i}, {j}, {k}, regardless of declared symmetry.

Per-operand symmetry metadata is still stored on the graph object for use by the σ-loop, but it no longer affects graph topology.

### Change 2: Generator-Based Expanded σ-Loop

Replace the current σ-loop (which only enumerates identical-operand permutations) with a generator-based loop that iterates over generators from two sources:

**Source A — Per-operand internal symmetry generators:** For each operand with a declared group G_i, each generator g ∈ generators(G_i) induces a row permutation on that operand's U-vertices. For example, if operand T has subscript "aijk" and C₃ on axes (1,2,3), the C₃ generator (0→1, 1→2, 2→0) permutes the U-vertices at positions 1,2,3 of that operand while fixing position 0.

**Source B — Identical-operand swap generators:** For each identical-operand group of size k, generate adjacent transpositions that swap entire operand blocks (all U-vertices of operand i with all U-vertices of operand i+1). For k operands, that's k-1 generators.

**The loop:**
1. Collect all generators from sources A and B
2. For each generator, lift to U-row permutation → compute σ(M) → derive π via fingerprint matching
3. Collect all valid π's as generators of the output group
4. Run Dimino on the π generators to get the full group

This is mathematically equivalent to enumerating the full Cartesian product of group elements (as the reviewer suggested), but efficient because we only process generators (typically 1-3 per source) and let Dimino close the group.

### Change 3: Remove Fingerprint Fast Path

The fingerprint fast path in `_compute_subset_symmetry` / `buildGroup` is removed entirely. It was a workaround for the case where the σ-loop found nothing (non-identical operands with declared symmetry). With the expanded σ-loop processing per-operand generators, the fast path is unnecessary — those generators naturally produce the correct π's.

Functions removed:
- Python: `_find_declared_group_for_labels`
- JS: `findDeclaredGroupForLabels`

The fast-path code blocks in `_compute_subset_symmetry` (Python) and `buildGroup` (JS) that fall back to S_k when no σ-loop generators are found are also removed.

## Files Changed

### Python (`src/whest/_opt_einsum/_subgraph_symmetry.py`)

| Function | Change |
|----------|--------|
| `_build_bipartite` | Remove orbit-based merging (lines 216-230). Each axis → own U-vertex. |
| `_collect_pi_permutations` | Add iteration over per-operand group generators. Accept `per_op_groups` from the graph. |
| `_lift_operand_perm_to_u` | Adjust to handle per-operand internal generators (not just identical-operand swaps). For a per-operand generator, the lift permutes U-vertices within that operand's block using the generator's array form mapped to global U-vertex indices. For an identical-operand swap, the lift swaps entire blocks (same as current). |
| `_compute_subset_symmetry` | Remove fingerprint fast path (lines 439-481). Only use σ-loop results. |
| `_find_declared_group_for_labels` | Delete entirely. |

### JS (`docs/visualization/symmetry-explorer/src/engine/algorithm.js`)

| Function | Change |
|----------|--------|
| `buildBipartite` | Remove axis merging (lines 34-46). Each axis → own U-vertex. |
| `runSigmaLoop` | Add per-operand generator iteration. Accept `perOpSymmetry` or the example to access declared groups. Build generators from symmetry type + axes, iterate them alongside identical-operand generators. |
| `buildGroup` | Remove fingerprint fast path (lines 431-496). |
| `findDeclaredGroupForLabels` | Delete entirely. |
| `declaredSymGenerators` | Keep — still used to build generators from symmetry type names. |

### JS (`docs/visualization/symmetry-explorer/src/engine/permutation.js`)

- Verify `Permutation` class supports the operations needed by the expanded σ-loop (array-form construction, composition). Add if missing.

## Tests

### Primary regression test

```python
def test_c3_no_false_s2():
    """C3 on T should NOT produce S2 on output of einsum('ijk,jki->ik', T, T)."""
    n = 4
    grp = PermutationGroup.cyclic(3, axes=(0,1,2))
    data = np.random.randn(n, n, n)
    sym_data = sum(np.transpose(data, g.array_form) for g in grp.elements()) / grp.order()
    T = as_symmetric(sym_data, symmetry=grp)
    
    path, info = einsum_path('ijk,jki->ik', T, T)
    # Must be trivial — no output symmetry
    assert info.steps[0].output_group is None or info.steps[0].output_group.order() == 1
    # Verify numerically
    result = einsum('ijk,jki->ik', T, T)
    assert not np.allclose(result, result.T)
```

### Existing tests that must still pass

All existing preset examples must produce the same group detections:

| Expression | Expected Group |
|-----------|----------------|
| `einsum('ia,ib->ab', X, X)` | S₂{a,b} |
| `einsum('ia,ib,ic->abc', X, X, X)` | S₃{a,b,c} |
| `einsum('ab,cd->abcd', X, X)` | S₂{a,c}×S₂{b,d} |
| `einsum('ij,jk,ki->ijk', A, A, A)` | C₃{i,j,k} |
| `einsum('ij,jk,kl,li->ijkl', S, S, S, S)` (S symmetric) | D₄{i,j,k,l} |
| `einsum('ij,ji->', A, A)` | W: S₂{i,j} |
| `einsum('aijk,ab->ijkb', T, W)` (T has C₃(1,2,3)) | C₃{i,j,k} |
| `einsum('aijkl,ab->ijklb', T, W)` (T has D₄(1,2,3,4)) | D₄{i,j,k,l} |
| `einsum('ij,jk->ik', A, A)` | trivial |

### New test: per-operand generators in σ-loop

```python
def test_c3_declared_uses_sigma_loop():
    """Declared C3 should be found via σ-loop generators, not fast path."""
    n = 4
    grp = PermutationGroup.cyclic(3, axes=(1,2,3))
    data = np.random.randn(n, n, n, n)
    sym_data = sum(np.transpose(data, [0] + [grp.axes[g.array_form[i]] for i in range(3)])
                   for g in grp.elements()) / grp.order()  # Reynolds on axes 1,2,3
    T = as_symmetric(sym_data, symmetry=grp)
    W = np.random.randn(n, n)

    path, info = einsum_path('aijk,ab->ijkb', T, W)
    step = info.steps[0] if len(info.steps) == 1 else info.steps[-1]
    assert step.output_group is not None
    assert step.output_group.order() == 3  # C3, not S3 (order 6)
```

## Documentation Updates

### `docs/explanation/subgraph-symmetry.md` (algorithm walkthrough)

This is the primary explanation of the algorithm. Changes needed:

1. **Bipartite graph section:** Update the description of U-vertex construction. Remove the explanation of orbit-based axis merging ("axes identified by that operand's declared symmetry partition"). Replace with: each axis always gets its own U-vertex; per-operand symmetry is handled by the σ-loop, not by graph topology.

2. **σ-loop section:** Expand to describe the two sources of generators (per-operand internal symmetry + identical-operand swaps). Remove any description of the fingerprint fast path as the primary detection mechanism for declared symmetries. The fast path is no longer used.

3. **Worked examples:** Update the Declared C₃ walkthrough (if present) to show the fully expanded graph with separate U-vertices for each axis, and show how the C₃ generator produces a valid π through the σ-loop.

4. **Add the bug example:** Include `einsum('ijk,jki->ik', T, T)` with C₃ as a cautionary note explaining why orbit-based merging was incorrect.

### `docs/explanation/symmetry-explorer.md` (visualization docs)

Update the description of the bipartite graph step to reflect that per-operand symmetry no longer merges axes. The visualization will now show one U-vertex per axis even for symmetric operands, with edges colored by variable.

### `src/whest/_opt_einsum/_subgraph_symmetry.py` (module docstring)

Update the module-level docstring (lines 1-16) to:
- Remove references to the "fingerprint fast-path" as a separate detection mechanism
- Describe the expanded σ-loop with per-operand generators
- Note that axis merging is no longer performed

### `docs/how-to/exploit-symmetry.md` (practical guide)

Review for any mentions of axis merging or the fast path that need updating. The user-facing API (`as_symmetric`, `PermutationGroup`) doesn't change, but any internal algorithm descriptions should be consistent.

## What Does NOT Change

- `PermutationGroup`, `Permutation`, `Cycle` classes
- Dimino's algorithm
- Burnside counting and cost reduction
- The explorer UI (ExampleChooser, variable cards, expression panel)
- `as_symmetric` API
- The 7-step pipeline visualization structure
- The π derivation logic (fingerprint matching on columns)
