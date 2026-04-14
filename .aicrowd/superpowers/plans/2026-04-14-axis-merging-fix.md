# Axis Merging Fix & Expanded σ-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix false symmetry detection caused by orbit-based axis merging in the subgraph symmetry algorithm, and expand the σ-loop to iterate over per-operand symmetry generators.

**Architecture:** Three changes: (1) remove axis merging in bipartite graph construction so each axis gets its own U-vertex, (2) expand the σ-loop to iterate over per-operand internal symmetry generators alongside identical-operand swap generators, (3) remove the fingerprint fast path which was a workaround that's no longer needed. Apply to both Python (`_subgraph_symmetry.py`) and JS (`algorithm.js`), update tests and docs.

**Tech Stack:** Python (whest library), JavaScript (React/Vite symmetry explorer), pytest

**Spec:** `.aicrowd/superpowers/specs/2026-04-14-transposition-merging-fix-design.md`

---

## File Structure

| File | Role | Change |
|------|------|--------|
| `tests/test_subgraph_symmetry.py` | Python tests | Add regression test, update axis-merging test |
| `src/whest/_opt_einsum/_subgraph_symmetry.py` | Core algorithm | Remove merging, expand σ-loop, remove fast path |
| `docs/visualization/symmetry-explorer/src/engine/algorithm.js` | JS algorithm | Mirror Python changes |
| `docs/explanation/subgraph-symmetry.md` | Algorithm docs | Update descriptions |
| `src/whest/_opt_einsum/_subgraph_symmetry.py` (docstring) | Module docs | Update |

---

## Task 1: Add Regression Test for False S₂ Detection

**Files:**
- Modify: `tests/test_subgraph_symmetry.py`

- [ ] **Step 1: Add the regression test**

Add this test class at the end of the file, after `TestDeclaredGroupNotPromoted`:

```python
class TestC3AxisMergingBug:
    """Regression test: C3 orbit-based merging must not produce false S2."""

    def test_c3_self_contraction_no_false_s2(self):
        """einsum('ijk,jki->ik', T, T) with C3 on T must be trivial.

        Bug: orbit-based merging collapsed {i,j,k} into one U-vertex,
        causing the fingerprint fast path to falsely detect S2{i,k}.
        The result is numerically NOT symmetric: Result[i,k] != Result[k,i].
        """
        from whest._perm_group import PermutationGroup

        n = 4
        c3 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        c3._labels = ("i", "j", "k")

        T = np.ones((n, n, n))
        # Use identical Python objects for the two T operands
        oracle = SubgraphSymmetryOracle(
            operands=[T, T],
            subscript_parts=["ijk", "jki"],
            per_op_groups=[[c3], [c3]],
            output_chars="ik",
        )
        sym = oracle.sym(frozenset({0, 1}))
        # Must NOT detect S2 — result is not symmetric
        if sym.output is not None:
            assert sym.output.order() == 1, (
                f"Expected trivial (order 1), got order {sym.output.order()} — "
                f"C3 orbit merging produced false symmetry"
            )
```

    def test_c3_declared_uses_sigma_loop(self):
        """Declared C3 on T in 'aijk,ab->ijkb' should be found via σ-loop
        generators, not the (now-removed) fingerprint fast path."""
        from whest._perm_group import PermutationGroup

        n = 4
        c3 = PermutationGroup.cyclic(3, axes=(1, 2, 3))
        c3._labels = ("a", "i", "j", "k")  # full operand labels

        T = np.ones((n, n, n, n))
        W = np.ones((n, n))

        oracle = SubgraphSymmetryOracle(
            operands=[T, W],
            subscript_parts=["aijk", "ab"],
            per_op_groups=[[c3], None],
            output_chars="ijkb",
        )
        sym = oracle.sym(frozenset({0, 1}))
        assert sym.output is not None
        assert sym.output.order() == 3, (
            f"Expected C3 (order 3), got order {sym.output.order()}"
        )

- [ ] **Step 2: Run the test to verify it fails (confirms the bug exists)**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest tests/test_subgraph_symmetry.py::TestC3AxisMergingBug -v`

Expected: FAIL — the current code falsely detects S₂.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_subgraph_symmetry.py
git commit -m "test: add regression test for false S2 detection with C3 merging"
```

---

## Task 2: Remove Axis Merging in Python `_build_bipartite`

**Files:**
- Modify: `src/whest/_opt_einsum/_subgraph_symmetry.py:206-230`

- [ ] **Step 1: Remove orbit-based merging**

In `_build_bipartite`, replace the equivalence-class block (lines 210-230) with code that gives each axis its own class unconditionally:

Find this block:

```python
        # Build equivalence classes on the axes of this operand.
        # Each axis (position in sub) starts in its own singleton class.
        # Declared symmetry groups merge axes into classes via orbit
        # analysis on the PermutationGroup.
        class_of_position: dict[int, int] = {k: k for k in range(len(sub))}

        if groups is not None:
            for group in groups:
                if group._labels is None:
                    continue
                for orbit in group.orbits():
                    if len(orbit) < 2:
                        continue
                    chars_in_orbit = {group._labels[i] for i in orbit}
                    positions_in_orbit = [
                        k for k, c in enumerate(sub) if c in chars_in_orbit
                    ]
                    if len(positions_in_orbit) >= 2:
                        canonical = positions_in_orbit[0]
                        for k in positions_in_orbit[1:]:
                            class_of_position[k] = class_of_position[canonical]
```

Replace with:

```python
        # Each axis gets its own U-vertex — no merging.
        # Per-operand symmetry is handled by the σ-loop (Source A generators),
        # not by graph topology.  Orbit-based merging was incorrect for
        # non-S_k groups (e.g. C_k) because it assumed all transpositions
        # within an orbit were present; see test_c3_self_contraction_no_false_s2.
        class_of_position: dict[int, int] = {k: k for k in range(len(sub))}
```

- [ ] **Step 2: Run the full test suite to see what breaks**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest tests/test_subgraph_symmetry.py -v --tb=short 2>&1 | tail -40`

Expected: Some tests will fail — especially `test_fully_symmetric_operand_collapses_to_one_u` (which asserts merging happens) and possibly the declared-group tests (which relied on the fast path).

- [ ] **Step 3: Fix `test_fully_symmetric_operand_collapses_to_one_u`**

This test asserts that a symmetric operand produces 1 U-vertex. With our change, it should produce 2 U-vertices (one per axis). Update the test:

Find in `tests/test_subgraph_symmetry.py`:

```python
    def test_fully_symmetric_operand_collapses_to_one_u(self):
        T = np.zeros((3, 3))
        per_op = [[_sym_group("i", "j")]]  # T symmetric in (i, j)
        g = _build_bipartite(
            operands=[T],
            subscript_parts=["ij"],
            per_op_groups=per_op,
            output_chars="ij",
        )
        # One U vertex for the class {i, j}
        assert len(g.u_vertices) == 1
        assert g.incidence[0] == {"i": 1, "j": 1}
        assert g.u_labels[0] == frozenset({"i", "j"})
```

Replace with:

```python
    def test_symmetric_operand_keeps_separate_u_vertices(self):
        """Each axis gets its own U-vertex even with declared symmetry.

        Per-operand symmetry is handled by the σ-loop, not graph topology.
        """
        T = np.zeros((3, 3))
        per_op = [[_sym_group("i", "j")]]  # T symmetric in (i, j)
        g = _build_bipartite(
            operands=[T],
            subscript_parts=["ij"],
            per_op_groups=per_op,
            output_chars="ij",
        )
        # Two U vertices — one per axis, no merging
        assert len(g.u_vertices) == 2
        assert g.incidence[0] == {"i": 1}
        assert g.incidence[1] == {"j": 1}
        assert g.u_labels[0] == frozenset({"i"})
        assert g.u_labels[1] == frozenset({"j"})
```

- [ ] **Step 4: Commit**

```bash
git add src/whest/_opt_einsum/_subgraph_symmetry.py tests/test_subgraph_symmetry.py
git commit -m "fix: remove orbit-based axis merging in _build_bipartite

Each axis gets its own U-vertex unconditionally. Per-operand symmetry
is handled by the σ-loop (Task 3), not graph topology."
```

---

## Task 3: Expand σ-Loop with Per-Operand Generators (Python)

**Files:**
- Modify: `src/whest/_opt_einsum/_subgraph_symmetry.py:73-131` (`_collect_pi_permutations`)

This is the core fix. The σ-loop must iterate over generators from two sources:
- **Source A:** Per-operand internal symmetry generators (new)
- **Source B:** Identical-operand swap generators (existing, but refactored to use generators only)

- [ ] **Step 1: Rewrite `_collect_pi_permutations`**

Replace the function (lines 73-131) with:

```python
def _collect_pi_permutations(
    graph: "EinsumBipartite",
    sub: "_Subgraph",
    row_order: tuple[int, ...],
    col_of: dict[str, tuple[int, ...]],
    fp_to_labels: dict[tuple[int, ...], set[str]],
) -> tuple[list[Perm], list[Perm]]:
    """Collect π generators from both per-operand and identical-operand sources.

    Iterates over generators (not all elements) of the combined row symmetry
    group, derives π for each, and returns the resulting V-side and W-side
    permutation generators.  Dimino's algorithm in the caller closes these
    into the full group.

    Generator sources:
      A) Per-operand internal symmetry: each generator of a declared group G_i
         induces a row permutation on that operand's U-vertices.
      B) Identical-operand swaps: for each group of k identical operands,
         k-1 adjacent transpositions swap entire operand blocks.
    """
    v_perms: list[Perm] = []
    w_perms: list[Perm] = []
    all_labels = sub.v_labels | sub.w_labels
    v_sorted = tuple(sorted(sub.v_labels))
    w_sorted = tuple(sorted(sub.w_labels))
    v_idx = {lbl: i for i, lbl in enumerate(v_sorted)}
    w_idx = {lbl: i for i, lbl in enumerate(w_sorted)}

    # Build operand → U-vertex index mapping within this subgraph
    op_to_u_indices: dict[int, list[int]] = {}
    for k, u_idx in enumerate(row_order):
        op_idx = graph.u_operand[u_idx]
        op_to_u_indices.setdefault(op_idx, []).append(k)

    # Collect row-permutation generators from both sources
    row_generators: list[tuple[int, ...]] = []

    # ── Source A: per-operand internal symmetry generators ──
    ops_in_subset = set()
    for u_idx in row_order:
        ops_in_subset.add(graph.u_operand[u_idx])

    for op_idx in sorted(ops_in_subset):
        groups = graph.per_op_groups[op_idx]
        if groups is None:
            continue
        u_positions = op_to_u_indices.get(op_idx, [])
        if len(u_positions) < 2:
            continue
        for group in groups:
            if group._labels is None:
                continue
            # Map group's label ordering to U-vertex positions within this operand.
            # group._labels[i] is the label at group position i.
            # We need to find which U-vertex position (in row_order) corresponds
            # to each group position, via the operand's subscript.
            sub_str = graph.operand_subscripts[op_idx]
            # label_to_positions: which positions in the subscript have this label
            # For each group position i, find the axis position in the subscript
            # that corresponds to group._labels[i]
            label_positions: dict[str, list[int]] = {}
            for pos, ch in enumerate(sub_str):
                label_positions.setdefault(ch, []).append(pos)

            # Map group indices to row_order positions
            # group acts on positions 0..degree-1 within the group's label set.
            # group._labels gives the label at each position.
            # We need: for group position g_pos, what's the row_order position?
            # The operand's axes map 1:1 to U-vertices (no merging), so
            # axis k of the operand is u_positions[k].
            axes = group.axes if group.axes is not None else tuple(range(group.degree))

            for gen in group.generators:
                row_perm = list(range(len(row_order)))
                valid = True
                for g_pos in range(gen.degree):
                    target = gen.array_form[g_pos]
                    if g_pos == target:
                        continue
                    # axes[g_pos] is the axis index within the operand
                    src_axis = axes[g_pos]
                    dst_axis = axes[target]
                    if src_axis >= len(u_positions) or dst_axis >= len(u_positions):
                        valid = False
                        break
                    row_perm[u_positions[src_axis]] = u_positions[dst_axis]
                if valid and tuple(row_perm) != tuple(range(len(row_order))):
                    row_generators.append(tuple(row_perm))

    # ── Source B: identical-operand swap generators ──
    # For each group of k identical operands, generate k-1 adjacent transpositions.
    for id_group in sub.id_groups:
        group_list = sorted(id_group)
        for idx in range(len(group_list) - 1):
            op_a = group_list[idx]
            op_b = group_list[idx + 1]
            u_a = op_to_u_indices.get(op_a, [])
            u_b = op_to_u_indices.get(op_b, [])
            if len(u_a) != len(u_b):
                continue  # different number of axes — can't swap
            row_perm = list(range(len(row_order)))
            for ka, kb in zip(u_a, u_b):
                row_perm[ka] = kb
                row_perm[kb] = ka
            if tuple(row_perm) != tuple(range(len(row_order))):
                row_generators.append(tuple(row_perm))

    # ── Derive π for each row-permutation generator ──
    for sigma_row_perm in row_generators:
        # Compute σ(M)'s column fingerprints
        sigma_col_of: dict[str, tuple[int, ...]] = {}
        for label in all_labels:
            sigma_col_of[label] = tuple(
                graph.incidence[row_order[sigma_row_perm[k]]].get(label, 0)
                for k in range(len(row_order))
            )

        # Derive π
        pi = _derive_pi_canonical(
            sigma_col_of, fp_to_labels, sub.v_labels, sub.w_labels
        )
        if pi is None:
            continue

        # Restrict π to V labels — emit Perm if non-identity
        if sub.v_labels and any(pi.get(lbl, lbl) != lbl for lbl in sub.v_labels):
            arr = [v_idx[pi.get(lbl, lbl)] for lbl in v_sorted]
            v_perms.append(Perm(arr))

        # Restrict π to W labels — emit Perm if non-identity
        if sub.w_labels and any(pi.get(lbl, lbl) != lbl for lbl in sub.w_labels):
            arr = [w_idx[pi.get(lbl, lbl)] for lbl in w_sorted]
            w_perms.append(Perm(arr))

    return v_perms, w_perms
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest tests/test_subgraph_symmetry.py -v --tb=short 2>&1 | tail -40`

Check which tests pass and which still fail (the fast-path-dependent tests may still fail until Task 4).

- [ ] **Step 3: Commit**

```bash
git add src/whest/_opt_einsum/_subgraph_symmetry.py
git commit -m "feat: expand σ-loop with per-operand symmetry generators

Source A: per-operand group generators induce row permutations.
Source B: identical-operand adjacent transpositions swap blocks.
Derives π for each generator; Dimino closes the group."
```

---

## Task 4: Remove Fingerprint Fast Path (Python)

**Files:**
- Modify: `src/whest/_opt_einsum/_subgraph_symmetry.py:370-405,438-481`

- [ ] **Step 1: Delete `_find_declared_group_for_labels`**

Delete the entire function (lines 370-405):

```python
def _find_declared_group_for_labels(
    graph: EinsumBipartite,
    subset: frozenset[int],
    target_labels: tuple[str, ...],
) -> PermutationGroup | None:
    ...
    return None
```

- [ ] **Step 2: Remove the fast-path blocks in `_compute_subset_symmetry`**

In `_compute_subset_symmetry`, find the V-side fast path (the `elif` block starting around line 439 with comment "Fingerprint fast path"):

```python
    elif sub.v_labels and len(sub.v_labels) >= 2:
        # Fingerprint fast path: labels that share a fingerprint with at least
        # one other V label are symmetry-related.  Check for a declared group
        # covering those labels before defaulting to S_k.
        fp_groups: dict[tuple[int, ...], list[str]] = {}
        for lbl in sub.v_labels:
            fp_groups.setdefault(col_of[lbl], []).append(lbl)
        v_equiv = sorted(
            lbl for grp in fp_groups.values() if len(grp) >= 2 for lbl in grp
        )
        if len(v_equiv) >= 2:
            declared = _find_declared_group_for_labels(graph, subset, tuple(v_equiv))
            if declared is not None:
                v_group = declared
                v_group._labels = tuple(v_equiv)
            else:
                v_group = PermutationGroup.symmetric(
                    len(v_equiv), axes=tuple(range(len(v_equiv)))
                )
                v_group._labels = tuple(v_equiv)
```

Delete this entire `elif` block. Do the same for the W-side equivalent block below it.

The resulting `_compute_subset_symmetry` should only build groups from σ-loop results — no fallback.

- [ ] **Step 3: Update module docstring**

Replace lines 1-16:

```python
"""Subset-keyed subgraph symmetry detection for einsum intermediates.

One oracle per contract_path call. Given the original operand list,
subscript parts, per-operand declared symmetries, and output subscript,
builds a bipartite graph once and exposes ``.sym(subset)`` which returns
a ``SubsetSymmetry`` with ``.output`` (V-side) and ``.inner`` (W-side)
symmetries, computed lazily on first access and cached.

The detection algorithm derives the induced column permutation π for
each operand permutation σ via column-fingerprint hash lookup, then
collects the resulting Permutation objects to build a PermutationGroup
directly. A fingerprint fast-path detects S_k symmetry without running
the σ-loop when all labels on a side share the same column fingerprint.

See docs/explanation/subgraph-symmetry.md for the algorithm walkthrough.
"""
```

Replace with:

```python
"""Subset-keyed subgraph symmetry detection for einsum intermediates.

One oracle per contract_path call. Given the original operand list,
subscript parts, per-operand declared symmetries, and output subscript,
builds a bipartite graph once and exposes ``.sym(subset)`` which returns
a ``SubsetSymmetry`` with ``.output`` (V-side) and ``.inner`` (W-side)
symmetries, computed lazily on first access and cached.

Each axis of each operand gets its own U-vertex in the bipartite graph
(no axis merging). The σ-loop iterates over generators from two sources:
(A) per-operand internal symmetry generators and (B) identical-operand
swap generators. For each generator, the induced column permutation π is
derived via fingerprint hash lookup. Collected π's are used as generators
for a PermutationGroup built via Dimino's algorithm.

See docs/explanation/subgraph-symmetry.md for the algorithm walkthrough.
"""
```

- [ ] **Step 4: Also remove the now-unused `_enumerate_id_group_permutations` function**

This function (lines 486-512) enumerated all permutations of identical operands. It's no longer called since the σ-loop now uses adjacent transposition generators instead. Delete it.

- [ ] **Step 5: Remove the unused import of `_lift_operand_perm_to_u`** if it's no longer called (the new σ-loop builds row permutations directly). Check if it's still referenced anywhere before deleting.

- [ ] **Step 6: Run the full test suite**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest tests/test_subgraph_symmetry.py -v --tb=short`

Expected: All tests pass, including the new regression test from Task 1.

- [ ] **Step 7: Run the broader einsum symmetry tests**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest tests/test_opt_einsum_symmetry.py tests/test_symmetric_einsum.py -v --tb=short 2>&1 | tail -30`

- [ ] **Step 8: Commit**

```bash
git add src/whest/_opt_einsum/_subgraph_symmetry.py
git commit -m "fix: remove fingerprint fast path and unused functions

The expanded σ-loop (per-operand generators) makes the fast path
unnecessary. Removes _find_declared_group_for_labels,
_enumerate_id_group_permutations, and _lift_operand_perm_to_u."
```

---

## Task 5: Mirror Changes in JS Algorithm Engine

**Files:**
- Modify: `docs/visualization/symmetry-explorer/src/engine/algorithm.js`

Apply the same three changes to the JS implementation.

- [ ] **Step 1: Remove axis merging in `buildBipartite`**

Find this block (lines 32-46):

```javascript
    const opSym = Array.isArray(perOpSymmetry) ? perOpSymmetry[opIdx] : perOpSymmetry;
    if (opSym === 'symmetric' || opSym === 'cyclic' || opSym === 'dihedral') {
      // All axes in one class (group acts on all axes of this operand)
      for (let k = 1; k < sub.length; k++) classOf[k] = 0;
    } else if (opSym && typeof opSym === 'object' && opSym.axes) {
      // Partial symmetry: only the specified axes collapse into one class
      const symAxes = opSym.axes;
      if (symAxes.length >= 2) {
        const target = symAxes[0];
        for (let j = 1; j < symAxes.length; j++) {
          classOf[symAxes[j]] = target;
        }
      }
    }
```

Replace with:

```javascript
    // No axis merging — each axis gets its own U-vertex.
    // Per-operand symmetry is handled by the σ-loop generators.
```

(Keep the `classOf` initialization above — each axis stays in its own class.)

- [ ] **Step 2: Expand `runSigmaLoop` to accept the example and iterate over per-operand generators**

The function signature needs to accept the `example` (to access `perOpSymmetry`). Add per-operand generator iteration (Source A) and refactor identical-operand iteration to use adjacent swap generators (Source B).

Update the function to:
1. Accept `example` parameter
2. Build row-permutation generators from Source A (per-operand symmetry) using `declaredSymGenerators` and the operand's axes
3. Build row-permutation generators from Source B (identical-operand adjacent transpositions)
4. For each generator, compute σ(M) column fingerprints and derive π

The key changes mirror the Python `_collect_pi_permutations` rewrite from Task 3.

- [ ] **Step 3: Remove the fingerprint fast path in `buildGroup`**

Delete the `findDeclaredGroupForLabels` function and the fast-path blocks (lines 431-496) that fall back to S_k when no σ-loop generators are found.

- [ ] **Step 4: Update `App.jsx` call to pass example to `runSigmaLoop`**

In `App.jsx`, the pipeline calls `runSigmaLoop(graph, matrixData)`. Update to `runSigmaLoop(graph, matrixData, normalizedExample)` to pass the example with `perOpSymmetry`.

- [ ] **Step 5: Verify the explorer works**

Start the dev server and test:
- Gram matrix → S₂
- Directed triangle → C₃
- Declared C₃ → C₃ (not S₃)
- Undirected 4-cycle → D₄

- [ ] **Step 6: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/engine/algorithm.js docs/visualization/symmetry-explorer/src/App.jsx
git commit -m "fix(explorer): remove axis merging, expand σ-loop with per-operand generators

Mirrors Python fix: each axis gets own U-vertex, σ-loop iterates over
per-operand symmetry generators + identical-operand swap generators.
Removes findDeclaredGroupForLabels and fingerprint fast path."
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `docs/explanation/subgraph-symmetry.md`

- [ ] **Step 1: Update the bipartite graph section**

Find the paragraph describing U-vertex construction (around line 31):

```
**Left vertices (U):** One U-vertex per equivalence class of axes within each
```

Update to explain that each axis gets its own U-vertex, and per-operand symmetry is handled by the σ-loop. Remove mentions of "equivalence class" merging based on declared symmetry.

- [ ] **Step 2: Update the σ-loop section**

Find the "σ loop" section (around line 198). Update to describe the two generator sources (Source A: per-operand internal, Source B: identical-operand swaps). Explain that the loop iterates over generators, not all elements, and Dimino closes the group.

- [ ] **Step 3: Update the fast-path section**

Find the "Fast path: fingerprint equivalences" section (around line 186). Either remove it or convert it to a historical note explaining why it was removed (orbit-based merging + fast path produced false positives for non-S_k groups like C₃).

- [ ] **Step 4: Add the C₃ bug as a cautionary note**

Add a brief section explaining the `einsum('ijk,jki->ik', T, T)` with C₃ case, showing why orbit-based merging was wrong and how the generator-based approach fixes it.

- [ ] **Step 5: Commit**

```bash
git add docs/explanation/subgraph-symmetry.md
git commit -m "docs: update subgraph symmetry walkthrough for expanded σ-loop

Remove axis-merging description, update σ-loop to describe per-operand
generators, remove fast-path section, add C3 bug cautionary note."
```

---

## Task 7: Run Full Test Suite and Verify

**Files:** None (verification only)

- [ ] **Step 1: Run the full subgraph symmetry tests**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest tests/test_subgraph_symmetry.py -v`

All tests must pass.

- [ ] **Step 2: Run the broader symmetry-related tests**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest tests/test_opt_einsum_symmetry.py tests/test_symmetric_einsum.py tests/test_dp_symmetry.py tests/test_random_greedy_symmetry.py -v --tb=short 2>&1 | tail -30`

- [ ] **Step 3: Run the full test suite**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run pytest --cov=whest -x --tb=short 2>&1 | tail -20`

All tests must pass with coverage ≥ 90%.

- [ ] **Step 4: Verify the concrete bug is fixed**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run python3 -c "
import whest as we
import numpy as np
np.random.seed(42)
n = 4

grp = we.PermutationGroup.cyclic(3, axes=(0,1,2))
data = np.random.randn(n, n, n)
sym_data = sum(np.transpose(data, g.array_form) for g in grp.elements()) / grp.order()
T = we.as_symmetric(sym_data, symmetry=grp)

path, info = we.einsum_path('ijk,jki->ik', T, T)
print(info)
for step in info.steps:
    og = step.output_group
    print(f'output_group: {og}, order={og.order() if og else \"none\"}')
"`

Expected: trivial (no output symmetry), NOT S₂.

- [ ] **Step 5: Verify existing examples still work**

Run: `cd /Users/mohanty/work/AIcrowd/challenges/alignment-research-center/whest && uv run python3 -c "
import whest as we
import numpy as np
n = 5

# Gram matrix
X = we.random.randn(n,n)
_, info = we.einsum_path('ia,ib->ab', X, X)
assert info.steps[0].output_group.order() == 2, 'Gram: expected S2'

# Directed triangle
A = we.random.randn(n,n)
_, info = we.einsum_path('ij,jk,ki->ijk', A, A, A)
assert info.steps[-1].output_group.order() == 3, 'Triangle: expected C3'

# Declared C3
grp = we.PermutationGroup.cyclic(3, axes=(1,2,3))
data = np.random.randn(n,n,n,n)
total = data.copy() * 0
for g in grp.elements():
    perm = list(range(4))
    for i in range(3):
        perm[grp.axes[i]] = grp.axes[g.array_form[i]]
    total = total + np.transpose(data, perm)
total = total / grp.order()
T = we.as_symmetric(total, symmetry=grp)
W = we.random.randn(n,n)
_, info = we.einsum_path('aijk,ab->ijkb', T, W)
step = info.steps[-1] if len(info.steps) > 1 else info.steps[0]
assert step.output_group is not None and step.output_group.order() == 3, f'Declared C3: expected order 3, got {step.output_group.order() if step.output_group else \"none\"}'

print('All assertions passed!')
"`

- [ ] **Step 6: Commit final state**

```bash
git add -A
git commit -m "fix: complete axis-merging fix — all tests pass"
```
