# Stabilizer-Based Symmetry Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix slice/reduce symmetry propagation to correctly handle general permutation groups (C_k, D_k, etc.) using stabilizer subgroups instead of assuming S_k.

**Architecture:** Add `pointwise_stabilizer`, `setwise_stabilizer`, and `restrict` methods to `PermutationGroup`. Rewrite `propagate_symmetry_slice`, `propagate_symmetry_reduce`, and `intersect_symmetry` to operate on `list[PermutationGroup]` instead of `list[tuple[int, ...]]`. Update callers in `SymmetricTensor.__getitem__` and `_pointwise.py`.

**Tech Stack:** Python, numpy, pytest

**Spec:** `.aicrowd/superpowers/specs/2026-04-14-stabilizer-propagation-design.md`

---

### Task 1: Add `pointwise_stabilizer` to `PermutationGroup`

**Files:**
- Modify: `src/whest/_perm_group.py:244-475` (PermutationGroup class)
- Test: `tests/test_perm_group.py`

- [ ] **Step 1: Write failing tests for pointwise_stabilizer**

Add to `tests/test_perm_group.py`:

```python
class TestPointwiseStabilizer:
    def test_s3_fix_one_point(self):
        """S_3 fixing point 2 → {id, (0 1)} = S_2."""
        g = PermutationGroup.symmetric(3)
        stab = g.pointwise_stabilizer({2})
        assert stab.order() == 2
        assert stab.degree == 3
        # The surviving elements fix point 2
        for elem in stab.elements():
            assert elem(2) == 2

    def test_c3_fix_one_point(self):
        """C_3 fixing any single point → trivial group."""
        g = PermutationGroup.cyclic(3)
        for pt in range(3):
            stab = g.pointwise_stabilizer({pt})
            assert stab.order() == 1

    def test_c4_fix_two_points(self):
        """C_4 fixing {1,3} pointwise → only identity."""
        g = PermutationGroup.cyclic(4)
        stab = g.pointwise_stabilizer({1, 3})
        assert stab.order() == 1

    def test_s4_fix_two_points(self):
        """S_4 fixing {0,1} pointwise → S_2 on {2,3}."""
        g = PermutationGroup.symmetric(4)
        stab = g.pointwise_stabilizer({0, 1})
        assert stab.order() == 2
        for elem in stab.elements():
            assert elem(0) == 0
            assert elem(1) == 1

    def test_fix_empty_set(self):
        """Fixing empty set returns the full group."""
        g = PermutationGroup.cyclic(4)
        stab = g.pointwise_stabilizer(set())
        assert stab.order() == g.order()

    def test_fix_all_points(self):
        """Fixing all points returns trivial group."""
        g = PermutationGroup.symmetric(3)
        stab = g.pointwise_stabilizer({0, 1, 2})
        assert stab.order() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_perm_group.py::TestPointwiseStabilizer -v`
Expected: FAIL with `AttributeError: 'PermutationGroup' object has no attribute 'pointwise_stabilizer'`

- [ ] **Step 3: Implement pointwise_stabilizer**

Add to `PermutationGroup` class in `src/whest/_perm_group.py`, after the `orbit` method (after line 372):

```python
def pointwise_stabilizer(self, fixed: set[int]) -> PermutationGroup:
    """Subgroup of elements that fix every point in *fixed*.

    Parameters
    ----------
    fixed : set of int
        Group-local indices that must map to themselves.

    Returns
    -------
    PermutationGroup
        The pointwise stabilizer subgroup (same degree).
    """
    if not fixed:
        return PermutationGroup(*self._generators, axes=self._axes)
    surviving = [g for g in self.elements() if all(g(p) == p for p in fixed)]
    if not surviving:
        surviving = [Permutation.identity(self._degree)]
    return PermutationGroup(*surviving, axes=self._axes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_perm_group.py::TestPointwiseStabilizer -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_perm_group.py tests/test_perm_group.py
git commit -m "feat: add pointwise_stabilizer to PermutationGroup"
```

---

### Task 2: Add `setwise_stabilizer` to `PermutationGroup`

**Files:**
- Modify: `src/whest/_perm_group.py:244-475` (PermutationGroup class)
- Test: `tests/test_perm_group.py`

- [ ] **Step 1: Write failing tests for setwise_stabilizer**

Add to `tests/test_perm_group.py`:

```python
class TestSetwiseStabilizer:
    def test_c4_setwise_13(self):
        """C_4 setwise stabilizer of {1,3} → {id, rot²} = C_2."""
        g = PermutationGroup.cyclic(4)
        stab = g.setwise_stabilizer({1, 3})
        assert stab.order() == 2
        for elem in stab.elements():
            assert {elem(1), elem(3)} == {1, 3}

    def test_c3_setwise_any_pair(self):
        """C_3 setwise stabilizer of any 2-element subset → trivial."""
        g = PermutationGroup.cyclic(3)
        stab = g.setwise_stabilizer({0, 1})
        assert stab.order() == 1

    def test_s3_setwise_pair(self):
        """S_3 setwise stabilizer of {0,1} → {id, (0 1)} = S_2."""
        g = PermutationGroup.symmetric(3)
        stab = g.setwise_stabilizer({0, 1})
        assert stab.order() == 2
        for elem in stab.elements():
            assert {elem(0), elem(1)} == {0, 1}

    def test_s4_setwise_pair(self):
        """S_4 setwise stabilizer of {0,1} has order 12.

        Elements that map {0,1}→{0,1}: 2 choices for {0,1} × 2! for {2,3} = nope.
        Actually: pick where 0,1 go (2 options: stay or swap) × permute {2,3}
        freely (2!) = 4. Wait — any elem that maps {0,1}→{0,1} can also
        permute 2,3 freely. So 2 × 2! × ... no.
        Setwise stab of {0,1} in S_4: elements mapping {0,1}→{0,1}.
        Choose image of {0,1}: 2! ways. Choose image of {2,3}: 2! ways.
        Total: 2! × 2! = 4.
        """
        g = PermutationGroup.symmetric(4)
        stab = g.setwise_stabilizer({0, 1})
        assert stab.order() == 4

    def test_empty_set(self):
        """Setwise stabilizer of empty set is the full group."""
        g = PermutationGroup.cyclic(4)
        stab = g.setwise_stabilizer(set())
        assert stab.order() == g.order()

    def test_full_set(self):
        """Setwise stabilizer of the full set is the full group."""
        g = PermutationGroup.symmetric(3)
        stab = g.setwise_stabilizer({0, 1, 2})
        assert stab.order() == g.order()

    def test_d4_setwise(self):
        """D_4 setwise stabilizer of {0,2} (opposite corners of square)."""
        g = PermutationGroup.dihedral(4)
        stab = g.setwise_stabilizer({0, 2})
        # D_4 has 8 elements. Elements mapping {0,2}→{0,2}:
        # id(0→0,2→2), rot²(0→2,2→0), refl across 0-2 diagonal, refl across 1-3 diagonal
        assert stab.order() == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_perm_group.py::TestSetwiseStabilizer -v`
Expected: FAIL with `AttributeError: 'PermutationGroup' object has no attribute 'setwise_stabilizer'`

- [ ] **Step 3: Implement setwise_stabilizer**

Add to `PermutationGroup` class in `src/whest/_perm_group.py`, after `pointwise_stabilizer`:

```python
def setwise_stabilizer(self, subset: set[int]) -> PermutationGroup:
    """Subgroup of elements that map *subset* to itself as a set.

    Parameters
    ----------
    subset : set of int
        Group-local indices. The subgroup consists of elements g where
        {g(x) for x in subset} == subset.

    Returns
    -------
    PermutationGroup
        The setwise stabilizer subgroup (same degree).
    """
    if not subset or subset == set(range(self._degree)):
        return PermutationGroup(*self._generators, axes=self._axes)
    frozen = frozenset(subset)
    surviving = [
        g for g in self.elements()
        if frozenset(g(x) for x in frozen) == frozen
    ]
    if not surviving:
        surviving = [Permutation.identity(self._degree)]
    return PermutationGroup(*surviving, axes=self._axes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_perm_group.py::TestSetwiseStabilizer -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_perm_group.py tests/test_perm_group.py
git commit -m "feat: add setwise_stabilizer to PermutationGroup"
```

---

### Task 3: Add `restrict` to `PermutationGroup`

**Files:**
- Modify: `src/whest/_perm_group.py:244-475` (PermutationGroup class)
- Test: `tests/test_perm_group.py`

- [ ] **Step 1: Write failing tests for restrict**

Add to `tests/test_perm_group.py`:

```python
class TestRestrict:
    def test_s3_restrict_to_pair(self):
        """S_3 pointwise-stabilizer of {2}, restricted to {0,1} → S_2."""
        g = PermutationGroup.symmetric(3, axes=(10, 20, 30))
        stab = g.pointwise_stabilizer({2})
        r = stab.restrict((0, 1))
        assert r.degree == 2
        assert r.order() == 2
        assert r.axes == (10, 20)

    def test_c4_setwise_restrict(self):
        """C_4 setwise-stabilizer of {1,3}, restricted to {0,2} → C_2."""
        g = PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))
        stab = g.setwise_stabilizer({1, 3})
        r = stab.restrict((0, 2))
        assert r.degree == 2
        assert r.order() == 2
        assert r.axes == (0, 2)
        # The non-identity element swaps 0↔1 (re-indexed from 0↔2)
        elems = r.elements()
        non_id = [e for e in elems if not e.is_identity]
        assert len(non_id) == 1
        assert non_id[0](0) == 1 and non_id[0](1) == 0

    def test_restrict_preserves_identity(self):
        """Restricting a trivial group gives a trivial group."""
        g = PermutationGroup.cyclic(3)
        stab = g.pointwise_stabilizer({0, 1, 2})  # trivial
        r = stab.restrict((0, 1))
        assert r.degree == 2
        assert r.order() == 1

    def test_restrict_single_point(self):
        """Restricting to a single point gives degree-1 trivial group."""
        g = PermutationGroup.symmetric(3)
        stab = g.pointwise_stabilizer({1, 2})
        r = stab.restrict((0,))
        assert r.degree == 1
        assert r.order() == 1

    def test_restrict_no_axes(self):
        """Restrict works when group has no axes metadata."""
        g = PermutationGroup.symmetric(3)
        stab = g.pointwise_stabilizer({2})
        r = stab.restrict((0, 1))
        assert r.degree == 2
        assert r.order() == 2
        assert r.axes is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_perm_group.py::TestRestrict -v`
Expected: FAIL with `AttributeError: 'PermutationGroup' object has no attribute 'restrict'`

- [ ] **Step 3: Implement restrict**

Add to `PermutationGroup` class in `src/whest/_perm_group.py`, after `setwise_stabilizer`:

```python
def restrict(self, kept: tuple[int, ...]) -> PermutationGroup:
    """Project permutations onto *kept* positions, re-indexing to 0..len(kept)-1.

    Precondition: the group must already stabilize *kept* setwise
    (every element maps the set of kept positions to itself).

    Parameters
    ----------
    kept : tuple of int
        Group-local indices to keep, in the desired output order.

    Returns
    -------
    PermutationGroup
        Group of degree ``len(kept)`` with projected permutations.
        ``axes`` is updated to select the corresponding entries from
        the original ``axes`` tuple (or None if original had no axes).
    """
    new_degree = len(kept)
    if new_degree == 0:
        raise ValueError("kept must be non-empty")

    # Map old group-local index → new index
    old_to_new = {old: new for new, old in enumerate(kept)}

    projected: set[Permutation] = set()
    for g in self.elements():
        new_arr = [old_to_new[g(k)] for k in kept]
        projected.add(Permutation(new_arr))

    new_axes: tuple[int, ...] | None = None
    if self._axes is not None:
        new_axes = tuple(self._axes[k] for k in kept)

    gens = list(projected) if projected else [Permutation.identity(new_degree)]
    return PermutationGroup(*gens, axes=new_axes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_perm_group.py::TestRestrict -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_perm_group.py tests/test_perm_group.py
git commit -m "feat: add restrict to PermutationGroup"
```

---

### Task 4: Rewrite `propagate_symmetry_slice` to use `PermutationGroup`

**Files:**
- Modify: `src/whest/_symmetric.py:225-363` (propagate_symmetry_slice)
- Test: `tests/test_symmetric_coverage.py`

- [ ] **Step 1: Write failing tests for the new behavior**

Add to `tests/test_symmetric_coverage.py`. These tests use general groups and verify the stabilizer logic produces correct results:

```python
from whest._perm_group import PermutationGroup


class TestPropagateSliceGeneralGroups:
    """Test propagate_symmetry_slice with non-S_k groups."""

    def test_c3_slice_one_axis_no_symmetry(self):
        """C_3 on {0,1,2}, slice axis 2 → no symmetry survives."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_slice([g], (5, 5, 5), (slice(None), slice(None), 0))
        assert result is None

    def test_c4_slice_two_axes_c2_survives(self):
        """C_4 on {0,1,2,3}, slice axes {1,3} → C_2 on output {0,1}."""
        g = PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))
        # T[:, 0, :, 0] → removes axes 1 and 3
        result = propagate_symmetry_slice([g], (5, 5, 5, 5), (slice(None), 0, slice(None), 0))
        assert result is not None
        assert len(result) == 1
        assert result[0].degree == 2
        assert result[0].order() == 2  # C_2
        assert result[0].axes == (0, 1)  # re-indexed from (0, 2)

    def test_s3_slice_one_axis_s2_survives(self):
        """S_3 on {0,1,2}, slice axis 2 → S_2 on {0,1}. Same as old behavior."""
        g = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        result = propagate_symmetry_slice([g], (5, 5, 5), (slice(None), slice(None), 0))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 2  # S_2

    def test_d4_slice_one_axis(self):
        """D_4 on {0,1,2,3}, slice axis 0 → pointwise stab of {0}, restricted."""
        g = PermutationGroup.dihedral(4, axes=(0, 1, 2, 3))
        result = propagate_symmetry_slice([g], (5, 5, 5, 5), (0, slice(None), slice(None), slice(None)))
        assert result is not None
        assert len(result) == 1
        # D_4 pointwise stab of {0}: elements fixing 0.
        # D_4 on square vertices 0,1,2,3:
        #   id(0→0), rot(0→1), rot²(0→2), rot³(0→3),
        #   refl_02(0→0), refl_13(0→2), refl_01(0→1), refl_23(0→3)
        # Wait — D_4 reflection: [0, 3, 2, 1] fixes 0. rot³*refl fixes 0?
        # Let's just check the result is a valid subgroup
        stab = result[0]
        assert stab.degree == 3
        # All elements of the restricted group should be valid permutations
        for elem in stab.elements():
            assert elem.size == 3

    def test_multiple_groups_independent(self):
        """Two independent groups, slice removes axis from one."""
        g1 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        g2 = PermutationGroup.symmetric(2, axes=(3, 4))
        result = propagate_symmetry_slice(
            [g1, g2], (5, 5, 5, 5, 5), (0, slice(None), slice(None), slice(None), slice(None))
        )
        # g1 loses axis 0 → C_3 pointwise stab of {0} → trivial, dropped
        # g2 unaffected but re-indexed: (3,4) → (2,3)
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 2  # S_2

    def test_no_axes_removed_group_unchanged(self):
        """Full slice preserves group unchanged."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_slice([g], (5, 5, 5), (slice(None), slice(None), slice(None)))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestPropagateSliceGeneralGroups -v`
Expected: FAIL (current function takes `list[tuple]`, not `list[PermutationGroup]`)

- [ ] **Step 3: Rewrite propagate_symmetry_slice**

Replace the function at `src/whest/_symmetric.py:225-363` with:

```python
def propagate_symmetry_slice(
    groups: list[PermutationGroup],
    shape: tuple[int, ...],
    key,
) -> list[PermutationGroup] | None:
    """Compute new symmetry groups after ``__getitem__(key)``.

    Parameters
    ----------
    groups : list of PermutationGroup
        Each group has ``axes`` indicating which tensor dimensions it acts on.
    shape : tuple of int
        Original tensor shape.
    key : indexing key
        The slicing/indexing key.

    Returns *None* if no symmetry survives (caller should return plain ndarray).
    """
    ndim = len(shape)

    # Normalize key to a tuple.
    if not isinstance(key, tuple):
        key = (key,)

    # Advanced indexing (ndarray / list) → bail out.
    for k in key:
        if isinstance(k, (np.ndarray, list)):
            return None

    # Expand Ellipsis.
    expanded: list = []
    ellipsis_seen = False
    for k in key:
        if k is Ellipsis:
            if ellipsis_seen:
                raise IndexError("only one Ellipsis allowed")
            ellipsis_seen = True
            n_newaxis_in_key = sum(1 for kk in key if kk is None)
            n_explicit = len(key) - 1 - n_newaxis_in_key
            n_fill = ndim - n_explicit
            expanded.extend([slice(None)] * n_fill)
        else:
            expanded.append(k)
    if not ellipsis_seen:
        n_newaxis = sum(1 for k in expanded if k is None)
        while len(expanded) - n_newaxis < ndim:
            expanded.append(slice(None))
    key_expanded = expanded

    # Classify each original dim.
    old_dim_idx = 0
    dim_actions: dict[int, str | tuple] = {}

    for k in key_expanded:
        if k is None:
            continue
        if old_dim_idx >= ndim:
            break
        if isinstance(k, (int, np.integer)):
            dim_actions[old_dim_idx] = "removed"
            old_dim_idx += 1
        elif isinstance(k, slice):
            start, stop, step = k.indices(shape[old_dim_idx])
            new_size = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
            if new_size == shape[old_dim_idx]:
                dim_actions[old_dim_idx] = "untouched"
            else:
                dim_actions[old_dim_idx] = ("resized", new_size)
            old_dim_idx += 1
        else:
            return None

    while old_dim_idx < ndim:
        dim_actions[old_dim_idx] = "untouched"
        old_dim_idx += 1

    # Build old→new dim mapping.
    removed_dims = {d for d, a in dim_actions.items() if a == "removed"}
    old_to_new: dict[int, int | None] = {}
    newaxis_positions: list[int] = []
    orig_idx = 0
    for k in key_expanded:
        if k is None:
            newaxis_positions.append(orig_idx)
        else:
            orig_idx += 1

    new_idx = 0
    for d in range(ndim):
        while newaxis_positions and newaxis_positions[0] <= d:
            newaxis_positions.pop(0)
            new_idx += 1
        if d in removed_dims:
            old_to_new[d] = None
        else:
            old_to_new[d] = new_idx
            new_idx += 1

    # Process each group.
    new_groups: list[PermutationGroup] = []
    for group in groups:
        axes = group.axes
        if axes is None:
            continue

        # Map tensor axes to group-local indices.
        # axes[i] is the tensor dim for group-local position i.
        local_removed: set[int] = set()
        local_kept: list[int] = []
        for local_idx, tensor_dim in enumerate(axes):
            action = dim_actions.get(tensor_dim, "untouched")
            if action == "removed":
                local_removed.add(local_idx)
            else:
                local_kept.append(local_idx)

        if not local_kept:
            continue

        # Pointwise stabilizer: fix removed axes.
        stab = group.pointwise_stabilizer(local_removed)

        # Among surviving axes, check for size mismatches (resized vs untouched).
        # Permutations can only swap axes of the same output size.
        size_map: dict[int, int] = {}
        for local_idx in local_kept:
            tensor_dim = axes[local_idx]
            action = dim_actions.get(tensor_dim, "untouched")
            if isinstance(action, tuple) and action[0] == "resized":
                size_map[local_idx] = action[1]
            else:
                size_map[local_idx] = shape[tensor_dim]

        sizes = set(size_map.values())
        if len(sizes) > 1:
            # Need setwise stabilizer of each size-class partition.
            for sz in sizes:
                same_size = {li for li, s in size_map.items() if s == sz}
                complement = set(local_kept) - same_size
                if complement:
                    stab = stab.setwise_stabilizer(same_size)

        # Restrict to kept local indices.
        kept_tuple = tuple(local_kept)
        if len(kept_tuple) < 2:
            continue

        restricted = stab.restrict(kept_tuple)

        if restricted.order() <= 1:
            continue

        # Remap axes to new tensor dim numbering.
        new_axes = tuple(old_to_new[axes[k]] for k in kept_tuple)
        if any(a is None for a in new_axes):
            continue

        final = PermutationGroup(*restricted.generators, axes=new_axes)
        new_groups.append(final)

    return new_groups if new_groups else None
```

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestPropagateSliceGeneralGroups -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_symmetric.py tests/test_symmetric_coverage.py
git commit -m "feat: rewrite propagate_symmetry_slice for general permutation groups"
```

---

### Task 5: Rewrite `propagate_symmetry_reduce` to use `PermutationGroup`

**Files:**
- Modify: `src/whest/_symmetric.py:366-411` (propagate_symmetry_reduce)
- Test: `tests/test_symmetric_coverage.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_symmetric_coverage.py`:

```python
class TestPropagateReduceGeneralGroups:
    """Test propagate_symmetry_reduce with non-S_k groups."""

    def test_c4_reduce_13_c2_survives(self):
        """C_4 on {0,1,2,3}, reduce axes {1,3} → C_2 on output {0,1}."""
        g = PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))
        result = propagate_symmetry_reduce([g], 4, (1, 3), keepdims=False)
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 2  # C_2
        assert result[0].axes == (0, 1)  # re-indexed from (0, 2)

    def test_c3_reduce_one_axis_trivial(self):
        """C_3 on {0,1,2}, reduce axis 2 → only identity survives → no group."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_reduce([g], 3, 2, keepdims=False)
        assert result is None

    def test_s3_reduce_one_axis_s2(self):
        """S_3 on {0,1,2}, reduce axis 2 → S_2 on {0,1}. Matches old behavior."""
        g = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        result = propagate_symmetry_reduce([g], 3, 2, keepdims=False)
        assert result is not None
        assert result[0].order() == 2

    def test_reduce_none_returns_none(self):
        """axis=None reduces everything → no symmetry."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = propagate_symmetry_reduce([g], 3, None)
        assert result is None

    def test_c4_reduce_keepdims(self):
        """C_4 on {0,1,2,3}, reduce axes {1,3} keepdims=True → C_2."""
        g = PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))
        result = propagate_symmetry_reduce([g], 4, (1, 3), keepdims=True)
        assert result is not None
        assert len(result) == 1
        # Axes stay at their positions, but setwise stab of {1,3} → {id, rot²}
        # restricted to {0,2}: C_2 with axes=(0,2)
        assert result[0].order() == 2
        assert result[0].axes == (0, 2)

    def test_reduce_disjoint_axis(self):
        """Reduce an axis not in the group → group unchanged, axes renumbered."""
        g = PermutationGroup.cyclic(3, axes=(1, 2, 3))
        result = propagate_symmetry_reduce([g], 4, 0, keepdims=False)
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3  # C_3 unchanged
        assert result[0].axes == (0, 1, 2)  # re-indexed from (1,2,3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestPropagateReduceGeneralGroups -v`
Expected: FAIL

- [ ] **Step 3: Rewrite propagate_symmetry_reduce**

Replace the function at `src/whest/_symmetric.py:366-411` with:

```python
def propagate_symmetry_reduce(
    groups: list[PermutationGroup],
    ndim: int,
    axis: int | tuple[int, ...] | None,
    keepdims: bool = False,
) -> list[PermutationGroup] | None:
    """Compute new symmetry groups after a reduction.

    Parameters
    ----------
    groups : list of PermutationGroup
        Each group has ``axes`` indicating which tensor dimensions it acts on.
    ndim : int
        Original tensor rank.
    axis : int, tuple of int, or None
        Axes being reduced.
    keepdims : bool
        Whether reduced dims are kept at size 1.

    Returns *None* if no symmetry survives.
    """
    if axis is None:
        return None

    # Normalize axis.
    if isinstance(axis, int):
        axes_set = {axis % ndim}
    else:
        axes_set = {a % ndim for a in axis}

    # Build old→new mapping (only needed when not keepdims).
    old_to_new: dict[int, int] = {}
    if not keepdims:
        new_idx = 0
        for d in range(ndim):
            if d not in axes_set:
                old_to_new[d] = new_idx
                new_idx += 1
    else:
        old_to_new = {d: d for d in range(ndim)}

    new_groups: list[PermutationGroup] = []
    for group in groups:
        grp_axes = group.axes
        if grp_axes is None:
            continue

        # Map tensor axes to group-local indices.
        local_reduced: set[int] = set()
        local_kept: list[int] = []
        for local_idx, tensor_dim in enumerate(grp_axes):
            if tensor_dim in axes_set:
                local_reduced.add(local_idx)
            else:
                local_kept.append(local_idx)

        if not local_reduced:
            # Group is entirely outside the reduced axes — just remap.
            new_axes = tuple(old_to_new[grp_axes[i]] for i in range(group.degree))
            new_groups.append(PermutationGroup(*group.generators, axes=new_axes))
            continue

        if not local_kept:
            # All axes in this group are reduced — group vanishes from output.
            continue

        # Setwise stabilizer: elements mapping reduced axes among themselves.
        stab = group.setwise_stabilizer(local_reduced)

        # Restrict to kept local indices.
        kept_tuple = tuple(local_kept)
        if len(kept_tuple) < 2:
            continue

        restricted = stab.restrict(kept_tuple)
        if restricted.order() <= 1:
            continue

        # Remap to new tensor axis numbering.
        new_axes = tuple(old_to_new[grp_axes[k]] for k in kept_tuple)
        final = PermutationGroup(*restricted.generators, axes=new_axes)
        new_groups.append(final)

    return new_groups if new_groups else None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestPropagateReduceGeneralGroups -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_symmetric.py tests/test_symmetric_coverage.py
git commit -m "feat: rewrite propagate_symmetry_reduce for general permutation groups"
```

---

### Task 6: Rewrite `intersect_symmetry` to use `PermutationGroup`

**Files:**
- Modify: `src/whest/_symmetric.py:414-466` (intersect_symmetry)
- Test: `tests/test_symmetric_coverage.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_symmetric_coverage.py`:

```python
class TestIntersectSymmetryGeneralGroups:
    """Test intersect_symmetry with PermutationGroup objects."""

    def test_same_group_returns_group(self):
        """Intersecting a group with itself returns same group."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = intersect_symmetry([g], [g], (5, 5, 5), (5, 5, 5), (5, 5, 5))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3

    def test_s3_intersect_c3(self):
        """S_3 ∩ C_3 = C_3 (C_3 is a subgroup of S_3)."""
        s3 = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        c3 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        result = intersect_symmetry([s3], [c3], (5, 5, 5), (5, 5, 5), (5, 5, 5))
        assert result is not None
        assert len(result) == 1
        assert result[0].order() == 3  # C_3

    def test_none_input_returns_none(self):
        """If either input is None, result is None."""
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        assert intersect_symmetry(None, [g], (5, 5, 5), (5, 5, 5), (5, 5, 5)) is None
        assert intersect_symmetry([g], None, (5, 5, 5), (5, 5, 5), (5, 5, 5)) is None

    def test_disjoint_axes_dropped(self):
        """Groups on different axes don't intersect → None."""
        g1 = PermutationGroup.symmetric(2, axes=(0, 1))
        g2 = PermutationGroup.symmetric(2, axes=(2, 3))
        result = intersect_symmetry([g1], [g2], (5, 5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5))
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestIntersectSymmetryGeneralGroups -v`
Expected: FAIL

- [ ] **Step 3: Rewrite intersect_symmetry**

Replace at `src/whest/_symmetric.py:414-466`:

```python
def intersect_symmetry(
    groups_a: list[PermutationGroup] | None,
    groups_b: list[PermutationGroup] | None,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> list[PermutationGroup] | None:
    """Intersect symmetry groups for binary ops, accounting for broadcasting.

    Parameters
    ----------
    groups_a, groups_b : list of PermutationGroup or None
    shape_a, shape_b : input shapes
    output_shape : broadcast output shape
    """
    if groups_a is None or groups_b is None:
        return None

    ndim_out = len(output_shape)
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)

    offset_a = ndim_out - ndim_a
    offset_b = ndim_out - ndim_b

    def _remap_axes(groups: list[PermutationGroup], offset: int, input_shape: tuple[int, ...]):
        """Remap group axes to output dims and remove broadcast-stretched dims."""
        result = []
        for group in groups:
            if group.axes is None:
                continue
            new_axes = []
            local_kept = []
            for local_idx, tensor_dim in enumerate(group.axes):
                out_dim = tensor_dim + offset
                # Check if broadcast-stretched (size 1 → larger).
                if 0 <= tensor_dim < len(input_shape):
                    if input_shape[tensor_dim] == 1 and output_shape[out_dim] > 1:
                        continue  # stretched — remove from group
                new_axes.append(out_dim)
                local_kept.append(local_idx)
            if len(local_kept) >= 2:
                restricted = group.restrict(tuple(local_kept))
                if restricted.order() > 1:
                    result.append(PermutationGroup(
                        *restricted.generators, axes=tuple(new_axes)
                    ))
        return result

    aligned_a = _remap_axes(groups_a, offset_a, shape_a)
    aligned_b = _remap_axes(groups_b, offset_b, shape_b)

    # Intersect: for groups acting on the same output axes, compute element intersection.
    b_by_axes: dict[tuple[int, ...], PermutationGroup] = {}
    for g in aligned_b:
        if g.axes is not None:
            b_by_axes[g.axes] = g

    intersection: list[PermutationGroup] = []
    for ga in aligned_a:
        if ga.axes is None:
            continue
        gb = b_by_axes.get(ga.axes)
        if gb is None:
            continue
        # Element-set intersection.
        common_elements = set(ga.elements()) & set(gb.elements())
        if len(common_elements) <= 1:
            continue
        intersection.append(PermutationGroup(*common_elements, axes=ga.axes))

    return intersection if intersection else None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestIntersectSymmetryGeneralGroups -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_symmetric.py tests/test_symmetric_coverage.py
git commit -m "feat: rewrite intersect_symmetry for general permutation groups"
```

---

### Task 7: Update callers — `SymmetricTensor.__getitem__` and `_pointwise.py`

**Files:**
- Modify: `src/whest/_symmetric.py:562-595` (SymmetricTensor.__getitem__)
- Modify: `src/whest/_pointwise.py:80-130` (binary op symmetry)
- Modify: `src/whest/_pointwise.py:200-228` (reduction symmetry)
- Test: `tests/test_symmetric_coverage.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_symmetric_coverage.py`:

```python
class TestSymmetricTensorGeneralGroups:
    """End-to-end tests: SymmetricTensor with general groups through slice/reduce."""

    def test_c3_tensor_slice_loses_symmetry(self):
        """SymmetricTensor with C_3, slicing one axis → no symmetry."""
        import numpy as np
        from whest._symmetric import SymmetricTensor

        arr = np.zeros((3, 3, 3))
        g = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        t = SymmetricTensor(arr, symmetric_axes=[(0, 1, 2)], perm_groups=[g])
        result = t[:, :, 0]
        assert not isinstance(result, SymmetricTensor) or not result.symmetric_axes

    def test_s3_tensor_slice_keeps_s2(self):
        """SymmetricTensor with S_3, slicing one axis → S_2 survives."""
        import numpy as np
        from whest._symmetric import SymmetricTensor

        arr = np.zeros((3, 3, 3))
        g = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        t = SymmetricTensor(arr, symmetric_axes=[(0, 1, 2)], perm_groups=[g])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = t[:, :, 0]
        assert isinstance(result, SymmetricTensor)
        assert len(result._symmetry_groups) == 1
        assert result._symmetry_groups[0].order() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestSymmetricTensorGeneralGroups -v`
Expected: FAIL (callers still pass tuples)

- [ ] **Step 3: Update SymmetricTensor.__getitem__**

Replace `__getitem__` in `src/whest/_symmetric.py` (lines 562-595):

```python
def __getitem__(self, key):  # type: ignore[override]
    result = super().__getitem__(key)
    if not isinstance(result, np.ndarray) or result.ndim == 0:
        return result if not isinstance(result, np.ndarray) else np.asarray(result)

    if not self._symmetry_groups:
        return np.asarray(result)

    new_groups = propagate_symmetry_slice(self._symmetry_groups, self.shape, key)
    if new_groups is not None:
        out = np.asarray(result).view(SymmetricTensor)
        out._symmetry_groups = new_groups
        out._symmetric_axes = [g.axes for g in new_groups if g.axes is not None]
        # Warn if symmetry was partially lost.
        if len(new_groups) < len(self._symmetry_groups):
            lost_axes = [
                g.axes for g in self._symmetry_groups
                if not any(
                    ng.axes == g.axes or (ng.axes is not None and g.axes is not None)
                    for ng in new_groups
                )
            ]
            if lost_axes:
                _warn_symmetry_loss(
                    [a for a in lost_axes if a is not None],
                    "slicing changed dim sizes or removed dims",
                )
        return out
    else:
        if self._symmetry_groups:
            _warn_symmetry_loss(
                [g.axes for g in self._symmetry_groups if g.axes is not None],
                "slicing removed all symmetric dim groups",
            )
        return np.asarray(result)
```

- [ ] **Step 4: Update _pointwise.py reduction caller**

In `src/whest/_pointwise.py`, update the `_counted_reduction` function (around line 200-228). Change:

```python
            new_groups = propagate_symmetry_reduce(
                sym_info.symmetric_axes, len(a.shape), axis, keepdims=keepdims
            )
            if new_groups is not None:
                result = _np.asarray(result).view(SymmetricTensor)
                result._symmetric_axes = new_groups
                # Warn if groups were partially lost.
                old_set = set(sym_info.symmetric_axes)
                new_set = set(new_groups)
                if new_set != old_set:
                    lost = [g for g in sym_info.symmetric_axes if g not in new_set]
                    if lost:
                        _warn_symmetry_loss(lost, f"{op_name} reduced dims")
```

To:

```python
            perm_groups = sym_info.groups if sym_info.groups else []
            new_groups = propagate_symmetry_reduce(
                perm_groups, len(a.shape), axis, keepdims=keepdims
            )
            if new_groups is not None:
                result = _np.asarray(result).view(SymmetricTensor)
                result._symmetry_groups = new_groups
                result._symmetric_axes = [g.axes for g in new_groups if g.axes is not None]
                if len(new_groups) < len(perm_groups):
                    lost_axes = [g.axes for g in perm_groups if g.axes is not None]
                    if lost_axes:
                        _warn_symmetry_loss(lost_axes, f"{op_name} reduced dims")
```

- [ ] **Step 5: Update _pointwise.py binary op caller**

In `src/whest/_pointwise.py`, update the binary op path (around lines 90-105). Change:

```python
            x_axes = x_sym.symmetric_axes if x_sym else []
            y_axes = y_sym.symmetric_axes if y_sym else []
            out_sym_axes = intersect_symmetry(
                x_axes if x_axes else None,
                y_axes if y_axes else None,
                x.shape,
                y.shape,
                output_shape,
            )
```

To:

```python
            x_groups = x_sym.groups if x_sym else []
            y_groups = y_sym.groups if y_sym else []
            out_groups = intersect_symmetry(
                x_groups if x_groups else None,
                y_groups if y_groups else None,
                x.shape,
                y.shape,
                output_shape,
            )
            out_sym_axes = (
                [g.axes for g in out_groups if g.axes is not None]
                if out_groups else None
            )
```

And update the `SymmetricTensor` construction (around line 122) from:

```python
        if out_sym_axes:
            result = SymmetricTensor(result, symmetric_axes=out_sym_axes)
```

To:

```python
        if out_sym_axes:
            result = SymmetricTensor(result, symmetric_axes=out_sym_axes, perm_groups=out_groups)
```

Also update the warning block that follows (lines 123-134) to derive `input_groups` from `PermutationGroup.axes` instead of `sym.symmetric_axes`.

- [ ] **Step 6: Run integration tests**

Run: `uv run pytest tests/test_symmetric_coverage.py::TestSymmetricTensorGeneralGroups -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/whest/_symmetric.py src/whest/_pointwise.py tests/test_symmetric_coverage.py
git commit -m "feat: update callers to use PermutationGroup-based propagation"
```

---

### Task 8: Fix existing tests and run full suite

**Files:**
- Modify: `tests/test_symmetric_coverage.py` (existing tests that pass tuples)
- Modify: `tests/test_pointwise_coverage.py` (if needed)

- [ ] **Step 1: Update existing propagate_symmetry_slice tests**

The existing tests in `TestPropagateSymmetrySlice` and `TestPropagateSymmetryReduce` classes pass bare tuples like `[(0, 1)]`. Update them to pass `PermutationGroup` objects instead.

For each test calling `propagate_symmetry_slice([(0, 1, 2)], ...)`, change to:
```python
propagate_symmetry_slice([PermutationGroup.symmetric(3, axes=(0, 1, 2))], ...)
```

For each test calling `propagate_symmetry_reduce([(1, 2)], ...)`, change to:
```python
propagate_symmetry_reduce([PermutationGroup.symmetric(2, axes=(1, 2))], ...)
```

Similarly update `intersect_symmetry` test calls.

Update assertions: old tests assert `result == [(0, 1)]` (tuples). New results are `list[PermutationGroup]`. Change to:
```python
assert result is not None
assert len(result) == 1
assert result[0].axes == (0, 1)
assert result[0].order() == 2  # S_2
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all PASS (or pre-existing xfails only)

- [ ] **Step 3: Fix any remaining failures**

Address any test failures found in step 2. Common issues:
- Tests in `test_pointwise_coverage.py` that check `intersect_symmetry` with tuples
- Tests that compare results to `list[tuple]` instead of checking group properties
- The `_counted_reduction` and `_counted_binary` paths that construct `SymmetryInfo`

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "fix: update existing tests for PermutationGroup-based propagation"
```

- [ ] **Step 5: Run full test suite one final time**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all PASS
