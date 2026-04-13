# Sympy API Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Permutation` and `PermutationGroup` feel native to sympy users by adding 14 missing API methods, dogfooding them in internal code, and documenting with examples.

**Architecture:** All additions go into the existing `_perm_group.py` (~375 lines currently). The `Cycle` class is added to the same file. Internal callsites are updated to use the cleaner APIs. Documentation is added to `docs/api/symmetric.md` and `docs/how-to/exploit-symmetry.md`.

**Tech Stack:** Pure Python (`math` stdlib), numpy for tests, sympy for bridge tests.

**Spec:** `.aicrowd/superpowers/specs/2026-04-13-sympy-api-parity-design.md`

---

## File Structure

| Action | File | What changes |
|--------|------|-------------|
| Modify | `src/whest/_perm_group.py` | Add `Cycle` class, 7 Permutation methods, 6 PermutationGroup methods |
| Modify | `tests/test_perm_group.py` | Tests for all 14 additions |
| Modify | `src/whest/__init__.py` | Export `Cycle` |
| Modify | `src/whest/_opt_einsum/_subgraph_symmetry.py` | Dogfood: use `perm(i)` and `Permutation(cycles)` where cleaner |
| Modify | `tests/test_subgraph_symmetry.py` | Dogfood: use `Cycle` and cycle notation in test setup |
| Modify | `docs/api/symmetric.md` | API reference for all additions |
| Modify | `docs/how-to/exploit-symmetry.md` | User-facing examples |

---

### Task 1: `Cycle` class and cycle-notation construction

**Files:**
- Modify: `src/whest/_perm_group.py`
- Modify: `tests/test_perm_group.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_perm_group.py`:

```python
from whest._perm_group import Cycle


class TestCycle:
    def test_single_cycle(self):
        c = Cycle(0, 2)
        p = Permutation(c)
        assert p.array_form == [2, 1, 0]

    def test_chained_cycles(self):
        c = Cycle(0, 2)(1, 3)
        p = Permutation(c)
        assert p.array_form == [2, 3, 0, 1]

    def test_three_cycle(self):
        c = Cycle(0, 1, 2)
        p = Permutation(c)
        assert p.array_form == [1, 2, 0]

    def test_chained_three_cycles(self):
        c = Cycle(0, 1, 2)(3, 4)
        p = Permutation(c)
        assert p.array_form == [1, 2, 0, 4, 3]

    def test_with_explicit_size(self):
        c = Cycle(0, 1)
        p = Permutation(c, size=5)
        assert p.size == 5
        assert p.array_form == [1, 0, 2, 3, 4]

    def test_empty_cycle(self):
        c = Cycle()
        p = Permutation(c, size=3)
        assert p.is_identity
        assert p.size == 3

    def test_cycle_list_method(self):
        c = Cycle(0, 2)(1, 3)
        assert c.list() == [2, 3, 0, 1]
        assert c.list(6) == [2, 3, 0, 1, 4, 5]


class TestPermutationCycleNotation:
    def test_list_of_lists(self):
        p = Permutation([[0, 2], [1, 3]])
        assert p.array_form == [2, 3, 0, 1]

    def test_single_cycle_list(self):
        p = Permutation([[0, 1, 2]])
        assert p.array_form == [1, 2, 0]

    def test_list_of_lists_with_size(self):
        p = Permutation([[0, 1]], size=5)
        assert p.size == 5
        assert p.array_form == [1, 0, 2, 3, 4]

    def test_array_form_still_works(self):
        p = Permutation([2, 0, 1])
        assert p.array_form == [2, 0, 1]

    def test_from_cycle_object(self):
        p = Permutation(Cycle(0, 2)(1, 3))
        assert p.array_form == [2, 3, 0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_perm_group.py::TestCycle tests/test_perm_group.py::TestPermutationCycleNotation -v`
Expected: ImportError / TypeError

- [ ] **Step 3: Implement `Cycle` class**

Add BEFORE the `Permutation` class in `src/whest/_perm_group.py`:

```python
class Cycle:
    """Composable cycle builder, matching sympy's Cycle API.

    Build permutations by composing disjoint or overlapping cycles::

        Cycle(0, 2)(1, 3)     # → (0 2)(1 3)
        Permutation(Cycle(0, 2)(1, 3))  # → Permutation([2, 3, 0, 1])
    """

    __slots__ = ("_mapping",)

    def __init__(self, *cycle: int) -> None:
        self._mapping: dict[int, int] = {}
        if cycle:
            for i in range(len(cycle)):
                self._mapping[cycle[i]] = cycle[(i + 1) % len(cycle)]

    def __call__(self, *cycle: int) -> Cycle:
        """Compose another cycle, returning a new Cycle."""
        new = Cycle()
        new._mapping = dict(self._mapping)
        if cycle:
            # Apply the new cycle on top of existing mapping.
            # For each point in the new cycle, follow existing mapping first,
            # then apply the new cycle.
            new_cycle_map: dict[int, int] = {}
            for i in range(len(cycle)):
                new_cycle_map[cycle[i]] = cycle[(i + 1) % len(cycle)]
            # Compose: result[x] = new_cycle_map[old[x]] if old[x] in new_cycle,
            # else old[x]. And for points only in new_cycle: result[x] = new_cycle[x].
            combined: dict[int, int] = {}
            all_points = set(new._mapping) | set(new_cycle_map)
            for x in all_points:
                y = new._mapping.get(x, x)
                z = new_cycle_map.get(y, y)
                if z != x:
                    combined[x] = z
            new._mapping = combined
        return new

    def list(self, size: int | None = None) -> list[int]:
        """Return array form. Size is inferred from max element + 1 if not given."""
        if size is None:
            size = max(self._mapping.keys(), default=-1) + 1
            size = max(size, max(self._mapping.values(), default=-1) + 1)
        arr = list(range(size))
        for k, v in self._mapping.items():
            if k < size:
                arr[k] = v
        return arr
```

- [ ] **Step 4: Update `Permutation.__init__` to accept cycles**

Replace the `__init__` method:

```python
    def __init__(
        self,
        array_form: list[int] | tuple[int, ...] | Cycle,
        size: int | None = None,
    ) -> None:
        if isinstance(array_form, Cycle):
            self._array_form = tuple(array_form.list(size))
        elif array_form and isinstance(array_form[0], (list, tuple)):
            # Cycle notation: list of lists, e.g., [[0, 2], [1, 3]]
            c = Cycle()
            for cycle in array_form:
                c = c(*cycle)
            self._array_form = tuple(c.list(size))
        else:
            arr = list(array_form)
            if size is not None and size > len(arr):
                arr.extend(range(len(arr), size))
            self._array_form = tuple(arr)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_perm_group.py -v`
Expected: All PASS (old + new).

- [ ] **Step 6: Commit**

```bash
git add src/whest/_perm_group.py tests/test_perm_group.py
git commit -m "feat: add Cycle class and cycle-notation construction for Permutation"
```

---

### Task 2: Permutation methods — `__call__`, `support`, `parity`, `signature`, `transpositions`

**Files:**
- Modify: `src/whest/_perm_group.py`
- Modify: `tests/test_perm_group.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_perm_group.py`:

```python
class TestPermutationNewMethods:
    def test_call(self):
        p = Permutation([1, 2, 0])
        assert p(0) == 1
        assert p(1) == 2
        assert p(2) == 0

    def test_call_identity(self):
        e = Permutation.identity(3)
        assert e(0) == 0
        assert e(2) == 2

    def test_support(self):
        assert Permutation([2, 3, 0, 1]).support() == {0, 1, 2, 3}
        assert Permutation([0, 2, 1, 3]).support() == {1, 2}

    def test_support_identity(self):
        assert Permutation.identity(5).support() == set()

    def test_parity_even(self):
        # (0 1 2) is a 3-cycle = 2 transpositions = even
        assert Permutation([1, 2, 0]).parity() == 0

    def test_parity_odd(self):
        # (0 1) is 1 transposition = odd
        assert Permutation([1, 0, 2]).parity() == 1

    def test_parity_identity(self):
        assert Permutation.identity(3).parity() == 0

    def test_parity_two_disjoint_transpositions(self):
        # (0 1)(2 3) = 2 transpositions = even
        assert Permutation([1, 0, 3, 2]).parity() == 0

    def test_signature(self):
        assert Permutation([1, 2, 0]).signature() == 1   # even → +1
        assert Permutation([1, 0, 2]).signature() == -1   # odd → -1
        assert Permutation.identity(3).signature() == 1

    def test_transpositions_3cycle(self):
        # (0 1 2) → [(0, 2), (0, 1)]
        t = Permutation([1, 2, 0]).transpositions()
        assert len(t) == 2
        # Verify: composing the transpositions gives back the original
        result = Permutation.identity(3)
        for a, b in t:
            result = Permutation.from_cycle(3, [a, b]) * result
        assert result == Permutation([1, 2, 0])

    def test_transpositions_identity(self):
        assert Permutation.identity(3).transpositions() == []

    def test_transpositions_single_swap(self):
        t = Permutation([1, 0, 2]).transpositions()
        assert t == [(0, 1)]

    def test_transpositions_two_disjoint(self):
        # (0 1)(2 3) → [(0, 1), (2, 3)]
        t = Permutation([1, 0, 3, 2]).transpositions()
        assert len(t) == 2
        result = Permutation.identity(4)
        for a, b in t:
            result = Permutation.from_cycle(4, [a, b]) * result
        assert result == Permutation([1, 0, 3, 2])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_perm_group.py::TestPermutationNewMethods -v`
Expected: AttributeError

- [ ] **Step 3: Implement the methods**

Add to the `Permutation` class in `src/whest/_perm_group.py`:

```python
    def __call__(self, i: int) -> int:
        """Apply the permutation: ``perm(i)`` returns the image of ``i``."""
        return self._array_form[i]

    def support(self) -> set[int]:
        """Set of non-fixed points."""
        return {i for i in range(self.size) if self._array_form[i] != i}

    def parity(self) -> int:
        """Parity: 0 if even, 1 if odd."""
        return sum(len(c) - 1 for c in self.cyclic_form) % 2

    def signature(self) -> int:
        """Signature: +1 if even, -1 if odd."""
        return 1 if self.parity() == 0 else -1

    def transpositions(self) -> list[tuple[int, int]]:
        """Decompose into a product of transpositions.

        Each cycle (a, b, c, ...) becomes [(a, ...), ..., (a, c), (a, b)].
        The rightmost transposition is applied first.
        """
        result: list[tuple[int, int]] = []
        for cycle in self.cyclic_form:
            for i in range(len(cycle) - 1, 0, -1):
                result.append((cycle[0], cycle[i]))
        return result
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_perm_group.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whest/_perm_group.py tests/test_perm_group.py
git commit -m "feat: add __call__, support, parity, signature, transpositions to Permutation"
```

---

### Task 3: PermutationGroup methods — `contains`, `is_transitive`, `is_abelian`, `identity`, `equals`, `orbit`

**Files:**
- Modify: `src/whest/_perm_group.py`
- Modify: `tests/test_perm_group.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_perm_group.py`:

```python
class TestPermutationGroupNewMethods:
    def test_contains_member(self):
        g = PermutationGroup.cyclic(3)
        assert g.contains(Permutation([1, 2, 0]))

    def test_contains_non_member(self):
        g = PermutationGroup.cyclic(3)
        assert not g.contains(Permutation([0, 2, 1]))  # transposition not in C_3

    def test_contains_identity(self):
        g = PermutationGroup.cyclic(3)
        assert g.contains(Permutation.identity(3))

    def test_is_transitive_true(self):
        assert PermutationGroup.symmetric(3).is_transitive
        assert PermutationGroup.cyclic(4).is_transitive

    def test_is_transitive_false(self):
        gen = Permutation.from_cycle(4, [0, 1])
        g = PermutationGroup(gen)
        assert not g.is_transitive  # orbits: {0,1}, {2}, {3}

    def test_is_abelian_cyclic(self):
        assert PermutationGroup.cyclic(4).is_abelian

    def test_is_abelian_s2(self):
        assert PermutationGroup.symmetric(2).is_abelian

    def test_is_abelian_s3_false(self):
        assert not PermutationGroup.symmetric(3).is_abelian

    def test_identity_property(self):
        g = PermutationGroup.symmetric(4)
        e = g.identity
        assert e.is_identity
        assert e.size == 4

    def test_equals_same_generators(self):
        g1 = PermutationGroup.symmetric(3)
        g2 = PermutationGroup.symmetric(3)
        assert g1.equals(g2)

    def test_equals_different_generators_same_group(self):
        # S_3 generated by (0 1) and (0 1 2) vs by adjacent transpositions
        g1 = PermutationGroup(
            Permutation.from_cycle(3, [0, 1]),
            Permutation.from_cycle(3, [0, 1, 2]),
        )
        g2 = PermutationGroup.symmetric(3)
        assert g1.equals(g2)

    def test_equals_different_groups(self):
        g1 = PermutationGroup.cyclic(3)
        g2 = PermutationGroup.symmetric(3)
        assert not g1.equals(g2)

    def test_orbit_transitive(self):
        g = PermutationGroup.symmetric(3)
        assert g.orbit(0) == frozenset({0, 1, 2})
        assert g.orbit(1) == frozenset({0, 1, 2})

    def test_orbit_intransitive(self):
        gen = Permutation.from_cycle(5, [0, 1])
        g = PermutationGroup(gen)
        assert g.orbit(0) == frozenset({0, 1})
        assert g.orbit(1) == frozenset({0, 1})
        assert g.orbit(2) == frozenset({2})
        assert g.orbit(4) == frozenset({4})

    def test_orbit_cyclic(self):
        g = PermutationGroup.cyclic(4)
        # C_4 is transitive: all points in one orbit
        assert g.orbit(0) == frozenset({0, 1, 2, 3})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_perm_group.py::TestPermutationGroupNewMethods -v`
Expected: AttributeError

- [ ] **Step 3: Implement the methods**

Add to the `PermutationGroup` class in `src/whest/_perm_group.py`:

```python
    def contains(self, perm: Permutation) -> bool:
        """Test whether *perm* is an element of this group."""
        return perm in set(self.elements())

    @property
    def is_transitive(self) -> bool:
        """True if the group acts transitively (single orbit)."""
        return len(self.orbits()) == 1

    @property
    def is_abelian(self) -> bool:
        """True if all generators commute (implies all elements commute)."""
        gens = self._generators
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                if gens[i] * gens[j] != gens[j] * gens[i]:
                    return False
        return True

    @property
    def identity(self) -> Permutation:
        """The identity element of the group."""
        return Permutation.identity(self._degree)

    def equals(self, other: PermutationGroup) -> bool:
        """True if *self* and *other* represent the same group (same elements)."""
        if self._degree != other._degree:
            return False
        if self.order() != other.order():
            return False
        return set(self.elements()) == set(other.elements())

    def orbit(self, alpha: int) -> frozenset[int]:
        """Orbit of a single point under the group action (BFS)."""
        visited: set[int] = {alpha}
        queue: list[int] = [alpha]
        while queue:
            point = queue.pop()
            for g in self._generators:
                image = g._array_form[point]
                if image not in visited:
                    visited.add(image)
                    queue.append(image)
        return frozenset(visited)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_perm_group.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whest/_perm_group.py tests/test_perm_group.py
git commit -m "feat: add contains, is_transitive, is_abelian, identity, equals, orbit to PermutationGroup"
```

---

### Task 4: Export `Cycle`, dogfood in internal code

**Files:**
- Modify: `src/whest/__init__.py`
- Modify: `src/whest/_opt_einsum/_subgraph_symmetry.py`
- Modify: `tests/test_subgraph_symmetry.py`

- [ ] **Step 1: Export `Cycle` from `__init__.py`**

Change the existing import line:
```python
from whest._perm_group import Permutation, PermutationGroup  # noqa: F401
```
to:
```python
from whest._perm_group import Cycle, Permutation, PermutationGroup  # noqa: F401
```

- [ ] **Step 2: Dogfood `perm(i)` in `_subgraph_symmetry.py`**

In `_collect_pi_permutations` and `_lift_operand_perm_to_u`, the code accesses `g._array_form[i]` directly. Replace with `g(i)` where it improves readability. Specifically in the `orbits()` method of `PermutationGroup` (line ~230):

```python
# Old:
if g._array_form[i] != i:
    union(i, g._array_form[i])

# New (cleaner):
if g(i) != i:
    union(i, g(i))
```

And in `orbit()` method:
```python
# Old:
image = g._array_form[point]

# New:
image = g(point)
```

- [ ] **Step 3: Dogfood cycle notation in tests**

In `tests/test_subgraph_symmetry.py`, find places where test setup creates permutations verbosely and use `Cycle` or cycle notation instead. For example, in `_sym_group` helpers or oracle construction where `PermutationGroup(Permutation([1, 0]))` could be `PermutationGroup(Permutation(Cycle(0, 1)))`.

Only change test setup code, not assertions. Don't force it where array form is already clear.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/numpy_compat -x -q`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whest/__init__.py src/whest/_perm_group.py src/whest/_opt_einsum/_subgraph_symmetry.py tests/test_subgraph_symmetry.py
git commit -m "feat: export Cycle, dogfood new API in internal code"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/api/symmetric.md`
- Modify: `docs/how-to/exploit-symmetry.md`

- [ ] **Step 1: Update `docs/api/symmetric.md`**

In the "Permutation Groups" section, add detailed API documentation. After the existing autodoc directive `::: whest._perm_group`, add an explicit API summary:

```markdown
### Cycle

Composable cycle builder matching sympy's `Cycle` API:

```python
from whest import Cycle, Permutation

# Single cycle
Permutation(Cycle(0, 2))                 # → (0 2)

# Chained cycles (block swap)
Permutation(Cycle(0, 2)(1, 3))           # → (0 2)(1 3)

# Equivalent list-of-lists notation
Permutation([[0, 2], [1, 3]])            # same result

# Explicit size (pad with fixed points)
Permutation(Cycle(0, 1), size=5)         # size 5, only 0↔1 moves
```

### Permutation API

Construction:
- `Permutation([2, 0, 1])` — array form
- `Permutation([[0, 2], [1, 3]])` — cycle notation
- `Permutation(Cycle(0, 2)(1, 3))` — from Cycle object
- `Permutation.identity(n)` — identity of size n
- `Permutation.from_cycle(n, [0, 1, 2])` — single cycle

Application and inspection:
- `perm(i)` — apply: image of point i
- `perm.support()` — set of non-fixed points
- `perm.parity()` — 0 (even) or 1 (odd)
- `perm.signature()` — +1 or -1
- `perm.transpositions()` — decompose into 2-cycles
- `perm.cyclic_form` — disjoint cycles (excluding fixed points)
- `perm.full_cyclic_form` — disjoint cycles (including 1-cycles)
- `perm.cycle_structure` — dict of cycle length → count
- `perm.order` — element order (lcm of cycle lengths)

Composition:
- `p * q` — compose (p after q)
- `~p` — inverse

### PermutationGroup API

Construction:
- `PermutationGroup(*generators)` — from generator permutations
- `PermutationGroup.symmetric(k)` — S_k
- `PermutationGroup.cyclic(k)` — C_k
- `PermutationGroup.dihedral(k)` — D_k

Queries:
- `group.order()` — number of elements
- `group.degree` — number of points acted on
- `group.elements()` — list of all elements (cached)
- `group.contains(perm)` — membership test
- `group.is_symmetric()` — is this S_k?
- `group.is_transitive` — single orbit?
- `group.is_abelian` — all elements commute?
- `group.identity` — identity element
- `group.equals(other)` — same group regardless of generators?

Orbits:
- `group.orbits()` — partition into orbits
- `group.orbit(i)` — orbit of a single point

Counting:
- `group.burnside_unique_count(size_dict)` — unique tensor elements via Burnside's lemma
```

- [ ] **Step 2: Update `docs/how-to/exploit-symmetry.md`**

In the "Declaring non-S_k symmetries" section, update the block swap example to use `Cycle`:

```markdown
Block swaps are natural with cycle notation:

```python
from whest import Cycle, Permutation, PermutationGroup

# Block swap: (i,j) ↔ (k,l) as a unit
block_swap = Permutation(Cycle(0, 2)(1, 3))
G = PermutationGroup(block_swap, axes=(0, 1, 2, 3))
T = we.as_symmetric(data, symmetry=G)
```

This reads naturally: "cycle 0 with 2, and cycle 1 with 3" — axes 0,1 swap
with axes 2,3 as blocks.

For inspection:

```python
p = Permutation(Cycle(0, 1, 2))
p(0)              # → 1 (where does axis 0 go?)
p.support()       # → {0, 1, 2} (which axes move?)
p.parity()        # → 0 (even permutation)
p.signature()     # → 1 (no sign change)
p.transpositions()  # → [(0, 2), (0, 1)]
```

Group queries:

```python
G = PermutationGroup.cyclic(3, axes=(0, 1, 2))
G.contains(Permutation([1, 2, 0]))  # → True (rotation is in C_3)
G.contains(Permutation([0, 2, 1]))  # → False (swap is NOT in C_3)
G.is_transitive  # → True (all axes connected)
G.is_abelian     # → True (cyclic groups are abelian)
G.orbit(0)       # → frozenset({0, 1, 2})
```
```

- [ ] **Step 3: Commit**

```bash
git add docs/api/symmetric.md docs/how-to/exploit-symmetry.md
git commit -m "docs: add comprehensive Permutation/PermutationGroup API reference with examples"
```

---

### Task 6: Lint, format, regression, push

**Files:** None (testing only)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/numpy_compat -v`
Expected: All PASS.

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
If issues: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 3: Commit fixes if needed**

```bash
git commit -am "style: lint and format"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```
