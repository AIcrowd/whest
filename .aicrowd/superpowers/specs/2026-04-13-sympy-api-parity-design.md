# Sympy API Parity for Permutation and PermutationGroup

**Date:** 2026-04-13
**Status:** Draft
**Scope:** `src/whest/_perm_group.py`, `tests/test_perm_group.py`, `docs/api/symmetric.md`

## Problem

A user familiar with sympy's `combinatorics.Permutation` and `PermutationGroup` encounters friction when using whest's versions. Key API surfaces they'd immediately reach for are missing:

- Cycle notation construction: `Permutation([[0, 2], [1, 3]])` for block swaps
- Callable syntax: `perm(3)` to apply a permutation
- Parity/signature: fundamental for antisymmetric tensors
- Membership test: `group.contains(perm)`
- Group properties: `is_transitive`, `is_abelian`
- Single-point orbit: `group.orbit(0)`

## Solution

Add 12 methods/properties to match the sympy API surface that users would expect. No new files — all changes in `_perm_group.py`.

### `Cycle` helper class — 1 addition

#### 0. `Cycle` — composable cycle builder

Sympy users construct permutations from cycles with:
```python
Cycle(0, 2)(1, 3)  # → the permutation (0 2)(1 3)
```

The `Cycle` class builds a permutation by composing cycles via `__call__`:

```python
c = Cycle(0, 2)       # creates (0 2)
c = Cycle(0, 2)(1, 3) # composes with (1 3) → (0 2)(1 3)
```

`Cycle` stores a dict mapping of moved points. Each `__call__` composes another cycle. Converting to `Permutation` uses `Permutation(cycle_obj)`:

```python
Permutation(Cycle(0, 2)(1, 3))         # size inferred from max element + 1
Permutation(Cycle(0, 2)(1, 3), size=6) # explicit size
```

The `Cycle` class has:
- `__init__(*cycle)` — initial cycle, e.g. `Cycle(0, 2)`
- `__call__(*cycle)` — compose another cycle, returns new `Cycle`
- `list(size=None)` — return array form as a list
- Internal storage: `dict[int, int]` of point mappings

~25 lines total.

### Permutation — 7 additions

#### 1. `__init__` accepts list-of-lists AND `Cycle` objects

Sympy users construct permutations three ways:

```python
# Array form (already supported)
Permutation([2, 3, 0, 1])

# Cycle notation via list-of-lists
Permutation([[0, 2], [1, 3]])           # size inferred as 4
Permutation([[0, 2], [1, 3]], size=6)   # pad with fixed points to size 6

# Cycle object
Permutation(Cycle(0, 2)(1, 3))          # size inferred
Permutation(Cycle(0, 2)(1, 3), size=6)  # explicit size
```

Detection logic:
- If input is a `Cycle` object → call `cycle.list(size)` to get array form
- If input is a list/tuple and first element is a list/tuple → cycle notation, apply each cycle
- Otherwise → array form (existing behavior)

#### 2. `__call__(i)` — apply the permutation

```python
perm = Permutation([1, 2, 0])
perm(0)  # → 1
perm(2)  # → 0
```

One line: `return self._array_form[i]`

#### 3. `support()` — non-fixed points

```python
Permutation([2, 3, 0, 1]).support()  # → {0, 1, 2, 3}
Permutation([0, 2, 1, 3]).support()  # → {1, 2}
```

Returns `set[int]` (sympy returns a set). Computed from array form: `{i for i in range(size) if array[i] != i}`.

#### 4. `parity()` — even (0) or odd (1)

```python
Permutation([1, 0, 2]).parity()  # → 1 (one transposition = odd)
Permutation([1, 2, 0]).parity()  # → 0 (two transpositions = even)
```

A permutation's parity equals `sum(len(cycle) - 1 for cycle in cyclic_form) % 2`.

#### 5. `signature()` — +1 or -1

```python
Permutation([1, 0, 2]).signature()  # → -1
Permutation([1, 2, 0]).signature()  # → 1
```

Just `1 if parity() == 0 else -1`. Useful for sign-weighted tensor operations.

#### 6. `transpositions()` — decompose into 2-cycles

```python
Permutation([1, 2, 0]).transpositions()  # → [(0, 1), (0, 2)]
```

Each cycle `(a, b, c, ...)` decomposes as `(a, c), (a, b)` (sympy convention: rightmost applied first). Returns `list[tuple[int, int]]`.

### PermutationGroup — 6 additions

#### 7. `contains(perm)` — membership test

```python
G = PermutationGroup.cyclic(3)
G.contains(Permutation([1, 2, 0]))  # → True
G.contains(Permutation([0, 2, 1]))  # → False (transposition not in C_3)
```

Enumerate elements (already cached via Dimino) and check membership.

#### 8. `is_transitive` — property

```python
PermutationGroup.symmetric(3).is_transitive  # → True (one orbit)
```

True when `orbits()` returns a single orbit. This means the group acts on all axes — no axis is isolated.

#### 9. `is_abelian` — property

```python
PermutationGroup.cyclic(4).is_abelian   # → True
PermutationGroup.symmetric(3).is_abelian # → False
```

Check: for all pairs of generators, `g1 * g2 == g2 * g1`. If generators commute, the whole group is abelian (generators generate all elements via products).

#### 10. `identity` — property

```python
G = PermutationGroup.symmetric(3)
G.identity  # → Permutation([0, 1, 2])
```

Returns `Permutation.identity(self.degree)`.

#### 11. `equals(other)` — group equality

```python
G1 = PermutationGroup(Permutation([1, 0, 2]), Permutation([0, 2, 1]))
G2 = PermutationGroup.symmetric(3)
G1.equals(G2)  # → True (different generators, same group)
```

Two groups are equal iff they have the same elements. Since both are enumerated via Dimino, compare `set(self.elements()) == set(other.elements())`.

#### 12. `orbit(alpha)` — single-point orbit

```python
G = PermutationGroup(Permutation.from_cycle(5, [0, 1]))
G.orbit(0)  # → frozenset({0, 1})
G.orbit(2)  # → frozenset({2})
```

BFS from `alpha` using generators. Returns `frozenset[int]`.

## Documentation

Update `docs/api/symmetric.md` to document all 12 additions in the Permutation Groups section. Include examples matching the sympy patterns above.

## Public exports

Add `Cycle` to `__init__.py`:
```python
from whest._perm_group import Cycle, Permutation, PermutationGroup
```

## Testing

Each addition gets at least 2 tests (normal case + edge case):

- Cycle class: single cycle, chained `__call__`, `list()` with/without size
- Permutation from Cycle: `Permutation(Cycle(0,2)(1,3))`, with size kwarg
- Cycle notation: single cycle, multiple cycles, with `size=` kwarg, empty cycles
- `__call__`: normal index, identity
- `support`: identity (empty), full support, partial
- `parity`/`signature`: even, odd, identity (even)
- `transpositions`: single cycle, multiple cycles, identity (empty list)
- `contains`: member, non-member, identity always member
- `is_transitive`: transitive (S_k), intransitive (single transposition on larger set)
- `is_abelian`: abelian (C_k), non-abelian (S_3)
- `identity`: correct size and is_identity
- `equals`: same group different generators, different groups
- `orbit`: single point in transitive group, isolated point
