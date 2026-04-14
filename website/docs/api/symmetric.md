---
sidebar_position: 3
sidebar_label: Symmetric Tensors
---
# Symmetric Tensors

First-class symmetric tensor support for automatic FLOP cost reductions.

`SymmetricTensor` is an `ndarray` subclass that carries symmetry metadata
through operations. When passed to any whest operation, the cost is
automatically reduced based on the number of unique elements.

See [Exploit Symmetry Savings](../how-to/exploit-symmetry.md) for usage patterns.

::: whest._symmetric

---

## Permutation Groups

First-class permutation group support for declaring and inspecting tensor
symmetries beyond the full symmetric group S_k.

::: whest._perm_group

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

### Sympy Interoperability

`Permutation` and `PermutationGroup` use the same internal representation
(integer array form) as sympy's `combinatorics` module. Convert with:

- `perm.as_sympy()` → `sympy.combinatorics.Permutation`
- `Permutation.from_sympy(sp)` → whest `Permutation`
- `group.as_sympy()` → `sympy.combinatorics.PermutationGroup`
- `PermutationGroup.from_sympy(spg, axes=...)` → whest `PermutationGroup`

Requires `pip install sympy`. Sympy is not a required dependency.

---

## PathInfo

Contraction path with per-step diagnostics. Returned by `we.einsum_path()`.

::: whest.PathInfo

| Field | Type | Description |
|-------|------|-------------|
| `path` | `list[tuple[int, ...]]` | Sequence of contraction index groups |
| `steps` | `list[StepInfo]` | Per-step diagnostics |
| `naive_cost` | `int` | FLOP cost without path optimization |
| `optimized_cost` | `int` | FLOP cost along the optimal path |
| `largest_intermediate` | `int` | Max number of elements in any intermediate tensor |
| `speedup` | `float` | `naive_cost / optimized_cost` |
| `input_subscripts` | `str` | Comma-separated input subscripts (e.g. `"ij,jk,kl"`) |
| `output_subscript` | `str` | Output subscript (e.g. `"il"`) |
| `size_dict` | `dict[str, int]` | Label → dimension size |
| `optimizer_used` | `str` | Name of the path-finder actually invoked. For `optimize='auto'`/`'auto-hq'` this resolves to the inner choice (e.g. `'optimal'`, `'branch_2'`, `'dynamic_programming'`, `'random_greedy_128'`). `'trivial'` for `num_ops ≤ 2` cases where no optimizer runs. |

### Methods

**`format_table(verbose: bool = False) -> str`**

Render the path as a printable table. `__str__` calls this with `verbose=False`.

When `verbose=True`, an indented second row is emitted under each step
showing the merged operand subset (`subset={0,1,2}` — the key the oracle
uses for its symmetry lookup), the intermediate's output shape
(`out_shape=(10,10,10)`), and the running cumulative cost (`cumulative=2,208`).
This is the most useful view when debugging why a particular step's
symmetry savings are what they are.

---

## StepInfo

Per-step contraction info within a `PathInfo`. Each step represents one
pairwise contraction along the optimal path.

::: whest.StepInfo

| Field | Type | Description |
|-------|------|-------------|
| `subscript` | `str` | Einsum subscript for this pairwise step (e.g., `'ijk,ai->ajk'`) |
| `flop_cost` | `int` | Symmetry-aware FLOP cost of this step, exploiting cross-group symmetry including contracted indices when applicable. See [Einsum cost model](../concepts/flop-counting-model.md#einsum-cost-model). |
| `dense_flop_cost` | `int` | FLOP cost without symmetry savings |
| `symmetry_savings` | `float` | `1 - (flop_cost / dense_flop_cost)` — fraction of cost saved by symmetry |
| `input_groups` | `list[PermutationGroup \| None]` | Symmetry of each input to this step. In the oracle-based flow this is typically `[None, None]` — the oracle computes each intermediate's output symmetry directly from its operand subset rather than chaining per-input groups through the tree. The field is retained for backward compatibility and for future per-input tracking. The informative side is `output_group`. |
| `output_group` | `PermutationGroup \| None` | Symmetry of the step's output, as derived by `SubgraphSymmetryOracle.sym(merged_subset)`. Displays as `S2`, `C3`, `D4`, `G(order)` etc. depending on the group structure. |
| `input_shapes` | `list[tuple[int, ...]]` | Shapes of input operands |
| `output_shape` | `tuple[int, ...]` | Shape of the output tensor |
| `blas_type` | `str \| bool` | BLAS classification (e.g. `'GEMM'`, `'TDOT'`, `'DOT'`) or `False` for non-BLAS steps |
| `path_indices` | `tuple[int, ...]` | The path-supplied contraction tuple for this step (the entry from `PathInfo.path[i]`). Lets you cross-reference the table with the raw path field. |
| `merged_subset` | `frozenset[int] \| None` | Subset of **original** operand positions that this step's output intermediate covers. For a step contracting two original operands `i` and `j`, this is `frozenset({i, j})`. For later steps it's the union of the subsets of all SSA inputs being contracted. This is the exact key `SubgraphSymmetryOracle.sym(...)` used to derive `output_symmetry`, so any symmetry shown in the table is directly attributable to this subset. |
