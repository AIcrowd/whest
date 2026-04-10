# Symmetric Tensors

First-class symmetric tensor support for automatic FLOP cost reductions.

`SymmetricTensor` is an `ndarray` subclass that carries symmetry metadata
through operations. When passed to any mechestim operation, the cost is
automatically reduced based on the number of unique elements.

See [Exploit Symmetry Savings](../how-to/exploit-symmetry.md) for usage patterns.

::: mechestim._symmetric

---

## Permutation Groups

First-class permutation group support for declaring and inspecting tensor
symmetries beyond the full symmetric group S_k.

::: mechestim._perm_group

### Sympy Interoperability

`Permutation` and `PermutationGroup` use the same internal representation
(integer array form) as sympy's `combinatorics` module. Convert with:

- `perm.as_sympy()` → `sympy.combinatorics.Permutation`
- `Permutation.from_sympy(sp)` → mechestim `Permutation`
- `group.as_sympy()` → `sympy.combinatorics.PermutationGroup`
- `PermutationGroup.from_sympy(spg, axes=...)` → mechestim `PermutationGroup`

Requires `pip install sympy`. Sympy is not a required dependency.

---

## PathInfo

Contraction path with per-step diagnostics. Returned by `me.einsum_path()`.

::: mechestim.PathInfo

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

::: mechestim.StepInfo

| Field | Type | Description |
|-------|------|-------------|
| `subscript` | `str` | Einsum subscript for this pairwise step (e.g., `'ijk,ai->ajk'`) |
| `flop_cost` | `int` | Symmetry-aware FLOP cost of this step: `min(direct, Φ)` where *direct* is `dense_cost × unique_output / total_output` and *Φ* is the symmetry-preserving bound (when applicable). See [Einsum cost model](../concepts/flop-counting-model.md#einsum-cost-model). |
| `dense_flop_cost` | `int` | FLOP cost without symmetry savings |
| `symmetry_savings` | `float` | `1 - (flop_cost / dense_flop_cost)` — fraction of cost saved by symmetry |
| `input_symmetries` | `list[IndexSymmetry \| None]` | Symmetry of each input to this step. In the oracle-based flow this is typically `[None, None]` — the oracle computes each intermediate's output symmetry directly from its operand subset rather than chaining per-input groups through the tree. The field is retained for backward compatibility and for future per-input tracking. The informative side is `output_symmetry`. |
| `output_symmetry` | `IndexSymmetry \| None` | Symmetry of the step's output, as derived by `SubgraphSymmetryOracle.sym(merged_subset)` |
| `input_shapes` | `list[tuple[int, ...]]` | Shapes of input operands |
| `output_shape` | `tuple[int, ...]` | Shape of the output tensor |
| `blas_type` | `str \| bool` | BLAS classification (e.g. `'GEMM'`, `'TDOT'`, `'DOT'`) or `False` for non-BLAS steps |
| `path_indices` | `tuple[int, ...]` | The path-supplied contraction tuple for this step (the entry from `PathInfo.path[i]`). Lets you cross-reference the table with the raw path field. |
| `merged_subset` | `frozenset[int] \| None` | Subset of **original** operand positions that this step's output intermediate covers. For a step contracting two original operands `i` and `j`, this is `frozenset({i, j})`. For later steps it's the union of the subsets of all SSA inputs being contracted. This is the exact key `SubgraphSymmetryOracle.sym(...)` used to derive `output_symmetry`, so any symmetry shown in the table is directly attributable to this subset. |
