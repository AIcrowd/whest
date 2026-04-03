# Symmetric Tensors

First-class symmetric tensor support for automatic FLOP cost reductions.

`SymmetricTensor` is an `ndarray` subclass that carries symmetry metadata
through operations. When passed to any mechestim operation, the cost is
automatically reduced based on the number of unique elements.

See [Exploit Symmetry Savings](../how-to/exploit-symmetry.md) for usage patterns.

::: mechestim._symmetric

---

## PathInfo

Contraction path with per-step diagnostics. Returned by `me.einsum_path()`.

::: mechestim.PathInfo

| Field | Type | Description |
|-------|------|-------------|
| `path` | `list[tuple[int, int]]` | Sequence of pairwise contraction indices |
| `steps` | `list[StepInfo]` | Per-step diagnostics |
| `naive_cost` | `int` | FLOP cost without path optimization |
| `optimized_cost` | `int` | FLOP cost along the optimal path |
| `largest_intermediate` | `int` | Max number of elements in any intermediate tensor |
| `speedup` | `float` | `naive_cost / optimized_cost` |

---

## StepInfo

Per-step contraction info within a `PathInfo`. Each step represents one
pairwise contraction along the optimal path.

::: mechestim.StepInfo

| Field | Type | Description |
|-------|------|-------------|
| `subscript` | `str` | Einsum subscript for this pairwise step (e.g., `'ijk,ai->ajk'`) |
| `flop_cost` | `int` | Symmetry-aware FLOP cost of this step |
| `dense_flop_cost` | `int` | FLOP cost without symmetry savings |
| `symmetry_savings` | `float` | `1 - (flop_cost / dense_flop_cost)` — fraction of cost saved by symmetry |
| `input_symmetries` | `list[IndexSymmetry | None]` | Symmetry of each input to this step |
| `output_symmetry` | `IndexSymmetry | None` | Symmetry of the step's output (propagated to next step) |
| `input_shapes` | `list[tuple[int, ...]]` | Shapes of input operands |
| `output_shape` | `tuple[int, ...]` | Shape of the output tensor |
