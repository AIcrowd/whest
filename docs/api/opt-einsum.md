# Path Optimizer (opt_einsum fork)

mechestim includes a vendored fork of [opt_einsum](https://github.com/dgasmith/opt_einsum) by Daniel G. A. Smith and Johnnie Gray (MIT license). The fork lives at [`src/mechestim/_opt_einsum/`](https://github.com/AIcrowd/mechestim/tree/main/src/mechestim/_opt_einsum) and is used internally by `me.einsum()` and `me.einsum_path()`.

## What is opt_einsum?

[opt_einsum](https://github.com/dgasmith/opt_einsum) is a library for optimizing the contraction order of Einstein summations. Given a multi-operand einsum like `ij,jk,kl->il`, it finds the pairwise contraction sequence that minimizes total FLOPs. For background, see the [opt_einsum documentation](https://dgasmith.github.io/opt_einsum/) and the paper [*Opt_einsum — A Python library for optimizing contraction order*](https://joss.theoj.org/papers/10.21105/joss.00753).

## What our fork adds

This fork extends opt_einsum with **symmetry-aware path finding**. When input tensors have permutation symmetry (e.g., a symmetric matrix where `A[i,j] = A[j,i]`) or the same Python object is passed at multiple operand positions, the fork:

1. **Uses symmetry to choose contraction order.** The path algorithms (optimal, branch-and-bound, greedy, random-greedy, and dynamic programming) all account for symmetry when scoring candidate contractions, preferring orders that exploit symmetric structure. Every optimizer uses the exact `unique/dense` ratio derived from the subgraph-symmetry oracle.

2. **Tracks symmetry by operand subset.** Each intermediate tensor encountered during path search has its symmetry derived by the `SubgraphSymmetryOracle` from the **subset of original operands it contracts** — not by restricting each input's symmetry groups step-by-step. The oracle runs a subgraph-level analysis on the bipartite graph for that subset and returns the output symmetry directly. Results are cached per subset for the duration of a `contract_path` call. See [the subgraph symmetry explanation](../explanation/subgraph-symmetry.md) for the algorithm.

3. **Reports symmetry-aware costs.** Each step's cost is reduced by the exact ratio of unique output elements to total output elements, computed via the stars-and-bars formula `C(B + k - 1, k)` where `k` is the number of interchangeable blocks in a symmetry group and `B` is the cardinality of one block (the product of axis sizes within the block). For per-index groups (block size 1) this reduces to `C(n + k - 1, k)`. Both the symmetry-reduced cost and the dense cost are reported in `PathInfo`, so you can see exactly where symmetry helps. See [Symmetry savings in the FLOP counting model](../concepts/flop-counting-model.md#symmetry-savings) for the full derivation including the heterogeneous-axis block case.

4. **Classifies symmetric BLAS operations.** Pairwise contractions where an input has a symmetric group covering 2+ of its indices are labelled with specialised BLAS types (`SYMM`, `SYMV`, `SYDT`) instead of the generic `GEMM`, `GEMV`/`EINSUM`, `DOT`. These labels are informational — they don't affect cost estimation but help identify where symmetric BLAS routines (like LAPACK's `dsymm`) could be dispatched.

## What was removed from upstream

- Execution logic (`contract`, `_core_contract`) — mechestim handles execution via `numpy.einsum`
- Backend dispatch (JAX, TensorFlow, PyTorch, Theano, CuPy)
- Intermediate caching/sharing layer

The fork is **self-contained** (zero imports from mechestim, depends only on Python stdlib + numpy) and could be contributed back upstream.

## Path algorithms

All algorithms are symmetry-aware — they receive symmetry information from the `symmetry_oracle` kwarg and use it to score candidate contractions.

| Algorithm | `optimize=` | Used by | Symmetry-aware ordering? |
|-----------|-------------|---------|--------------------------|
| Optimal (brute-force DFS) | `'optimal'` | `'auto'` for 1-4 operands | Yes |
| Branch-and-bound | `'branch-all'`, `'branch-2'` | `'auto'` for 5-14 operands | Yes |
| Greedy | `'greedy'` | `'auto'` for 15+ operands | Yes |
| Dynamic programming | `'dp'` | `'auto-hq'` for 6-16 operands | Yes |
| Random greedy | `'random-greedy'`, `'random-greedy-128'` | `'auto-hq'` for 17+ operands | Yes (via greedy) |
| Auto | `'auto'` (default) | — | Dispatches to above |

## Key types

### `IndexSymmetry`

```python
IndexSymmetry = list[frozenset[tuple[str, ...]]]
```

The fork's native symmetry representation. Each `frozenset` names blocks of
einsum index characters that are symmetric under permutation. Per-index groups
use 1-tuples: `frozenset({('i',), ('j',)})` means S₂ on `{i, j}`. Block groups
use k-tuples: `frozenset({('a','b'), ('c','d')})` means the two 2-label blocks
can swap as a unit.

mechestim's `_einsum.py` converts between positional `SymmetryInfo` (used by `SymmetricTensor`) and character-based `IndexSymmetry` at the boundary.

### `SubsetSymmetry`

```python
@dataclass(frozen=True)
class SubsetSymmetry:
    output: IndexSymmetry | None  # V-side: output tensor symmetry
    inner: IndexSymmetry | None   # W-side: inner summation symmetry
```

Returned by `SubgraphSymmetryOracle.sym(subset)`. The `.output` field carries
the same V-side symmetry that consumers use for cost reduction. The `.inner`
field carries W-side symmetry among contracted labels, used when
`use_inner_symmetry=True`.

### `PathInfo` and `StepInfo`

See [Symmetric Tensors API](./symmetric.md#pathinfo) for the full dataclass reference.

## Path finding parameters

### `symmetry_oracle` parameter

`contract_path` and related path algorithms accept an optional `symmetry_oracle` keyword argument. This is a `SubgraphSymmetryOracle` instance (from `mechestim._opt_einsum._subgraph_symmetry`) that provides symmetry information for each intermediate tensor encountered during path search.

The oracle is constructed once per `contract_path` call by `me.einsum` and `me.einsum_path`. It is plumbed through `_PATH_OPTIONS` so that every algorithm receives it. Most users never interact with this directly.

The oracle's `sym()` method returns a `SubsetSymmetry` dataclass. Access
`.output` for the output tensor's symmetry (used by all path algorithms) and
`.inner` for inner-sum symmetry (used when `use_inner_symmetry=True`).

## Deviations from upstream opt_einsum

The fork diverges from upstream opt_einsum in the following ways that are visible to users and contributors.

### Every path optimizer is symmetry-aware

In upstream opt_einsum, path algorithms operate purely on tensor shapes with no knowledge of symmetry. In this fork, every algorithm (optimal, branch-\*, greedy, dp, random-greedy) receives a `symmetry_oracle` and uses it to score candidate contractions. Symmetry information propagates through the candidate evaluation loop so that early contractions producing symmetric intermediates are preferred.

The `symmetry_oracle` kwarg is plumbed through `_PATH_OPTIONS` in `_paths.py`. Any algorithm that calls `_PATH_OPTIONS[alg](*args, **kwargs)` automatically inherits it.

### No silent symmetry fallback

Upstream opt_einsum silently ignores unknown kwargs. This fork enforces that `symmetry_oracle` is consumed. The absence of silent fallback is verified by `tests/test_no_silent_symmetry_drop.py`.

## Attribution

See the [NOTICE](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_opt_einsum/NOTICE) file for a detailed file-by-file changelog of all modifications from upstream.
