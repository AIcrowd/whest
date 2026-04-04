# Path Optimizer (opt_einsum fork)

mechestim includes a vendored fork of [opt_einsum](https://github.com/dgasmith/opt_einsum) by Daniel G. A. Smith and Johnnie Gray (MIT license). The fork lives at [`src/mechestim/_opt_einsum/`](https://github.com/AIcrowd/mechestim/tree/main/src/mechestim/_opt_einsum) and is used internally by `me.einsum()` and `me.einsum_path()`.

## What is opt_einsum?

[opt_einsum](https://github.com/dgasmith/opt_einsum) is a library for optimizing the contraction order of Einstein summations. Given a multi-operand einsum like `ij,jk,kl->il`, it finds the pairwise contraction sequence that minimizes total FLOPs. For background, see the [opt_einsum documentation](https://dgasmith.github.io/opt_einsum/) and the paper [*Opt_einsum — A Python library for optimizing contraction order*](https://joss.theoj.org/papers/10.21105/joss.00753).

## What our fork adds

This fork extends opt_einsum with **symmetry-aware path finding**. When input tensors have permutation symmetry (e.g., a symmetric matrix where `A[i,j] = A[j,i]`), the fork:

1. **Uses symmetry to choose contraction order.** The path algorithms (greedy, optimal, branch-and-bound) account for symmetry when scoring candidate contractions, preferring orders that exploit symmetric structure.

2. **Propagates symmetry through intermediates.** After each pairwise contraction, the result's symmetry is computed by restricting each input's symmetry groups to the surviving indices. This propagated symmetry influences subsequent ordering decisions.

3. **Reports symmetry-aware costs.** Each step in the contraction path includes both the symmetry-reduced cost and the dense cost, so you can see exactly where symmetry helps.

4. **Classifies symmetric BLAS operations.** Pairwise contractions involving symmetric inputs are labeled with symmetric BLAS types (SYMM, SYMV, SYDT) in addition to standard types (GEMM, DOT, TDOT).

## What was removed from upstream

- Execution logic (`contract`, `_core_contract`) — mechestim handles execution via `numpy.einsum`
- Backend dispatch (JAX, TensorFlow, PyTorch, Theano, CuPy)
- Intermediate caching/sharing layer

The fork is **self-contained** (zero imports from mechestim, depends only on Python stdlib + numpy) and could be contributed back upstream.

## Path algorithms

All algorithms accept an optional `input_symmetries` parameter for symmetry-aware path finding.

| Algorithm | `optimize=` | Used by | Symmetry-aware ordering? |
|-----------|-------------|---------|--------------------------|
| Optimal (brute-force DFS) | `'optimal'` | `'auto'` for 1-4 operands | Yes |
| Branch-and-bound | `'branch-all'`, `'branch-2'` | `'auto'` for 5-14 operands | Yes |
| Greedy | `'greedy'` | `'auto'` for 15+ operands | Yes |
| Dynamic programming | `'dp'` | `'auto-hq'` for 6-16 operands | No (dense ordering, symmetric cost reporting) |
| Random greedy | `'random-greedy'`, `'random-greedy-128'` | `'auto-hq'` for 17+ operands | Yes (via greedy) |
| Auto | `'auto'` (default) | — | Dispatches to above |

## Key types

### `IndexSymmetry`

```python
IndexSymmetry = list[frozenset[str]]
```

The fork's native symmetry representation. Each `frozenset` names einsum index characters that are symmetric under permutation. Example: `[frozenset("ijk")]` means S_3 symmetry on indices i, j, k.

mechestim's `_einsum.py` converts between positional `SymmetryInfo` (used by `SymmetricTensor`) and character-based `IndexSymmetry` at the boundary.

### `PathInfo` and `StepInfo`

See [Symmetric Tensors API](./symmetric.md#pathinfo) for the full dataclass reference.

## Attribution

See the [NOTICE](https://github.com/AIcrowd/mechestim/blob/main/src/mechestim/_opt_einsum/NOTICE) file for a detailed file-by-file changelog of all modifications from upstream.
