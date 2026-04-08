# Changelog

## Unreleased

### Changed

- **Symmetry detection rewritten** — the induced-symmetry mechanism is replaced
  by a subset-keyed subgraph symmetry oracle (`SubgraphSymmetryOracle`). The
  oracle analyses the bipartite structure of the einsum expression, evaluates
  symmetry lazily per operand subset, and caches results. This correctly handles
  intermediates (not just the top-level contraction) and eliminates over-eager
  per-step propagation.

- **Every optimizer is symmetry-aware** — the `symmetry_oracle` kwarg is plumbed
  through `_PATH_OPTIONS` so that optimal, branch-\*, greedy, and random-greedy
  algorithms all receive symmetry information. DP uses a conservative 2× reduction
  heuristic (`TODO(dp-symmetry)`). Previously only greedy received symmetry info
  in some code paths.

- **Silent fallback deleted** — the previous code silently fell back to dense
  costs when detection produced no result. The oracle now enforces that symmetry
  information is consumed. Enforcement is verified by
  `tests/test_no_silent_symmetry_drop.py`.

### Removed

- `symmetric_flop_count`'s `input_symmetries` parameter (high-level API)
- `propagate_symmetry` and related helpers
- `_detect_induced_output_symmetry` and related helpers
- `induced_output_symmetry` kwarg on `contract_path`

### Fixed

- **Heterogeneous block dimensions in `unique_elements`** — the stars-and-bars
  block-cardinality calculation assumed all axes within a block had the same
  dimension. For rectangular block-symmetric tensors (e.g.
  `einsum('ab,cd->abcd', X, X)` with `X` of shape `(3, 4)`), it computed
  `n**s = 3**2 = 9` instead of the correct product `3*4 = 12`, silently
  underestimating the unique-element count by up to ~8× on rank-3 cases.
  `block_card` is now computed as `prod(size_dict[c] for c in blocks[0])`,
  which reduces to the old formula for per-index groups and gives the
  correct product for block groups with differing axis sizes.

### Added

- **Enriched `PathInfo` display** — `me.einsum_path().format_table(verbose=False)`
  (called by `__str__`) now shows an `Optimizer:` header line resolving
  `optimize='auto'`/`'auto-hq'` to the inner choice that actually ran
  (e.g. `optimal`, `dynamic_programming`, `random_greedy_128`), a
  `contract` column giving the path-supplied contraction tuple, and a
  `unique/dense` column showing the bare element counts that the
  symmetry savings derive from. Call `format_table(verbose=True)` for an
  indented detail row per step showing the merged operand subset, the
  intermediate's output shape, and the running cumulative cost — the
  most useful view when debugging why a particular step's savings are
  what they are.
- **New `PathInfo.optimizer_used: str`** field and new `StepInfo` fields
  `path_indices: tuple[int, ...]` and `merged_subset: frozenset[int] | None`.
  The `merged_subset` field is the exact key
  `SubgraphSymmetryOracle.sym(...)` uses for its lookups, making the
  symmetry column directly attributable to the oracle's view of each
  intermediate.

## 0.2.0 (2026-04-03)

Second release with unified einsum cost model, NumPy compatibility testing, and expanded operation coverage.

### New features

- **Unified einsum cost model** — all einsum-like operations (einsum, dot, matmul, tensordot) now share a single cost model based on opt_einsum's contraction path optimizer
- **Symmetry-aware path finding** — the opt_einsum path optimizer now factors symmetry savings into contraction ordering decisions, producing different (cheaper) paths for symmetric inputs
- **NumPy compatibility test harness** — run NumPy's own test suite against mechestim via monkeypatching; 7,300+ tests passing across 7 NumPy test modules
- **Polynomial operations** — `polyval`, `polyfit`, `polymul`, `polydiv`, `polyadd`, `polysub`, `poly`, `roots`, `polyder`, `polyint` with analytical FLOP costs
- **Window functions** — `bartlett`, `hamming`, `hanning`, `blackman`, `kaiser` with per-function cost formulas
- **FFT module** — `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfftn`, `irfftn` and free helpers (`fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`)
- **Client-server architecture** — `mechestim-client` and `mechestim-server` packages for sandboxed competition evaluation over ZMQ
- **Global default budget** — a 1e15 FLOP budget auto-activates on first use, so explicit `BudgetContext` is no longer required for quick scripts
- **`MECHESTIM_DEFAULT_BUDGET` env var** — configure the global default budget amount
- **`budget_live()`** — Rich-based live-updating budget display context manager
- **`einsum_path()`** — inspect contraction plans with per-step symmetry savings without spending budget
- **90%+ test coverage gate** enforced in CI

### Breaking changes

- Einsum cost formula now uses `product_of_all_index_dims × op_factor` (op_factor=2 for inner products, 1 for outer products), matching opt_einsum convention. Previously used a different formula.
- `me.dot` and `me.matmul` costs are now computed via the einsum cost model instead of separate formulas.

### Bug fixes

- Accept scalars and array-likes in all mechestim functions
- Fix symmetry-aware greedy algorithm to actually use symmetry in path selection
- Fix `contract_path` cost reporting for output indices
- Correctly handle `symmetric_dims` propagation through multi-step contraction paths

### Documentation

- Comprehensive how-to guides for einsum, symmetry, linalg, budget planning, and debugging
- Architecture docs for client-server model and Docker deployment
- AI agent guide with `llms.txt`, `ops.json`, and cheat sheet
- NumPy compatibility testing methodology docs

## 0.1.0 (2026-04-01)

Initial release for warm-up round.

- Einsum with symmetry detection and FLOP counting
- Pointwise operations (exp, log, add, multiply, etc.)
- Reductions (sum, mean, max, etc.)
- SVD with truncated top-k
- Free tensor creation and manipulation ops
- Budget enforcement via BudgetContext
- FLOP cost query API
- NumPy-compatible API (`import mechestim as me`)
