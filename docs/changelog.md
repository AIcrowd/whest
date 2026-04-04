# Changelog

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
