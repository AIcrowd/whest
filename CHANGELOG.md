# Changelog

## Unreleased

### Changed (BREAKING)

- **Vendored opt_einsum replaced with runtime dependency.** flopscope now
  depends on `opt_einsum>=3.3.0,<4.0.0` instead of vendoring its source.
  The remaining `flopscope._opt_einsum` is a slim ~830-line shim that
  adapts upstream's `PathInfo` to flopscope's expected shape and recomputes
  per-step FLOP costs using flopscope's FMA convention.
  - `flopscope._opt_einsum.contract_path` still returns a flopscope
    `PathInfo` with the same essential fields (`path`, `steps`,
    `optimized_cost`, etc.).
  - `StepInfo` no longer carries the dead symmetry-related fields
    (`input_groups`, `output_group`, `inner_group`, `inner_applied`,
    `dense_flop_cost`, `symmetry_savings`). It now has 4-5 fields:
    `subscript`, `flop_count`, `input_shapes`, `output_shape` (plus
    `merged_subset` if still used by display).

- **Einsum cost model rewritten** to mirror the JS Symmetry-Aware Einsum
  Contractions explorer's α/M direct-event model. The charged FLOP cost is
  now path-independent: `(k - 1) · ∏ M_a + ∏ α_a` summed across components.
  - `path_info.optimized_cost` returns the new whole-expression cost. For
    expressions with declared symmetry, this number differs from the old
    per-step `cost · unique/total` formula. See migration notes below.
  - Path optimization no longer uses symmetry; `opt_einsum.contract_path`
    behaves like upstream stock opt_einsum.
  - Per-step `path_info.steps[i].flop_count` reverts to dense (no symmetry
    adjustment per step).

### Added

- `flopscope.einsum_accumulation_cost(subscripts, *operands, partition_budget=None)`
  — public inspection function returning the new `AccumulationCost` decomposition
  (path-independent, per-component breakdown, regime trace).
- `flopscope.AccumulationCost`, `flopscope.ComponentCost`, `flopscope.RegimeStep`
  — public dataclasses.
- New settings:
  - `partition_budget` (default 100 000): per-component typed-partition cap.
  - `dimino_budget` (default 500 000): whole-expression `G_pt` closure cap.
- `CostFallbackWarning` now also fires when a partition counter exceeds its
  budget; total falls back to `k · dense_baseline` (the no-symmetry direct-
  event count).
- New configurable setting `fma_cost` (default 1). Counts a fused
  multiply-add as 1 op (hardware convention). Set to 2 to get the
  textbook / opt_einsum convention.
- `flopscope._cost_model.fma_cost()` function replaces the
  `FMA_COST` constant. The constant is removed.

### Removed

- `flopscope._opt_einsum._paths` — now upstream
- `flopscope._opt_einsum._path_random` — now upstream
- `flopscope._opt_einsum._parser` — now upstream (re-exported via shim)
- `flopscope._opt_einsum._blas` — was unused dead code
- `flopscope._opt_einsum._testing` — was unused dead code
- `flopscope._opt_einsum._typing` — now upstream
- `flopscope._cost_model.FMA_COST` constant — replaced by `fma_cost()`
- `flopscope._opt_einsum._subgraph_symmetry` — internal module deleted.
- `flopscope._opt_einsum._symmetry` — internal module deleted (was mostly
  `symmetric_flop_count`, `unique_elements`, `SubsetSymmetry` — all only used
  by the deleted oracle).
- `use_inner_symmetry` setting — was a knob on the deleted oracle.

---

### BREAKING

- `BudgetContext.untracked_time` and `summary_dict()["untracked_time_s"]` now
  subtract the new `flopscope_overhead_time_s` bucket. Consumers that previously
  read `untracked_time` as a proxy for "all unattributed time" should switch to
  `flopscope_overhead_time + untracked_time` for the old value, or stay on
  `untracked_time` for the new (more meaningful) "genuinely neither" residual.

- `BudgetContext.tracked_time` semantics tightened: now reflects ONLY the wall
  time of the inner numpy call, not the entire `with budget.deduct(...):`
  block. Time spent in flopscope's pre/post-numpy work inside the block (view-
  casts, copyto fallbacks, dispatch logic) is now attributed to
  `flopscope_overhead_time`.

- `BudgetContext.wall_time_s` now spans `__init__` start through the end of
  `__exit__` (was `__enter__` start through start of `__exit__` body). For an
  empty `with BudgetContext(): pass` the value grows by ~3 µs (was ~0.4 µs,
  now ~2.7 µs); real workloads see noise-level changes. The pre-`__enter__`
  slice (`__init__` body + banner print) and the post-`__exit__` body slice
  (accumulator-record, active-budget restore) are now attributed to
  `flopscope_overhead_time` rather than falling out of any bucket. Issue #82.

### Added

- `BudgetContext.flopscope_overhead_time` property — measured wall-clock time
  spent inside flopscope's own dispatch code.
- `summary_dict()["flopscope_overhead_time_s"]` and the same key per namespace
  bucket in `summary_dict(by_namespace=True)`.
- `OpRecord.flopscope_overhead` — per-op overhead in seconds.

### Changed

- `flopscope.configure(check_nan_inf=True)` — opt-in scan time is now attributed
  to `flopscope_overhead_time` instead of `untracked_time`.

- `_NamespaceScope.__enter__/__exit__` brackets now use `try/finally` so that
  push/pop time is billed to `flopscope_overhead_time` even when the underlying
  `_push_namespace`/`_pop_namespace` raises. Issue #82.
