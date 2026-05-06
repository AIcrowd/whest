# Changelog

## Unreleased

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
