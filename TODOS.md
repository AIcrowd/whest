# Polishing-sym-aware-einsum-contractions-js — Deferred Items

Captured at the end of the C5 (engine↔UI integration) commit. These are
explicit follow-ups that were triaged during implementation and code review
but not landed in commits C1–C5.

## Engine

### Cleanup: delete `classificationSpec.js` and `casePresentation.js` shim
- `website/components/symmetry-aware-einsum-contractions/engine/classificationSpec.js`
  now carries an `@deprecated` JSDoc header but is still referenced by
  `symmetry-explorer.classification-spec.test.mjs` (70+ assertions) and by the
  legacy-backward-compat path in `regimePresentation.js`. Plan for removal:
  1. Migrate `symmetry-explorer.classification-spec.test.mjs` to assert against
     `SHAPE_SPEC` + `REGIME_SPEC` (or replace with shape-layer-specific tests).
  2. Drop the `LEGACY_CASE_PRESENTATION` block from `regimePresentation.js`.
  3. Delete `classificationSpec.js`.
- The file `casePresentation.js` was renamed to `regimePresentation.js` in C5.5
  via `git mv`. The old filename is gone from HEAD. No further rename work.

### `diagonalSimultaneous` refusal message when m > 4
- Currently returns "no diagonal V↔W pairing works" whenever m > 4, even though
  the search is capped (not actually attempted). Users/debuggers may see this
  and wrongly conclude diagonal does not apply.
- Fix: hoist the `m > 4` guard to `recognize()` and return a distinct reason
  like `"m = ${m} exceeds diagonal search cap of 4; falling through to
  vSetwiseStable"`.
- File: `website/components/symmetry-aware-einsum-contractions/engine/regimes/diagonalSimultaneous.js`.

### `wreath.compute` re-runs `recognize` (contract quirk)
- `wreath.compute` internally calls `this.recognize(ctx)` to recover
  `{s, b, u, baseGens}`. This is duplicate work (recognition cost paid twice
  per fire) and a soft violation of the plugin contract (no other regime does
  this).
- Fix: thread the verdict through `computeAccumulation` into `compute(ctx, verdict)`.
  Touches the plugin contract, so it's a cross-cutting change.

### `vSetwiseStable` kernel sub-trace is narrative-only
- The sub-trace adds a `kernelReduction` entry when |K| > 1 but the algorithm
  does NOT actually use K to accelerate the hidden Burnside. The spec's
  `|[n]^W / G_u| = |([n]^W / K) / (G_u / K)|` reduction is described but not
  implemented.
- Fix: implement the quotient reduction, caching hidden-orbit counts across
  visible orbits. Real speedup when |K| is large.

### `bruteForceEstimate` overflow short-circuit
- `bruteForceEstimate` returns `groupOrder * Π sizes`. When both are large
  (e.g., sizes [10,10,10,10,10] × |G|=720), the product exceeds `2^53` and
  Number precision drops. The `<= budget` comparison still returns `false`
  correctly, but the returned value shown in UI / logged is imprecise.
- Fix: short-circuit `if (total > budget) break;` inside the loop.
- File: `website/components/symmetry-aware-einsum-contractions/engine/budget.js`.

### Extract duplicate combinatorics helpers
- Four regime files (`fullSymmetric`, `alternating`, `wreath`, `diagonalSimultaneous`)
  each define their own `binomial`, `factorial`, or `fallingFactorial`.
- Fix: extract to `engine/math/combinatorics.js` and import uniformly.

### subTrace contract documentation
- The `subTrace` field of a regime's `compute` return is undocumented.
  `bruteForceOrbit` sets it to `undefined` explicitly; `directProduct`,
  `singleton`, etc. omit it (→ `undefined`); `vSetwiseStable` sets it when
  `|K| > 1`.
- Fix: document the shape in the rationale block of `regimeRegistry.js` or in
  a `types.js` file.

### Enrich property test with multi-regime overlap
- `symmetry-explorer.regime-property.test.mjs` covers 6 contexts. Add contexts
  where multiple non-adjacent regimes in the ladder fire (e.g., S_3 with |V|=1
  triggers both `singleton` and `fullSymmetric`; a diagonal-on-block structure
  would trigger both `wreath` and `vSetwiseStable`).
- Catches future priority-order regressions.

## UI

### Multi-component ladder highlighting
- `ComponentCostView` passes only `components[0]?.accumulation?.regimeId` to
  `DecisionLadder`. Multi-component examples lose ladder context for components
  2+.
- Fix: either render one ladder per component, or an aggregate ladder with
  tabs / indicators.

### Equality-pattern (partition) backend regime
- Explicitly cut from this pass (see spec). Reintroduce as a regime between
  `vSetwiseStable` and `bruteForceOrbit`, budget-capped at
  `Bell(|L|) · |G| ≤ 1.5e6`. Widens closed-form coverage on instances where
  no other regime fires and brute-force exceeds the orbit budget.

### Copy-citation button
- One-click copy of a LaTeX block containing the expression, declared
  symmetry, detected group, regime, count, and verification pointer. Useful
  for researchers dropping results into papers or issues.

### Dark mode
- Add `next-themes` integration and Tailwind `dark:` variants across the page.

### Persistent regime-badge legend
- Corner overlay listing each regime's color + shortLabel, persistent across
  scroll.

### Proof-sketch deep-dive pages per regime
- Each regime's `FormulaPopover` is a one-line hint. A long-form article per
  regime (derivation + example + counter-example) would give researchers a
  proper reference.

### Comparison mode
- Two presets side-by-side with deltas in count, regime, and formula. Pairs
  well with URL-shareable state from C8.

### Counter-example button per regime
- Given a preset where `fullSymmetric` fires, show "what would break this?"
  by mutating the example until the regime refuses and the ladder falls through.
  Teaches the boundary of each regime viscerally.

## Docs / process

### Update design doc with post-ship notes
- `.aicrowd/superpowers/specs/2026-04-16-polishing-sym-aware-einsum-contractions-design.md`
  is gitignored (local-only per repo convention). Capture any design deltas
  discovered during C1–C5 there so the spec stays truthful.
- Known deltas to log: `generatorIsLocalBlockInternal` in wreath (extension);
  `generators` threaded through `computeAccumulation` context (absent from the
  plan's original prose); `restrictToW` upgrade from duck-type to real
  `Permutation`; `classificationSpec.js` kept as `@deprecated` shim not deleted.

### Verification script alignment
- The external `followup_verification/verify_families.py` checks uniform-size
  cases only. Our JS engine is heterogeneous-first. A Python harness extension
  taking `sizes: Sequence[int]` would let us keep running Python ground-truth
  as we extend the regime ladder.

## Follow-ups from final UI code review

Captured after the branch-wide review post-C9. None block the merge, but worth
tracking.

### `useKeyboardShortcuts` re-subscribe churn
- `lib/useKeyboardShortcuts.js` calls `useEffect` with the `bindings` object in
  the dep array. The consumer in `SymmetryAwareEinsumContractionsApp.jsx` passes
  an inline object literal, so the effect tears down + reattaches the listener
  on every render.
- Fix: stash `bindings` in a `useRef` updated on every render, run the effect
  with an empty dep array, and dispatch through the ref. One-line change.

### Playground URL sync should debounce
- `Playground.jsx` pushes `history.replaceState` on every keystroke in the
  subscripts/output inputs. `replaceState` is cheap but noisy for extensions
  watching history.
- Fix: wrap the URL update in a 150 ms debounce.

### FormulaPopover is not keyboard-closable
- Current implementation closes on mousedown-outside but not on Escape.
- Fix: add a `keydown` listener for `Escape` in the same effect.

### Animation ignores `prefers-reduced-motion`
- `@keyframes trace-in` fires unconditionally. Add a `@media
  (prefers-reduced-motion: reduce) { .animate-trace-in { animation: none; } }`
  block in `styles.css`.

### Wire BipartiteGraph.highlightedLabels from orbit hover
- The `highlightedLabels` prop was added in C9 but has no active caller. Plan
  called for "orbit hover coloring" — need a state handler in the orbit
  inspector that sets the labels of the hovered orbit and passes them down.

### Extract `PLAYGROUND_SUBSCRIPT_INPUT_ID` constant
- Currently the id `'playground-subscripts'` is hard-coded in two places
  (the Playground input + the `'/'` keyboard shortcut handler in the app).
  Extract to a named constant to avoid silent breakage on rename.

### Clamp Playground operand rank to [1, 8]
- `updateOperandRank` clamps to `max(1, floor(v))` but the input advertises
  `max={8}`. Upper-bound the clamp so the pipeline doesn't receive very large
  ranks.

### Focus trap for ExplorerModal
- Current `role="dialog"` + `aria-modal="true"` but no tab-focus trap inside.
  For accessibility completeness, add a focus trap (or use a library like
  focus-trap-react already in the tailwind ecosystem).
