import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('Acts 2-4 are sequenced around the inline savings narrative', () => {
  const appSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');
  const componentCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx', import.meta.url), 'utf8');
  const totalCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx', import.meta.url), 'utf8');

  assert.match(appSource, /EXPLORER_ACTS\[1\]\.heading/);
  assert.match(appSource, /EXPLORER_ACTS\[2\]\.heading/);
  assert.match(appSource, /EXPLORER_ACTS\[3\]\.heading/);
  // Act 4 must read its bridge text from EXPLORER_ACTS instead of hardcoding,
  // for symmetry with Acts 2/3 (single source of truth in explorerNarrative.js).
  assert.match(appSource, /EXPLORER_ACTS\[3\]\.bridge/);
  // Interaction Graph card caption must name the math (edge = co-permuted)
  // AND the consequence (components factor the cost). See fizzy-nibbling-turtle plan.
  assert.match(componentCostSource, /moves together/);
  assert.match(componentCostSource, /factor the cost into independent sub-problems/);
  assert.match(appSource, /EXPLORER_ACTS\[4\]\.question/);
  assert.doesNotMatch(totalCostSource, /payoff of the previous acts/);
});

test('Act 4 no longer carries the Mental Framework modal — it is now the preamble', () => {
  const appSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');

  // Mental Framework is now a permanent preamble section, not a modal on Act 4.
  assert.doesNotMatch(appSource, /Open Mental Framework/);
  assert.doesNotMatch(appSource, /showMentalModel/);
  assert.doesNotMatch(appSource, /reduceMentalModelVisibility/);
  assert.doesNotMatch(appSource, /buildMentalModelCode/);
  // The app renders the new preamble above Act 1.
  assert.match(appSource, /import AlgorithmAtAGlance/);
  assert.match(appSource, /<AlgorithmAtAGlance/);
});

test('TotalCostView renders the current savings metric cards', () => {
  const totalCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx', import.meta.url), 'utf8');

  assert.match(totalCostSource, /space-y-8/);
  assert.match(totalCostSource, /ExplorerMetricCard/);
  assert.match(totalCostSource, /% savings/);
  assert.match(totalCostSource, /Cost:/);
  assert.match(totalCostSource, /Speedup:/);
  assert.doesNotMatch(totalCostSource, /TableHeader/);
  assert.doesNotMatch(totalCostSource, /TableBody/);
  assert.doesNotMatch(totalCostSource, /text-xs font-semibold uppercase tracking-wide/);
  assert.match(totalCostSource, /border-coral\/30 bg-coral-light/);
  assert.match(totalCostSource, /border-green-600\/20 bg-green-600\/5/);
});

test('ComponentCostView renders the decision ladder and component table', () => {
  const componentCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx', import.meta.url), 'utf8');

  // DecisionTree was replaced by DecisionLadder in C5 (shape + regime ladder).
  assert.match(componentCostSource, /DecisionLadder/);
  // Trivial case is still a recognized branch (e.g. trivial orbit enumeration is disabled).
  assert.match(componentCostSource, /isTrivial\(comp\)/);
  // Per-component Mₐ and αₐ come from the engine fields populated by
  // decomposeClassifyAndCount + accumulationCount — the column values
  // displayed in the table must match the hero formula (∏_a Mₐ, ∏_a αₐ).
  assert.match(componentCostSource, /multiplicationCount\(comp\)/);
  assert.match(componentCostSource, /accumulationCount\(comp\)/);
  assert.match(componentCostSource, /Orbits \(Mₐ\)/);
  assert.match(componentCostSource, /Accumulation \(αₐ\)/);
  // The per-component table must be able to horizontally scroll on narrow
  // viewports instead of silently overflowing the page.
  assert.match(componentCostSource, /overflow-x-auto rounded-xl/);
  assert.match(componentCostSource, /min-w-0 space-y-6/);
});

test('TotalCostView explains how per-component costs aggregate into the global total', () => {
  const totalCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx', import.meta.url), 'utf8');

  // Helpers the explainer block depends on.
  assert.match(totalCostSource, /import GlossaryProse from '\.\/GlossaryProse\.jsx'/);
  assert.match(totalCostSource, /import Latex from '\.\/Latex\.jsx'/);

  // The explainer block itself.
  assert.match(totalCostSource, /How components combine/);
  assert.match(totalCostSource, /AggregationExplainer/);

  // Top-line formula: Burnside unrolled on the μ arm, ∏_a α_a on the α arm.
  // The hero renders the real machinery now, not a three-term shorthand.
  assert.match(totalCostSource, /AGGREGATION_FORMULA/);
  assert.match(totalCostSource, /\\text\{Total\}\s*\\;=\\;/);
  assert.match(totalCostSource, /\(k-1\)\s*\\cdot\s*\\prod_\{a\}/);
  assert.match(totalCostSource, /\\tfrac\{1\}\{\|G_a\|\}/);
  assert.match(totalCostSource, /\\sum_\{g\s*\\in\s*G_a\}/);
  assert.match(totalCostSource, /\\prod_\{c\}\s*n_c/);
  assert.match(totalCostSource, /\\prod_\{a\}\s*\\alpha_a/);

  // Glossary — rendered as a definition list. Hybrid policy: covers every
  // symbol in the top line plus any piecewise symbol that appears in two or
  // more rows. One-off symbols live in leaf-badge tooltips.
  assert.match(totalCostSource, /AGGREGATION_LEGEND/);
  assert.match(totalCostSource, /number of operand tensors/);
  assert.match(totalCostSource, /symmetry group acting on component/);
  // V and W — phrases wrapped in JSX spans so each can carry its own color
  // (the V/W coloring matches the Interaction Graph legend).
  assert.match(totalCostSource, /free \(output\) labels/);
  assert.match(totalCostSource, /summed \(contracted\) labels/);
  assert.match(totalCostSource, /orbit decomposition/);
  assert.match(totalCostSource, /accumulation cost/);
  assert.match(totalCostSource, /<dl/);
  assert.match(totalCostSource, /<dt/);
  assert.match(totalCostSource, /<dd/);

  // Six leaves from the current SHAPE × REGIME classification
  // (shapeSpec.js + regimeSpec.js). Leaf ids match the canonical regime/shape
  // ids so CaseBadge resolves color + tooltip from the live spec — no
  // duplicated content in this file.
  assert.match(totalCostSource, /AGGREGATION_LEAVES/);
  for (const leaf of [
    "id: 'trivial'",
    "id: 'allVisible'",
    "id: 'allSummed'",
    "id: 'singleton'",
    "id: 'directProduct'",
    "id: 'bruteForceOrbit'",
  ]) {
    assert.ok(totalCostSource.includes(leaf), `expected leaf ${leaf} in TotalCostView`);
  }

  // Leaf badges reuse CaseBadge so their tooltip content stays in sync with
  // the regime/shape specs and the rest of the page.
  assert.match(totalCostSource, /CaseBadge\s+regimeId=\{leaf\.id\}/);
});
