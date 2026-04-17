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
  assert.match(appSource, /We now decompose the detected global action/);
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

  assert.match(componentCostSource, /TableHeader/);
  assert.match(componentCostSource, /TableBody/);
  // DecisionTree was replaced by DecisionLadder in C5 (shape + regime ladder).
  assert.match(componentCostSource, /DecisionLadder/);
  // Trivial case has a dedicated code path (not a Burnside degeneration).
  assert.match(componentCostSource, /isTrivial\(comp\)/);
  assert.match(componentCostSource, /directCount\(comp/);
  assert.match(componentCostSource, /Orbit Enumeration/);
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

  // Formula pinning — catches silent edits that break the aggregation story.
  // Notation updated to match the Counting Convention band: μ = total
  // multiplication cost, α = total accumulation cost, M_a = per-component
  // orbit count (Burnside). μ_a is no longer used at the aggregation layer.
  assert.match(totalCostSource, /AGGREGATION_FORMULA/);
  assert.match(totalCostSource, /\\mu\s*=\s*\(k\s*-\s*1\)/);
  assert.match(totalCostSource, /\\prod_\{a\}\\!M_a/);
  assert.match(totalCostSource, /\\prod_\{a\}\\!\\alpha_a/);
  assert.match(totalCostSource, /\\text\{Total\}\s*=\s*\\mu\s*\+\s*\\alpha/);

  // Glossary legend for the formula variables — rendered as a definition list,
  // one row per symbol so the math is readable on small screens.
  assert.match(totalCostSource, /AGGREGATION_LEGEND/);
  assert.match(totalCostSource, /orbit count per component/);
  assert.match(totalCostSource, /total multiplication cost/);
  assert.match(totalCostSource, /accumulation cost per component/);
  assert.match(totalCostSource, /number of operand tensors/);
  assert.match(totalCostSource, /<dl/);
  assert.match(totalCostSource, /<dt/);
  assert.match(totalCostSource, /<dd/);
});
