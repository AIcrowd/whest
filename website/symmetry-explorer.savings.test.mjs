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
  assert.match(appSource, /EXPLORER_ACTS\[1\]\.introParagraphs/);
  assert.match(appSource, /EXPLORER_ACTS\[2\]\.introParagraphs/);
  assert.match(appSource, /EXPLORER_ACTS\[3\]\.introParagraphs/);
  assert.match(appSource, /EXPLORER_ACTS\[4\]\.heading/);
  assert.match(appSource, /EXPLORER_ACTS\[4\]\.question/);
  assert.match(appSource, /EXPLORER_ACTS\[4\]\.introParagraphs/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.supportingSentence/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[[1-4]\]\.bridge/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.why/);
  // Interaction Graph card caption must name the math (edge = co-permuted)
  // AND the consequence (components factor the cost). See fizzy-nibbling-turtle plan.
  assert.match(componentCostSource, /moves together/);
  assert.match(componentCostSource, /factor the cost into independent sub-problems/);
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

test('Section 5 renders the exact Option B-style Cost Savings spread', () => {
  const appSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');
  const totalCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx', import.meta.url), 'utf8');
  const denseIndex = totalCostSource.indexOf("label: 'Dense Cost'");
  const symmetryIndex = totalCostSource.indexOf("label: 'Symmetry-Aware Cost'");
  const muIndex = totalCostSource.indexOf("label: 'Multiplication Cost (μ)'");
  const alphaIndex = totalCostSource.indexOf("label: 'Accumulation Cost (α)'");
  const speedupIndex = totalCostSource.indexOf("label: 'Speedup'");
  const savingsIndex = totalCostSource.indexOf("label: '% Savings'");

  assert.match(appSource, /title={EXPLORER_ACTS\[4\]\.heading}/);
  assert.match(appSource, /description={<InlineMathText>{EXPLORER_ACTS\[4\]\.question}<\/InlineMathText>}/);
  assert.match(totalCostSource, /section5-editorial-spread/);
  assert.match(totalCostSource, /section5-editorial-header/);
  assert.match(totalCostSource, /section5-band-top/);
  assert.match(totalCostSource, /section5-band-bottom/);
  assert.match(totalCostSource, /section5-floating-sep section5-pair-sep/);
  assert.match(totalCostSource, /section5-floating-sep section5-quad-sep-1/);
  assert.match(totalCostSource, /section5-floating-sep section5-quad-sep-2/);
  assert.match(totalCostSource, /section5-floating-sep section5-quad-sep-3/);
  assert.match(totalCostSource, /TOP_COMPARISON_METRICS/);
  assert.match(totalCostSource, /SUPPORTING_METRICS/);
  assert.doesNotMatch(totalCostSource, /ExplorerMetricCard/);
  assert.doesNotMatch(totalCostSource, /label="Total Cost"/);
  assert.ok(denseIndex < symmetryIndex && symmetryIndex < muIndex && muIndex < alphaIndex && alphaIndex < speedupIndex && speedupIndex < savingsIndex);
  assert.match(totalCostSource, /Cost Savings/);
  assert.doesNotMatch(totalCostSource, /supportingSentence/);
  assert.match(totalCostSource, /mx-auto max-w-\[44rem\] bg-white .* text-center/);
  assert.match(totalCostSource, /text-\[24px\] text-gray-800/);
  assert.doesNotMatch(totalCostSource, /max-w-\[420px\] font-serif italic text-\[14px\] leading-\[1\.5\] text-gray-500/);
  assert.match(totalCostSource, /h-px w-\[64px\] bg-\[\#f3c5bf\]/);
  assert.doesNotMatch(totalCostSource, /bg-\[\#fffdfa\]/);
  assert.doesNotMatch(totalCostSource, /border-\[\#ece7e2\]/);
  assert.doesNotMatch(totalCostSource, /rounded-\[14px\]/);
  assert.match(totalCostSource, /section5-band-top relative grid grid-cols-1 sm:grid-cols-2/);
  assert.match(totalCostSource, /section5-band-bottom relative border-t border-b border-gray-100/);
  assert.match(totalCostSource, /bg-gray-100/);
  assert.doesNotMatch(totalCostSource, /border-y border-black/);
  assert.doesNotMatch(totalCostSource, /bg-black/);
  assert.match(totalCostSource, /bottom-\[18%\].*top-\[18%\]/);
  assert.match(totalCostSource, /left-1\/2/);
  assert.match(totalCostSource, /left-1\/4/);
  assert.match(totalCostSource, /left-3\/4/);
  assert.match(totalCostSource, /grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4/);
  assert.match(totalCostSource, /text-\[10px\] font-semibold uppercase tracking-\[0\.22em\] text-gray-600/);
  assert.match(totalCostSource, /font-serif text-\[54px\] leading-\[0\.95\] tracking-\[-0\.03em\]/);
  assert.match(totalCostSource, /font-serif text-\[24px\] leading-none tracking-\[-0\.02em\]/);
  assert.match(totalCostSource, /text-coral/);
  assert.match(totalCostSource, /text-green-700/);
});

test('ComponentCostView renders the decision ladder and component table', () => {
  const componentCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx', import.meta.url), 'utf8');
  const roleBadgeSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/RoleBadge.jsx', import.meta.url), 'utf8');

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
  assert.match(componentCostSource, /<RoleBadge key=\{/);
  assert.match(roleBadgeSource, /notationIdForRole/);
  assert.match(roleBadgeSource, /notationColor\(notationId\)/);
  assert.match(roleBadgeSource, /notationTint\(notationId,\s*0\.28\)/);
  assert.match(roleBadgeSource, /notationTint\(notationId,\s*0\.1\)/);
  assert.doesNotMatch(roleBadgeSource, /sky-/);
});

test('TotalCostView explains how per-component costs aggregate into the global total', () => {
  const totalCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx', import.meta.url), 'utf8');
  const recapIndex = totalCostSource.indexOf('<ComponentRecap components={components} />');
  const introIndex = totalCostSource.indexOf('<SectionFiveIntro />');
  const aggregationIndex = totalCostSource.indexOf('<AggregationExplainer />');
  const spreadIndex = totalCostSource.indexOf('<EditorialComparisonSpread');

  // Helpers the explainer block depends on.
  assert.match(totalCostSource, /import GlossaryProse from '\.\/GlossaryProse\.jsx'/);
  assert.match(totalCostSource, /import Latex from '\.\/Latex\.jsx'/);
  assert.equal(introIndex, -1);
  assert.doesNotMatch(totalCostSource, /The earlier sections identified the symmetry group, decomposed its action into components, and computed the local quantities that govern work\./);
  assert.doesNotMatch(totalCostSource, /We do that by combining the multiplication cost and the accumulation cost\./);
  assert.doesNotMatch(totalCostSource, /\\mathrm\{Total\} = \\mu \+ \\alpha/);

  // The explainer block itself.
  assert.match(totalCostSource, /How components combine/);
  assert.match(totalCostSource, /AggregationExplainer/);
  assert.match(totalCostSource, /getRegimePresentation/);
  assert.match(totalCostSource, /regimeColor \?\? componentColor\(idx\)/);

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
  assert.doesNotMatch(totalCostSource, /larger formal group/);
  assert.match(totalCostSource, /Symmetry-Aware Cost/);
  assert.ok(recapIndex !== -1 && aggregationIndex !== -1 && spreadIndex !== -1);
  assert.equal(introIndex, -1);
  assert.ok(recapIndex < aggregationIndex && aggregationIndex < spreadIndex);
  assert.doesNotMatch(totalCostSource, /Seven paths through Section 5/);
});
