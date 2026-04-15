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
  assert.match(appSource, /NarrativeCallout label="Why this matters">\{EXPLORER_ACTS\[3\]\.why\}/);
  assert.match(componentCostSource, /independent components/);
  assert.match(totalCostSource, /payoff of the previous acts/);
});

test('Act 4 opens the Mental Framework in a modal using the shared code block', () => {
  const appSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url), 'utf8');

  assert.doesNotMatch(appSource, /<PseudocodeRail/);
  assert.match(appSource, /showMentalModel/);
  assert.match(appSource, /Open Mental Framework/);
  assert.match(appSource, /buildMentalModelCode/);
  assert.match(appSource, /PythonCodeBlock/);
  assert.match(appSource, /reduceMentalModelVisibility/);
  assert.match(appSource, /setShowMentalModel\(\(isOpen\) => reduceMentalModelVisibility\(isOpen, 'selectPreset'\)\)/);
  assert.match(appSource, /setShowMentalModel\(\(isOpen\) => reduceMentalModelVisibility\(isOpen, 'customMode'\)\)/);
  assert.match(appSource, /setShowMentalModel\(\(isOpen\) => reduceMentalModelVisibility\(isOpen, 'customExample'\)\)/);
});

test('TotalCostView renders savings totals with shared table and metric primitives', () => {
  const totalCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx', import.meta.url), 'utf8');

  assert.match(totalCostSource, /space-y-8/);
  assert.match(totalCostSource, /TableHeader/);
  assert.match(totalCostSource, /TableBody/);
  assert.match(totalCostSource, /ExplorerMetricCard/);
  assert.doesNotMatch(totalCostSource, /text-xs font-semibold uppercase tracking-wide/);
  assert.match(totalCostSource, /px-4 py-3 text-sm font-semibold uppercase tracking-\[0\.16em\] text-muted-foreground/);
  assert.match(totalCostSource, /px-4 py-3 text-sm/);
  assert.match(totalCostSource, /font-mono text-sm text-foreground/);
  assert.match(totalCostSource, /border-coral\/30 bg-coral-light/);
  assert.match(totalCostSource, /border-green-600\/20 bg-green-600\/5/);
});

test('ComponentCostView composes component summaries from shared metric primitives', () => {
  const componentCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx', import.meta.url), 'utf8');

  assert.doesNotMatch(componentCostSource, /function MetricCard/);
  assert.doesNotMatch(componentCostSource, /text-\[11px\]/);
  assert.match(componentCostSource, /ExplorerMetricCard/);
  assert.match(componentCostSource, /text-sm font-semibold uppercase tracking-\[0\.16em\] text-muted-foreground/);
  assert.match(componentCostSource, /label=\"Method\"/);
  assert.match(componentCostSource, /DecisionTree/);
});
