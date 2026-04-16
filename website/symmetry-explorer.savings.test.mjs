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
  assert.match(componentCostSource, /independent components/);
  assert.match(appSource, /EXPLORER_ACTS\[4\]\.question/);
  assert.doesNotMatch(totalCostSource, /payoff of the previous acts/);
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

test('TotalCostView renders the current savings metric cards', () => {
  const totalCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx', import.meta.url), 'utf8');

  assert.match(totalCostSource, /space-y-8/);
  assert.match(totalCostSource, /ExplorerMetricCard/);
  assert.match(totalCostSource, /%age Savings/);
  assert.match(totalCostSource, /Cost:/);
  assert.match(totalCostSource, /Speedup:/);
  assert.doesNotMatch(totalCostSource, /TableHeader/);
  assert.doesNotMatch(totalCostSource, /TableBody/);
  assert.doesNotMatch(totalCostSource, /text-xs font-semibold uppercase tracking-wide/);
  assert.match(totalCostSource, /border-coral\/30 bg-coral-light/);
  assert.match(totalCostSource, /border-green-600\/20 bg-green-600\/5/);
});

test('ComponentCostView renders the current decision tree and component table', () => {
  const componentCostSource = fs.readFileSync(new URL('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx', import.meta.url), 'utf8');

  assert.match(componentCostSource, /TableHeader/);
  assert.match(componentCostSource, /TableBody/);
  assert.match(componentCostSource, /DecisionTree/);
  assert.match(componentCostSource, /Direct count \(trivial\)/);
  assert.match(componentCostSource, /Orbit Enumeration/);
});
