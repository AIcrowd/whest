import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function read(relativePath) {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf8');
}

test('symmetry explorer is pinned to shared primitives', () => {
  const app = read('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  const exampleChooser = read('./components/symmetry-aware-einsum-contractions/components/ExampleChooser.jsx');
  const caseBadge = read('./components/symmetry-aware-einsum-contractions/components/CaseBadge.jsx');
  const componentCostView = read('./components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  const totalCostView = read('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');
  const styles = read('./components/symmetry-aware-einsum-contractions/styles.css');

  assert.match(app, /ExplorerSectionCard/);
  assert.match(exampleChooser, /ExplorerField/);
  assert.match(exampleChooser, /Button/);
  assert.match(exampleChooser, /Input/);
  assert.match(exampleChooser, /ExplorerSectionCard/);
  assert.match(caseBadge, /Badge/);
  assert.match(totalCostView, /Table/);
  assert.match(totalCostView, /ExplorerMetricCard/);
  assert.match(componentCostView, /ExplorerMetricCard/);
  assert.doesNotMatch(componentCostView, /function MetricCard/);

  assert.equal(styles.includes("@import url('https://fonts.googleapis.com"), false);
  assert.equal(styles.includes('--font-accent:'), false);
  assert.equal(styles.includes('--coral:'), false);
  assert.equal(styles.includes('.example-grid'), false);
  assert.equal(styles.includes('.custom-builder'), false);
  assert.equal(styles.includes('.python-preview'), false);
  assert.equal(styles.includes('.dimension-slider'), false);
  assert.equal(styles.includes('.var-input'), false);
  assert.equal(styles.includes('.analyze-btn'), false);

  assert.equal(styles.includes('.app-header'), false);
  assert.equal(styles.includes('.einsum-banner'), false);
  assert.equal(styles.includes('.subtitle'), false);
  assert.equal(styles.includes('.step-nav'), false);
  assert.equal(styles.includes('.section-header'), false);
});
