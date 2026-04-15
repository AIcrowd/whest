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
  assert.match(app, /import '\.\/styles\.css';/);

  assert.doesNotMatch(styles, /\.example-grid/);
  assert.doesNotMatch(styles, /\.custom-builder/);
  assert.doesNotMatch(styles, /\.python-preview/);
  assert.doesNotMatch(styles, /\.dimension-slider/);
  assert.doesNotMatch(styles, /\.var-input/);
  assert.doesNotMatch(styles, /\.analyze-btn/);

  assert.doesNotMatch(styles, /\.app-header/);
  assert.doesNotMatch(styles, /\.einsum-banner/);
  assert.doesNotMatch(styles, /\.subtitle/);
  assert.doesNotMatch(styles, /\.step-nav/);
  assert.doesNotMatch(styles, /\.section-header/);
});
