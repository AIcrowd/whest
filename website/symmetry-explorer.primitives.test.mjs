import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function read(relativePath) {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf8');
}

test('symmetry explorer is pinned to shared primitives', () => {
  const app = read('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  const exampleChooser = read('./components/symmetry-aware-einsum-contractions/components/ExampleChooser.jsx');
  const totalCostView = read('./components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');
  const styles = read('./components/symmetry-aware-einsum-contractions/styles.css');

  assert.match(app, /ExplorerSectionCard/);
  assert.match(exampleChooser, /ExplorerField/);
  assert.match(totalCostView, /Table/);
  assert.match(totalCostView, /ExplorerMetricCard/);

  assert.equal(styles.includes('.app-header'), false);
  assert.equal(styles.includes('.einsum-banner'), false);
  assert.equal(styles.includes('.subtitle'), false);
});
