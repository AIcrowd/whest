import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function read(relativePath) {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf8');
}

test('legacy explorer shell selectors have been removed from styles.css', () => {
  const app = read('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  const styles = read('./components/symmetry-aware-einsum-contractions/styles.css');

  assert.match(app, /import '\.\/styles\.css';/);

  assert.doesNotMatch(styles, /\.example-grid/);
  assert.doesNotMatch(styles, /\.custom-builder/);
  assert.doesNotMatch(styles, /\.python-preview/);
  assert.doesNotMatch(styles, /\.dimension-slider/);
  assert.doesNotMatch(styles, /\.var-input/);
  assert.doesNotMatch(styles, /\.analyze-btn/);
});
