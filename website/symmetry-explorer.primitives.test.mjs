import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function read(relativePath) {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf8');
}

test('legacy explorer shell selectors have been removed from styles.css', () => {
  const app = read('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  const groupView = read('./components/symmetry-aware-einsum-contractions/components/GroupView.jsx');
  const styles = read('./components/symmetry-aware-einsum-contractions/styles.css');

  assert.match(app, /import '\.\/styles\.css';/);
  assert.doesNotMatch(app, /section-desc/);
  assert.doesNotMatch(app, /pill pill-v/);
  assert.doesNotMatch(app, /pill pill-w/);
  assert.doesNotMatch(groupView, /pill pill-v/);
  assert.doesNotMatch(groupView, /pill pill-w/);

  assert.doesNotMatch(styles, /\.example-grid/);
  assert.doesNotMatch(styles, /\.custom-builder/);
  assert.doesNotMatch(styles, /\.python-preview/);
  assert.doesNotMatch(styles, /\.dimension-slider/);
  assert.doesNotMatch(styles, /\.var-input/);
  assert.doesNotMatch(styles, /\.analyze-btn/);
  assert.doesNotMatch(styles, /^\.section\s*\{/m);
  assert.doesNotMatch(styles, /\.section-desc/);
  assert.doesNotMatch(styles, /^\.pill\s*\{/m);
  assert.doesNotMatch(styles, /\.pill-v/);
  assert.doesNotMatch(styles, /\.pill-w/);

  // The pseudocode-* IDE-chrome styles were removed when Mental Framework
  // moved from a modal into the AlgorithmAtAGlance preamble with its own
  // editorial-light code card. These must no longer be present.
  assert.doesNotMatch(styles, /\.pseudocode-editor/);
  assert.doesNotMatch(styles, /\.pseudocode-code-line/);

  // Still-used visualization styles.
  assert.match(styles, /\.orbit-inspector-kicker/);
  assert.match(styles, /\.matrix-table/);
  assert.match(styles, /\.orbit-detail-card/);
});
