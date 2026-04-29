import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('worked-example primitives are exported from a shared module', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/workedExample/index.jsx');
  for (const name of [
    'WorkedExampleTensorRef',
    'WorkedExampleCoords',
    'WorkedExampleTensorProduct',
    'WorkedExampleDisplayEquation',
    'WorkedExampleEquation',
    'WorkedExampleEquationLedger',
    'WorkedExampleNote',
    'APPENDIX_PROSE_CLASS',
    'APPENDIX_MONO_LEDGER_CLASS',
  ]) {
    assert.match(src, new RegExp(`export\\s+(function|const)\\s+${name}\\b`));
  }
});

test('ExpressionLevelModal imports from workedExample/ rather than redefining', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx');
  assert.match(src, /from '\.\/workedExample\/index\.jsx'/);
  assert.doesNotMatch(src, /^function WorkedExampleTensorRef/m);
  assert.doesNotMatch(src, /^function WorkedExampleEquation\(/m);
  assert.doesNotMatch(src, /^function WorkedExampleNote/m);
});

test('WorkedExamplePanel renders three states distinguishable in source', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/WorkedExamplePanel.jsx');
  assert.match(src, /data-testid="worked-example-panel"/);
  // Empty state: white-on-white with a visible "Worked example" eyebrow +
  // hint copy so the column reads as a real surface.
  assert.match(src, /Hover any cell on the left/);
  // Hovered/pinned state also shows the "Worked example" eyebrow.
  assert.match(src, /Worked example/);
  assert.match(src, /clear pin|× clear/i);
});

test('WorkedExamplePanel renders mini row preview, projection sketch, and two ledgers', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/WorkedExamplePanel.jsx');
  assert.match(src, /data-testid="worked-example-row-preview"/);
  assert.match(src, /data-testid="worked-example-projection"/);
  assert.match(src, /this Q|THIS Q/);
  assert.match(src, /other reached|OTHER Q/);
});

test('WorkedExamplePanel reuses the workedExample/ primitives + Latex', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/WorkedExamplePanel.jsx');
  // Latex import for the einsum equation
  assert.match(src, /import Latex from '\.\.\/Latex\.jsx'/);
  // Pulls helpers from the layout module (labelledTuple, tupleKey)
  assert.match(src, /from '\.\/orbitRepMatrixLayout\.js'/);
});
