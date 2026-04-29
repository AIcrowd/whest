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
