import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx'),
  'utf-8',
);

test('app imports Playground + useKeyboardShortcuts', () => {
  assert.match(source, /import Playground/);
  assert.match(source, /useKeyboardShortcuts/);
});

test('app registers ArrowLeft/ArrowRight/r/slash shortcuts', () => {
  assert.match(source, /ArrowLeft/);
  assert.match(source, /ArrowRight/);
  assert.match(source, /playground-subscripts/);
});

test('app renders Playground inside a dedicated section', () => {
  assert.match(source, /id="playground"/);
  assert.match(source, /<Playground/);
});
