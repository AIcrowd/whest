import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/Playground.jsx'),
  'utf-8',
);

test('Playground exports default React component', () => {
  assert.match(source, /export default function Playground/);
});

test('Playground reads URL state on mount', () => {
  assert.match(source, /decodePlaygroundState/);
  assert.match(source, /URLSearchParams/);
});

test('Playground pushes state to URL on change', () => {
  assert.match(source, /encodePlaygroundState/);
  assert.match(source, /replaceState/);
});

test('Playground renders subscripts and output inputs', () => {
  assert.match(source, /state\.subscripts/);
  assert.match(source, /state\.output/);
});

test('Playground composes analyzeExample with LabelClusterPanel and RegimeTrace', () => {
  assert.match(source, /analyzeExample/);
  assert.match(source, /LabelClusterPanel/);
  assert.match(source, /RegimeTrace/);
});

test('Playground shows regime id + count for each component', () => {
  assert.match(source, /accumulation\?\.regimeId/);
  assert.match(source, /accumulation\?\.count/);
});
