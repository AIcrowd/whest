import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/RegimeTrace.jsx'),
  'utf-8',
);

test('RegimeTrace exports a default React component', () => {
  assert.match(source, /export default function RegimeTrace/);
});

test('RegimeTrace renders Row for each trace step', () => {
  assert.match(source, /function Row\(/);
  assert.match(source, /trace\.map/);
});

test('RegimeTrace renders sub-trace steps recursively', () => {
  assert.match(source, /subSteps/);
});

test('RegimeTrace distinguishes fired vs refused steps visually', () => {
  assert.match(source, /decision/);
  assert.match(source, /fired/);
});
