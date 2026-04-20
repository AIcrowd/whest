import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/ProgressiveMath.jsx'),
  'utf-8',
);

test('ProgressiveMath exports default React component', () => {
  assert.match(source, /export default function ProgressiveMath/);
});

test('ProgressiveMath uses IntersectionObserver for reveal-on-scroll', () => {
  assert.match(source, /IntersectionObserver/);
});

test('ProgressiveMath disconnects observer after first reveal', () => {
  assert.match(source, /observer\.disconnect/);
});

test('ProgressiveMath applies opacity transition classes', () => {
  assert.match(source, /opacity-0/);
  assert.match(source, /opacity-100/);
});
