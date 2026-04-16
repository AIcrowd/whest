import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('RegimeTrace animates rows with staggered delay', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/RegimeTrace.jsx');
  assert.match(src, /animationDelay/);
  assert.match(src, /animate-trace-in/);
});

test('styles.css defines the trace-in keyframes', () => {
  const src = read('components/symmetry-aware-einsum-contractions/styles.css');
  assert.match(src, /@keyframes trace-in/);
  assert.match(src, /\.animate-trace-in/);
});

test('FormulaPopover exports a default function and reads REGIME_SPEC', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/FormulaPopover.jsx');
  assert.match(src, /export default function FormulaPopover/);
  assert.match(src, /REGIME_SPEC/);
});

test('BipartiteGraph accepts a highlightedLabels prop', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx');
  assert.match(src, /highlightedLabels/);
});
