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

test('BipartiteGraph renders notation headers as math, not plain text pills', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx');
  assert.match(src, /import PanZoomCanvas/);
  assert.match(src, /<PanZoomCanvas/);
  assert.match(src, /ariaLabel="Bipartite graph \(zoomable\)"/);
  assert.match(src, /import Latex/);
  assert.match(src, /foreignObject/);
  assert.match(src, /notationLatex\('v_free'\)/);
  assert.match(src, /notationLatex\('w_summed'\)/);
  assert.match(src, /notationLatex\('u_axis_classes'\)/);
  assert.doesNotMatch(src, /text=\{notationText\('v_free'\)\}/);
  assert.doesNotMatch(src, /text=\{notationText\('w_summed'\)\}/);
});

test('IncidenceMatrix exposes notation-aware row and column legends', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/IncidenceMatrix.jsx');
  assert.match(src, /import Latex/);
  assert.match(src, /notationLatex\('u_axis_classes'\)/);
  assert.match(src, /notationLatex\('v_free'\)/);
  assert.match(src, /notationLatex\('w_summed'\)/);
  assert.match(src, /rows:/i);
  assert.match(src, /columns:/i);
});
