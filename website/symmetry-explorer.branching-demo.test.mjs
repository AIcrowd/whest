import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('BranchingDemo exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /export default function BranchingDemo/);
});

test('BranchingDemo uses theme helpers and contains zero raw hex codes', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /explorerThemeColor|explorerThemeTint|notationColor/);
  assert.doesNotMatch(src, /#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b/);
});

test('BranchingDemo carries the relocated section4 ¶3 prose verbatim', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /A product-orbit representative can contain many full index assignments/);
  assert.match(src, /That is exactly why accumulation counting is harder than multiplication counting\./);
});

test('BranchingDemo carries the relocated PartitionCountingExplainer body ¶1, ¶2 prose verbatim', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /A product orbit may contain many full assignments\. When those assignments are projected to the output labels/);
  assert.match(src, /Counting product orbits alone is therefore not enough: a single product orbit can update multiple stored output representatives/);
});

test('BranchingDemo renders four view-mode tabs (fan / arcs / grids / pile-buckets)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /data-view-id="fan"/);
  assert.match(src, /data-view-id="arcs"/);
  assert.match(src, /data-view-id="grids"/);
  assert.match(src, /data-view-id="pile-buckets"/);
});

test('FanView exports a default React component with no raw hex', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/FanView.jsx');
  assert.match(src, /export default function FanView/);
  assert.doesNotMatch(src, /#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b/);
  assert.match(src, /<svg/);
});

test('ArcsView exports a default React component with no raw hex', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/ArcsView.jsx');
  assert.match(src, /export default function ArcsView/);
  assert.doesNotMatch(src, /#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b/);
});

test('BranchingDemo wires ArcsView for view-id="arcs"', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /import ArcsView from '\.\/branchingViews\/ArcsView\.jsx'/);
  assert.match(src, /activeView === 'arcs' && <ArcsView/);
});
