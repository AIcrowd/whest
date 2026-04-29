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

test('BranchingDemo renders the new title "The O → Q matrix"', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  // The matrix title is rendered via <Latex math="\to" /> between O and Q,
  // so assert just the surrounding "O" and "Q matrix" tokens.
  assert.match(src, /The O\s/);
  assert.match(src, /Q matrix/);
});

test('BranchingDemo renders the deck defining O, Q, filled cells, and α', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /product orbit/);
  assert.match(src, /stored output representative/);
  assert.match(src, /Counting filled cells gives/);
});

test('BranchingDemo mounts OrbitRepMatrix + WorkedExamplePanel side-by-side', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /import OrbitRepMatrix from '\.\/branchingViews\/OrbitRepMatrix\.jsx'/);
  assert.match(src, /import WorkedExamplePanel from '\.\/branchingViews\/WorkedExamplePanel\.jsx'/);
  assert.match(src, /<OrbitRepMatrix/);
  assert.match(src, /<WorkedExamplePanel/);
});

test('BranchingDemo holds pin state and skips per-hover propagation for perf', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  // Pin state: still React state (drives the WorkedExamplePanel and modal).
  assert.match(src, /useState[\s\S]{0,200}pin/);
  // Hover does NOT live in BranchingDemo state — it's a ref inside
  // OrbitRepMatrix that paints the canvas marker without React.
  assert.doesNotMatch(src, /useState\([^)]*\)[^;]{0,80}hover/);
});

test('BranchingDemo renders the modal trigger and mounts OrbitRepMatrixModal', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /data-action="open-modal"/);
  assert.match(src, /import OrbitRepMatrixModal from '\.\/branchingViews\/OrbitRepMatrixModal\.jsx'/);
  assert.match(src, /<OrbitRepMatrixModal/);
});

test('BranchingDemo derives reps + cells from costModel.orbitRows', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /costModel\?\.orbitRows/);
  assert.match(src, /derivePreReps\(/);
  assert.match(src, /deriveCells\(/);
});

test('BranchingDemo emits a live α total via data-testid="branching-alpha-total"', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /data-testid="branching-alpha-total"/);
});

test('BranchingDemo uses no raw hex outside design tokens', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  const allowed = new Set([
    '#F0524D', '#FEF2F1', '#D9DCDC', '#F8F9F9', '#FFFFFF', '#1F2526',
    '#ECEFEF', '#4A7CFF', '#64748B', '#9AA0A0', '#B23E3A',
  ]);
  const hexes = src.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  for (const h of hexes) {
    assert.ok(allowed.has(h.toUpperCase()), `disallowed hex ${h} — use a design token`);
  }
});

test('§4 ComponentCostView no longer mounts LabelInteractionGraph or TypedPartitionDemo', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.doesNotMatch(src, /<LabelInteractionGraph/);
  assert.doesNotMatch(src, /<TypedPartitionDemo/);
});

test('§3 mounts LabelInteractionGraph in the App', () => {
  const src = read('components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  assert.match(src, /import \{[^}]*LabelInteractionGraph/);
  assert.match(src, /<LabelInteractionGraph/);
});

test('ComponentCostView forwards dimensionN to BranchingDemo', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /dimensionN=\{dimensionN\}[\s\S]{0,200}<BranchingDemo|<BranchingDemo[\s\S]{0,200}dimensionN=\{dimensionN\}/);
});
