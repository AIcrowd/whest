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

test('BranchingDemo renders OrbitRepMatrix as a full-width figure (no 2-col grid wrapping the matrix)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  // No more 2-col grid for matrix + panel.
  assert.doesNotMatch(src, /lg:grid-cols-\[400px_minmax\(0,1fr\)\]/);
  // No WorkedExamplePanel anywhere.
  assert.doesNotMatch(src, /WorkedExamplePanel/);
});

test('BranchingDemo no longer mounts MatrixHoverTooltip (hover surfaces in OrbitDetailCard)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.doesNotMatch(src, /MatrixHoverTooltip/);
  assert.doesNotMatch(src, /tooltipRef/);
});

test('BranchingDemo wires hover-driven OrbitDetailCard', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /import OrbitDetailCard from '\.\/branchingViews\/OrbitDetailCard\.jsx'/);
  assert.match(src, /<OrbitDetailCard\b[\s\S]*?mode=['"]floating['"]/);
  assert.match(src, /hover=\{hover\}/);
});

test('BranchingDemo holds hover in React state (no click-pin)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /\[hover[\s\S]{0,30}\]\s*=\s*useState/);
  assert.doesNotMatch(src, /\[pin,/);
});

test('BranchingDemo passes a matrixRef to OrbitDetailCard so it can auto-dismiss on scroll-out', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  assert.match(src, /matrixRef/);
  assert.match(src, /<OrbitDetailCard\b[\s\S]{0,300}matrixRef=/);
});

test('BranchingDemo wires the modal trigger via onExpand and mounts OrbitRepMatrixModal', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx');
  // The expand button itself lives inside OrbitRepMatrix as a corner overlay
  // (separate test in orbit-rep-matrix.test.mjs); BranchingDemo just passes
  // an onExpand callback that opens the modal.
  assert.match(src, /onExpand=\{handleOpenModal\}/);
  assert.match(src, /import OrbitRepMatrixModal from '\.\/branchingViews\/OrbitRepMatrixModal\.jsx'/);
  assert.match(src, /<OrbitRepMatrixModal/);
});

test('OrbitRepMatrix renders the prominent expand button as a canvas-corner overlay', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitRepMatrix.jsx');
  assert.match(src, /data-action="open-modal"/);
  assert.match(src, /onExpand/);
  // Stop propagation so the click doesn't accidentally pin a cell.
  assert.match(src, /e\.stopPropagation\(\);\s*onExpand\(\)/);
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
