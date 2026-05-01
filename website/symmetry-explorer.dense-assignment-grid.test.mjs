// website/symmetry-explorer.dense-assignment-grid.test.mjs
//
// Source-grep coverage for V3.1 §6 — C06 DenseAssignmentGrid (NEW).
//
// The dense assignment grid is the FIRST concrete object readers see in §3
// (Projection): a 2D faceted grid over (rowLabel, colLabel) faceted by any
// remaining labels (e.g. k for Cross S2). Each cell is the FULL assignment
// (i,j,k) — no symmetry collapse yet. Selected cells expand into the V3.1
// detail block (`full assignment:` / `product:` / `output:`).
//
// These tests pin the contract that survives future refactoring; behavior is
// exercised by the Storybook stories.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const COMPONENT_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/DenseAssignmentGrid.jsx',
);
const APP_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
);
const STORIES_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/DenseAssignmentGrid.stories.jsx',
);

test('DenseAssignmentGrid.jsx exists and exports default', () => {
  assert.ok(existsSync(COMPONENT_PATH), 'expected DenseAssignmentGrid.jsx to exist');
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Default export of the component (named DenseAssignmentGrid).
  assert.match(src, /export default DenseAssignmentGrid/);
  // Component is declared as a function (arrow or named) so the prop contract
  // is greppable.
  assert.match(src, /function DenseAssignmentGrid\(/);
});

test('DenseAssignmentGrid renders the V3.1 verbatim "full assignment:" / "product:" / "output:" labels', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Selected-cell detail block uses the V3.1 §6 labels verbatim.
  assert.match(src, /full assignment:/);
  assert.match(src, /product:/);
  assert.match(src, /output:/);
  // The detail block carries a stable testid.
  assert.match(src, /data-testid="dense-assignment-grid-detail"/);
});

test('DenseAssignmentGrid renders the "show products" / "show outputs" toggles', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Verbatim toggle labels.
  assert.match(src, /show products/);
  assert.match(src, /show outputs/);
  // Each toggle is wired to its own React state.
  assert.match(src, /setShowProducts/);
  assert.match(src, /setShowOutputs/);
  // Stable testids so the harness can flip the toggles.
  assert.match(src, /data-testid="dense-assignment-grid-toggle-products"/);
  assert.match(src, /data-testid="dense-assignment-grid-toggle-outputs"/);
});

test('DenseAssignmentGrid caps rendering at n > 4 with a verbatim cap message', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Verbatim cap-message phrasing matching the spec:
  // "Grid hidden — n=N is too large for visual rendering. Reduce n to see the dense grid."
  assert.match(src, /Grid hidden/);
  assert.match(src, /is too large for visual rendering/);
  assert.match(src, /Reduce n to see the dense grid/);
  // The cap is keyed off RENDER_CAP = 4 (n > RENDER_CAP).
  assert.match(src, /RENDER_CAP\s*=\s*4/);
  assert.match(src, /tooLarge\s*=\s*n\s*>\s*RENDER_CAP/);
  // The cap branch sets a stable data attribute so harnesses can detect it.
  assert.match(src, /data-cap-state="too-large"/);
});

test('DenseAssignmentGrid cells expose role="button", tabIndex, and aria-label', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Each cell is a clickable button-role with keyboard-reachable tabIndex.
  assert.match(src, /role="button"/);
  assert.match(src, /tabIndex=\{0\}/);
  // aria-label includes the full assignment + product + output description.
  assert.match(src, /aria-label=\{ariaLabel\}/);
  assert.match(src, /assignment \$\{tupleLabel\}/);
  assert.match(src, /product \$\{productLabel\}/);
  assert.match(src, /output \$\{outputLabel\}/);
  // Cells stamp aria-pressed so screen readers announce the selected cell.
  assert.match(src, /aria-pressed=\{isSelected\}/);
  // Cells are reachable by stable testid.
  assert.match(src, /data-testid="dense-assignment-grid-cell"/);
});

test('App mounts DenseAssignmentGrid in §3 (Projection) before ComponentCostView', () => {
  const src = readFileSync(APP_PATH, 'utf-8');
  // Default import.
  assert.match(src, /import DenseAssignmentGrid from '\.\/components\/DenseAssignmentGrid\.jsx';/);
  // Mounted JSX (with the prop contract from the spec).
  assert.match(src, /<DenseAssignmentGrid[\s\S]*?dimensionN=\{dimensionN\}[\s\S]*?\/>/);
  assert.match(src, /<DenseAssignmentGrid[\s\S]*?allLabels=\{group\?\.allLabels \?\? \[\]\}[\s\S]*?\/>/);
  // Mount order: DenseAssignmentGrid must appear BEFORE ComponentCostView so
  // the dense view introduces the assignment space before orbits collapse it.
  const dagIdx = src.indexOf('<DenseAssignmentGrid');
  const ccvIdx = src.indexOf('<ComponentCostView');
  assert.ok(dagIdx > 0, 'expected DenseAssignmentGrid to be mounted');
  assert.ok(ccvIdx > 0, 'expected ComponentCostView to be mounted');
  assert.ok(dagIdx < ccvIdx, 'DenseAssignmentGrid must mount before ComponentCostView in §3');
});

test('DenseAssignmentGrid stories file declares ≥3 stories', () => {
  assert.ok(existsSync(STORIES_PATH), 'expected DenseAssignmentGrid.stories.jsx to exist');
  const src = readFileSync(STORIES_PATH, 'utf-8');
  // Default-export Storybook meta with the right title prefix.
  assert.match(src, /title:\s*'Section3\/DenseAssignmentGrid'/);
  // Count `export const X = {` blocks → must be at least 3.
  const matches = src.match(/export const \w+\s*=\s*\{/g) ?? [];
  assert.ok(matches.length >= 3, `expected ≥3 stories, got ${matches.length}`);
});
