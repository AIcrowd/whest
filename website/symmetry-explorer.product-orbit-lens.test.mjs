// website/symmetry-explorer.product-orbit-lens.test.mjs
//
// Source-grep coverage for V3.1 §8 — C08 Product-Orbit Lens.
//
// The Product-Orbit Lens augments the §3 DenseAssignmentGrid (C06) with
// product-orbit overlays + a side-card that reveals the orbit's
// Representative / Members / Size / Reason fields when a cell is clicked.
//
// These tests pin the contract: prop shape, toggle, data-* attributes, and
// App-level wiring. Behaviour itself is exercised by the storybook stories
// and live in the explorer.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const GRID_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/DenseAssignmentGrid.jsx',
);
const LENS_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/ProductOrbitLens.jsx',
);
const APP_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
);

test('DenseAssignmentGrid accepts orbitsByAssignment + onOrbitSelect props', () => {
  assert.ok(existsSync(GRID_PATH), 'expected DenseAssignmentGrid.jsx to exist');
  const src = readFileSync(GRID_PATH, 'utf-8');
  // Both props must be declared in the destructured signature.
  assert.match(src, /orbitsByAssignment\s*=\s*null/);
  assert.match(src, /onOrbitSelect\b/);
  // Click handler must invoke onOrbitSelect with the resolved orbit id.
  assert.match(src, /onOrbitSelect\(id\)/);
  // Orbit-id resolver helper must be present (so cells can stamp data-orbit-id).
  assert.match(src, /orbitIdFor\b/);
});

test('DenseAssignmentGrid exposes a "Show all orbits" toggle with aria-label', () => {
  const src = readFileSync(GRID_PATH, 'utf-8');
  // Verbatim toggle label.
  assert.match(src, /show all orbits/);
  // ARIA label is set on both the wrapping label AND the input itself, so
  // accessibility tooling can find the toggle reliably.
  assert.match(src, /aria-label="Show all orbits"/);
  // Stable testid for the harness.
  assert.match(src, /data-testid="dense-assignment-grid-toggle-all-orbits"/);
  // Toggle is wired to its own React state.
  assert.match(src, /setShowAllOrbits/);
});

test('DenseAssignmentGrid cells stamp data-orbit-id and react to hover', () => {
  const src = readFileSync(GRID_PATH, 'utf-8');
  // Each cell exposes its orbit id via a data-* attribute.
  assert.match(src, /data-orbit-id=\{orbitId/);
  // Hover handlers update the shared hoveredOrbitId state.
  assert.match(src, /onMouseEnter=\{\(\)\s*=>\s*onHoverOrbit/);
  assert.match(src, /onMouseLeave=\{\(\)\s*=>\s*onHoverOrbit/);
  // The hovered orbit lights up via data-orbit-hovered.
  assert.match(src, /data-orbit-hovered=/);
  // Module exports the colour helper so the lens / stories can match palettes.
  assert.match(src, /export function orbitColor/);
});

test('ProductOrbitLens renders V3.1 §8 fields (Representative / Members / Orbit size / Reason)', () => {
  assert.ok(existsSync(LENS_PATH), 'expected ProductOrbitLens.jsx to exist');
  const src = readFileSync(LENS_PATH, 'utf-8');
  // V3.1 §8 verbatim field labels.
  assert.match(src, /Representative/);
  assert.match(src, /Members/);
  assert.match(src, /Orbit size/);
  assert.match(src, /Reason/);
  // Fixed-point vs Non-fixed orbit label.
  assert.match(src, /Fixed point/);
  assert.match(src, /Non-fixed orbit/);
  // Stable testids so behaviour can be exercised.
  assert.match(src, /data-testid="product-orbit-lens"/);
  assert.match(src, /data-testid="product-orbit-lens-members"/);
  // Default reason builder is the "product equality" sentence.
  assert.match(src, /Product equality/);
});

test('App computes orbitsByAssignment from cost.orbitRows', () => {
  const src = readFileSync(APP_PATH, 'utf-8');
  // Map is built keyed by the same tuple-string the grid produces (label-
  // values joined by commas, in `group.allLabels` order).
  assert.match(src, /const orbitsByAssignment = useMemo/);
  // Build loops over orbitRows + their orbitTuples.
  assert.match(src, /orbitRows\[orbitId\]\?\.orbitTuples/);
  assert.match(src, /map\.set\(key, orbitId\)/);
  // App wires the map + lock callback through the DenseAssignmentGrid props.
  assert.match(src, /orbitsByAssignment=\{orbitsByAssignment\}/);
  assert.match(src, /onOrbitSelect=\{setLockedProductOrbitId\}/);
});

test('App mounts ProductOrbitLens alongside DenseAssignmentGrid in §3', () => {
  const src = readFileSync(APP_PATH, 'utf-8');
  // Default import.
  assert.match(src, /import ProductOrbitLens from '\.\/components\/ProductOrbitLens\.jsx';/);
  // Mounted JSX with the full prop contract from the spec.
  assert.match(src, /<ProductOrbitLens[\s\S]*?orbit=\{lockedProductOrbit\}[\s\S]*?\/>/);
  assert.match(src, /<ProductOrbitLens[\s\S]*?orbitId=\{lockedProductOrbitId\}[\s\S]*?\/>/);
  // Mount order: ProductOrbitLens must appear AFTER DenseAssignmentGrid so
  // it visually pairs with the grid that drives its selection.
  const dagIdx = src.indexOf('<DenseAssignmentGrid');
  const polIdx = src.indexOf('<ProductOrbitLens');
  assert.ok(dagIdx > 0, 'expected DenseAssignmentGrid to be mounted');
  assert.ok(polIdx > 0, 'expected ProductOrbitLens to be mounted');
  assert.ok(dagIdx < polIdx, 'ProductOrbitLens must mount AFTER DenseAssignmentGrid in §3');
});

test('App resets lockedProductOrbitId on preset / custom-example switches', () => {
  const src = readFileSync(APP_PATH, 'utf-8');
  // The lock state is declared.
  assert.match(src, /const \[lockedProductOrbitId, setLockedProductOrbitId\] = useState\(null\);/);
  // It is cleared by all three example-switch handlers, so a stale orbit-id
  // can never out-live the cost.orbitRows it indexed into.
  const occurrences = src.match(/setLockedProductOrbitId\(null\)/g) ?? [];
  assert.ok(
    occurrences.length >= 3,
    `expected ≥3 lock-resets across handleSelect/handleCustomExample/handleCustomMode, got ${occurrences.length}`,
  );
});
