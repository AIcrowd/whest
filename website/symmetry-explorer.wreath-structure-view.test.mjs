// website/symmetry-explorer.wreath-structure-view.test.mjs
//
// Smoke tests for WreathStructureView.
// - Structure: read the .jsx as text and verify it exports a default
//   React component (no JSX runtime imported in this test file).
// - Data contract: verify wreathElements is present on every preset
//   and each element carries a valid classification tag.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

const WIDGET_SRC = readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/WreathStructureView.jsx', import.meta.url),
  'utf8',
);

test('WreathStructureView exports a default React component', () => {
  assert.match(
    WIDGET_SRC,
    /export default function WreathStructureView\s*\(/,
    'default export must be function WreathStructureView(...)',
  );
});

test('WreathStructureView reads analysis.symmetry.wreathElements and identicalGroups', () => {
  assert.match(WIDGET_SRC, /wreathElements/, 'widget must consume wreathElements');
  assert.match(WIDGET_SRC, /identicalGroups/, 'widget must consume identicalGroups');
});

test('WreathStructureView uses the cleaned editorial labels and no stale modal-section link', () => {
  assert.match(WIDGET_SRC, /matching relabeling <Latex math=\{String\.raw`\\pi`\} \/>/);
  assert.match(WIDGET_SRC, /no matching relabeling/);
  assert.doesNotMatch(WIDGET_SRC, /derivePi\(σ\)/);
  assert.doesNotMatch(WIDGET_SRC, /Formal argument → modal §2/);
  assert.doesNotMatch(WIDGET_SRC, /onOpenModalSection/);
  assert.doesNotMatch(WIDGET_SRC, /notationColoredLatex\('g_wreath',/);
});

test('WreathStructureView uses the same card styling language as the rest of the explorer', () => {
  assert.match(WIDGET_SRC, /<div className="bg-white p-4">/);
  assert.match(WIDGET_SRC, /<div className="mt-4 overflow-x-auto bg-white">/);
  assert.match(WIDGET_SRC, /rounded-full border border-gray-200 bg-gray-50/);
  assert.doesNotMatch(WIDGET_SRC, /rounded-xl border border-gray-200 bg-white p-4/);
  assert.doesNotMatch(WIDGET_SRC, /mt-4 overflow-x-auto rounded-lg border border-gray-200 bg-white/);
});

test('WreathStructureView caps the initial table at 10 rows, opens a modal for the rest, and uses neutral rows with outcome-only status cues', () => {
  assert.match(WIDGET_SRC, /INITIAL_ROW_LIMIT\s*=\s*10/);
  assert.match(WIDGET_SRC, /Click to see/);
  assert.match(WIDGET_SRC, /ExplorerModal/);
  assert.match(WIDGET_SRC, /explorerThemeColor/);
  assert.match(WIDGET_SRC, /explorerThemeTint/);
  assert.match(WIDGET_SRC, /getActiveExplorerThemeId/);
  assert.doesNotMatch(WIDGET_SRC, /Aggregated Summary/);
  assert.doesNotMatch(WIDGET_SRC, /bg-emerald-50/);
  assert.doesNotMatch(WIDGET_SRC, /bg-rose-50/);
  assert.match(WIDGET_SRC, /✓ kept in G/);
  assert.match(WIDGET_SRC, /✗ no matching relabeling/);
  assert.match(WIDGET_SRC, /style=\{\{ color: notationColor\('sigma_row_move'\) \}\}/);
  assert.match(WIDGET_SRC, /const matrixEffect = element\.matrixPreserving\s*\?\s*String\.raw`\\sigma\(M\) = M`/);
  assert.match(WIDGET_SRC, /Latex math=\{matrixEffect\}/);
  assert.doesNotMatch(WIDGET_SRC, /const matrixEffect = element\.matrixPreserving\s*\?\s*<Latex/);
  assert.match(WIDGET_SRC, /sigma_row_move/);
});

test('wreathElements is present and non-empty on every preset at n=3', () => {
  for (const preset of EXAMPLES) {
    const r = analyzeExample(preset, 3);
    const elements = r.symmetry.wreathElements;
    assert.ok(Array.isArray(elements), `${preset.id}: wreathElements missing`);
    assert.ok(elements.length >= 1, `${preset.id}: wreathElements empty`);
    for (const e of elements) {
      assert.ok(
        ['valid', 'matrix-preserving', 'rejected'].includes(e.classification),
        `${preset.id}: invalid classification ${e.classification}`,
      );
    }
  }
});
