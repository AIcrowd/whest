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
