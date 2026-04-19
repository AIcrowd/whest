// website/symmetry-explorer.wreath-classification.test.mjs
//
// The 3-way classification of wreath elements:
//   - matrix-preserving (M_σ = M elementwise → derivePi = identity → discarded)
//   - rejected (M_σ ≠ M but derivePi returns null → discarded)
//   - valid (M_σ ≠ M AND derivePi returns non-identity → contributes to G_pt)
//
// Counts per gallery preset at n=3 are pinned by the audit's
// REVIEW_RESPONSE.md §4.

import test from 'node:test';
import assert from 'node:assert/strict';

import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

function countsFor(presetId) {
  const preset = EXAMPLES.find((p) => p.id === presetId);
  if (!preset) throw new Error(`preset ${presetId} not found`);
  const r = analyzeExample(preset, 3);
  const elements = r.symmetry.wreathElements;
  const counts = { 'valid': 0, 'matrix-preserving': 0, 'rejected': 0 };
  for (const e of elements) counts[e.classification] += 1;
  counts.total = elements.length;
  return counts;
}

test('Frobenius: 2 elements, 0 valid, 2 matrix-preserving, 0 rejected', () => {
  const c = countsFor('frobenius');
  assert.equal(c.total, 2);
  assert.equal(c.valid, 0);
  assert.equal(c['matrix-preserving'], 2);
  assert.equal(c.rejected, 0);
});

test('trace-product: 2 elements, 1 valid, 1 matrix-preserving, 0 rejected', () => {
  const c = countsFor('trace-product');
  assert.equal(c.total, 2);
  assert.equal(c.valid, 1);
  assert.equal(c['matrix-preserving'], 1);
  assert.equal(c.rejected, 0);
});

test('triangle: 6 elements, 2 valid, 1 matrix-preserving, 3 rejected', () => {
  const c = countsFor('triangle');
  assert.equal(c.total, 6);
  assert.equal(c.valid, 2);
  assert.equal(c['matrix-preserving'], 1);
  assert.equal(c.rejected, 3);
});

test('young-s3: 6 elements, 5 valid, 1 matrix-preserving, 0 rejected', () => {
  const c = countsFor('young-s3');
  assert.equal(c.total, 6);
  assert.equal(c.valid, 5);
  assert.equal(c['matrix-preserving'], 1);
  assert.equal(c.rejected, 0);
});

test('four-cycle: 384 elements, 7 valid, class-1 and class-3 counts computed from engine', () => {
  const c = countsFor('four-cycle');
  assert.equal(c.total, 384);
  assert.equal(c.valid, 7);
  assert.equal(c['matrix-preserving'] + c.rejected, 384 - 7);
  // Don't pin the split between class 1 and class 3; engine-computed.
});
