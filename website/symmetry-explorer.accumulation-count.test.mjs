// website/symmetry-explorer.accumulation-count.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import {
  computeAccumulation,
} from './components/symmetry-aware-einsum-contractions/engine/accumulationCount.js';

test('accumulationCount for trivial group equals |X| (all assignments, no savings)', () => {
  const e = Permutation.identity(3);
  const r = computeAccumulation({
    labels: ['i', 'j', 'k'],
    va: ['i'],
    wa: ['j', 'k'],
    elements: [e],
    sizes: [3, 4, 5],
    visiblePositions: [0],
  });
  assert.equal(r.regimeId, 'trivial');
  // A = |X| = 3*4*5 = 60; each singleton orbit has |π_V(O)| = 1.
  assert.equal(r.count, 60);
  assert.ok(Array.isArray(r.trace));
  assert.equal(r.trace[0].regimeId, 'trivial');
  assert.equal(r.trace[0].decision, 'fired');
});

test('accumulationCount routes allVisible shape to Π_V n_ℓ', () => {
  const e = Permutation.identity(2);
  const swap = new Permutation([1, 0]);
  const r = computeAccumulation({
    labels: ['a', 'b'],
    va: ['a', 'b'],
    wa: [],
    elements: [e, swap],
    sizes: [3, 3],
    visiblePositions: [0, 1],
  });
  assert.equal(r.regimeId, 'allVisible');
  assert.equal(r.count, 9);
});

test('accumulationCount routes allSummed shape to Burnside on G', () => {
  const e = Permutation.identity(2);
  const swap = new Permutation([1, 0]);
  const r = computeAccumulation({
    labels: ['a', 'b'],
    va: [],
    wa: ['a', 'b'],
    elements: [e, swap],
    sizes: [3, 3],
    visiblePositions: [],
  });
  assert.equal(r.regimeId, 'allSummed');
  // Burnside (3^2 + 3)/2 = 6.
  assert.equal(r.count, 6);
});
