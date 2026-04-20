// website/symmetry-explorer.young-regime.test.mjs
//
// Unit tests for the Young regime.

import test from 'node:test';
import assert from 'node:assert/strict';
import { youngRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/young.js';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';

function s3OnThreeLabels() {
  return [
    new Permutation([0, 1, 2]), new Permutation([1, 0, 2]),
    new Permutation([2, 1, 0]), new Permutation([0, 2, 1]),
    new Permutation([1, 2, 0]), new Permutation([2, 0, 1]),
  ];
}

test('Young fires: abc→ab, G = S₃, |V|=2, cross-V/W', () => {
  const verdict = youngRegime.recognize({
    labels: ['a', 'b', 'c'],
    va: ['a', 'b'],
    wa: ['c'],
    elements: s3OnThreeLabels(),
    sizes: [3, 3, 3],
  });
  assert.equal(verdict.fired, true, `expected fired, got reason: ${verdict.reason}`);
});

test('Young declines: |V|=1 (singleton handles it)', () => {
  const verdict = youngRegime.recognize({
    labels: ['a', 'b', 'c'],
    va: ['a'],
    wa: ['b', 'c'],
    elements: s3OnThreeLabels(),
    sizes: [3, 3, 3],
  });
  assert.equal(verdict.fired, false);
});

test('Young declines: no cross-V/W element', () => {
  // G = {e, (a b)} — only V-only swap, no cross.
  const identity = new Permutation([0, 1, 2]);
  const swapAB = new Permutation([1, 0, 2]);
  const verdict = youngRegime.recognize({
    labels: ['a', 'b', 'c'],
    va: ['a', 'b'],
    wa: ['c'],
    elements: [identity, swapAB],
    sizes: [3, 3, 3],
  });
  assert.equal(verdict.fired, false);
});

test('Young declines: G not full Sym(L_c)', () => {
  // G = C₃ ≤ S₃, only 3 elements not 6.
  const c3 = [
    new Permutation([0, 1, 2]),
    new Permutation([1, 2, 0]),
    new Permutation([2, 0, 1]),
  ];
  const verdict = youngRegime.recognize({
    labels: ['a', 'b', 'c'],
    va: ['a', 'b'],
    wa: ['c'],
    elements: c3,
    sizes: [3, 3, 3],
  });
  assert.equal(verdict.fired, false);
});

test('Young declines: mixed label sizes', () => {
  const verdict = youngRegime.recognize({
    labels: ['a', 'b', 'c'],
    va: ['a', 'b'],
    wa: ['c'],
    elements: s3OnThreeLabels(),
    sizes: [3, 4, 3],
  });
  assert.equal(verdict.fired, false);
});

test('Young compute at n=2: α = 2^2 · C(2+1-1, 1) = 4 · 2 = 8', () => {
  const result = youngRegime.compute({
    labels: ['a', 'b', 'c'],
    va: ['a', 'b'],
    wa: ['c'],
    elements: s3OnThreeLabels(),
    sizes: [2, 2, 2],
  });
  assert.equal(result.count, 8);
});

test('Young compute at n=3: α = 3^2 · C(3, 1) = 9 · 3 = 27', () => {
  const result = youngRegime.compute({
    labels: ['a', 'b', 'c'],
    va: ['a', 'b'],
    wa: ['c'],
    elements: s3OnThreeLabels(),
    sizes: [3, 3, 3],
  });
  assert.equal(result.count, 27);
});

test('Young compute: abcd→ab at n=3 = 3^2 · C(4, 2) = 9 · 6 = 54', () => {
  const result = youngRegime.compute({
    labels: ['a', 'b', 'c', 'd'],
    va: ['a', 'b'],
    wa: ['c', 'd'],
    elements: new Array(24), // only length matters in compute
    sizes: [3, 3, 3, 3],
  });
  assert.equal(result.count, 54);
});
