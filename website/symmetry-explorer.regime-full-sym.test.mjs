import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { fullSymmetricRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/fullSymmetric.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

function symmetricGroupOn(k) {
  const gens = [];
  for (let i = 0; i < k - 1; i += 1) {
    const arr = Array.from({ length: k }, (_, j) => j);
    arr[i] = i + 1; arr[i + 1] = i;
    gens.push(new Permutation(arr));
  }
  return { elements: dimino(gens), generators: gens };
}

test('fullSymmetric recognizes G = S_L on uniform sizes', () => {
  const { elements } = symmetricGroupOn(3);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, sizes: [4, 4, 4], visiblePositions: [0],
  };
  assert.equal(fullSymmetricRegime.recognize(ctx).fired, true);
});

test('fullSymmetric refuses on heterogeneous sizes', () => {
  const { elements } = symmetricGroupOn(3);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, sizes: [3, 3, 4], visiblePositions: [0],
  };
  assert.equal(fullSymmetricRegime.recognize(ctx).fired, false);
});

test('fullSymmetric refuses when |G| != |L|!', () => {
  const e = Permutation.identity(3);
  const swap = new Permutation([1, 0, 2]);
  const elements = [e, swap];
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, sizes: [3, 3, 3], visiblePositions: [0],
  };
  assert.equal(fullSymmetricRegime.recognize(ctx).fired, false);
});

test('fullSymmetric count matches oracle: S_3, m=1, r=2, n=3', () => {
  const { elements } = symmetricGroupOn(3);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, sizes: [3, 3, 3], visiblePositions: [0],
  };
  const { count } = fullSymmetricRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('fullSymmetric count matches verified formula: S_4, m=2, r=2, n=3 gives 9*C(4,2)=54', () => {
  const { elements } = symmetricGroupOn(4);
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, sizes: [3, 3, 3, 3], visiblePositions: [0, 1],
  };
  const { count } = fullSymmetricRegime.compute(ctx);
  assert.equal(count, 54);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});
