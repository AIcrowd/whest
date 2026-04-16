import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { diagonalSimultaneousRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/diagonalSimultaneous.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

function buildDiagonalS2() {
  // V = {0,1}, W = {2,3}. Generator swaps (0↔1) AND (2↔3) simultaneously.
  const gen = new Permutation([1, 0, 3, 2]);
  return { elements: dimino([gen]), generators: [gen] };
}

test('diagonal recognizes diagonal S_2 action on |V|=|W|=2', () => {
  const { elements, generators } = buildDiagonalS2();
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, generators, sizes: [3, 3, 5, 5], visiblePositions: [0, 1],
  };
  assert.equal(diagonalSimultaneousRegime.recognize(ctx).fired, true);
});

test('diagonal count agrees with oracle on diagonal S_2 with n_V=n_W=3', () => {
  const { elements, generators } = buildDiagonalS2();
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, generators, sizes: [3, 3, 3, 3], visiblePositions: [0, 1],
  };
  const { count } = diagonalSimultaneousRegime.compute(ctx);
  assert.equal(count, 72);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('diagonal count agrees with oracle on heterogeneous n_V=2, n_W=3', () => {
  const { elements, generators } = buildDiagonalS2();
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, generators, sizes: [2, 2, 3, 3], visiblePositions: [0, 1],
  };
  const { count } = diagonalSimultaneousRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('diagonal refuses when |V| != |W|', () => {
  const gen = new Permutation([1, 0, 2]);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a', 'b'], wa: ['c'],
    elements: dimino([gen]), generators: [gen], sizes: [3, 3, 3], visiblePositions: [0, 1],
  };
  assert.equal(diagonalSimultaneousRegime.recognize(ctx).fired, false);
});
