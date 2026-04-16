import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { alternatingRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/alternating.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

function alternatingGroupOn(k) {
  const gens = [];
  for (let i = 0; i <= k - 3; i += 1) {
    const arr = Array.from({ length: k }, (_, j) => j);
    arr[i] = i + 1; arr[i + 1] = i + 2; arr[i + 2] = i;
    gens.push(new Permutation(arr));
  }
  return { elements: dimino(gens), generators: gens };
}

test('alternating recognizes A_4 on uniform sizes', () => {
  const { elements } = alternatingGroupOn(4);
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, sizes: [4, 4, 4, 4], visiblePositions: [0, 1],
  };
  assert.equal(alternatingRegime.recognize(ctx).fired, true);
});

test('alternating count matches oracle and verified formula on A_4, m=2, r=2, n=4', () => {
  const { elements } = alternatingGroupOn(4);
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, sizes: [4, 4, 4, 4], visiblePositions: [0, 1],
  };
  const { count } = alternatingRegime.compute(ctx);
  assert.equal(count, 172);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('alternating count matches oracle on A_3 with m=1, r=2, n=3', () => {
  const { elements } = alternatingGroupOn(3);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, sizes: [3, 3, 3], visiblePositions: [0],
  };
  const { count } = alternatingRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('alternating refuses on heterogeneous sizes', () => {
  const { elements } = alternatingGroupOn(3);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, sizes: [3, 3, 4], visiblePositions: [0],
  };
  assert.equal(alternatingRegime.recognize(ctx).fired, false);
});

test('alternating refuses when a transposition is present (group is S_L not A_L)', () => {
  const transposition = new Permutation([1, 0, 2]);
  const cyc3 = new Permutation([1, 2, 0]);
  const elements = dimino([transposition, cyc3]);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, sizes: [3, 3, 3], visiblePositions: [0],
  };
  assert.equal(alternatingRegime.recognize(ctx).fired, false);
});
