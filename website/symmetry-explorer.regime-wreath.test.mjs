import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { wreathRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/wreath.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

function buildS2wrS3() {
  // Labels 0,1 | 2,3 | 4,5. Blocks of size 2.
  const w0 = new Permutation([1, 0, 2, 3, 4, 5]); // swap within block 0
  const w1 = new Permutation([0, 1, 3, 2, 4, 5]); // swap within block 1
  const w2 = new Permutation([0, 1, 2, 3, 5, 4]); // swap within block 2
  const b01 = new Permutation([2, 3, 0, 1, 4, 5]); // swap block 0 with block 1
  const b12 = new Permutation([0, 1, 4, 5, 2, 3]); // swap block 1 with block 2
  const gens = [w0, w1, w2, b01, b12];
  return { elements: dimino(gens), generators: gens };
}

test('wreath recognizes S_2 wr S_3 with V = first 2 blocks', () => {
  const { elements, generators } = buildS2wrS3();
  const ctx = {
    labels: ['a', 'b', 'c', 'd', 'e', 'f'], va: ['a', 'b', 'c', 'd'], wa: ['e', 'f'],
    elements, generators, sizes: [2, 2, 2, 2, 2, 2], visiblePositions: [0, 1, 2, 3],
  };
  assert.equal(wreathRegime.recognize(ctx).fired, true);
});

test('wreath refuses on heterogeneous sizes', () => {
  const { elements, generators } = buildS2wrS3();
  const ctx = {
    labels: ['a', 'b', 'c', 'd', 'e', 'f'], va: ['a', 'b', 'c', 'd'], wa: ['e', 'f'],
    elements, generators, sizes: [2, 2, 2, 2, 3, 3], visiblePositions: [0, 1, 2, 3],
  };
  assert.equal(wreathRegime.recognize(ctx).fired, false);
});

test('wreath count matches verified formula and oracle for S_2 wr S_3, V = 2 blocks, n=2', () => {
  const { elements, generators } = buildS2wrS3();
  const ctx = {
    labels: ['a', 'b', 'c', 'd', 'e', 'f'], va: ['a', 'b', 'c', 'd'], wa: ['e', 'f'],
    elements, generators, sizes: [2, 2, 2, 2, 2, 2], visiblePositions: [0, 1, 2, 3],
  };
  const { count } = wreathRegime.compute(ctx);
  // verify_families.py: S2 on 2-point blocks, blocks=3, visible=2 blocks, n=2 → 48.
  assert.equal(count, 48);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('wreath refuses when V is not whole blocks', () => {
  const { elements, generators } = buildS2wrS3();
  const ctx = {
    labels: ['a', 'b', 'c', 'd', 'e', 'f'], va: ['a'], wa: ['b', 'c', 'd', 'e', 'f'],
    elements, generators, sizes: [2, 2, 2, 2, 2, 2], visiblePositions: [0],
  };
  assert.equal(wreathRegime.recognize(ctx).fired, false);
});
