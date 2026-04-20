// website/symmetry-explorer.regime-direct-product.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { directProductRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/directProduct.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

test('directProduct recognizes S_V × G_W when every generator moves only V or only W', () => {
  // V = {a,b}, W = {c,d}
  const g1 = new Permutation([1, 0, 2, 3]); // swap a↔b
  const g2 = new Permutation([0, 1, 3, 2]); // swap c↔d
  const elements = dimino([g1, g2]);
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, sizes: [3, 3, 4, 4], visiblePositions: [0, 1],
    generators: [g1, g2],
  };
  const v = directProductRegime.recognize(ctx);
  assert.equal(v.fired, true);
});

test('directProduct refuses when a generator crosses V/W', () => {
  const cross = new Permutation([2, 1, 0, 3]); // maps a↔c
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements: [Permutation.identity(4), cross], sizes: [3, 3, 3, 3], visiblePositions: [0, 1],
    generators: [cross],
  };
  assert.equal(directProductRegime.recognize(ctx).fired, false);
});

test('directProduct count agrees with oracle on S_V × S_W uniform n=3', () => {
  const g1 = new Permutation([1, 0, 2, 3]);
  const g2 = new Permutation([0, 1, 3, 2]);
  const elements = dimino([g1, g2]);
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, sizes: [3, 3, 3, 3], visiblePositions: [0, 1],
    generators: [g1, g2],
  };
  const { count } = directProductRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('directProduct count agrees with oracle on heterogeneous sizes', () => {
  const g1 = new Permutation([1, 0, 2, 3]);
  const g2 = new Permutation([0, 1, 3, 2]);
  const elements = dimino([g1, g2]);
  const ctx = {
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements, sizes: [2, 2, 3, 3], visiblePositions: [0, 1],
    generators: [g1, g2],
  };
  const { count } = directProductRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});
