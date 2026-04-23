// website/symmetry-explorer.regime-singleton.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { singletonRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/singleton.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

function S3On3Labels() {
  // Generators (0 1) and (1 2)
  const a = new Permutation([1, 0, 2]);
  const b = new Permutation([0, 2, 1]);
  return dimino([a, b]);
}

test('singleton recognizes |V| = 1', () => {
  const elements = S3On3Labels();
  const ctx = {
    labels: ['v', 'w1', 'w2'], va: ['v'], wa: ['w1', 'w2'],
    elements, sizes: [3, 3, 3], visiblePositions: [0],
  };
  assert.equal(singletonRegime.recognize(ctx).fired, true);
});

test('singleton refuses |V| != 1', () => {
  const elements = S3On3Labels();
  const ctx = {
    labels: ['v1', 'v2', 'w'], va: ['v1', 'v2'], wa: ['w'],
    elements, sizes: [3, 3, 3], visiblePositions: [0, 1],
  };
  assert.equal(singletonRegime.recognize(ctx).fired, false);
});

test('singleton count matches oracle on S3 on 3 labels with single visible', () => {
  const elements = S3On3Labels();
  const ctx = {
    labels: ['v', 'w1', 'w2'], va: ['v'], wa: ['w1', 'w2'],
    elements, sizes: [3, 3, 3], visiblePositions: [0],
  };
  const { count } = singletonRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('singleton count matches oracle on heterogeneous sizes (Ω common + R different)', () => {
  // S_2 on {0,1} × trivial on {2}. n_0=n_1=3, n_2=5.
  const swap01 = new Permutation([1, 0, 2]);
  const elements = [Permutation.identity(3), swap01];
  const ctx = {
    labels: ['v', 'w1', 'w2'], va: ['v'], wa: ['w1', 'w2'],
    elements, sizes: [3, 3, 5], visiblePositions: [0],
  };
  const { count } = singletonRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});
