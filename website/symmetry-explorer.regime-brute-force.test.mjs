// website/symmetry-explorer.regime-brute-force.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { bruteForceOrbitRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/bruteForceOrbit.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

test('bruteForceOrbit recognizes when within budget', () => {
  const e = Permutation.identity(3);
  const swap = new Permutation([0, 2, 1]);
  const v = bruteForceOrbitRegime.recognize({
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements: [e, swap], sizes: [2, 2, 2], visiblePositions: [0],
  });
  assert.equal(v.fired, true);
});

test('bruteForceOrbit refuses when over budget', () => {
  const e = Permutation.identity(3);
  const v = bruteForceOrbitRegime.recognize({
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements: [e], sizes: [10_000, 10_000, 10_000], visiblePositions: [0],
  });
  assert.equal(v.fired, false);
});

test('bruteForceOrbit count matches oracle on Gram symmetry', () => {
  const e = Permutation.identity(3);
  const swap12 = new Permutation([0, 2, 1]);
  const ctx = {
    labels: ['x', 'a', 'b'], va: ['a', 'b'], wa: ['x'],
    elements: [e, swap12], sizes: [2, 2, 2], visiblePositions: [1, 2],
  };
  const { count } = bruteForceOrbitRegime.compute(ctx);
  const expected = computeABruteforce(ctx.elements, ctx.sizes, ctx.visiblePositions);
  assert.equal(count, expected);
});

test('bruteForceOrbit count matches oracle on heterogeneous sizes', () => {
  const e = Permutation.identity(3);
  const swap12 = new Permutation([0, 2, 1]);
  const ctx = {
    labels: ['x', 'a', 'b'], va: ['a', 'b'], wa: ['x'],
    elements: [e, swap12], sizes: [3, 2, 2], visiblePositions: [1, 2],
  };
  const { count } = bruteForceOrbitRegime.compute(ctx);
  const expected = computeABruteforce(ctx.elements, ctx.sizes, ctx.visiblePositions);
  assert.equal(count, expected);
});
