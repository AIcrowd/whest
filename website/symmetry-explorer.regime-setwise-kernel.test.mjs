import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { vSetwiseStableRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/vSetwiseStable.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

test('vSetwiseStable recognizes when every generator preserves V as a set', () => {
  // V = {a, b}, W = {c}. G swaps a↔b only (fixes c).
  const swap = new Permutation([1, 0, 2]);
  const elements = dimino([swap]);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a', 'b'], wa: ['c'],
    elements, generators: [swap], sizes: [3, 3, 5], visiblePositions: [0, 1],
  };
  assert.equal(vSetwiseStableRegime.recognize(ctx).fired, true);
});

test('vSetwiseStable refuses when a generator crosses V ↔ W', () => {
  const cross = new Permutation([2, 1, 0]);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a', 'b'], wa: ['c'],
    elements: [Permutation.identity(3), cross], generators: [cross],
    sizes: [3, 3, 3], visiblePositions: [0, 1],
  };
  assert.equal(vSetwiseStableRegime.recognize(ctx).fired, false);
});

test('vSetwiseStable count matches oracle on heterogeneous sizes', () => {
  const swap = new Permutation([1, 0, 2]);
  const elements = dimino([swap]);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a', 'b'], wa: ['c'],
    elements, generators: [swap], sizes: [3, 3, 5], visiblePositions: [0, 1],
  };
  const { count } = vSetwiseStableRegime.compute(ctx);
  assert.equal(count, computeABruteforce(elements, ctx.sizes, ctx.visiblePositions));
});

test('vSetwiseStable adds kernel sub-trace when |K| > 1', () => {
  // V = {a}, W = {b, c}. G swaps b↔c only. Then ρ_V(G) = {e}, K = G.
  const swapWW = new Permutation([0, 2, 1]);
  const elements = dimino([swapWW]);
  const ctx = {
    labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'],
    elements, generators: [swapWW], sizes: [3, 3, 3], visiblePositions: [0],
  };
  const { subTrace } = vSetwiseStableRegime.compute(ctx);
  assert.ok(subTrace, 'sub-trace should be present');
  const kernelStep = subTrace.find((s) => s.regimeId === 'kernelReduction');
  assert.ok(kernelStep, `expected kernelReduction sub-step, got ${JSON.stringify(subTrace)}`);
});
