// website/symmetry-explorer.heterogeneous-sizes.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { computeAccumulation } from './components/symmetry-aware-einsum-contractions/engine/accumulationCount.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

function suite() {
  const cases = [];
  // Matmul: trivial group, heterogeneous
  cases.push({
    labels: ['i', 'j', 'k'], va: ['i', 'k'], wa: ['j'],
    elements: [Permutation.identity(3)], generators: [], sizes: [2, 3, 4],
    visiblePositions: [0, 2],
  });
  // Gram setwise, heterogeneous
  const swap01 = new Permutation([1, 0, 2]);
  cases.push({
    labels: ['a', 'b', 'c'], va: ['a', 'b'], wa: ['c'],
    elements: dimino([swap01]), generators: [swap01], sizes: [3, 3, 5],
    visiblePositions: [0, 1],
  });
  // Direct product with different sizes per factor
  const g1 = new Permutation([1, 0, 2, 3]);
  const g2 = new Permutation([0, 1, 3, 2]);
  cases.push({
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements: dimino([g1, g2]), generators: [g1, g2], sizes: [2, 2, 4, 4],
    visiblePositions: [0, 1],
  });
  // Diagonal with n_V=2, n_W=3
  const diag = new Permutation([1, 0, 3, 2]);
  cases.push({
    labels: ['a', 'b', 'c', 'd'], va: ['a', 'b'], wa: ['c', 'd'],
    elements: dimino([diag]), generators: [diag], sizes: [2, 2, 3, 3],
    visiblePositions: [0, 1],
  });
  return cases;
}

test('accumulation count matches brute-force oracle across heterogeneous cases', () => {
  for (const ctx of suite()) {
    const r = computeAccumulation(ctx);
    const oracle = computeABruteforce(ctx.elements, ctx.sizes, ctx.visiblePositions);
    assert.equal(r.count, oracle, `regime ${r.regimeId} disagreed on ${JSON.stringify({labels: ctx.labels, va: ctx.va, sizes: ctx.sizes})}`);
  }
});
