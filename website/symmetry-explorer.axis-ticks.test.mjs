import test from 'node:test';
import assert from 'node:assert/strict';
import { computeAxisTicks } from './components/symmetry-aware-einsum-contractions/components/branchingViews/orbitRepMatrixLayout.js';

test('computeAxisTicks: 0 → empty', () => {
  assert.deepEqual(computeAxisTicks(0), []);
});
test('computeAxisTicks: 1 → [0]', () => {
  assert.deepEqual(computeAxisTicks(1), [0]);
});
test('computeAxisTicks: small n returns every index', () => {
  assert.deepEqual(computeAxisTicks(5, 6), [0, 1, 2, 3, 4]);
  assert.deepEqual(computeAxisTicks(6, 6), [0, 1, 2, 3, 4, 5]);
});
test('computeAxisTicks: includes 0 and n-1', () => {
  for (const n of [9, 10, 27, 100, 136, 165, 336]) {
    const ticks = computeAxisTicks(n, 6);
    assert.equal(ticks[0], 0, `n=${n} first should be 0`);
    assert.equal(ticks[ticks.length - 1], n - 1, `n=${n} last should be ${n - 1}`);
  }
});
test('computeAxisTicks: respects maxTicks cap', () => {
  for (const n of [27, 100, 136, 165, 336]) {
    const ticks = computeAxisTicks(n, 6);
    assert.ok(ticks.length <= 6 + 1, `n=${n} should produce at most 7 ticks (maxTicks + last); got ${ticks.length}`);
  }
});
test('computeAxisTicks: monotonic increasing + unique', () => {
  for (const n of [27, 100, 136, 165, 336]) {
    const ticks = computeAxisTicks(n, 6);
    const sorted = [...ticks].sort((a, b) => a - b);
    assert.deepEqual(ticks, sorted, `n=${n} should be sorted`);
    const unique = [...new Set(ticks)];
    assert.deepEqual(ticks, unique, `n=${n} should be unique`);
  }
});
