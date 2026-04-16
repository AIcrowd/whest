// website/symmetry-explorer.budget.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BRUTE_FORCE_BUDGET,
  bruteForceEstimate,
  withinBruteForceBudget,
} from './components/symmetry-aware-einsum-contractions/engine/budget.js';

test('BRUTE_FORCE_BUDGET is 1_500_000', () => {
  assert.equal(BRUTE_FORCE_BUDGET, 1_500_000);
});

test('bruteForceEstimate = Π sizes × |G|', () => {
  assert.equal(bruteForceEstimate([3, 4, 5], 6), 360);
});

test('withinBruteForceBudget true for small, false for large', () => {
  assert.equal(withinBruteForceBudget([3, 4, 5], 6), true);
  assert.equal(withinBruteForceBudget([10, 10, 10, 10, 10], 720), false);
});

test('withinBruteForceBudget accepts custom budget', () => {
  assert.equal(withinBruteForceBudget([3, 4, 5], 6, 100), false);
  assert.equal(withinBruteForceBudget([3, 4, 5], 6, 1000), true);
});
