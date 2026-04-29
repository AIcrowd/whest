import test from 'node:test';
import assert from 'node:assert/strict';

import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import {
  canonicalTupleUnderGroup,
  preservesPositionSet,
  restrictStabilizerToPositions,
  restrictToPositions,
} from './components/symmetry-aware-einsum-contractions/engine/outputOrbit.js';
import { computeExactCostModel } from './components/symmetry-aware-einsum-contractions/engine/costModel.js';

test('preservesPositionSet recognizes setwise preservation, not pointwise fixing', () => {
  const coupled = new Permutation([1, 0, 3, 2]);
  assert.equal(preservesPositionSet(coupled, [0, 1]), true);
  assert.equal(preservesPositionSet(coupled, [0, 2]), false);
});

test('restrictToPositions uses local coordinates', () => {
  const coupled = new Permutation([1, 0, 3, 2]);
  const restricted = restrictToPositions(coupled, [0, 1]);
  assert.deepEqual(restricted.arr, [1, 0]);
});

test('restrictStabilizerToPositions includes coupled stabilizer restrictions', () => {
  const gen = new Permutation([1, 0, 3, 2]);
  const elements = dimino([gen]);
  const h = restrictStabilizerToPositions(elements, [0, 1]);
  const keys = new Set(h.map((perm) => perm.key()));
  assert.deepEqual(keys, new Set(['0,1', '1,0']));
});

test('canonicalTupleUnderGroup canonicalizes output representatives', () => {
  const h = [new Permutation([0, 1]), new Permutation([1, 0])];
  assert.equal(canonicalTupleUnderGroup([7, 3], h), '3|7');
  assert.equal(canonicalTupleUnderGroup([3, 7], h), '3|7');
  assert.equal(canonicalTupleUnderGroup([5, 5], h), '5|5');
});

test('exact oracle quotients projected outputs by H for S3 to S2 reduction', () => {
  const s01 = new Permutation([1, 0, 2]);
  const s12 = new Permutation([0, 2, 1]);
  const groupElements = dimino([s01, s12]);
  const result = computeExactCostModel({
    labels: ['i', 'j', 'k'],
    vLabels: ['i', 'j'],
    groupElements,
    sizes: [4, 4, 4],
    dimensionN: 4,
    numTerms: 1,
  });

  assert.equal(result.orbitCount, 20);
  assert.equal(result.reductionCostExact, 40);
});

test('exact oracle handles heterogeneous no-cross output representatives', () => {
  const gen = new Permutation([1, 0, 3, 2]);
  const groupElements = dimino([gen]);
  const result = computeExactCostModel({
    labels: ['i', 'j', 'k', 'l'],
    vLabels: ['i', 'j'],
    groupElements,
    sizes: [3, 3, 5, 5],
    dimensionN: 3,
    numTerms: 1,
  });

  assert.equal(result.reductionCostExact, result.orbitCount);
});

import { computeAccumulation } from './components/symmetry-aware-einsum-contractions/engine/accumulationCount.js';

test('all-visible accumulation equals product orbit count, not raw output cells', () => {
  const gen = new Permutation([1, 0]);
  const elements = dimino([gen]);
  const result = computeAccumulation({
    labels: ['i', 'j'],
    va: ['i', 'j'],
    wa: [],
    elements,
    sizes: [5, 5],
    visiblePositions: [0, 1],
    generators: [gen],
  });
  assert.equal(result.count, 15);
  assert.equal(result.regimeId, 'functionalProjection');
});

test('all-summed accumulation remains product orbit count', () => {
  const gen = new Permutation([1, 0]);
  const elements = dimino([gen]);
  const result = computeAccumulation({
    labels: ['i', 'j'],
    va: [],
    wa: ['i', 'j'],
    elements,
    sizes: [5, 5],
    visiblePositions: [],
    generators: [gen],
  });
  assert.equal(result.count, 15);
  assert.equal(result.regimeId, 'functionalProjection');
});

test('mixed no-cross coupled action has alpha equal M', () => {
  const gen = new Permutation([1, 0, 3, 2]);
  const elements = dimino([gen]);
  const result = computeAccumulation({
    labels: ['i', 'j', 'k', 'l'],
    va: ['i', 'j'],
    wa: ['k', 'l'],
    elements,
    sizes: [3, 3, 5, 5],
    visiblePositions: [0, 1],
    generators: [gen],
  });
  assert.equal(result.count, 120);
  assert.equal(result.regimeId, 'functionalProjection');
});
