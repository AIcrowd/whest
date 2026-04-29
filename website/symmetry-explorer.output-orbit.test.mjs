import test from 'node:test';
import assert from 'node:assert/strict';

import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import {
  canonicalTupleUnderGroup,
  preservesPositionSet,
  restrictStabilizerToPositions,
  restrictToPositions,
} from './components/symmetry-aware-einsum-contractions/engine/outputOrbit.js';

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
