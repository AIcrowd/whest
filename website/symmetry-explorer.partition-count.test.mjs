import test from 'node:test';
import assert from 'node:assert/strict';

import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import {
  countMapOrbitsUnderH,
  fallingFactorial,
  generateTypedSetPartitions,
  inducedBlockActionSize,
  inducedPrefixMap,
  numBlocks,
  partitionOrbitReps,
} from './components/symmetry-aware-einsum-contractions/engine/partition/typedPartitions.js';

test('fallingFactorial supports heterogeneous domain factors', () => {
  assert.equal(fallingFactorial(5, 0), 1);
  assert.equal(fallingFactorial(5, 1), 5);
  assert.equal(fallingFactorial(5, 2), 20);
  assert.equal(fallingFactorial(1, 2), 0);
});

test('typed partitions never merge positions with different sizes', () => {
  const partitions = generateTypedSetPartitions([2, 3]);
  assert.deepEqual(partitions, [[0, 1]]);
});

test('typed partitions allow equal-size positions to merge or split', () => {
  const partitions = generateTypedSetPartitions([4, 4]);
  const keys = partitions.map((partition) => partition.join('|')).sort();
  assert.deepEqual(keys, ['0|0', '0|1']);
});

test('partition orbit reps collapse discrete two-block partition under S2', () => {
  const elements = dimino([new Permutation([1, 0])]);
  const partitions = generateTypedSetPartitions([5, 5]);
  const reps = partitionOrbitReps(partitions, elements);
  assert.equal(reps.length, 2);
});

test('induced block action size uses action on blocks, not raw stabilizer size', () => {
  const elements = dimino([new Permutation([1, 0])]);
  assert.equal(inducedBlockActionSize([0, 0], elements), 1);
  assert.equal(inducedBlockActionSize([0, 1], elements), 2);
});

test('inducedPrefixMap uses g inverse convention', () => {
  const swap = new Permutation([1, 0, 2]);
  const partition = [0, 1, 2];
  assert.deepEqual(inducedPrefixMap(partition, swap, [0, 1]), [1, 0]);
});

test('countMapOrbitsUnderH quotients maps by output action', () => {
  const h = [new Permutation([0, 1]), new Permutation([1, 0])];
  const maps = new Set(['0|1', '1|0', '0|2']);
  assert.equal(countMapOrbitsUnderH(maps, h), 2);
});
