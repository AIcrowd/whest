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

import { partitionCountRegime } from './components/symmetry-aware-einsum-contractions/engine/regimes/partitionCount.js';

test('partition count gives n^2 for d=2 r=1 S2 example', () => {
  const elements = dimino([new Permutation([1, 0])]);
  const ctx = {
    labels: ['i', 'j'],
    va: ['i'],
    wa: ['j'],
    elements,
    sizes: [6, 6],
    visiblePositions: [0],
    generators: [new Permutation([1, 0])],
  };
  assert.equal(partitionCountRegime.recognize(ctx).fired, true);
  const result = partitionCountRegime.compute(ctx);
  assert.equal(result.count, 36);
});

test('partition count gives n^2(n+1)/2 for S3 to S2 reduction', () => {
  const s01 = new Permutation([1, 0, 2]);
  const s12 = new Permutation([0, 2, 1]);
  const elements = dimino([s01, s12]);
  const ctx = {
    labels: ['i', 'j', 'k'],
    va: ['i', 'j'],
    wa: ['k'],
    elements,
    sizes: [4, 4, 4],
    visiblePositions: [0, 1],
    generators: [s01, s12],
  };
  const result = partitionCountRegime.compute(ctx);
  assert.equal(result.count, 40);
});

test('partition count supports heterogeneous typed partitions', () => {
  // Independent S2 x S2 on the (3,3,5,5) shape: matches the plan's stated
  // expectation C(3+2-1,2) * C(5+2-1,2) = 6 * 15 = 90. The plan-as-written
  // used a single coupled generator (1,0,3,2) which gives alpha = M = 120,
  // not 90; that pairing does not correspond to the cited multiset formula.
  const visSwap = new Permutation([1, 0, 2, 3]);
  const sumSwap = new Permutation([0, 1, 3, 2]);
  const elements = dimino([visSwap, sumSwap]);
  const ctx = {
    labels: ['i', 'j', 'k', 'l'],
    va: ['i', 'j'],
    wa: ['k', 'l'],
    elements,
    sizes: [3, 3, 5, 5],
    visiblePositions: [0, 1],
    generators: [visSwap, sumSwap],
  };
  const result = partitionCountRegime.compute(ctx);
  assert.equal(result.count, 90);
});
