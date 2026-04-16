// website/symmetry-explorer.oracle.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import {
  allAssignments,
  applyPermToTuple,
  computeABruteforce,
  computeMBruteforce,
} from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

test('allAssignments enumerates the Cartesian product of per-label sizes', () => {
  const tuples = allAssignments([2, 3]);
  assert.equal(tuples.length, 6);
  assert.deepEqual(tuples[0], [0, 0]);
  assert.deepEqual(tuples[tuples.length - 1], [1, 2]);
});

test('applyPermToTuple applies π by position', () => {
  const perm = new Permutation([1, 0]);
  assert.deepEqual(applyPermToTuple([5, 7], perm), [7, 5]);
});

test('computeMBruteforce for trivial group: M = Π n_ℓ', () => {
  const e = Permutation.identity(3);
  assert.equal(computeMBruteforce([e], [2, 3, 4]), 24);
});

test('computeMBruteforce for S_2 on 2 labels uniform n=3: M = 6', () => {
  const e = Permutation.identity(2);
  const swap = new Permutation([1, 0]);
  assert.equal(computeMBruteforce([e, swap], [3, 3]), 6);
});

test('computeABruteforce on Gram symmetry with V = {1, 2} and uniform n=2: A = 8', () => {
  const e = Permutation.identity(3);
  const swap12 = new Permutation([0, 2, 1]);
  const visible = [1, 2];
  assert.equal(computeABruteforce([e, swap12], [2, 2, 2], visible), 8);
});

test('computeABruteforce on trivial group: A = Π_V n_ℓ for trivial group restricted', () => {
  // Clarification: for trivial group, each assignment is its own singleton orbit,
  // |π_V(O)| = 1 per orbit. Number of orbits = |X| = Π n_ℓ. So A = Π over ALL L, not Π over V.
  const e = Permutation.identity(3);
  // V = {0, 2}, all labels size: [3, 4, 5] → orbits = 60 singletons, A = 60.
  assert.equal(computeABruteforce([e], [3, 4, 5], [0, 2]), 60);
});

test('computeABruteforce with a non-trivial swap merges orbits but preserves π_V count', () => {
  // G = <(0 2)>, V = {1}. Orbits of X: pairs {(a,b,c), (c,b,a)}. |π_V(O)|=1 always.
  // Number of orbits ≈ Π / 2 when (a,c) distinct, or = 1 when a=c. With sizes [3,4,5]:
  // pairs (a,c) with a≠c: C(3,2)*...wait sizes differ: n_0=3, n_2=5. Pairs (a,c) in [0,3)×[0,5).
  // Orbits = floor(3*5 / 2) when applying only first and last dim... but sizes mismatch.
  // Simplify: use uniform sizes for this test.
  const e = Permutation.identity(3);
  const swap02 = new Permutation([2, 1, 0]);
  // Sizes [3,4,3] (first and last match).
  // |X|=3*4*3=36. Orbits under {e, swap02}: assignments (a,b,c) with a=c form 3*4=12 fixed orbits size 1.
  // Others: (a,b,c) and (c,b,a) merge → (36-12)/2 = 12 orbits size 2. Total orbits = 24.
  // Each orbit projects to π_V = {b}, |π_V| = 1. So A = 24.
  assert.equal(computeABruteforce([e, swap02], [3, 4, 3], [1]), 24);
});
