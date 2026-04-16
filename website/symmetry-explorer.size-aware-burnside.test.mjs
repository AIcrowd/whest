// website/symmetry-explorer.size-aware-burnside.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import {
  cyclesOfG,
  sizeAwareBurnside,
} from './components/symmetry-aware-einsum-contractions/engine/sizeAware/burnside.js';

test('sizeAwareBurnside matches n^L for trivial group on uniform sizes', () => {
  const e = Permutation.identity(3);
  const M = sizeAwareBurnside([e], [5, 5, 5]);
  assert.equal(M, 125);
});

test('sizeAwareBurnside matches n_1 * n_2 * n_3 for trivial group on heterogeneous sizes', () => {
  const e = Permutation.identity(3);
  assert.equal(sizeAwareBurnside([e], [3, 4, 5]), 60);
});

test('sizeAwareBurnside with (01)(2) and uniform n=4 gives (4^2 + 4^3) / 2 = 40', () => {
  const swap = new Permutation([1, 0, 2]);
  const e = Permutation.identity(3);
  assert.equal(sizeAwareBurnside([e, swap], [4, 4, 4]), 40);
});

test('sizeAwareBurnside with (01) and sizes [3,3,5] gives (3*3*5 + 3*5)/2 = 30', () => {
  const swap = new Permutation([1, 0, 2]);
  const e = Permutation.identity(3);
  assert.equal(sizeAwareBurnside([e, swap], [3, 3, 5]), 30);
});

test('cyclesOfG returns [[0,1], [2]] for swap(0,1) on 3 labels', () => {
  const swap = new Permutation([1, 0, 2]);
  const cycles = cyclesOfG(swap);
  assert.deepEqual(
    cycles.map(c => [...c].sort((a, b) => a - b)),
    [[0, 1], [2]],
  );
});

test('sizeAwareBurnside throws on size mismatch within a cycle', () => {
  const swap = new Permutation([1, 0, 2]);
  assert.throws(
    () => sizeAwareBurnside([swap], [3, 4, 5]),
    /cycle size mismatch/,
  );
});
