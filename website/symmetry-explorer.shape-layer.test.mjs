// website/symmetry-explorer.shape-layer.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { detectShape } from './components/symmetry-aware-einsum-contractions/engine/shapeLayer.js';

function makeFacts({ labels, va, wa, elements }) {
  return { labels, va, wa, elements };
}

test('trivial shape when |G| = 1', () => {
  const e = Permutation.identity(2);
  const f = makeFacts({ labels: ['i', 'j'], va: ['i'], wa: ['j'], elements: [e] });
  assert.equal(detectShape(f).kind, 'trivial');
});

test('allVisible when W = ∅ and |G| > 1', () => {
  const e = Permutation.identity(2);
  const swap = new Permutation([1, 0]);
  const f = makeFacts({ labels: ['a', 'b'], va: ['a', 'b'], wa: [], elements: [e, swap] });
  assert.equal(detectShape(f).kind, 'allVisible');
});

test('allSummed when V = ∅ and |G| > 1', () => {
  const e = Permutation.identity(2);
  const swap = new Permutation([1, 0]);
  const f = makeFacts({ labels: ['a', 'b'], va: [], wa: ['a', 'b'], elements: [e, swap] });
  assert.equal(detectShape(f).kind, 'allSummed');
});

test('mixed when V and W both nonempty and |G| > 1', () => {
  const e = Permutation.identity(3);
  const swap = new Permutation([1, 0, 2]);
  const f = makeFacts({ labels: ['a', 'b', 'c'], va: ['a'], wa: ['b', 'c'], elements: [e, swap] });
  assert.equal(detectShape(f).kind, 'mixed');
});
