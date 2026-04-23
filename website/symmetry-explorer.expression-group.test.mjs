// website/symmetry-explorer.expression-group.test.mjs
//
// Unit tests for buildExpressionGroup (V-sub × S(W)).

import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { buildExpressionGroup } from './components/symmetry-aware-einsum-contractions/engine/expressionGroup.js';

test('Frobenius (V=∅, W={i,j}): G_expr = Sym(W) = {e, (i j)}', () => {
  const identity = Permutation.identity(2);
  const result = buildExpressionGroup({
    perTupleElements: [identity],
    vLabels: [],
    wLabels: ['i', 'j'],
    allLabels: ['i', 'j'],
  });
  assert.equal(result.order, 2, 'Sym(W) order 2');
  assert.equal(result.vSub.length, 1, 'V-sub has only the empty-V identity');
  assert.equal(result.sw.length, 2, 'Sym(W) = {e, (i j)}');
});

test('bilinear-trace: G_pt={e,(i j)(k l)} ⇒ G_expr = Z₂ × Z₂ order 4', () => {
  const identity = Permutation.identity(4);
  const ijkl = new Permutation([1, 0, 3, 2]); // (i j)(k l) on [i,j,k,l]
  const result = buildExpressionGroup({
    perTupleElements: [identity, ijkl],
    vLabels: ['i', 'j'],
    wLabels: ['k', 'l'],
    allLabels: ['i', 'j', 'k', 'l'],
  });
  assert.equal(result.order, 4);
  assert.equal(result.vSub.length, 2, 'V-sub = {e, (i j)}');
  assert.equal(result.sw.length, 2, 'S(W) = {e, (k l)}');
});

test('triple-outer: G_pt=S₃ on V={a,b,c}, |W|=1 ⇒ G_expr = S₃', () => {
  const s3 = [
    new Permutation([0, 1, 2, 3]), // e
    new Permutation([1, 0, 2, 3]), // (a b)
    new Permutation([2, 1, 0, 3]), // (a c)
    new Permutation([0, 2, 1, 3]), // (b c)
    new Permutation([1, 2, 0, 3]), // (a b c)
    new Permutation([2, 0, 1, 3]), // (a c b)
  ];
  const result = buildExpressionGroup({
    perTupleElements: s3,
    vLabels: ['a', 'b', 'c'],
    wLabels: ['i'],
    allLabels: ['a', 'b', 'c', 'i'],
  });
  assert.equal(result.order, 6, 'S₃ × trivial = S₃');
  assert.equal(result.vSub.length, 6);
  assert.equal(result.sw.length, 1, 'S(W) trivial when |W|=1');
});

test('G_pt with cross-V/W elements: V-sub drops them but S(W) is still full', () => {
  // Labels [a, b, c]. V = {a, b}. W = {c}. G_pt includes (a c) which
  // crosses V/W. The V-sub projection should drop it; keep only elements
  // that preserve V.
  const identity = Permutation.identity(3);
  const swapAB = new Permutation([1, 0, 2]); // (a b), V-preserving
  const swapAC = new Permutation([2, 1, 0]); // (a c), cross-V/W

  const result = buildExpressionGroup({
    perTupleElements: [identity, swapAB, swapAC],
    vLabels: ['a', 'b'],
    wLabels: ['c'],
    allLabels: ['a', 'b', 'c'],
  });

  // V-sub: only identity and (a b) — cross element dropped.
  assert.equal(result.vSub.length, 2, 'V-sub contains only V-preserving projections');
  // S(W) = Sym({c}) = trivial.
  assert.equal(result.sw.length, 1);
  // G_expr = vSub × sw = 2 × 1 = 2.
  assert.equal(result.order, 2);
});

test('dummy-renaming factor only permutes same-size summed-label blocks', () => {
  const identity = Permutation.identity(3);
  const result = buildExpressionGroup({
    perTupleElements: [identity],
    vLabels: [],
    wLabels: ['i', 'j', 'k'],
    allLabels: ['i', 'j', 'k'],
    sizeByLabel: new Map([
      ['i', 2],
      ['j', 2],
      ['k', 3],
    ]),
  });

  assert.equal(result.sw.length, 2, 'S({i,j}) × S({k}) has order 2');
  assert.equal(result.order, 2);
  assert.deepEqual(
    result.wBlocks.map((block) => block.labels),
    [['i', 'j'], ['k']],
  );

  for (const elem of result.elements) {
    assert.equal(elem.arr[2], 2, 'size-3 label k must not be swapped with size-2 labels');
  }
});

test('dummy-renaming remains full S(W) when no sizes are supplied', () => {
  const identity = Permutation.identity(3);
  const result = buildExpressionGroup({
    perTupleElements: [identity],
    vLabels: [],
    wLabels: ['i', 'j', 'k'],
    allLabels: ['i', 'j', 'k'],
  });

  assert.equal(result.sw.length, 6, 'legacy equal-domain case remains full S(W)');
  assert.equal(result.order, 6);
});
