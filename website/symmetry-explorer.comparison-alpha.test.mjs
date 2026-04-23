import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { Permutation } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import {
  computeExpressionAlphaComparison,
  computeExpressionAlphaTotal,
} from './components/symmetry-aware-einsum-contractions/engine/comparisonAlpha.js';

test('comparison α for Frobenius at n=2 = 3 (G_expr over-compresses)', () => {
  const ex = EXAMPLES.find((e) => e.id === 'frobenius');
  assert.ok(ex, 'frobenius example not found');
  const analysis = analyzeExample(ex, 2);
  const exprAlpha = computeExpressionAlphaTotal({ analysis });
  // Frobenius 'ij,ij->' at n=2: G_pt is trivial (order 1), G_expr = S({i})×S({j})
  // but since there are no W labels (output is empty scalar), the expression
  // group captures the dummy-rename symmetry over {i,j}. Under G_expr, orbit
  // enumeration on [2]^{i,j} groups (0,1) and (1,0) together → 3 orbits.
  // Correct α under G_pt: every tuple alone = 4.
  assert.equal(exprAlpha, 3);
});

test('comparison α for matrix-chain (G_pt = G_expr) returns null', () => {
  const ex = EXAMPLES.find((e) => e.id === 'matrix-chain');
  assert.ok(ex, 'matrix-chain example not found');
  const analysis = analyzeExample(ex, 3);
  const exprAlpha = computeExpressionAlphaTotal({ analysis });
  // matrix-chain has trivial G_pt and G_expr — they have the same order.
  assert.equal(exprAlpha, null);
});

test('comparison α for bilinear-trace at n=2 is less than the correct α', () => {
  const ex = EXAMPLES.find((e) => e.id === 'bilinear-trace');
  assert.ok(ex, 'bilinear-trace example not found');
  const analysis = analyzeExample(ex, 2);
  const exprAlpha = computeExpressionAlphaTotal({ analysis });
  const correctAlpha = analysis.componentData.components.reduce(
    (s, c) => s + (c.accumulation?.count ?? 0), 0,
  );
  // G_expr applied via orbit compression under-counts distinct accumulations.
  assert.ok(exprAlpha !== null, 'expected non-null comparison alpha');
  assert.ok(exprAlpha < correctAlpha, `expected exprAlpha (${exprAlpha}) < correctAlpha (${correctAlpha})`);
});

test('comparison helper reports none state when there is no meaningful G_f vs G_pt count difference to show', () => {
  const ex = EXAMPLES.find((e) => e.id === 'matrix-chain');
  assert.ok(ex, 'matrix-chain example not found');
  const analysis = analyzeExample(ex, 3);
  const comparison = computeExpressionAlphaComparison({ analysis, example: ex });

  assert.equal(comparison.state, 'none');
  assert.equal(comparison.exprAlpha, null);
  assert.equal(comparison.witness, null);
});

test('comparison helper reports coincident state when G_f is larger but α happens to match numerically', () => {
  const ex = EXAMPLES.find((e) => e.id === 'young-s3');
  assert.ok(ex, 'young-s3 example not found');
  const analysis = analyzeExample(ex, 5);
  const comparison = computeExpressionAlphaComparison({ analysis, example: ex });

  assert.equal(comparison.state, 'coincident');
  assert.equal(comparison.exprAlpha, comparison.correctAlpha);
  assert.equal(comparison.exprAlpha, 125);
  assert.equal(comparison.witness, null);
});

test('comparison helper reports mismatch state and exposes a structural witness for bilinear-trace', () => {
  const ex = EXAMPLES.find((e) => e.id === 'bilinear-trace');
  assert.ok(ex, 'bilinear-trace example not found');
  const analysis = analyzeExample(ex, 2);
  const comparison = computeExpressionAlphaComparison({ analysis, example: ex });

  assert.equal(comparison.state, 'mismatch');
  assert.ok(comparison.exprAlpha !== null, 'expected non-null expression alpha');
  assert.ok(comparison.exprAlpha < comparison.correctAlpha);
  assert.ok(comparison.witness, 'expected a structural witness');
  assert.notEqual(comparison.witness.tupleA.join('|'), comparison.witness.tupleB.join('|'));
  assert.equal(comparison.witness.outputA.join('|'), comparison.witness.outputB.join('|'));
  assert.notEqual(comparison.witness.summandA, comparison.witness.summandB);
  assert.match(comparison.witness.summandA, /^A\[/);
  assert.match(comparison.witness.summandB, /^A\[/);
});

test('comparison helper exposes a structural witness for a non-bilinear mismatch preset', () => {
  const ex = EXAMPLES.find((e) => e.id === 'direct-s2-c3');
  assert.ok(ex, 'direct-s2-c3 example not found');
  const analysis = analyzeExample(ex, 3);
  const comparison = computeExpressionAlphaComparison({ analysis, example: ex });

  assert.equal(comparison.state, 'mismatch');
  assert.ok(comparison.witness, 'expected a structural witness');
  assert.notEqual(comparison.witness.tupleA.join('|'), comparison.witness.tupleB.join('|'));
  assert.equal(comparison.witness.outputA.join('|'), comparison.witness.outputB.join('|'));
  assert.notEqual(comparison.witness.summandA, comparison.witness.summandB);
  assert.match(comparison.witness.summandA, /^T\[/);
  assert.match(comparison.witness.summandB, /^T\[/);
});

test('comparison helper throws when a formal action leaves the assignment domain', () => {
  const analysis = {
    expressionGroup: {
      elements: [Permutation.identity(2), new Permutation([1, 0])],
    },
    symmetry: {
      allLabels: ['i', 'j'],
      vLabels: [],
      fullElements: [Permutation.identity(2)],
    },
    clusters: [
      { labels: ['i'], size: 1 },
      { labels: ['j'], size: 2 },
    ],
    componentCosts: {
      alpha: 0,
    },
  };

  assert.throws(
    () => computeExpressionAlphaComparison({ analysis }),
    /group action mapped tuple outside the assignment domain/,
  );
});
