import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { computeExpressionAlphaTotal } from './components/symmetry-aware-einsum-contractions/engine/comparisonAlpha.js';

test('comparison α for Frobenius at n=2 = 3 (G_EXPR over-compresses)', () => {
  const ex = EXAMPLES.find((e) => e.id === 'frobenius');
  assert.ok(ex, 'frobenius example not found');
  const analysis = analyzeExample(ex, 2);
  const exprAlpha = computeExpressionAlphaTotal({ analysis });
  // Frobenius 'ij,ij->' at n=2: G_PT is trivial (order 1), G_EXPR = S({i})×S({j})
  // but since there are no W labels (output is empty scalar), the expression
  // group captures the dummy-rename symmetry over {i,j}. Under G_EXPR, orbit
  // enumeration on [2]^{i,j} groups (0,1) and (1,0) together → 3 orbits.
  // Correct α under G_PT: every tuple alone = 4.
  assert.equal(exprAlpha, 3);
});

test('comparison α for matrix-chain (G_PT = G_EXPR) returns null', () => {
  const ex = EXAMPLES.find((e) => e.id === 'matrix-chain');
  assert.ok(ex, 'matrix-chain example not found');
  const analysis = analyzeExample(ex, 3);
  const exprAlpha = computeExpressionAlphaTotal({ analysis });
  // matrix-chain has trivial G_PT and G_EXPR — they have the same order.
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
  // G_EXPR applied via orbit compression under-counts distinct accumulations.
  assert.ok(exprAlpha !== null, 'expected non-null comparison alpha');
  assert.ok(exprAlpha < correctAlpha, `expected exprAlpha (${exprAlpha}) < correctAlpha (${correctAlpha})`);
});
