// website/symmetry-explorer.regime-property.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { MIXED_REGIMES } from './components/symmetry-aware-einsum-contractions/engine/regimes/index.js';
import { computeABruteforce } from './components/symmetry-aware-einsum-contractions/engine/__tests__/oracle.mjs';

function allV(labels, vCount) { return labels.slice(0, vCount); }
function allW(labels, vCount) { return labels.slice(vCount); }

function buildContexts() {
  const cases = [];

  // S_3 on 3 labels
  const s3Gens = [new Permutation([1, 0, 2]), new Permutation([0, 2, 1])];
  const s3 = dimino(s3Gens);
  for (const vCount of [1, 2]) {
    for (const sizes of [[2, 2, 2], [3, 3, 3]]) {
      const labels = ['a', 'b', 'c'];
      cases.push({
        labels,
        va: allV(labels, vCount),
        wa: allW(labels, vCount),
        elements: s3,
        generators: s3Gens,
        sizes,
        visiblePositions: labels.slice(0, vCount).map((_, i) => i),
      });
    }
  }

  // Trivial group, heterogeneous. Shape layer would handle this, but regime ladder
  // should also produce a consistent count via bruteForceOrbit.
  cases.push({
    labels: ['a', 'b', 'c'],
    va: ['a'],
    wa: ['b', 'c'],
    elements: [Permutation.identity(3)],
    generators: [],
    sizes: [2, 3, 4],
    visiblePositions: [0],
  });

  // S_2 × S_2 direct product
  const sp1 = new Permutation([1, 0, 2, 3]);
  const sp2 = new Permutation([0, 1, 3, 2]);
  const sps = dimino([sp1, sp2]);
  for (const sizes of [[3, 3, 3, 3], [2, 2, 4, 4]]) {
    cases.push({
      labels: ['a', 'b', 'c', 'd'],
      va: ['a', 'b'],
      wa: ['c', 'd'],
      elements: sps,
      generators: [sp1, sp2],
      sizes,
      visiblePositions: [0, 1],
    });
  }

  return cases;
}

test('every regime that fires agrees with the oracle', () => {
  for (const ctx of buildContexts()) {
    const oracle = computeABruteforce(ctx.elements, ctx.sizes, ctx.visiblePositions);
    for (const regime of MIXED_REGIMES) {
      const v = regime.recognize(ctx);
      if (!v.fired) continue;
      const { count } = regime.compute(ctx);
      assert.equal(
        count, oracle,
        `regime ${regime.id} produced ${count}, oracle says ${oracle} for ctx ${JSON.stringify({labels: ctx.labels, va: ctx.va, sizes: ctx.sizes})}`,
      );
    }
  }
});

test('at least one regime fires on every ctx (bruteForce is the floor)', () => {
  for (const ctx of buildContexts()) {
    const fired = MIXED_REGIMES.some((r) => r.recognize(ctx).fired);
    assert.ok(fired, `no regime fired for ctx ${JSON.stringify({labels: ctx.labels, va: ctx.va, sizes: ctx.sizes})}`);
  }
});
