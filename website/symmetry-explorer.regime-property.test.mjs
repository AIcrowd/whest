// website/symmetry-explorer.regime-property.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { Permutation, dimino } from './components/symmetry-aware-einsum-contractions/engine/permutation.js';
import { MIXED_REGIMES } from './components/symmetry-aware-einsum-contractions/engine/regimes/index.js';
import { REGIME_SPEC } from './components/symmetry-aware-einsum-contractions/engine/regimeSpec.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
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

test('regime tooltip copy reflects the updated direct-product, Young, and brute-force descriptions', () => {
  assert.equal(
    REGIME_SPEC.directProduct.when,
    'No element crosses V/W and the materialized group satisfies |G| = |G_V| · |G_W|, with both projections nontrivial.',
  );
  assert.equal(
    REGIME_SPEC.directProduct.description,
    'The action factors independently over visible and summed labels. Visible assignments still choose output bins directly, while the summed side is quotient-counted by size-aware Burnside.',
  );
  assert.equal(
    REGIME_SPEC.young.when,
    'Cross-V/W elements present, G is the full symmetric group on the component labels, all component label sizes agree, and |V| ≥ 2.',
  );
  assert.equal(
    REGIME_SPEC.young.description,
    'The full symmetric action lets α be counted through the pointwise V-stabilizer. The summed labels contribute a multiset count, giving the closed form shown here.',
  );
  assert.equal(
    REGIME_SPEC.bruteForceOrbit.description,
    'Exact enumeration of product orbits and their visible projections. It is used only when no analytic regime applies and the pair-touch budget is small enough for the interactive page.',
  );
  assert.equal(
    REGIME_SPEC.bruteForceOrbit.glossary.find((entry) => entry.term === 'budget').definition,
    'The cap on |X| · |G| pair-touches. Below the cap this regime is exact; above the cap the UI reports the component count as unavailable rather than guessing.',
  );
});

test('preset descriptions use the updated explanatory copy', () => {
  const byId = new Map(EXAMPLES.map((example) => [example.id, example.description]));

  assert.equal(
    byId.get('mixed-chain'),
    'A appears twice, but the middle B pins the incidence pattern; the A-swap does not induce a non-identity pointwise label relabeling.',
  );
  assert.equal(
    byId.get('bilinear-trace-3'),
    'Three identical ops → diagonal S₃ on V and W; no cross-V/W elements, but direct-product check fails because |G|=6 while |G_V||G_W|=36.',
  );
  assert.equal(
    byId.get('frobenius'),
    'The operand swap is admissible but induces the identity label relabeling; no non-identity pointwise symmetry appears unless A itself is declared symmetric.',
  );
});
