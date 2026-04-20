// Engine ↔ page narrative lock-in.
//
// The Act 5 hero claims:
//   Total = (k - 1) · ∏_a M_a + ∏_a α_a
//
// The displayed μ and α come from `aggregateComponentCosts`, which walks the
// per-component decomposition. This test verifies, preset by preset, that
// that per-component aggregation matches the brute-force ground truth from
// `computeExactCostModel` (a global enumeration of orbits and output bins).
//
// If decomposition ever produces components that aren't truly independent —
// or if the per-component M_a / α_a fields drift — this test fails before
// the page can lie to a reader.
import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';

const N = 4; // small enough that brute-force enumeration terminates for every preset

for (const preset of EXAMPLES) {
  test(`${preset.id}: per-component μ and α aggregate to brute-force values`, () => {
    const analysis = analyzeExample(preset, N);
    assert.ok(analysis, `analyzeExample returned null for ${preset.id}`);
    const { costModel, componentCosts } = analysis;
    assert.ok(componentCosts, `componentCosts missing for ${preset.id}`);
    assert.equal(
      componentCosts.mu,
      costModel.evaluationCostExact,
      `μ from per-component (${componentCosts.mu}) ≠ brute-force μ (${costModel.evaluationCostExact}) for ${preset.id}`,
    );
    assert.equal(
      componentCosts.alpha,
      costModel.reductionCostExact,
      `α from per-component (${componentCosts.alpha}) ≠ brute-force α (${costModel.reductionCostExact}) for ${preset.id}`,
    );
  });
}
