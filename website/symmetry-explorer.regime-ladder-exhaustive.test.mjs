// website/symmetry-explorer.regime-ladder-exhaustive.test.mjs
//
// Invariants that should hold for every preset:
//   - Every component in the decomposition has exactly one dispatched regime.
//   - No component carries a legacy caseType field (that classification
//     taxonomy was retired in commit 3023cb09 / 99a32e41).

import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';

const DIMENSION_N = 3;

test('every preset dispatches a regime on every component', () => {
  for (const example of EXAMPLES) {
    const analysis = analyzeExample(example, DIMENSION_N);
    for (const comp of analysis.componentData.components) {
      assert.ok(
        comp.accumulation?.regimeId,
        `${example.id}: component ${JSON.stringify(comp.labels)} has no regimeId`,
      );
    }
  }
});

test('no component carries a legacy caseType field', () => {
  for (const example of EXAMPLES) {
    const analysis = analyzeExample(example, DIMENSION_N);
    for (const comp of analysis.componentData.components) {
      assert.equal(
        comp.caseType, undefined,
        `${example.id}: stale caseType field present on component`,
      );
    }
  }
});

test('no preset carries a legacy caseType field', () => {
  for (const example of EXAMPLES) {
    assert.equal(
      example.caseType, undefined,
      `${example.id}: preset still has caseType`,
    );
  }
});
