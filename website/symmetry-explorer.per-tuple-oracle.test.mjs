// website/symmetry-explorer.per-tuple-oracle.test.mjs
//
// Regression: engine's detected per-tuple group size matches the Python
// prototype's oracle baseline (see alpha_mu_audit_v2_output.txt).
//
// Expected: after Source C is removed (Task 5), every preset in
// EXPECTED_GROUP_ORDERS produces a group of the expected order.
// Before Source C removal, Frobenius and edge-contracted-symmetric-trio
// will over-report (|G| larger than expected). This test pins the
// correct state.

import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXPECTED_GROUP_ORDERS } from './symmetry-explorer.oracle-helpers.mjs';

const DIMENSION_N = 3;

for (const example of EXAMPLES) {
  const expected = EXPECTED_GROUP_ORDERS[example.id];
  if (expected === undefined) continue;

  test(`per-tuple group order for preset '${example.id}' (expected |G|=${expected})`, () => {
    const analysis = analyzeExample(example, DIMENSION_N);
    const actual = analysis.symmetry?.fullElements?.length ?? 1;
    assert.equal(
      actual, expected,
      `Expected |G_PT|=${expected} for ${example.id}, got ${actual}`,
    );
  });
}
