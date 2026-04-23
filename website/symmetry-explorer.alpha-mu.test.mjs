// website/symmetry-explorer.alpha-mu.test.mjs
//
// Regression: engine-computed α matches orbit-theoretic ground truth for
// each preset's per-component groups.
//
// Ground truth: α = Σ_{O ∈ X/G} |π_V(O)|  (bruteForceOrbit's definition,
// also equal to α under the closed-form regimes on the test cases).
//
// Run at n=2 and n=3 to catch any formulas that only coincidentally match
// at the smaller dim.

import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { muAlphaGroundTruth } from './symmetry-explorer.oracle-helpers.mjs';

function componentsHaveUniformSize(component) {
  if (!component.sizes) return false;
  const s0 = component.sizes[0];
  return component.sizes.every((s) => s === s0);
}

for (const example of EXAMPLES) {
  for (const n of [2, 3]) {
    test(`α matches orbit-theoretic ground truth: ${example.id} at n=${n}`, () => {
      const analysis = analyzeExample(example, n);
      for (const comp of analysis.componentData.components) {
        // Skip components with mixed label sizes — ground-truth helper
        // requires uniform.
        if (!componentsHaveUniformSize(comp)) continue;

        const { labels, va, elements, sizes, accumulation } = comp;
        const vPos = va.map((l) => labels.indexOf(l));
        const { alpha: alphaGT } = muAlphaGroundTruth(elements, sizes, vPos);
        const alphaEngine = accumulation.count;
        assert.equal(
          alphaEngine, alphaGT,
          `${example.id} comp[${labels.join(',')}] α: engine=${alphaEngine}, ground=${alphaGT} (regime=${accumulation.regimeId})`,
        );
      }
    });
  }
}
