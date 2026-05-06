// website/symmetry-explorer.classification.test.mjs
//
// Preset annotation vs runtime classification regression.
//
// For each preset with an annotated regimeId, verify at least one
// per-component dispatch matches the annotation. The engine decomposes
// each einsum into bipartite components and classifies each separately;
// the preset's regimeId reflects the "headline" component (typically the
// non-trivial one). This test catches drift between annotations and the
// engine's actual dispatch.
//
// Preset annotations are still expressed as SHAPES (allVisible / allSummed
// / mixed / trivial) for the case-badge UI. Under the unified output-orbit
// metric, shapes 'allVisible' and 'allSummed' both dispatch through the
// functionalProjection regime (every g preserves V as a set when V = L or
// V = ∅). The compatibility map below records that promotion.

import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';

const DIMENSION_N = 3;

// Annotation → set of dispatch regimes that satisfy it. Defaults to the
// annotation itself when not listed here.
const SHAPE_PROMOTIONS = {
  allVisible: ['allVisible', 'functionalProjection', 'trivial'],
  allSummed: ['allSummed', 'functionalProjection', 'trivial'],
};

function annotationMatches(annotation, dispatchedRegimes) {
  const compatible = new Set(SHAPE_PROMOTIONS[annotation] ?? [annotation]);
  return dispatchedRegimes.some((id) => compatible.has(id));
}

for (const example of EXAMPLES) {
  if (!example.regimeId) continue;

  test(`annotation '${example.regimeId}' matches at least one component's dispatch: ${example.id}`, () => {
    const analysis = analyzeExample(example, DIMENSION_N);
    const dispatchedRegimes = analysis.componentData.components.map(
      (c) => c.accumulation?.regimeId,
    );
    assert.ok(
      annotationMatches(example.regimeId, dispatchedRegimes),
      `${example.id}: annotation '${example.regimeId}' not among dispatched regimes [${dispatchedRegimes.join(', ')}]`,
    );
  });
}
