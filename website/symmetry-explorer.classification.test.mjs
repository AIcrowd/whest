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

import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';

const DIMENSION_N = 3;

for (const example of EXAMPLES) {
  if (!example.regimeId) continue;

  test(`annotation '${example.regimeId}' matches at least one component's dispatch: ${example.id}`, () => {
    const analysis = analyzeExample(example, DIMENSION_N);
    const dispatchedRegimes = analysis.componentData.components.map(
      (c) => c.accumulation?.regimeId,
    );
    const matches = dispatchedRegimes.includes(example.regimeId);
    assert.ok(
      matches,
      `${example.id}: annotation '${example.regimeId}' not among dispatched regimes [${dispatchedRegimes.join(', ')}]`,
    );
  });
}
