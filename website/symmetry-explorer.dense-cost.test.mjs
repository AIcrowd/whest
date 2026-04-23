import test from 'node:test';
import assert from 'node:assert/strict';

import {
  denseTupleCountFromComponents,
  denseDirectEventCostFromComponents,
  denseGridScalingLatex,
} from './components/symmetry-aware-einsum-contractions/engine/denseCost.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import { buildSection1ExampleView } from './components/symmetry-aware-einsum-contractions/lib/section1ExampleView.js';

function preset(id) {
  const found = EXAMPLES.find((example) => example.id === id);
  assert.ok(found, `missing preset ${id}`);
  return found;
}

test('dense tuple count uses actual heterogeneous label sizes', () => {
  const outer = analyzeExample(preset('outer'), 3);
  assert.equal(denseTupleCountFromComponents(outer.componentData.components), 4 * 4 * 4 * 4);
  assert.equal(denseDirectEventCostFromComponents(outer.componentData.components, 2), 512);
});

test('dense tuple count keeps uniform behavior for ordinary presets', () => {
  const matrix = analyzeExample(preset('matrix-chain'), 3);
  assert.equal(denseTupleCountFromComponents(matrix.componentData.components), 27);
  assert.equal(denseDirectEventCostFromComponents(matrix.componentData.components, 2), 54);
});

test('dense scaling latex reflects selected label count, not a hard-coded n^5', () => {
  assert.equal(denseGridScalingLatex({ labelCount: 4, hasHeterogeneousSizes: false }), String.raw`n^{4}`);
  assert.equal(denseGridScalingLatex({ labelCount: 5, hasHeterogeneousSizes: false }), String.raw`n^{5}`);
  assert.equal(denseGridScalingLatex({ labelCount: 4, hasHeterogeneousSizes: true }), String.raw`\prod_{\ell \in L} n_\ell`);
});

test('section-one view exposes dense scaling for the selected expression', () => {
  const view = buildSection1ExampleView(preset('triple-outer'));
  assert.equal(view.labelCount, 4);
  assert.equal(view.denseGridScalingLatex, String.raw`\prod_{\ell \in L} n_\ell`);
  assert.equal(view.hasHeterogeneousSizes, true);
});
