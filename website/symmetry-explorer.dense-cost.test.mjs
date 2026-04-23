import test from 'node:test';
import assert from 'node:assert/strict';

import {
  denseTupleCountFromComponents,
  denseDirectEventCostFromComponents,
  denseGridScalingLatex,
  hasHeterogeneousLabelSizesFromOverrides,
} from './components/symmetry-aware-einsum-contractions/engine/denseCost.js';
import { analyzeExample } from './components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { EXAMPLES } from './components/symmetry-aware-einsum-contractions/data/examples.js';
import {
  buildSection1ExampleView,
  selectSection1PreambleExample,
} from './components/symmetry-aware-einsum-contractions/lib/section1ExampleView.js';

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

test('heterogeneous-size detection stays false for uniform explicit overrides', () => {
  assert.equal(hasHeterogeneousLabelSizesFromOverrides({ i: 3, j: 3 }), false);
  assert.equal(hasHeterogeneousLabelSizesFromOverrides({ i: 3, j: 4 }), true);
  assert.equal(hasHeterogeneousLabelSizesFromOverrides({ i: 7 }, 5), true);
});

test('section-one view exposes dense scaling for the selected expression', () => {
  const view = buildSection1ExampleView(preset('triple-outer'));
  assert.equal(view.labelCount, 4);
  assert.equal(view.denseGridScalingLatex, String.raw`n^{4}`);
  assert.equal(view.hasHeterogeneousSizes, false);
});

test('preamble source merges live cluster-size overrides when the page is clean', () => {
  const outer = preset('outer');
  const preambleExample = selectSection1PreambleExample({
    example: outer,
    clusterSizes: { 'b,d': 7 },
    isDirty: false,
    defaultSize: 5,
  });
  const view = buildSection1ExampleView(preambleExample);

  assert.equal(view.labelCount, 4);
  assert.equal(view.hasHeterogeneousSizes, true);
  assert.equal(view.denseGridScalingLatex, String.raw`\prod_{\ell \in L} n_\ell`);
});

test('preamble source preserves dirty preview behavior without merging stale cluster sizes', () => {
  const outer = preset('outer');
  const previewExample = {
    ...outer,
    labelSizes: { 'a,c': 4, 'b,d': 4 },
  };
  const preambleExample = selectSection1PreambleExample({
    example: outer,
    previewExample,
    clusterSizes: { 'b,d': 7 },
    isDirty: true,
    defaultSize: 4,
  });
  const view = buildSection1ExampleView(preambleExample);

  assert.equal(preambleExample.defaultSize, 4);
  assert.deepEqual(preambleExample.labelSizes, previewExample.labelSizes);
  assert.equal(view.hasHeterogeneousSizes, false);
  assert.equal(view.denseGridScalingLatex, String.raw`n^{4}`);
});

test('preamble source treats partial cluster-size overrides as heterogeneous against default size', () => {
  const matrix = preset('matrix-chain');
  const preambleExample = selectSection1PreambleExample({
    example: matrix,
    clusterSizes: { i: 7 },
    defaultSize: 5,
    isDirty: false,
  });
  const view = buildSection1ExampleView(preambleExample);

  assert.equal(view.hasHeterogeneousSizes, true);
  assert.equal(view.denseGridScalingLatex, String.raw`\prod_{\ell \in L} n_\ell`);
});
