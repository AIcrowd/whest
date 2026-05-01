import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const read = (rel) => readFileSync(resolve(__dirname, rel), 'utf-8');

const ANCHOR_HOST_FILES = [
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
  'components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx',
  'components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx',
  'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx',
  'components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx',
  'components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx',
  'components/symmetry-aware-einsum-contractions/components/AccumulationHardCard.jsx',
  'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx',
];

test('visual pass 2 uses a sticky-aware scroll-margin utility instead of scroll-mt-24 anchors', () => {
  for (const file of ANCHOR_HOST_FILES) {
    const src = read(file);
    assert.doesNotMatch(src, /scroll-mt-24/, `${file} still uses the old 96px anchor offset`);
  }

  const appSrc = read('components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  assert.match(appSrc, /scroll-mt-sticky/, 'the main article sections should use the sticky-aware anchor utility');

  const surfaceCss = read('app/design-system/einsum-surface.css');
  assert.match(surfaceCss, /--einsum-sticky-anchor-offset:\s*11rem/);
  assert.match(surfaceCss, /\.symmetry-aware-einsum-contractions-page-shell\s+\.scroll-mt-sticky/);
  assert.match(surfaceCss, /scroll-margin-top:\s*var\(--einsum-sticky-anchor-offset\)/);
});
