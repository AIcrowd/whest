import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx'),
  'utf-8',
);

test('app imports and renders LabelClusterPanel', () => {
  assert.match(source, /import LabelClusterPanel/);
  assert.match(source, /<LabelClusterPanel/);
});

test('app maintains defaultSize + clusterSizes state (replaces the raw dimensionN slider)', () => {
  assert.match(source, /defaultSize/);
  assert.match(source, /clusterSizes/);
});

test('app seeds clusterSizes from analysis.clusters', () => {
  assert.match(source, /analysis\?\.clusters/);
});

test('app passes labelSizes into analyzeExample', () => {
  assert.match(source, /labelSizes:\s*mergedLabelSizes/);
});
