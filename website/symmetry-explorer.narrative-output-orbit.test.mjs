import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve('components/symmetry-aware-einsum-contractions');

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), 'utf8');
}

test('main narrative uses output representatives, not output-bin updates', () => {
  const files = [
    'SymmetryAwareEinsumContractionsApp.jsx',
    'components/MentalFrameworkCode.jsx',
    'components/TotalCostView.jsx',
    'content/main/preamble.js',
    'content/main/certification.js',
    'content/main/rowsCols.js',
    'content/main/assembleCost.js',
  ];
  const joined = files.map(read).join('\n');

  assert.match(joined, /stored output representatives/);
  assert.match(joined, /Stab/);
  assert.match(joined, /product-orbit/);
  assert.doesNotMatch(joined, /output-bin update/i);
  assert.doesNotMatch(joined, /storage-only/i);
});

test('mental framework pseudocode names output representatives', () => {
  const source = read('components/MentalFrameworkCode.jsx');
  assert.match(source, /out_rep/);
  assert.match(source, /R\[out_rep\]/);
  assert.match(source, /coeff\(rep, out_rep\)/);
});
