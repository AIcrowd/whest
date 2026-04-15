import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

test('standalone route renders the renamed symmetry-aware app', async () => {
  const routeFile = path.join(
    websiteRoot,
    'app',
    'symmetry-aware-einsum-contractions',
    'page.tsx',
  );
  const routeSource = await readFile(routeFile, 'utf8');

  assert.match(routeSource, /SymmetryAwareEinsumContractionsApp/);
  assert.match(routeSource, /title:\s*'Symmetry Aware Einsum Contractions'/);
});

test('renamed feature entry exports the symmetry-aware app component', async () => {
  const entryFile = path.join(
    websiteRoot,
    'components',
    'symmetry-aware-einsum-contractions',
    'index.tsx',
  );
  const entrySource = await readFile(entryFile, 'utf8');

  assert.match(entrySource, /SymmetryAwareEinsumContractionsApp/);
  assert.doesNotMatch(entrySource, /SymmetryExplorer/);
});
