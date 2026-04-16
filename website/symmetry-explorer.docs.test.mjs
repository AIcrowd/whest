import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

test('legacy symmetry-explorer docs page no longer embeds the live app', async () => {
  const file = path.join(websiteRoot, 'content/docs/understanding/symmetry-explorer.mdx');
  const source = await readFile(file, 'utf8');

  assert.doesNotMatch(source, /<SymmetryExplorer \/>/);
  assert.match(source, /\/symmetry-aware-einsum-contractions/);
  assert.match(source, /new tab/i);
});

test('understanding meta no longer exposes the embedded symmetry-explorer page', async () => {
  const file = path.join(websiteRoot, 'content/docs/understanding/meta.json');
  const source = await readFile(file, 'utf8');

  assert.doesNotMatch(source, /"symmetry-explorer"/);
});

test('MDX SymmetryExplorer resolves to a standalone launch handoff instead of the live app', async () => {
  const file = path.join(websiteRoot, 'components', 'mdx.tsx');
  const source = await readFile(file, 'utf8');

  assert.match(source, /function SymmetryExplorerStandaloneHandoff/);
  assert.match(source, /SymmetryExplorer:\s*SymmetryExplorerStandaloneHandoff/);
  assert.match(source, /import Link from 'next\/link'/);
  assert.match(source, /href=\{STANDALONE_SYMMETRY_AWARE_EINSUM_URL\}/);
  assert.doesNotMatch(source, /SymmetryExplorer:\s*SymmetryAwareEinsumContractionsApp/);
});
