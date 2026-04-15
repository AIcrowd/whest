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

test('cleanup references only the standalone symmetry-aware route', async () => {
  const filesToScan = [
    'symmetry-explorer.act1.test.mjs',
    'symmetry-explorer.narrative.test.mjs',
    'symmetry-explorer.savings.test.mjs',
    'symmetry-explorer.state.test.mjs',
    'symmetry-explorer.teaching-model.test.mjs',
    'app/symmetry-aware-einsum-contractions/page.tsx',
    'app/(home)/page.tsx',
    'content/docs/understanding/symmetry-detection.mdx',
  ];

  for (const relativePath of filesToScan) {
    const source = await readFile(path.join(websiteRoot, relativePath), 'utf8');
    assert.doesNotMatch(
      source,
      /components\/symmetry-explorer/,
      `${relativePath} still references the legacy symmetry-explorer component tree`,
    );
  }

  const docsSource = await readFile(
    path.join(websiteRoot, 'content/docs/understanding/symmetry-detection.mdx'),
    'utf8',
  );
  assert.match(docsSource, /\/symmetry-aware-einsum-contractions/);
  assert.doesNotMatch(docsSource, /\/docs\/understanding\/symmetry-explorer/);

  const homeSource = await readFile(path.join(websiteRoot, 'app/(home)/page.tsx'), 'utf8');
  assert.match(homeSource, /\/symmetry-aware-einsum-contractions/);
  assert.doesNotMatch(homeSource, /\/docs\/explanation\/symmetry-explorer/);
});
