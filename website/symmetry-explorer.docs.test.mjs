import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

test('symmetry-explorer docs page embeds the component and points local workflow at website', async () => {
  const file = path.join(websiteRoot, 'content/docs/understanding/symmetry-explorer.mdx');
  const source = await readFile(file, 'utf8');

  assert.match(source, /<SymmetryExplorer \/>/);
  assert.match(source, /cd website/);
  assert.doesNotMatch(source, /docs\/visualization\/symmetry-explorer/);
});
