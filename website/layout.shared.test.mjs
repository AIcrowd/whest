import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

test('shared layout logo path is base-path neutral', async () => {
  const file = path.join(websiteRoot, 'lib', 'layout.shared.tsx');
  const source = await readFile(file, 'utf8');

  assert.match(source, /withBasePath/);
  assert.doesNotMatch(source, /src="\/logo\.png"/);
  assert.doesNotMatch(source, /src="\/whest\/logo\.png"/);
});
