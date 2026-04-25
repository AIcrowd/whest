import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

test('shared layout nav uses the wordmark, not a raster logo', async () => {
  const file = path.join(websiteRoot, 'lib', 'layout.shared.tsx');
  const source = await readFile(file, 'utf8');

  assert.match(source, /flopscope-wordmark/);
  assert.doesNotMatch(source, /src="\/logo\.png"/);
  assert.doesNotMatch(source, /src="\/flopscope\/logo\.png"/);
});

test('root layout metadata wires the favicon through withBasePath', async () => {
  const file = path.join(websiteRoot, 'app', 'layout.tsx');
  const source = await readFile(file, 'utf8');

  assert.match(source, /export const metadata/);
  assert.match(source, /icons:\s*\{/);
  // Primary favicon: the coral-dot SVG from the design-system size ladder
  assert.match(source, /withBasePath\('\/favicon\.svg'\)/);
  // PNG fallback and apple touch icon still point at the brush logo
  assert.match(source, /withBasePath\('\/logo\.png'\)/);
});
