import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('/docs/api table links rows to standalone op pages', async () => {
  const source = await readSource('components/api-reference/index.tsx');

  assert.match(source, /detail_href/);
  assert.match(source, /href:\s*op\.blocked \? undefined : op\.detail_href/);
});

test('op route is present and avoids runtime filesystem access', async () => {
  const source = await readSource('app/docs/api/ops/[slug]/page.tsx');

  assert.match(source, /generateStaticParams/);
  assert.match(source, /opDocImports/);
  assert.doesNotMatch(source, /node:fs|node:fs\/promises|from 'fs'|from "fs"/);
  assert.doesNotMatch(source, /process\.cwd\(/);
});

test('op pages use the dedicated numpy-style renderer stack', async () => {
  const pageSource = await readSource('components/api-reference/OperationDocPage.tsx');

  assert.match(pageSource, /OperationDocHeader/);
  assert.match(pageSource, /OperationDocOverlay/);
  assert.match(pageSource, /OperationDocBody/);
  assert.match(pageSource, /OperationDocSignature/);
});
