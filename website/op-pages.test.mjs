import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('canonical api catch-all route exists and uses the shared leaf renderer', async () => {
  const source = await readSource('app/docs/api/[[...slug]]/page.tsx');

  assert.match(source, /OperationDocPage/);
  assert.match(source, /publicApiLeafImports/);
  assert.match(source, /publicApiRouteMap/);
  assert.match(source, /generateStaticParams/);
  assert.match(source, /source\.getPage\(\['api'/);
  assert.doesNotMatch(source, /node:fs|node:fs\/promises|process\.cwd\(/);
});

test('legacy ops route file has been removed', async () => {
  await assert.rejects(
    readSource('app/docs/api/ops/[slug]/page.tsx'),
    /ENOENT|no such file/i,
  );
});

test('canonical api route prefers authored docs pages before generated leaves', async () => {
  const source = await readSource('app/docs/api/[[...slug]]/page.tsx');

  assert.match(source, /const authoredPage = source\.getPage\(\['api', \.{3}slug\]\)/);
  assert.match(source, /if \(authoredPage\)/);
  assert.match(source, /createRelativeLink/);
  assert.match(source, /getMDXComponents/);
});
