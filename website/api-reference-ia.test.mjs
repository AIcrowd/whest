import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('api docs tree exposes hub, namespaces, and operation cost index pages', async () => {
  const meta = JSON.parse(await readSource('content/docs/api/meta.json'));
  assert.deepEqual(meta.pages, [
    'index',
    'operation-cost-index',
    'numpy',
    'flopscope',
    'stats',
    'accounting',
    'for-agents',
  ]);
});

test('hub page is namespace-first and no longer embeds the op table directly', async () => {
  const source = await readSource('content/docs/api/index.mdx');
  assert.match(source, /ApiNamespaceHub/);
  assert.doesNotMatch(source, /<ApiReference\s*\/>/);
});

test('operation cost index page owns the dense operation table', async () => {
  const source = await readSource('content/docs/api/operation-cost-index.mdx');
  assert.match(source, /OperationCostIndex/);
});

test('namespace landing pages use the shared namespace inventory component', async () => {
  for (const page of ['numpy', 'flopscope', 'stats', 'accounting']) {
    const source = await readSource(`content/docs/api/${page}.mdx`);
    assert.match(source, /NamespaceInventory/);
  }
});
