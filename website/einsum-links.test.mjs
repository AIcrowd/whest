import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const einsumGuidePath = path.join(
  websiteRoot,
  'content/docs/guides/einsum.mdx',
);

test('einsum guide links to the current symmetry guide path', async () => {
  const source = await readFile(einsumGuidePath, 'utf8');

  assert.doesNotMatch(source, /exploit-symmetry\.md/);
  assert.doesNotMatch(source, /\[Exploit Symmetry(?: Savings)?\]\(\.\/symmetry\)/);
  assert.match(source, /\[Symmetry Savings\]\(\/docs\/guides\/symmetry\/\)/);
});
