import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

const targetedDocs = [
  'content/docs/getting-started/quickstart.mdx',
  'content/docs/guides/symmetry.mdx',
  'content/docs/guides/einsum.mdx',
  'content/docs/guides/linalg.mdx',
];

for (const relativePath of targetedDocs) {
  test(`${relativePath} uses flopscope-only operations in runnable examples`, async () => {
    const file = path.join(websiteRoot, relativePath);
    const source = await readFile(file, 'utf8');
    const pythonFences = [...source.matchAll(/```python\n([\s\S]*?)```/g)].map(
      ([, block]) => block,
    );

    assert.ok(pythonFences.length > 0);

    for (const block of pythonFences) {
      assert.doesNotMatch(block, /\bimport math\b/);
      assert.doesNotMatch(block, /\bmath\./);
      assert.doesNotMatch(block, /\bimport numpy as np\b/);
      assert.doesNotMatch(block, /\bnp\./);
    }
  });
}
