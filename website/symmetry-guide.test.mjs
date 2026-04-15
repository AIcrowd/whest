import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const symmetryGuidePath = path.join(
  websiteRoot,
  'content/docs/guides/symmetry.mdx',
);

async function readSymmetryGuide() {
  return readFile(symmetryGuidePath, 'utf8');
}

test('symmetry guide documents every declaration style', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /symmetric_axes=\(0, 1\)/);
  assert.match(source, /symmetric_axes=\[\(0, 1\), \(2, 3\)\]/);
  assert.match(
    source,
    /PermutationGroup\.symmetric\(3, axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /PermutationGroup\.cyclic\(3, axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /PermutationGroup\.dihedral\(4, axes=\(0, 1, 2, 3\)\)/,
  );
  assert.match(
    source,
    /we\.PermutationGroup\(\s*[\s\S]*we\.Permutation\(we\.Cycle\(0, 2\)\(1, 3\)\)/,
  );
});

test('symmetry guide includes the Reynolds helper and whest-only indexing examples', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /def symmetrize_with_group\(shape, symmetry_group\):/);
  assert.match(source, /we\.newaxis/);
  assert.match(source, /we\.array\(\[0, 1\]\)/);
});

test('symmetry guide includes all major propagation sections', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /## Symmetry propagation at a glance/);
  assert.match(source, /## Slicing rules/);
  assert.match(source, /## Reduction rules/);
  assert.match(source, /## Binary pointwise ops and broadcasting/);
  assert.match(source, /## Warnings, conservative behavior, and re-tagging/);
  assert.match(source, /## Under the hood/);
});
