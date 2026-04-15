import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('operation doc parity components are split into dedicated renderers', async () => {
  const typeSource = await readSource('components/api-reference/op-doc-types.ts');
  const headerSource = await readSource('components/api-reference/OperationDocHeader.tsx');
  const overlaySource = await readSource('components/api-reference/OperationDocOverlay.tsx');
  const sectionSource = await readSource('components/api-reference/OperationDocSection.tsx');
  const exampleSource = await readSource('components/api-reference/OperationDocExample.tsx');
  const navSource = await readSource('components/api-reference/OperationDocNav.tsx');

  assert.match(typeSource, /export interface OperationDocRecord/);
  assert.match(headerSource, /show source/i);
  assert.match(overlaySource, /Weight/);
  assert.match(sectionSource, /Parameters/);
  assert.match(sectionSource, /Returns/);
  assert.match(exampleSource, /CodeBlock/);
  assert.match(navSource, /Previous/);
  assert.match(navSource, /Next/);
});

test('histogram structured manifest carries parity-ready content', async () => {
  const opDocs = JSON.parse(await readFile(path.join(websiteRoot, '.generated/op-docs.json'), 'utf8'));
  const histogram = opDocs.histogram;

  assert.equal(histogram.summary, 'Compute the histogram of a dataset.');
  assert.equal(histogram.provenance_label, 'Adapted from NumPy docs');
  assert.match(histogram.signature, /^we\.histogram\(/);
  assert.ok(histogram.parameters.length >= 4);
  assert.ok(histogram.returns.length >= 2);
  assert.ok(histogram.see_also.length >= 2);
  assert.ok(histogram.notes_sections.length >= 1);
  assert.ok(histogram.example);
  assert.match(histogram.example.code, /import whest as we/);
  assert.ok(histogram.previous);
  assert.ok(histogram.next);
});
