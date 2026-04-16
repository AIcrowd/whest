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
  const signatureSource = await readSource('components/api-reference/OperationDocSignature.tsx');
  const overlaySource = await readSource('components/api-reference/OperationDocOverlay.tsx');
  const sectionSource = await readSource('components/api-reference/OperationDocSection.tsx');
  const fieldListSource = await readSource('components/api-reference/OperationDocFieldList.tsx');
  const exampleSource = await readSource('components/api-reference/OperationDocExample.tsx');
  const navSource = await readSource('components/api-reference/OperationDocNav.tsx');
  const linkSource = await readSource('components/api-reference/OperationDocLink.tsx');

  assert.match(typeSource, /export interface OperationDocRecord/);
  assert.match(signatureSource, /whest source/i);
  assert.match(signatureSource, /numpy source/i);
  assert.doesNotMatch(headerSource, /\[source\]/i);
  assert.match(overlaySource, /Weight/);
  assert.match(sectionSource, /Parameters/);
  assert.match(sectionSource, /Returns/);
  assert.match(sectionSource, /OperationDocFieldList/);
  assert.match(fieldListSource, /docDefinitionName/);
  assert.match(sectionSource, /OperationDocLink/);
  assert.match(exampleSource, /CodeBlock/);
  assert.match(linkSource, /canonical/i);
  assert.match(navSource, /Previous/);
  assert.match(navSource, /Next/);
  assert.match(navSource, /next\/link/);
});

test('histogram structured manifest carries parity-ready content', async () => {
  const opDocs = JSON.parse(await readFile(path.join(websiteRoot, '.generated/op-docs.json'), 'utf8'));
  const docCoverage = JSON.parse(await readFile(path.join(websiteRoot, '.generated/op-doc-coverage.json'), 'utf8'));
  const histogram = opDocs.histogram;
  const histogramBinEdges = opDocs.histogram_bin_edges;
  const nanvar = opDocs.nanvar;

  assert.equal(histogram.summary, 'Compute the histogram of a dataset.');
  assert.equal(histogram.provenance_label, 'Adapted from NumPy docs');
  assert.match(histogram.signature, /^we\.histogram\(/);
  assert.ok(histogram.parameters.length >= 4);
  assert.ok(histogram.returns.length >= 2);
  assert.ok(histogram.see_also.length >= 2);
  assert.ok(histogram.notes_sections.length >= 1);
  assert.ok(histogram.example);
  assert.ok(Array.isArray(histogram.body_sections));
  assert.ok(histogram.body_sections.length >= 4);
  assert.ok(histogram.body_sections.some((section) => section.title === 'Summary'));
  assert.ok(histogram.body_sections.some((section) => section.title === 'Parameters'));
  assert.ok(histogram.body_sections.some((section) => section.title === 'Returns'));
  assert.ok(histogram.body_sections.some((section) => section.title === 'Examples'));
  assert.match(histogram.example.code, /import whest as we/);
  assert.ok(histogram.previous);
  assert.ok(histogram.next);

  const notesSection = histogramBinEdges.body_sections.find((section) => section.title === 'Notes');
  assert.ok(notesSection);
  assert.ok(notesSection.blocks.some((block) => block.type === 'definition_list'));
  assert.ok(notesSection.blocks.some((block) => block.type === 'math_block'));

  const nanvarParameters = nanvar.body_sections.find((section) => section.title === 'Parameters');
  assert.ok(nanvarParameters);
  const whereField = nanvarParameters.blocks[0].items.find((item) => item.name === 'where');
  assert.ok(whereField);
  assert.ok(
    whereField.body_blocks.some(
      (block) =>
        block.type === 'paragraph' &&
        block.inline.some(
          (inline) =>
            inline.kind === 'link' &&
            inline.external_url?.endsWith('numpy.ufunc.reduce.html'),
        ),
    ),
  );
  assert.equal(docCoverage.nanvar.has_issues, false);
});

test('operation doc examples reuse the shared docs output renderer', async () => {
  const exampleSource = await readSource('components/api-reference/OperationDocExample.tsx');

  assert.match(exampleSource, /github-dark/);
  assert.match(exampleSource, /CodeBlock/);
  assert.match(exampleSource, /Pre/);
  assert.match(exampleSource, /Terminal/);
  assert.match(exampleSource, /fumadocs-core\/highlight/);
  assert.match(exampleSource, /highlightedOutput/);
  assert.doesNotMatch(exampleSource, /docOutputBlock/);
  assert.doesNotMatch(exampleSource, /docOutputLabel/);
});
