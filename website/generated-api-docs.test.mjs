import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('mdx components register the standalone operation doc renderer', async () => {
  const source = await readSource('components/mdx.tsx');

  assert.match(source, /OperationDocPage/);
  assert.match(source, /OperationDocPage,/);
});

test('operation doc page renderer keeps the required section order', async () => {
  const source = await readSource('components/api-reference/OperationDocPage.tsx');

  const quickInfo = source.indexOf('Quick Info');
  const whestInfo = source.indexOf('Whest-Specific Info');
  const apiDocs = source.indexOf('API Docs');
  const examples = source.indexOf('Whest Examples');

  assert.ok(quickInfo >= 0, 'missing Quick Info section');
  assert.ok(whestInfo > quickInfo, 'Whest-Specific Info should follow Quick Info');
  assert.ok(apiDocs > whestInfo, 'API Docs should follow Whest-Specific Info');
  assert.ok(examples > apiDocs, 'Whest Examples should be last');
  assert.match(source, /Latex/);
  assert.match(source, /No API docs available/i);
  assert.match(source, /No examples available/i);
});

test('api reference table is flattened and link-aware', async () => {
  const apiReference = await readSource('components/api-reference/ApiReference.tsx');
  const operationRow = await readSource('components/api-reference/OperationRow.tsx');
  const filterBar = await readSource('components/api-reference/FilterBar.tsx');
  const styles = await readSource('components/api-reference/styles.module.css');

  assert.match(apiReference, /<th>Area<\/th>/);
  assert.match(apiReference, /<th>Weight<\/th>/);
  assert.match(apiReference, /<th>Cost Formula<\/th>/);
  assert.doesNotMatch(apiReference, /expandedIdx|onToggle|detailRow|opArrow/);

  assert.match(operationRow, /from 'next\/link'/);
  assert.match(operationRow, /href/);
  assert.doesNotMatch(operationRow, /expanded|onToggle|detailRow/);

  assert.match(filterBar, /All areas/);
  assert.match(filterBar, /Area/);
  assert.doesNotMatch(filterBar, /moduleSelect|module filter|All modules/);

  assert.match(styles, /\.areaChip/);
  assert.match(styles, /\.typeChip/);
  assert.doesNotMatch(styles, /detailRow|arrowOpen|opArrow/);
});

test('generated manifests keep supported operation links and exclude blocked ops', async () => {
  const opDocs = JSON.parse(await readFile(path.join(websiteRoot, '.generated/op-docs.json'), 'utf8'));
  const opRefs = JSON.parse(await readFile(path.join(websiteRoot, '.generated/op-refs.json'), 'utf8'));
  const stubSource = await readSource('content/docs/api/ops/absolute.mdx');

  const absolute = opDocs.absolute;

  assert.equal(absolute.area, 'core');
  assert.equal(absolute.display_type, 'counted');
  assert.equal(absolute.href, '/docs/api/ops/absolute');
  assert.match(absolute.cost_formula_latex, /^\\?\$/);
  assert.equal(opRefs.absolute.href, '/docs/api/ops/absolute');
  assert.equal(opRefs.absolute.canonical_name, 'absolute');
  assert.equal(opRefs.save, undefined);
  assert.equal(opDocs.save, undefined);
  assert.doesNotMatch(stubSource, /<!--/);
});
