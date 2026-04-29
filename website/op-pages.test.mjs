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

test('api runtime code blocks use shared flopscope themes and preserve shiki backgrounds', async () => {
  const bodySource = await readSource('components/api-reference/OperationDocBody.tsx');
  const exampleSource = await readSource('components/api-reference/OperationDocExample.tsx');
  const themeSource = await readSource('components/api-reference/runtime-shiki-themes.ts');

  assert.match(bodySource, /from '\.\/runtime-shiki-themes'/);
  assert.match(exampleSource, /from '\.\/runtime-shiki-themes'/);
  assert.match(bodySource, /ServerCodeBlock/);
  assert.match(exampleSource, /ServerCodeBlock/);
  assert.doesNotMatch(bodySource, /github-light|github-dark/);
  assert.doesNotMatch(exampleSource, /github-light|github-dark/);
  assert.match(bodySource, /keepBackground/);
  assert.match(exampleSource, /keepBackground/);
  assert.match(themeSource, /flopscopeLight/);
  assert.match(themeSource, /flopscopeDark/);
  assert.match(themeSource, /editor\.background': '#232628'/);
});

test('api dark-mode styling is explicitly defined for shared chrome', async () => {
  const cssSource = await readSource('components/api-reference/styles.module.css');
  const navSource = await readSource('components/api-reference/OperationDocNav.tsx');
  const globalSource = await readSource('app/global.css');

  assert.match(cssSource, /:global\(\.dark\)\s+\.apiReference/);
  assert.match(cssSource, /--api-link-color/);
  assert.match(cssSource, /--api-nav-card-border/);
  assert.match(cssSource, /--api-chip-counted-bg/);
  assert.match(cssSource, /--api-summary-fg/);
  assert.match(cssSource, /--api-signature-ink/);
  assert.match(cssSource, /:global\(\.dark\)\s+\.docHeader/);
  assert.match(cssSource, /\.apiCodeFigure/);
  assert.match(cssSource, /\.apiCodeViewport/);
  assert.match(cssSource, /\.apiCodeFigure\s+:global\(pre\)/);
  assert.match(cssSource, /\.apiCodeFigure\s+:global\(\.line\)/);
  assert.match(cssSource, /color:\s*var\(--shiki-dark/);
  assert.match(navSource, /styles\.docNav/);
  assert.match(navSource, /styles\.docNavCard/);
  assert.match(globalSource, /\.dark article > div\.prose :not\(pre\) > code/);
  assert.match(globalSource, /border-color:\s*rgba\(232,\s*234,\s*235,\s*0\.26\)/);
});

test('api billing presentation is unified around a single cost field', async () => {
  const tableSource = await readSource('components/api-reference/ApiReference.tsx');
  const rowSource = await readSource('components/api-reference/OperationRow.tsx');
  const overlaySource = await readSource('components/api-reference/OperationDocOverlay.tsx');
  const billedCostSource = await readSource('components/api-reference/BilledCost.tsx');

  assert.match(tableSource, /<th>Cost<\/th>/);
  assert.doesNotMatch(tableSource, /<th>Weight<\/th>/);
  assert.doesNotMatch(tableSource, /<th>Cost Formula<\/th>/);
  assert.match(tableSource, /colSpan=\{4\}/);

  assert.match(overlaySource, /Cost/);
  assert.doesNotMatch(overlaySource, /Weight/);
  assert.doesNotMatch(overlaySource, /Cost Formula/);

  assert.match(rowSource, /BilledCost/);
  assert.match(overlaySource, /BilledCost/);
  assert.match(billedCostSource, /weight !== 1/);
  assert.match(billedCostSource, /display_type === 'free'/);
  assert.match(billedCostSource, /display_type === 'blocked'/);
  assert.match(billedCostSource, /Weight multiplier/);
  assert.match(billedCostSource, /formatWeight/);
  assert.match(billedCostSource, /stripMathDelimiters/);
  assert.match(billedCostSource, /op\.cost_formula/);
  assert.match(billedCostSource, />0</);
  assert.match(billedCostSource, /&mdash;|\\u2014/);
});
