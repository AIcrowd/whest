import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';

const WEBSITE_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname));
const CONTENT_ROOT = path.join(
  WEBSITE_ROOT,
  'components',
  'symmetry-aware-einsum-contractions',
  'content',
);

function read(relativePath) {
  return readFileSync(path.join(CONTENT_ROOT, relativePath), 'utf8');
}

function readFromWebsiteRoot(relativePath) {
  return readFileSync(path.join(WEBSITE_ROOT, relativePath), 'utf8');
}

function collectFiles(root, predicate, acc = []) {
  for (const entry of readdirSync(root)) {
    const fullPath = path.join(root, entry);
    const stats = statSync(fullPath);
    if (stats.isDirectory()) {
      collectFiles(fullPath, predicate, acc);
      continue;
    }
    if (predicate(fullPath)) acc.push(fullPath);
  }
  return acc;
}

test('copy registry schema stays prose-only and string-first', () => {
  const schema = read('schema.ts');

  assert.match(schema, /export\s+type\s+ProseBlock/);
  assert.match(schema, /paragraph/);
  assert.match(schema, /callout/);
  assert.match(schema, /label/);
  assert.match(schema, /caption/);
  assert.match(schema, /text:\s*string/);
  assert.match(schema, /export\s+type\s+SectionCopy/);
  assert.match(schema, /blocks\?:\s*ProseBlock\[\]/);
  assert.match(schema, /slots\?:\s*Record<string,\s*ProseBlock\[\]>/);
  assert.doesNotMatch(schema, /ReactNode|JSX|className/);
});

test('renderProseBlocks is a thin math adapter without layout shells', () => {
  const renderer = read('renderProseBlocks.jsx');

  assert.match(renderer, /import\s+\{\s*Fragment\s*\}\s+from\s+'react'/);
  assert.match(renderer, /InlineMathText/);
  assert.match(renderer, /function\s+renderProseBlocks/);
  assert.doesNotMatch(renderer, /grid-cols|rounded-2xl|px-6|border-gray-200/);
  assert.doesNotMatch(renderer, /className=/);
});

test('content barrel wires the registry and leaf modules stay layout-free', () => {
  const index = read('index.ts');
  const mainIndex = read('main/index.ts');
  const expectedMappings = [
    ['main', 'preamble', 'mainPreamble'],
    ['main', 'section1', 'mainSection1'],
    ['main', 'section2', 'mainSection2'],
    ['main', 'section3', 'mainSection3'],
    ['main', 'section4', 'mainSection4'],
    ['main', 'section5', 'mainSection5'],
    ['appendix', 'section1', 'appendixSection1'],
    ['appendix', 'section2', 'appendixSection2'],
    ['appendix', 'section3', 'appendixSection3'],
    ['appendix', 'section4', 'appendixSection4'],
    ['appendix', 'section5', 'appendixSection5'],
    ['appendix', 'section6', 'appendixSection6'],
  ];

  assert.match(index, /export\s+const\s+contentRegistry\s*=\s*\{/);
  assert.match(
    mainIndex,
    /@ts-expect-error -- Node test runner needs explicit \.ts specifiers/,
    'main/index.ts should document the Node-vs-Next import compatibility shim',
  );
  for (const [group, section, binding] of expectedMappings) {
    assert.match(
      index,
      new RegExp(`${group}:\\s*\\{[\\s\\S]*?\\b${section}:\\s*${binding}\\b`),
      `${group}.${section} should point at ${binding}`,
    );
  }

  const modules = [
    'main/preamble.ts',
    'main/section1.ts',
    'main/section2.ts',
    'main/section3.ts',
    'main/section4.ts',
    'main/section5.ts',
    'appendix/section1.ts',
    'appendix/section2.ts',
    'appendix/section3.ts',
    'appendix/section4.ts',
    'appendix/section5.ts',
    'appendix/section6.ts',
  ];

  for (const relativePath of modules) {
    const source = read(relativePath);
    assert.match(source, /slots\s*:/, `${relativePath} should expose slots`);
    assert.doesNotMatch(source, /className\s*=/, `${relativePath} should stay layout-free`);
    assert.doesNotMatch(
      source,
      /<\/?(?:div|span|p|section|article|header|footer|table|thead|tbody|tr|td|th|button|ul|ol|li|figure|figcaption|svg|path|g|circle|rect|line|polyline|polygon|text)\b/,
      `${relativePath} should not contain JSX layout tags`,
    );
  }
});

test('appendix copy modules use canonical notation keys for pointwise and formal groups', () => {
  const appendixModules = [
    'appendix/section1.ts',
    'appendix/section2.ts',
    'appendix/section3.ts',
    'appendix/section4.ts',
    'appendix/section5.ts',
    'appendix/section6.ts',
  ];

  for (const relativePath of appendixModules) {
    const source = read(relativePath);
    assert.doesNotMatch(
      source,
      /G_\{\\\\mathrm\{pt\}\}/,
      `${relativePath} should not use raw \\mathrm form for G_pt`,
    );
    assert.doesNotMatch(
      source,
      /G_\{\\\\mathrm\{f\}\}/,
      `${relativePath} should not use raw \\mathrm form for G_f`,
    );
  }
});

test('COPY_MAP documents the copy surface and the JSX-owned exceptions', () => {
  const copyMap = read('COPY_MAP.md');
  const assertMapped = (from, to) => {
    const escapedFrom = from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const escapedTo = to.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    assert.match(copyMap, new RegExp(`\`?${escapedFrom}\`?\\s+→\\s+\`?${escapedTo}\`?`));
  };

  assert.match(copyMap, /^# Copy Map$/m);
  assertMapped('main/preamble.ts', 'AlgorithmAtAGlance.jsx');
  assertMapped('main/section1.ts', 'explorerNarrative.js');
  assertMapped('main/section2.ts', 'explorerNarrative.js');
  assertMapped('main/section3.ts', 'explorerNarrative.js');
  assertMapped('main/section4.ts', 'explorerNarrative.js');
  assertMapped('main/section5.ts', 'explorerNarrative.js');
  assertMapped('appendix/section1.ts', 'ExpressionLevelModal.jsx');
  assertMapped('appendix/section2.ts', 'ExpressionLevelModal.jsx');
  assertMapped('appendix/section3.ts', 'ExpressionLevelModal.jsx');
  assertMapped('appendix/section4.ts', 'ExpressionLevelModal.jsx');
  assertMapped('appendix/section5.ts', 'ExpressionLevelModal.jsx');
  assertMapped('appendix/section6.ts', 'ExpressionLevelModal.jsx');
  assertMapped('index.ts', 'shared barrel and registry only');
  assertMapped('schema.ts', 'copy schema only');
  assertMapped('renderProseBlocks.jsx', 'shared prose renderer only');
  assert.match(copyMap, /Stays in JSX/);
  assert.match(copyMap, /layout, spacing, card shells, and responsive structure/i);
  assert.match(copyMap, /`?Latex`?, links, inline code, strong\/em emphasis, and other non-string tokens/i);
  assert.match(copyMap, /example-driven formulas, counts, badges, tables, and tooltips/i);
});

test('registry consumers stay constrained to the documented copy surface', () => {
  const algorithm = readFromWebsiteRoot('components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx');
  const narrative = readFromWebsiteRoot('components/symmetry-aware-einsum-contractions/components/explorerNarrative.js');
  const appendix = readFromWebsiteRoot('components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx');
  const contentImportPattern = /(?:^|['"])(?:\.\.\/)+content\//;
  const componentFiles = collectFiles(
    path.join(WEBSITE_ROOT, 'components', 'symmetry-aware-einsum-contractions'),
    (fullPath) => /\.(?:jsx|js|ts|tsx)$/.test(fullPath),
  );
  const importers = componentFiles
    .filter((fullPath) => !fullPath.includes(`${path.sep}content${path.sep}`))
    .map((fullPath) => path.relative(WEBSITE_ROOT, fullPath).replaceAll(path.sep, '/'))
    .filter((relativePath) => contentImportPattern.test(readFromWebsiteRoot(relativePath)))
    .sort();

  assert.match(`import demo from '../content/main/preamble.ts';`, contentImportPattern);
  assert.match(`import demo from '../../content/main/preamble.ts';`, contentImportPattern);
  assert.match(algorithm, /import\s+mainPreamble\s+from\s+'\.\.\/content\/main\/preamble\.ts'/);
  assert.match(narrative, /from\s+'\.\.\/content\/main\/index\.ts'/);
  assert.match(appendix, /import\s+appendixSection1\s+from\s+'\.\.\/content\/appendix\/section1\.ts'/);
  assert.match(appendix, /import\s+appendixSection6\s+from\s+'\.\.\/content\/appendix\/section6\.ts'/);
  assert.deepEqual(importers, [
    'components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx',
    'components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx',
    'components/symmetry-aware-einsum-contractions/components/explorerNarrative.js',
  ]);
});
