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

test('copy renderers thread stable key prefixes through singleton prose helpers', () => {
  const renderer = read('renderProseBlocks.jsx');
  const algorithm = readFromWebsiteRoot('components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx');
  const appendix = readFromWebsiteRoot('components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx');

  assert.match(renderer, /renderProseBlocks\(\s*blocks = \[\],\s*\{ renderCallout, strongClassName = null, keyPrefix = 'prose', themeOverride = null \} = \{\},\s*\)/);
  assert.match(renderer, /const blockKey = `\$\{keyPrefix\}-\$\{block\.kind\}-\$\{index\}`;/);
  assert.match(renderer, /<Fragment key=\{blockKey\}>/);
  assert.match(renderer, /<InlineMathText key=\{blockKey\} strongClassName=\{strongClassName\} themeOverride=\{themeOverride\}>/);
  assert.match(algorithm, /function renderSingleProseBlock\(blocks = \[\], keyPrefix = 'main-prose-block'\)/);
  assert.match(algorithm, /return renderProseBlocks\(blocks, \{ keyPrefix \}\)\[0\] \?\? null;/);
  assert.match(appendix, /return renderProseBlocks\(normalizedBlocks, \{ renderCallout, strongClassName, keyPrefix: slotKey, themeOverride \}\);/);
  assert.match(appendix, /return renderAppendixSlot\(\[block\], \{ \.\.\.options, slotKey: `\$\{slotKey\}-\$\{index\}` \}\)\[0\] \?\? null;/);
});

test('content barrel wires the registry and leaf modules stay layout-free', () => {
  const index = read('index.ts');
  const mainIndex = read('main/index.js');
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
  assert.match(mainIndex, /export \{ default as mainPreamble \} from '\.\/preamble\.js';/);
  assert.match(mainIndex, /export \{ default as mainSection5 \} from '\.\/section5\.js';/);
  for (const [group, section, binding] of expectedMappings) {
    assert.match(
      index,
      new RegExp(`${group}:\\s*\\{[\\s\\S]*?\\b${section}:\\s*${binding}\\b`),
      `${group}.${section} should point at ${binding}`,
    );
  }

  const modules = [
    'main/preamble.js',
    'main/section1.js',
    'main/section2.js',
    'main/section3.js',
    'main/section4.js',
    'main/section5.js',
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

test('content registry does not wrap inline math in markdown code ticks', () => {
  const modules = [
    'main/preamble.js',
    'main/section1.js',
    'main/section2.js',
    'main/section3.js',
    'main/section4.js',
    'main/section5.js',
    'appendix/section1.ts',
    'appendix/section2.ts',
    'appendix/section3.ts',
    'appendix/section4.ts',
    'appendix/section5.ts',
    'appendix/section6.ts',
  ];

  for (const relativePath of modules) {
    const source = read(relativePath);
    assert.doesNotMatch(
      source,
      /`\$[^`]+\$`/,
      `${relativePath} should not wrap inline math in code ticks`,
    );
  }
});

test('appendix section 3 takeaway introduces H = Stab_{G_pt}(V)|_V as the output action', () => {
  const source = read('appendix/section3.ts');

  // Section 3 was renamed to 'Output representatives induced by the
  // product-side group' (V4) and now defines H rather than describing
  // G_out as a separate storage-only concept.
  assert.match(
    source,
    /Output representatives induced by the product-side group/,
  );
  assert.match(source, /\$H = \\\\mathrm\{Stab\}_\{G_\{\\\\text\{pt\}\}\}\(V\)\|_V\$/);
  assert.match(source, /already present in \$\\\\alpha = \\\\#\\\\\{\(O,Q\)/);
  assert.doesNotMatch(source, /direct dense-output evaluator/);
  assert.doesNotMatch(source, /symmetry-aware accumulation model/);
});

test('appendix section 6 explains the unified output-orbit accumulation count', () => {
  const source = read('appendix/section6.ts');

  // V4 turned section 6 into the partition-counting theorem section. The
  // older 'unified accumulation count' framing has been folded into the
  // theorem's intro and the per-section explanations live in sections 3, 4, 5.
  assert.match(source, /Partition-counting theorem for branching projections/);
  assert.match(source, /typed equality pattern/);
  assert.match(source, /\\\\mathrm\{Stab\}_G\(V\)/);
  assert.match(source, /\\\\sum_\{\\\\tilde\{x\}/);
  assert.doesNotMatch(source, /Model 1/);
  assert.doesNotMatch(source, /Model 2/);
  assert.doesNotMatch(source, /Model 3/);
});

test('copy modules use canonical notation keys for pointwise and formal groups', () => {
  const modules = [
    'main/preamble.js',
    'main/section3.js',
    'main/section4.js',
    'main/section5.js',
    'appendix/section1.ts',
    'appendix/section2.ts',
    'appendix/section3.ts',
    'appendix/section4.ts',
    'appendix/section5.ts',
    'appendix/section6.ts',
  ];

  for (const relativePath of modules) {
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
  assertMapped('main/preamble.js', 'AlgorithmAtAGlance.jsx');
  assertMapped('main/section1.js', 'explorerNarrative.js');
  assertMapped('main/section2.js', 'explorerNarrative.js');
  assertMapped('main/section3.js', 'explorerNarrative.js');
  assertMapped('main/section4.js', 'explorerNarrative.js');
  assertMapped('main/section5.js', 'explorerNarrative.js');
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

  assert.match(`import demo from '../content/main/preamble.js';`, contentImportPattern);
  assert.match(`import demo from '../../content/main/preamble.js';`, contentImportPattern);
  assert.match(algorithm, /import\s+mainPreamble\s+from\s+'\.\.\/content\/main\/preamble\.js'/);
  assert.match(narrative, /from\s+'\.\.\/content\/main\/index\.js'/);
  assert.match(appendix, /import\s+appendixSection1\s+from\s+'\.\.\/content\/appendix\/section1\.ts'/);
  assert.match(appendix, /import\s+appendixSection6\s+from\s+'\.\.\/content\/appendix\/section6\.ts'/);
  assert.deepEqual(importers, [
    'components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx',
    'components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx',
    'components/symmetry-aware-einsum-contractions/components/explorerNarrative.js',
  ]);
});
