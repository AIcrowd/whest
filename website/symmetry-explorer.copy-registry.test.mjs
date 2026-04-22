import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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

test('copy registry schema stays prose-only and string-first', () => {
  const schema = read('schema.ts');

  assert.match(schema, /type\s+ProseBlock\s*=/);
  assert.match(schema, /'paragraph'/);
  assert.match(schema, /'callout'/);
  assert.match(schema, /'label'/);
  assert.match(schema, /'caption'/);
  assert.match(schema, /text:\s*string/);
  assert.match(schema, /type\s+SectionCopy\s*=\s*\{/);
  assert.match(schema, /blocks\?:\s*ProseBlock\[\]/);
  assert.match(schema, /slots\?:\s*Record<string,\s*ProseBlock\[\]>/);
  assert.doesNotMatch(schema, /ReactNode|JSX|className/);
});

test('renderProseBlocks is a thin math adapter without layout shells', () => {
  const renderer = read('renderProseBlocks.jsx');

  assert.match(renderer, /import\s+InlineMathText\s+from\s+'\.\.\/components\/InlineMathText\.jsx'/);
  assert.match(renderer, /function\s+renderProseBlocks/);
  assert.doesNotMatch(renderer, /grid-cols|rounded-2xl|px-6|border-gray-200/);
  assert.doesNotMatch(renderer, /className=/);
});

test('content modules expose slots without JSX layout markup', () => {
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
