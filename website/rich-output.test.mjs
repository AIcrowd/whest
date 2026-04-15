import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import React from 'react';

import {
  decodeAnsiEscapes,
  RICH_OUTPUT_TITLE,
  extractRichOutputText,
  isRichOutputBlock,
  parseAnsiRichText,
} from './components/rich-output.mjs';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

test('rich output blocks are identified by the shared title convention', () => {
  assert.equal(RICH_OUTPUT_TITLE, '__rich_output__');
  assert.equal(isRichOutputBlock({ title: RICH_OUTPUT_TITLE }), true);
  assert.equal(isRichOutputBlock({ title: 'Run it' }), false);
});

test('rich output text extraction preserves line structure', () => {
  const code = React.createElement('code', {}, [
    React.createElement('span', { className: 'line', key: '1' }, [
      '╭ header',
      React.createElement('span', { key: '1a' }, ' ╮'),
    ]),
    React.createElement('span', { className: 'line', key: '2' }, '│ body │'),
    React.createElement('span', { className: 'line', key: '3' }, '╰──────╯'),
  ]);

  assert.equal(
    extractRichOutputText(code),
    ['╭ header ╮', '│ body │', '╰──────╯'].join('\n'),
  );
});

test('rich output ANSI escapes are decoded and parsed into styled segments', () => {
  const decoded = decodeAnsiEscapes(
    String.raw`\u001b[36mTitle\u001b[0m \u001b[32m42\u001b[0m`,
  );
  const lines = parseAnsiRichText(decoded);

  assert.equal(lines.length, 1);
  assert.deepEqual(lines[0], [
    { text: 'Title', color: 'cyan', bold: false, dim: false },
    { text: ' ', color: null, bold: false, dim: false },
    { text: '42', color: 'green', bold: false, dim: false },
  ]);
});

test('quickstart marks the budget summary as rich output', async () => {
  const file = path.join(
    websiteRoot,
    'content/docs/getting-started/quickstart.mdx',
  );
  const source = await readFile(file, 'utf8');

  assert.match(source, /```(?:text|plaintext)?\s+title="__rich_output__"/);
  assert.match(source, /\\x1b\[[0-9;]*m/);
});

test('competition guide uses rich output for the budget summary snapshot', async () => {
  const file = path.join(
    websiteRoot,
    'content/docs/getting-started/competition.mdx',
  );
  const source = await readFile(file, 'utf8');

  assert.match(source, /```(?:text|plaintext)?\s+title="__rich_output__"/);
  assert.match(source, /\\x1b\[[0-9;]*m/);
});

test('einsum guide shows real path output instead of explanatory comments in the code block', async () => {
  const file = path.join(websiteRoot, 'content/docs/guides/einsum.mdx');
  const source = await readFile(file, 'utf8');

  assert.doesNotMatch(source, /# Prints a multi-line table with a header/);
  assert.doesNotMatch(source, /# Call info\.format_table\(verbose=True\)/);
  assert.match(source, /```text[\s\S]*Complete contraction:/);
});
