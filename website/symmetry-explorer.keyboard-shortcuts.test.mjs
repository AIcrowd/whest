import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/lib/useKeyboardShortcuts.js'),
  'utf-8',
);

test('useKeyboardShortcuts exports the hook', () => {
  assert.match(source, /export function useKeyboardShortcuts/);
});

test('useKeyboardShortcuts guards against text inputs', () => {
  assert.match(source, /INPUT|TEXTAREA|isContentEditable/);
});

test('useKeyboardShortcuts adds and removes keydown listener', () => {
  assert.match(source, /addEventListener\(['"]keydown/);
  assert.match(source, /removeEventListener\(['"]keydown/);
});
