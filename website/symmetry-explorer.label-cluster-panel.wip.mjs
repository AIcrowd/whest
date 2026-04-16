import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/LabelClusterPanel.jsx'),
  'utf-8',
);

test('LabelClusterPanel exports a default React component', () => {
  assert.match(source, /export default function LabelClusterPanel/);
});

test('LabelClusterPanel renders one row per cluster with labels + size', () => {
  assert.match(source, /clusters\.map/);
  assert.match(source, /cluster\.labels/);
  assert.match(source, /cluster\.size/);
});

test('LabelClusterPanel calls onSizeChange on numeric input change', () => {
  assert.match(source, /onSizeChange/);
  assert.match(source, /onChange/);
});

test('LabelClusterPanel uses RoleBadge to color labels by V/W', () => {
  assert.match(source, /RoleBadge/);
});

test('LabelClusterPanel supports a Set-all-n shortcut (onSetAll)', () => {
  assert.match(source, /onSetAll/);
});
