import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/CompactCaseTable.jsx',
);

function read() {
  return readFileSync(SRC, 'utf-8');
}

test('CompactCaseTable declares exactly the 6 supported cases in order', () => {
  const src = read();
  const ids = ['trivial', 'allVisible', 'allSummed', 'singleton', 'directProduct', 'bruteForceOrbit'];
  let lastIdx = -1;
  for (const id of ids) {
    const idx = src.indexOf(`id: '${id}'`);
    assert.ok(idx > 0, `missing CASES entry id: '${id}'`);
    assert.ok(idx > lastIdx, `CASES entries out of expected order near '${id}'`);
    lastIdx = idx;
  }
  // And no removed regimes leak back in.
  for (const gone of ['fullSymmetric', 'alternating', 'wreath', 'diagonalSimultaneous', 'vSetwiseStable']) {
    assert.doesNotMatch(src, new RegExp(`id: '${gone}'`), `unexpected stale regime '${gone}' in CompactCaseTable`);
  }
});

test('CompactCaseTable carries the benchmark empirical timings', () => {
  const src = read();
  // Spot-check a few non-trivial timings from the benchmark table so a stale
  // copy-paste can't pass silently.
  assert.match(src, /median: '0\.38 μs'/);      // singleton detection
  assert.match(src, /max: '3\.9 μs'/);          // singleton detection
  assert.match(src, /median: '48 μs'/);         // singleton computation
  assert.match(src, /max: '258 μs'/);           // singleton computation
  assert.match(src, /median: '95 μs'/);         // directProduct computation
  assert.match(src, /max: '266 μs'/);
  assert.match(src, /max: '5095 μs \(budget-capped\)'/);
});

test('CompactCaseTable applies active-highlight styles when a row id is in activeRegimeIds', () => {
  const src = read();
  // Row className branches on isActive; spot-check for the coral-accent ring
  // and the accent background used elsewhere for active state.
  assert.match(src, /ring-primary/);
  assert.match(src, /bg-accent/);
  assert.match(src, /data-active=\{isActive \? 'true' : 'false'\}/);
});

test('CompactCaseTable reuses shared primitives (Table, CaseBadge)', () => {
  const src = read();
  assert.match(src, /from '@\/components\/ui\/table'/);
  assert.match(src, /import CaseBadge/);
});

test('ComponentCostView renders DecisionLadder + CompactCaseTable as a 30% / 70% two-column row', () => {
  const appSrc = readFileSync(
    resolve(
      __dirname,
      'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx',
    ),
    'utf-8',
  );
  assert.match(appSrc, /import CompactCaseTable/);
  assert.match(appSrc, /<CompactCaseTable/);
  assert.match(appSrc, /lg:grid-cols-\[30%_1fr\]/);
});
