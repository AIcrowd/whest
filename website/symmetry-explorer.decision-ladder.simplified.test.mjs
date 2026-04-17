import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx',
);

function read() {
  return readFileSync(SRC, 'utf-8');
}

test('DecisionLadder QUESTIONS array has exactly 5 entries (one per surviving branch point)', () => {
  const src = read();
  const ids = ['q_hasG', 'q_hasW', 'q_hasV', 'q_singleton', 'q_direct'];
  let lastIdx = -1;
  for (const id of ids) {
    const idx = src.indexOf(`id: '${id}'`);
    assert.ok(idx > 0, `missing QUESTIONS entry id: '${id}'`);
    assert.ok(idx > lastIdx, `QUESTIONS entries out of expected order near '${id}'`);
    lastIdx = idx;
  }
});

test('DecisionLadder no longer references the 5 deleted branch questions', () => {
  const src = read();
  for (const gone of ['q_fullSym', 'q_alt', 'q_wreath', 'q_diag', 'q_setwise']) {
    assert.doesNotMatch(src, new RegExp(gone), `stale question '${gone}' still present in DecisionLadder`);
  }
});

test('DecisionLadder no longer references the 5 deleted regime leaves', () => {
  const src = read();
  for (const gone of ['fullSymmetric', 'alternating', 'wreath', 'diagonalSimultaneous', 'vSetwiseStable']) {
    assert.doesNotMatch(src, new RegExp(gone), `stale regime '${gone}' still referenced in DecisionLadder`);
  }
});

test('DecisionLadder routes q_singleton.onFalse straight to q_direct, and q_direct.onFalse to bruteForceOrbit', () => {
  const src = read();
  // Keep these as the structural contract the 6-case partition depends on.
  assert.match(src, /id:\s*'q_singleton',[\s\S]*?onTrue:\s*'singleton',\s*onFalse:\s*'q_direct'/);
  assert.match(src, /id:\s*'q_direct',[\s\S]*?onTrue:\s*'directProduct',\s*onFalse:\s*'bruteForceOrbit'/);
});
