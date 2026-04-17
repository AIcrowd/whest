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

// DecisionLadder has been promoted to the two-stage hybrid layout:
//   STAGE 1 — dimino-free structural checks. Four questions whose decisions
//             only need (V, W, generators).
//             q_hasW  → W ≠ ∅ ?             (leaf on "no": allVisible)
//             q_hasV  → V ≠ ∅ ?             (leaf on "no": allSummed)
//             q_trivial → |G| = 1 ?         (leaf on "yes": trivial)
//             q_direct  → all gens V-only / W-only ?  (leaf on "yes": directProduct;
//                         "no" crosses into the ENUMERATE divider)
//
//   STAGE 2 — after enumerating G via Dimino. One question with two leaves.
//             q_singleton → |V| = 1 ?       (yes: singleton; no: bruteForceOrbit)

test('DecisionLadder QUESTIONS array has exactly 5 entries, in stage order', () => {
  const src = read();
  const ids = ['q_hasW', 'q_hasV', 'q_trivial', 'q_direct', 'q_singleton'];
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

test('DecisionLadder Stage-1 routing: q_direct feeds the ENUMERATE divider on the "no" branch', () => {
  const src = read();
  assert.match(src, /id:\s*'q_direct',[\s\S]*?onTrue:\s*'directProduct',\s*onFalse:\s*'ENUMERATE'/);
});

test('DecisionLadder Stage-2 routing: q_singleton splits into singleton (yes) and bruteForceOrbit (no)', () => {
  const src = read();
  assert.match(src, /id:\s*'q_singleton',[\s\S]*?onTrue:\s*'singleton',\s*onFalse:\s*'bruteForceOrbit'/);
});

test('DecisionLadder marks questions with their stage (1 structural, 2 symmetry)', () => {
  const src = read();
  // Every question entry should name which stage it belongs to, so the
  // layout code can group them under the correct band.
  assert.match(src, /stage:\s*1/);
  assert.match(src, /stage:\s*2/);
});

test('DecisionLadder includes the ENUMERATE divider node type + tooltip entry', () => {
  const src = read();
  // Visual artefact that sits between the two stages.
  assert.match(src, /enumerate G via dimino/i);
  // Custom ReactFlow node type for the divider.
  assert.match(src, /type:\s*'enumerate'/);
  // Stage bands render as their own ReactFlow node type.
  assert.match(src, /type:\s*'stageBand'/);
});
