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

// DecisionLadder follows a two-stage hybrid layout whose Stage-1 ordering
// mirrors engine/shapeLayer.js's `detectShape` — trivial first, then
// W-empty, then V-empty — so the reader's narrative and the engine's
// actual code path agree leaf by leaf.
//
//   STAGE 1 — dimino-free structural checks. Three questions whose decisions
//             only need (V, W, generators). No element iteration required.
//             q_trivial → |G| = 1 ?         (yes: trivial; no: q_hasW)
//             q_hasW    → W ≠ ∅ ?           (yes: q_hasV; no: allVisible)
//             q_hasV    → V ≠ ∅ ?           (yes: ENUMERATE; no: allSummed)
//
//   STAGE 2 — after enumerating G via Dimino. Four questions and four leaves.
//             q_singleton → |V| = 1 ?       (yes: singleton; no: q_direct)
//             q_direct    → F-check passes? (yes: directProduct; no: q_crossVW)
//                           — moved from Stage 1 because the element-level
//                           F-check requires the materialised group.
//             q_crossVW   → Cross-V/W element? (yes: q_fullSym; no: bruteForceOrbit)
//             q_fullSym   → G = Sym(L_c)?   (yes: young; no: bruteForceOrbit)

test('DecisionLadder QUESTIONS array has 7 entries, in stage order', () => {
  const src = read();
  // Stage-1 order mirrors engine/shapeLayer.js: trivial → hasW → hasV.
  const ids = ['q_trivial', 'q_hasW', 'q_hasV', 'q_singleton', 'q_direct', 'q_crossVW', 'q_fullSym'];
  let lastIdx = -1;
  for (const id of ids) {
    const idx = src.indexOf(`id: '${id}'`);
    assert.ok(idx > 0, `missing QUESTIONS entry id: '${id}'`);
    assert.ok(idx > lastIdx, `QUESTIONS entries out of expected order near '${id}'`);
    lastIdx = idx;
  }
});

test('DecisionLadder no longer references the 4 deleted branch questions', () => {
  const src = read();
  for (const gone of ['q_alt', 'q_wreath', 'q_diag', 'q_setwise']) {
    assert.doesNotMatch(src, new RegExp(gone), `stale question '${gone}' still present in DecisionLadder`);
  }
});

test('DecisionLadder no longer references the 5 deleted regime leaves', () => {
  const src = read();
  // Match as quoted leaf-id strings (e.g. onTrue: 'wreath', specFor('wreath')).
  // 'wreath' and 'wreath product' may appear in prose after the Task 5
  // nomenclature update; only the bare quoted id form is stale.
  for (const gone of ['fullSymmetric', 'alternating', 'diagonalSimultaneous', 'vSetwiseStable']) {
    assert.doesNotMatch(src, new RegExp(gone), `stale regime '${gone}' still referenced in DecisionLadder`);
  }
  // 'wreath' as a leaf id: quoted, followed by word boundary (not 'wreath product' etc.)
  assert.doesNotMatch(src, /'wreath'/, "stale regime 'wreath' still referenced as a leaf id in DecisionLadder");
});

test('DecisionLadder Stage-1 routing: q_trivial splits into trivial (yes) and q_hasW (no)', () => {
  // q_trivial is now the first Stage-1 question — matches engine order where
  // |G|≤1 is the fast-exit check before wa.length / va.length are inspected.
  const src = read();
  assert.match(src, /id:\s*'q_trivial',[\s\S]*?onTrue:\s*'trivial',\s*onFalse:\s*'q_hasW'/);
});

test('DecisionLadder Stage-1 routing: q_hasV feeds the ENUMERATE divider on the "yes" branch', () => {
  // q_hasV is now the LAST Stage-1 question; "yes" (V ≠ ∅ with both W and
  // the group non-trivial) is the only path that escapes Stage 1 into
  // element-level checks, so this is the edge that crosses into enumerate-G.
  const src = read();
  assert.match(src, /id:\s*'q_hasV',[\s\S]*?onTrue:\s*'ENUMERATE',\s*onFalse:\s*'allSummed'/);
});

test('DecisionLadder Stage-2 routing: q_singleton splits into singleton (yes) and q_direct (no)', () => {
  const src = read();
  assert.match(src, /id:\s*'q_singleton',[\s\S]*?onTrue:\s*'singleton',\s*onFalse:\s*'q_direct'/);
});

test('DecisionLadder Stage-2 routing: q_direct splits into directProduct (yes) and q_crossVW (no)', () => {
  const src = read();
  assert.match(src, /id:\s*'q_direct',[\s\S]*?onTrue:\s*'directProduct',\s*onFalse:\s*'q_crossVW'/);
});

test('DecisionLadder q_direct is in Stage 2 (F-check inspects post-dimino elements)', () => {
  const src = read();
  const match = src.match(/id:\s*'q_direct'[\s\S]*?stage:\s*(\d)/);
  assert.ok(match, 'q_direct entry missing or has no stage marker');
  assert.equal(match[1], '2', 'q_direct must be in Stage 2 after the element-level F-check upgrade');
});

test('DecisionLadder Stage-2 routing: q_crossVW splits into q_fullSym (yes) and bruteForceOrbit (no)', () => {
  const src = read();
  assert.match(src, /id:\s*'q_crossVW',[\s\S]*?onTrue:\s*'q_fullSym',[\s\S]*?onFalse:\s*'bruteForceOrbit'/);
});

test('DecisionLadder Stage-2 routing: q_fullSym splits into young (yes) and bruteForceOrbit (no)', () => {
  const src = read();
  assert.match(src, /id:\s*'q_fullSym',[\s\S]*?onTrue:\s*'young',[\s\S]*?onFalse:\s*'bruteForceOrbit'/);
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
