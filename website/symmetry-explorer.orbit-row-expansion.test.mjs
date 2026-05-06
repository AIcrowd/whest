// V3.1 §13 — C13 O→Q Row Expansion Card. Source-grep tests for the
// OrbitDetailCard polish: V3.1 caption (branching + non-branching forms),
// THIS Q badge marker on the user-clicked destination, and a
// coefficient-view toggle (default off → members→destination, on →
// destination→coefficient pairs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const CARD_PATH = 'components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx';

test('OrbitDetailCard renders V3.1 branching caption (N stored-output updates + contributes N to alpha)', () => {
  const src = read(CARD_PATH);
  // Branching caption uses the V3.1 verbatim fragments. Use simple substring
  // checks instead of strict regexes so JSX whitespace doesn't trip the match.
  assert.ok(
    src.includes('stored-output update'),
    'expected V3.1 branching caption to mention "stored-output update(s)"',
  );
  assert.ok(
    src.includes('This row contributes'),
    'expected V3.1 branching caption to start "This row contributes ... to alpha"',
  );
  assert.ok(
    src.includes('to alpha'),
    'expected V3.1 caption tail "to alpha"',
  );
  // The branching branch keys off branchingDegree (was: branchCount). Check
  // both that branchingDegree exists and that it gates the multi-update form.
  assert.match(src, /branchingDegree/);
  assert.match(src, /branchingDegree\s*>\s*1/);
});

test('OrbitDetailCard renders V3.1 non-branching caption (1 representative product → 1 stored-output update)', () => {
  const src = read(CARD_PATH);
  // Non-branching branch keeps the singular phrasing verbatim.
  assert.ok(
    src.includes('1 representative product'),
    'expected V3.1 caption to begin "1 representative product → ..."',
  );
  assert.ok(
    src.includes('1 stored-output update.'),
    'expected non-branching V3.1 caption to say "1 stored-output update."',
  );
  assert.ok(
    src.includes('This row contributes 1 to alpha.'),
    'expected non-branching V3.1 caption tail "This row contributes 1 to alpha."',
  );
});

test('OrbitDetailCard renders "THIS Q" badge marker for the selected destination', () => {
  const src = read(CARD_PATH);
  // The THIS Q marker is a quiet inline marker with a stable testid so the
  // matrix can mark the destination matching the hovered cell's repTuple.
  assert.match(src, /data-testid="orbit-detail-this-q-badge"/);
  assert.match(src, />\s*· this Q\s*</);
  assert.doesNotMatch(src, /rounded-full px-1\.5 py-0\.5/);
  // The badge must be selectivity-driven: there is a per-row data-this-q
  // attribute so tests + screen readers can pick out the marked row.
  assert.match(src, /data-this-q=/);
});

test('OrbitDetailCard removes the coefficient-view toggle from the main projection sketch', () => {
  const src = read(CARD_PATH);
  assert.doesNotMatch(src, /data-testid="orbit-detail-coefficient-toggle"/);
  assert.doesNotMatch(src, /aria-label="Toggle coefficient view"/);
});

test('OrbitDetailCard does not expose coefficient view as a disclosure state', () => {
  const src = read(CARD_PATH);
  assert.doesNotMatch(src, /aria-expanded=\{coefficientView\}/);
  assert.doesNotMatch(src, /aria-pressed=\{coefficientView\}/);
});

test('OrbitDetailCard keeps coefficient as an explanatory note, not a second visual mode', () => {
  const src = read(CARD_PATH);
  assert.doesNotMatch(src, /data-coefficient-row="true"/);
  assert.doesNotMatch(src, /coefficientView\s*\?/);
  assert.match(src, /data-testid="orbit-detail-coefficient-note"/);
  assert.match(src, /change the coefficient, not the number of update events/);
});
