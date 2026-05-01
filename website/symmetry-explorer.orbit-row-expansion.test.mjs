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
  // The THIS Q marker is rendered as a coral pill badge with a stable testid
  // so the matrix can mark the row whose destination matches the user-clicked
  // cell's repTuple.
  assert.match(src, /data-testid="orbit-detail-this-q-badge"/);
  // The visible label must be exactly "THIS Q" (V3.1 §13 wording).
  assert.match(src, />\s*THIS Q\s*</);
  // The badge must be selectivity-driven: there is a per-row data-this-q
  // attribute so tests + screen readers can pick out the marked row.
  assert.match(src, /data-this-q=/);
});

test('OrbitDetailCard exposes a coefficient-view toggle button with aria-label', () => {
  const src = read(CARD_PATH);
  assert.match(src, /data-testid="orbit-detail-coefficient-toggle"/);
  // The toggle must declare an aria-label so screen readers describe its purpose.
  assert.match(src, /aria-label="Toggle coefficient view"/);
  // It must be a real <button type="button"> for keyboard focus + correct semantics.
  assert.match(src, /<button[\s\S]{0,200}data-testid="orbit-detail-coefficient-toggle"[\s\S]{0,400}type="button"|<button[\s\S]{0,200}type="button"[\s\S]{0,400}data-testid="orbit-detail-coefficient-toggle"/);
});

test('OrbitDetailCard coefficient-view toggle declares aria-expanded so its open/closed state is exposed to AT', () => {
  const src = read(CARD_PATH);
  // The toggle is a disclosure-style control: aria-expanded mirrors the
  // coefficientView state. (We also add aria-pressed for toggle-button AT,
  // but spec specifically wants aria-expanded.)
  assert.match(src, /aria-expanded=\{coefficientView\}/);
});

test('OrbitDetailCard coefficient view rows are tagged with data-coefficient-row so the renderer flips presentation when toggle is on', () => {
  const src = read(CARD_PATH);
  // When the toggle is on, the rendered rows are (destination Q, coefficient)
  // pairs. They are tagged with data-coefficient-row so the visual mode is
  // distinguishable from the default per-member rows.
  assert.match(src, /data-coefficient-row="true"/);
  // The view also surfaces the literal "coefficient(O, Q)" label so the
  // user knows what the right-hand value represents.
  assert.match(src, /coefficient\(O, ?Q\)/);
  // The toggle gates rendering — the source must branch on coefficientView
  // (from useState) at the projection-sketch site.
  assert.match(src, /useState/);
  assert.match(src, /coefficientView\s*\?/);
});
