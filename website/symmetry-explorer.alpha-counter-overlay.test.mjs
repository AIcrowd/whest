// V3.1 §15 — α-counter overlay (L5.T2.12 / C15).
// Source-grep tests for the M/α counter strip rendered near the O→Q matrix
// inside BranchingDemo. The strip mirrors V3.1's verbatim labels in Total
// mode, swaps to "rows counted so far" / "filled cells counted so far" in
// Cumulative mode, and surfaces a per-row contribution hint on hover. The
// toggle is keyboard-focusable and exposes pressed-state via aria-pressed.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const BRANCHING_DEMO_PATH =
  'components/symmetry-aware-einsum-contractions/components/BranchingDemo.jsx';

test('BranchingDemo renders verbatim "M = number of rows" V3.1 label', () => {
  const src = read(BRANCHING_DEMO_PATH);
  assert.match(src, /M = number of rows = /);
});

test('BranchingDemo renders verbatim "alpha = number of filled cells" V3.1 label', () => {
  const src = read(BRANCHING_DEMO_PATH);
  assert.match(src, /alpha = number of filled cells = /);
});

test('BranchingDemo renders the row-hover hint phrase "selected row contributes"', () => {
  const src = read(BRANCHING_DEMO_PATH);
  assert.match(src, /selected row contributes/);
  // The hint references the per-row count flowing into alpha.
  assert.match(src, /to alpha/);
});

test('BranchingDemo cumulative-mode toggle exists with descriptive aria-label', () => {
  const src = read(BRANCHING_DEMO_PATH);
  assert.match(src, /data-testid="branching-alpha-counter-toggle"/);
  assert.match(src, /aria-label="Toggle cumulative versus total counter mode"/);
});

test('BranchingDemo cumulative-mode toggle exposes pressed state via aria-pressed', () => {
  const src = read(BRANCHING_DEMO_PATH);
  // aria-pressed mirrors the cumulativeMode boolean — toggle button pattern.
  assert.match(src, /aria-pressed=\{cumulativeMode\}/);
});

test('BranchingDemo renders verbatim cumulative-mode labels for rows and filled cells', () => {
  const src = read(BRANCHING_DEMO_PATH);
  assert.match(src, /rows counted so far/);
  assert.match(src, /filled cells counted so far/);
});

test('BranchingDemo strip is reachable via data-testid="branching-alpha-counter-strip"', () => {
  const src = read(BRANCHING_DEMO_PATH);
  assert.match(src, /data-testid="branching-alpha-counter-strip"/);
});

test('BranchingDemo strip wires the cumulativeMode useState toggle', () => {
  const src = read(BRANCHING_DEMO_PATH);
  // The toggle must flip a real React state — not a static label.
  assert.match(src, /\[cumulativeMode\s*,\s*setCumulativeMode\]\s*=\s*useState\(false\)/);
  assert.match(src, /setCumulativeMode\(\(v\)\s*=>\s*!v\)/);
});

test('BranchingDemo derives the running prefix totals from the hovered row', () => {
  const src = read(BRANCHING_DEMO_PATH);
  // Cumulative counters use prefix sums so dense matrices do not re-scan rows
  // on every hover transition.
  assert.match(src, /prefixAlpha/);
  assert.match(src, /prefixAlpha\[cumulativeRows\]/);
  assert.match(src, /hoveredRow\s*>=\s*0\s*\?\s*hoveredRow\s*\+\s*1/);
});

test('BranchingDemo strip is keyboard-focusable via a real <button>', () => {
  const src = read(BRANCHING_DEMO_PATH);
  // The toggle is a <button type="button"> so it's reachable by Tab and
  // activatable via Space/Enter without explicit tabIndex wiring.
  assert.match(
    src,
    /<button[\s\S]{0,300}data-testid="branching-alpha-counter-toggle"[\s\S]{0,300}type="button"|<button[\s\S]{0,300}type="button"[\s\S]{0,300}data-testid="branching-alpha-counter-toggle"/,
  );
});
