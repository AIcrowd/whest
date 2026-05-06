// V3.1 §C22 — Incidence Fingerprint Visual (L5.T2.7).
//
// Source-grep tests pinning the polish added on top of the existing 3-stage
// `M → σ(M) → π(σ(M))` σ-loop animation:
//
//   - Failed (rejected) third stage turns red and shows a "✗ no compatible π"
//     register, so the user can tell rejected from accepted at a glance.
//   - Domain-mismatch labels (columns of σ(M) whose fingerprint has no partner
//     in M) are marked with a red border + × overlay in both the matrix grid
//     and the fingerprint pill row.
//   - "Show fingerprints" toggle collapses the matrix grid into its
//     column-signature summary; it is keyboard-focusable and exposes
//     aria-label / aria-expanded.
//   - Row and column labels are hover-help surfaces with definition tooltips
//     (cursor-help + tabIndex=0 + title).

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const SIGMA_LOOP = 'components/symmetry-aware-einsum-contractions/components/SigmaLoop.jsx';
const INCIDENCE_MATRIX = 'components/symmetry-aware-einsum-contractions/components/IncidenceMatrix.jsx';
const STYLES = 'components/symmetry-aware-einsum-contractions/styles.css';

// ─── Failed-path turn-red branch ───────────────────────────────────────────

test('C22 — SigmaLoop renders failed-path turn-red branch on the third stage', () => {
  const sigmaLoop = read(SIGMA_LOOP);
  const incidence = read(INCIDENCE_MATRIX);
  // SigmaLoop must forward `rejected` to IncidenceMatrix only when the
  // selected pair is invalid AND we are at the verdict stage (stage 2).
  assert.match(sigmaLoop, /rejected=\{stage === 2 && !selected\.isValid\}/);
  // IncidenceMatrix accepts the `rejected` prop and exposes a
  // data-rejected="true" attribute + an inc-matrix-rejected class so styles
  // and tests can latch on without reading internal state.
  assert.match(incidence, /^\s*rejected,/m);
  assert.match(incidence, /data-rejected=\{rejected \? 'true' : undefined\}/);
  assert.match(incidence, /inc-matrix-rejected/);
  // Styles for the turn-red verdict.
  const styles = read(STYLES);
  assert.match(styles, /\.inc-matrix-rejected \.inc-matrix \{[^}]*var\(--coral\)/s);
});

// ─── Domain-mismatch label marking ─────────────────────────────────────────

test('C22 — SigmaLoop marks domain-mismatch labels with red border / × overlay', () => {
  const sigmaLoop = read(SIGMA_LOOP);
  const incidence = read(INCIDENCE_MATRIX);
  const styles = read(STYLES);
  // SigmaLoop computes the mismatched-label set via a dedicated helper and
  // forwards it to IncidenceMatrix on rejected pairs starting at stage 1.
  assert.match(sigmaLoop, /computeDomainMismatchedLabels/);
  assert.match(sigmaLoop, /mismatchedLabels=\{[^}]*computeDomainMismatchedLabels\(/);
  // IncidenceMatrix accepts `mismatchedLabels` and applies an
  // inc-col-mismatch class + × overlay span.
  assert.match(incidence, /mismatchedLabels/);
  assert.match(incidence, /inc-col-mismatch/);
  assert.match(incidence, /inc-col-mismatch-mark/);
  // The fingerprint pill row also marks mismatched columns.
  assert.match(incidence, /inc-fp-mismatch-mark/);
  // Red border styling for the mismatched header.
  assert.match(styles, /\.inc-col-mismatch \{[^}]*var\(--coral\)/s);
});

// ─── Show fingerprints toggle — REMOVED (user feedback) ────────────────────
//
// V3.1 §C22 originally listed a "show fingerprints" toggle that hid the
// matrix grid in favour of the per-column signature strip. In practice
// the column-fingerprint strip already renders unconditionally beneath
// the matrix (IncidenceMatrix.jsx ~line 211), so the toggle's only effect
// was to hide the grid above. Multiple readers found the affordance
// jargon-heavy and unclear ("what is a fingerprint? why would I hide the
// matrix?"). The toggle has been removed; explicit doesNotMatch guards
// keep it from regressing.

test('C22 — Show fingerprints toggle is removed (clutter-reduction)', () => {
  const sigmaLoop = read(SIGMA_LOOP);
  assert.doesNotMatch(sigmaLoop, /Show fingerprints/,
    'The "Show fingerprints" toggle was removed; column fingerprints are always visible below the matrix.');
  assert.doesNotMatch(sigmaLoop, /showFingerprintsOnly/,
    'showFingerprintsOnly state should be removed along with the toggle.');
  assert.doesNotMatch(sigmaLoop, /ctrl-fingerprints-toggle/,
    'The toggle button class should be removed.');
});

test('C22 — IncidenceMatrix still renders the column-fingerprint strip unconditionally', () => {
  const incidence = read(INCIDENCE_MATRIX);
  // The strip lives below the matrix grid and is always rendered — that's
  // why the toggle could be removed without losing teaching content.
  assert.match(incidence, /Column Fingerprints/i,
    'IncidenceMatrix must always render the column-fingerprint strip header.');
  assert.match(incidence, /inc-fingerprints/,
    'IncidenceMatrix must render the inc-fingerprints container unconditionally.');
});

// ─── Failed-path "no compatible π" indicator ───────────────────────────────

test('C22 — Failed path surfaces a "no compatible π" / ✗ indicator', () => {
  const sigmaLoop = read(SIGMA_LOOP);
  const incidence = read(INCIDENCE_MATRIX);
  // SigmaLoop renders an explicit ✗ verdict badge below the matrix when the
  // pair is rejected at stage 2. The data-rejected-verdict attribute lets
  // tests find it without depending on Latex rendering.
  assert.match(sigmaLoop, /data-rejected-verdict="true"/);
  assert.match(sigmaLoop, /✗/);
  // IncidenceMatrix surfaces an inline reject badge with the literal phrase
  // "no compatible π" inside the matrix label area.
  assert.match(incidence, /inc-matrix-reject-badge/);
  assert.match(incidence, /no compatible π/);
});

// ─── Row / column hover-help tooltips ──────────────────────────────────────

test('C22 — Row and column labels are hover-help surfaces with definitions', () => {
  const incidence = read(INCIDENCE_MATRIX);
  // Both row and column labels are cursor-help surfaces with title= attrs
  // (the platform-native definition tooltip). They are keyboard-focusable
  // (tabIndex=0) so the same affordance reaches users without a pointer.
  assert.match(incidence, /className="inc-row-label cursor-help"/);
  assert.match(incidence, /title=\{rowTitle\}/);
  // Column header builds its className list including 'cursor-help'.
  assert.match(incidence, /'cursor-help'/);
  // Both label kinds receive tabIndex=0.
  const rowLabelTabIndex = incidence.match(/inc-row-label cursor-help[\s\S]{0,200}tabIndex=\{0\}/);
  assert.ok(rowLabelTabIndex, 'row label is focusable (tabIndex=0)');
  const colHeaderTabIndex = incidence.match(/headerClass[\s\S]{0,400}tabIndex=\{0\}/);
  assert.ok(colHeaderTabIndex, 'column header is focusable (tabIndex=0)');
});
