/**
 * Tests for CertificationCard (C21) — source-grep style.
 *
 * Mirrors symmetry-explorer.unavailable-state.test.mjs and other Act-3
 * source-grep suites. These tests parse the JSX source as text and assert
 * structural properties — no DOM, no React renderer required.
 *
 * Coverage (V3.1 §21):
 *   1. Component file exists and exports default
 *   2. Renders all five V3.1 verbatim labels (σ row move, π relabeling,
 *      domain check, incidence recovery, result)
 *   3. Renders the "Show in matrix" CTA button
 *   4. Renders the Appendix A link with href="#appendix-section-1"
 *   5. Has hover handlers wired (onMouseEnter on σ row + onMouseEnter on π row)
 *   6. SigmaLoop imports + mounts CertificationCard for accepted pairs
 *   7. SigmaLoop wires onHoverSigma / onHoverPi into a hover bus that
 *      bridges into IncidenceMatrix highlight props
 *   8. SigmaLoop attaches a ref so "Show in matrix" can scroll-into-view
 *   9. Reject case is handled (kind === 'rejected' returns null)
 *  10. No raw notation hex literals in the card source
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const CARD_PATH =
  'components/symmetry-aware-einsum-contractions/components/CertificationCard.jsx';
const SIGMA_LOOP_PATH =
  'components/symmetry-aware-einsum-contractions/components/SigmaLoop.jsx';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Component file exists and exports default React component
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — component file exists and exports default function', () => {
  const src = read(CARD_PATH);
  assert.match(src, /export default function CertificationCard/);
});

test('CertificationCard — accepts pair, uLabels, onHoverSigma, onHoverPi, onScrollToMatrix props', () => {
  const src = read(CARD_PATH);
  const start = src.indexOf('export default function CertificationCard(');
  assert.ok(start > -1, 'default export signature not found');
  const end = src.indexOf(') {', start);
  const sig = src.slice(start, end);
  assert.match(sig, /pair/);
  assert.match(sig, /uLabels/);
  assert.match(sig, /onHoverSigma/);
  assert.match(sig, /onHoverPi/);
  assert.match(sig, /onScrollToMatrix/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. V3.1 verbatim field labels — all five must appear
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — renders V3.1 "candidate row move" label', () => {
  const src = read(CARD_PATH);
  assert.match(src, /candidate row move/);
});

test('CertificationCard — renders V3.1 "matching label relabeling" label', () => {
  const src = read(CARD_PATH);
  assert.match(src, /matching label relabeling/);
});

test('CertificationCard — renders V3.1 "domain check" label + "passed" value', () => {
  const src = read(CARD_PATH);
  assert.match(src, /domain check/);
  assert.match(src, /passed/);
});

test('CertificationCard — renders V3.1 "incidence recovery" label + π(σ(M)) = M value', () => {
  const src = read(CARD_PATH);
  assert.match(src, /incidence recovery/);
  assert.match(src, /π\(σ\(M\)\) = M/);
});

test('CertificationCard — renders V3.1 "result" label + "accepted into G_pt" value', () => {
  const src = read(CARD_PATH);
  assert.match(src, /result/);
  assert.match(src, /accepted into G_pt/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. "Show in matrix" CTA button
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — renders "Show in matrix" button wired to onScrollToMatrix', () => {
  const src = read(CARD_PATH);
  // Button text appears verbatim.
  assert.match(src, /Show in matrix/);
  // onClick handler delegates to onScrollToMatrix.
  assert.match(src, /onClick=\{\(\)\s*=>\s*\{[\s\S]*?onScrollToMatrix[\s\S]*?\}\}/);
  // The element is a <button> (attributes may span multiple lines).
  assert.match(src, /<button[\s\S]*?data-testid="cert-show-in-matrix-btn"/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Appendix A link — href="#appendix-section-1"
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — Appendix A link points to #appendix-section-1', () => {
  const src = read(CARD_PATH);
  assert.match(src, /href="#appendix-section-1"/);
  assert.match(src, /Read Appendix A/);
  // Guard against drift to other appendix anchors.
  assert.doesNotMatch(src, /href="#appendix-section-7"/);
  assert.doesNotMatch(src, /href="#appendix-section-2"/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Hover handlers wired — onMouseEnter on σ row + onMouseEnter on π row
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — σ field row has onMouseEnter wired to onHoverSigma', () => {
  const src = read(CARD_PATH);
  // Locate the σ field row by its testId, then verify the row above it
  // wires onMouseEnter to fireHoverSigma with the moved-row Set.
  const sigmaIdx = src.indexOf('cert-field-sigma');
  assert.ok(sigmaIdx > -1, 'cert-field-sigma row not found');
  // Look backward and forward to capture the FieldRow block for σ.
  const blockStart = src.lastIndexOf('<FieldRow', sigmaIdx);
  const blockEnd = src.indexOf('/>', sigmaIdx);
  assert.ok(blockStart > -1 && blockEnd > blockStart, 'σ FieldRow block not isolated');
  const sigmaBlock = src.slice(blockStart, blockEnd);
  assert.match(sigmaBlock, /onMouseEnter=\{\(\)\s*=>\s*fireHoverSigma\(sigmaRowSet\)\}/);
  assert.match(sigmaBlock, /onMouseLeave=\{\(\)\s*=>\s*fireHoverSigma\(null\)\}/);
});

test('CertificationCard — π field row has onMouseEnter wired to onHoverPi', () => {
  const src = read(CARD_PATH);
  const piIdx = src.indexOf('cert-field-pi');
  assert.ok(piIdx > -1, 'cert-field-pi row not found');
  const blockStart = src.lastIndexOf('<FieldRow', piIdx);
  const blockEnd = src.indexOf('/>', piIdx);
  assert.ok(blockStart > -1 && blockEnd > blockStart, 'π FieldRow block not isolated');
  const piBlock = src.slice(blockStart, blockEnd);
  assert.match(piBlock, /onMouseEnter=\{\(\)\s*=>\s*fireHoverPi\(piLabelSet\)\}/);
  assert.match(piBlock, /onMouseLeave=\{\(\)\s*=>\s*fireHoverPi\(null\)\}/);
});

test('CertificationCard — fireHoverSigma + fireHoverPi delegate to onHover{Sigma,Pi} props', () => {
  const src = read(CARD_PATH);
  assert.match(src, /onHoverSigma\(set\)/);
  assert.match(src, /onHoverPi\(set\)/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. SigmaLoop mounts CertificationCard for accepted pairs
// ─────────────────────────────────────────────────────────────────────────────
test('SigmaLoop — imports CertificationCard from sibling file', () => {
  const src = read(SIGMA_LOOP_PATH);
  assert.match(
    src,
    /import\s+CertificationCard\s+from\s+['"]\.\/CertificationCard\.jsx['"]/,
  );
});

test('SigmaLoop — mounts <CertificationCard for accepted pairs only', () => {
  const src = read(SIGMA_LOOP_PATH);
  // Mount appears in the JSX.
  assert.match(src, /<CertificationCard/);
  // Mount is gated by selected.isValid + selected.pi.
  const mountIdx = src.indexOf('<CertificationCard');
  assert.ok(mountIdx > -1, '<CertificationCard mount not found in SigmaLoop');
  // Look back ~600 chars for the gating condition.
  const before = src.slice(Math.max(0, mountIdx - 800), mountIdx);
  assert.match(before, /selected\.isValid && selected\.pi/);
});

test('SigmaLoop — passes pair, uLabels, onHoverSigma, onHoverPi, onScrollToMatrix to CertificationCard', () => {
  const src = read(SIGMA_LOOP_PATH);
  const mountIdx = src.indexOf('<CertificationCard');
  assert.ok(mountIdx > -1, '<CertificationCard mount not found');
  // Capture the mount block to its closing `/>`
  const closeIdx = src.indexOf('/>', mountIdx);
  assert.ok(closeIdx > mountIdx, '<CertificationCard mount has no self-closing />');
  const mountBlock = src.slice(mountIdx, closeIdx + 2);
  assert.match(mountBlock, /pair=\{selected\}/);
  assert.match(mountBlock, /uLabels=\{uLabels\}/);
  assert.match(mountBlock, /onHoverSigma=\{setCertHoverRows\}/);
  assert.match(mountBlock, /onHoverPi=\{setCertHoverLabels\}/);
  assert.match(mountBlock, /onScrollToMatrix=\{/);
});

test('SigmaLoop — wires hover bus state (certHoverRows / certHoverLabels) into IncidenceMatrix', () => {
  const src = read(SIGMA_LOOP_PATH);
  // useState declarations exist
  assert.match(src, /const \[certHoverRows, setCertHoverRows\] = useState\(null\)/);
  assert.match(src, /const \[certHoverLabels, setCertHoverLabels\] = useState\(null\)/);
  // The hover state is spliced into IncidenceMatrix's movedRows / movedCols
  // on stage 0 (original M view).
  assert.match(src, /stage === 0 && certHoverRows/);
  assert.match(src, /stage === 0 && certHoverLabels/);
});

test('SigmaLoop — matrix container ref enables scroll-into-view from the card', () => {
  const src = read(SIGMA_LOOP_PATH);
  assert.match(src, /matrixContainerRef\s*=\s*useRef\(null\)/);
  assert.match(src, /ref=\{matrixContainerRef\}/);
  assert.match(src, /matrixContainerRef\.current\.scrollIntoView/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Reject case — kind === 'rejected' returns null
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — reject case (kind === "rejected") returns null', () => {
  const src = read(CARD_PATH);
  assert.match(src, /pair\.kind === 'rejected'/);
  // The early-return null path is present.
  assert.match(src, /if \(pair\.kind === 'rejected'\) return null/);
});

test('CertificationCard — null pair returns null (defensive)', () => {
  const src = read(CARD_PATH);
  assert.match(src, /if \(!pair\) return null/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. Accessibility
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — section uses role="region" + aria-label', () => {
  const src = read(CARD_PATH);
  assert.match(src, /role="region"/);
  assert.match(src, /aria-label="Certification card — accepted \(σ, π\) witness"/);
});

test('CertificationCard — every field row is keyboard-focusable (tabIndex=0) with aria-label', () => {
  const src = read(CARD_PATH);
  // FieldRow component sets tabIndex on its root element.
  assert.match(src, /tabIndex=\{0\}/);
  // The aria-label prop is threaded into FieldRow.
  assert.match(src, /aria-label=\{ariaLabel\}/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 9. No raw notation hex literals
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationCard — no raw notation hex literals', () => {
  const src = read(CARD_PATH);
  const FORBIDDEN_NOTATION_HEXES = [
    '#F0524D', // coral
    '#64748B', // ein-w (summed)
    '#4A7CFF', // ein-v / info
    '#FA9E33', // warning
    '#23B761', // success
    '#292C2D', // gray-900
    '#5D5F60', // gray-600
  ];
  for (const hex of FORBIDDEN_NOTATION_HEXES) {
    assert.equal(
      src.includes(hex),
      false,
      `Raw notation hex ${hex} must not appear in CertificationCard source — use a CSS variable instead`,
    );
  }
});
