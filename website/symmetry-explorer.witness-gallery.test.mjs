/**
 * Tests for WitnessGallery (C25 — V3.1 §25) — source-grep style.
 *
 * Mirrors symmetry-explorer.certification-card.test.mjs (the C21 sibling).
 * These tests parse the JSX source as text and assert structural properties
 * — no DOM, no React renderer required.
 *
 * Coverage (V3.1 §25):
 *   1. Component file exists and exports default React component
 *   2. Renders V3.1 verbatim labels — "candidate", "sigma row move",
 *      "pi label relabeling", "attempted pi", "reason", "result"
 *   3. Accepted card renders 5 fields (incl. "group contribution")
 *   4. Rejected card renders 5 fields (incl. "result: rejected")
 *   5. "Switch to Directed triangle" CTA is rendered when applicable
 *   6. "why rejected?" inline expand button + diagnostic panel
 *   7. Hover handlers wired on both accepted and rejected sigma rows
 *   8. SigmaLoop imports + mounts WitnessGallery as a top-level surface
 *   9. SigmaLoop forwards onSwitchToDirectedTriangle to the gallery
 *  10. App-level mount wires the CTA to handleSelect on the triangle preset
 *  11. No raw notation hex literals
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

const GALLERY_PATH =
  'components/symmetry-aware-einsum-contractions/components/WitnessGallery.jsx';
const SIGMA_LOOP_PATH =
  'components/symmetry-aware-einsum-contractions/components/SigmaLoop.jsx';
const APP_PATH =
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Component file exists and exports default React component
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — component file exists and exports default function', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /export default function WitnessGallery/);
});

test('WitnessGallery — accepts pairs, uLabels, group, onHoverSigma, onHoverPi, onSwitchToDirectedTriangle props', () => {
  const src = read(GALLERY_PATH);
  const start = src.indexOf('export default function WitnessGallery(');
  assert.ok(start > -1, 'default export signature not found');
  const end = src.indexOf(') {', start);
  const sig = src.slice(start, end);
  assert.match(sig, /pairs/);
  assert.match(sig, /uLabels/);
  assert.match(sig, /group/);
  assert.match(sig, /onHoverSigma/);
  assert.match(sig, /onHoverPi/);
  assert.match(sig, /onSwitchToDirectedTriangle/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. V3.1 §25 verbatim labels — accepted + rejected card
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — renders V3.1 "candidate" label', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /fieldLabel="candidate"/);
});

test('WitnessGallery — renders V3.1 "sigma row move" label', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /fieldLabel="sigma row move"/);
});

test('WitnessGallery — renders V3.1 "pi label relabeling" label on the accepted card', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /fieldLabel="pi label relabeling"/);
});

test('WitnessGallery — renders V3.1 "attempted pi" label on the rejected card', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /fieldLabel="attempted pi"/);
  // Value mirrors the spec ("none compatible") to make the rejected verdict
  // legible without expanding the diagnostic.
  assert.match(src, /none compatible/);
});

test('WitnessGallery — renders V3.1 "reason" label on the rejected card', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /fieldLabel="reason"/);
  // The spec phrasing for the headline reason is "incidence fingerprint mismatch".
  assert.match(src, /incidence fingerprint mismatch/);
});

test('WitnessGallery — renders V3.1 "result" label on both cards', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /fieldLabel="result"/);
  // Accepted-side value
  assert.match(src, /Result: accepted/);
  // Rejected-side value
  assert.match(src, /Result: rejected/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Accepted card — 5 fields (V3.1 §25)
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — accepted card has 5 testid-tagged fields', () => {
  const src = read(GALLERY_PATH);
  for (const id of [
    'witness-gallery-accepted-candidate',
    'witness-gallery-accepted-sigma',
    'witness-gallery-accepted-pi',
    'witness-gallery-accepted-result',
    'witness-gallery-accepted-contribution',
  ]) {
    assert.ok(src.includes(`testId="${id}"`), `missing testId="${id}"`);
  }
});

test('WitnessGallery — accepted card surfaces "group contribution" field with "generator of"', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /fieldLabel="group contribution"/);
  assert.match(src, /generator of /);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Rejected card — 5 fields (V3.1 §25)
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — rejected card has 5 testid-tagged fields', () => {
  const src = read(GALLERY_PATH);
  for (const id of [
    'witness-gallery-rejected-candidate',
    'witness-gallery-rejected-sigma',
    'witness-gallery-rejected-attempted-pi',
    'witness-gallery-rejected-reason',
    'witness-gallery-rejected-result',
  ]) {
    assert.ok(src.includes(`testId="${id}"`), `missing testId="${id}"`);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. "Switch to Directed triangle" CTA is rendered when applicable
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — renders "Switch to Directed triangle" CTA', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /Switch to Directed triangle/);
  // The CTA is a real <button> with a stable testid for the dual mount points
  // (empty state and one-sided state).
  assert.match(src, /data-testid="witness-gallery-switch-preset-btn"/);
});

test('WitnessGallery — CTA invokes onSwitchToDirectedTriangle prop on click', () => {
  const src = read(GALLERY_PATH);
  // The onClick handler delegates to onSwitchToDirectedTriangle when provided.
  assert.match(src, /onSwitchToDirectedTriangle\(\)/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. "why rejected?" expand button + diagnostic panel
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — renders "why rejected?" expand button on the rejected card', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /why rejected\?/);
  assert.match(src, /data-testid="witness-gallery-why-rejected-btn"/);
  // The button is a real <button type="button"> with aria-expanded driven by state
  assert.match(src, /aria-expanded=\{showWhy\}/);
});

test('WitnessGallery — diagnostic panel toggles via showWhy state', () => {
  const src = read(GALLERY_PATH);
  // The state hook is present.
  assert.match(src, /const \[showWhy, setShowWhy\] = useState\(false\)/);
  // The panel mounts conditionally on showWhy.
  assert.match(src, /\{showWhy && \(/);
  // The panel exposes a stable testid and a role for screen readers.
  assert.match(src, /data-testid="witness-gallery-rejected-diagnostic"/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Hover handlers wired on both cards
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — accepted sigma row wires onMouseEnter to fireSigma(sigmaRowSet)', () => {
  const src = read(GALLERY_PATH);
  const idx = src.indexOf('witness-gallery-accepted-sigma');
  assert.ok(idx > -1, 'accepted sigma row not found');
  const blockStart = src.lastIndexOf('<FieldRow', idx);
  const blockEnd = src.indexOf('/>', idx);
  assert.ok(blockStart > -1 && blockEnd > blockStart, 'accepted σ FieldRow block not isolated');
  const block = src.slice(blockStart, blockEnd);
  assert.match(block, /onMouseEnter=\{\(\)\s*=>\s*fireSigma\(sigmaRowSet\)\}/);
  assert.match(block, /onMouseLeave=\{\(\)\s*=>\s*fireSigma\(null\)\}/);
});

test('WitnessGallery — accepted pi row wires onMouseEnter to firePi(piLabelSet)', () => {
  const src = read(GALLERY_PATH);
  const idx = src.indexOf('witness-gallery-accepted-pi');
  assert.ok(idx > -1, 'accepted pi row not found');
  const blockStart = src.lastIndexOf('<FieldRow', idx);
  const blockEnd = src.indexOf('/>', idx);
  assert.ok(blockStart > -1 && blockEnd > blockStart, 'accepted π FieldRow block not isolated');
  const block = src.slice(blockStart, blockEnd);
  assert.match(block, /onMouseEnter=\{\(\)\s*=>\s*firePi\(piLabelSet\)\}/);
  assert.match(block, /onMouseLeave=\{\(\)\s*=>\s*firePi\(null\)\}/);
});

test('WitnessGallery — rejected sigma row wires onMouseEnter to fireSigma(sigmaRowSet)', () => {
  const src = read(GALLERY_PATH);
  const idx = src.indexOf('witness-gallery-rejected-sigma');
  assert.ok(idx > -1, 'rejected sigma row not found');
  const blockStart = src.lastIndexOf('<FieldRow', idx);
  const blockEnd = src.indexOf('/>', idx);
  assert.ok(blockStart > -1 && blockEnd > blockStart, 'rejected σ FieldRow block not isolated');
  const block = src.slice(blockStart, blockEnd);
  assert.match(block, /onMouseEnter=\{\(\)\s*=>\s*fireSigma\(sigmaRowSet\)\}/);
  assert.match(block, /onMouseLeave=\{\(\)\s*=>\s*fireSigma\(null\)\}/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. SigmaLoop mounts WitnessGallery
// ─────────────────────────────────────────────────────────────────────────────
test('SigmaLoop — imports WitnessGallery from sibling file', () => {
  const src = read(SIGMA_LOOP_PATH);
  assert.match(
    src,
    /import\s+WitnessGallery\s+from\s+['"]\.\/WitnessGallery\.jsx['"]/,
  );
});

test('SigmaLoop — mounts <WitnessGallery as a top-level surface', () => {
  const src = read(SIGMA_LOOP_PATH);
  assert.match(src, /<WitnessGallery/);
  // Mount should appear above the existing summary stats so the gallery
  // acts as an overview header. We check that the gallery JSX appears
  // before the "sigma-summary" block.
  const galleryIdx = src.indexOf('<WitnessGallery');
  const summaryIdx = src.indexOf('sigma-summary');
  assert.ok(galleryIdx > -1, '<WitnessGallery> mount not found');
  assert.ok(summaryIdx > -1, 'sigma-summary block not found');
  assert.ok(galleryIdx < summaryIdx, '<WitnessGallery> must be rendered before the summary stats');
});

test('SigmaLoop — passes pairs, uLabels, group, hover handlers, and switch CTA to WitnessGallery', () => {
  const src = read(SIGMA_LOOP_PATH);
  const mountIdx = src.indexOf('<WitnessGallery');
  assert.ok(mountIdx > -1, '<WitnessGallery mount not found');
  const closeIdx = src.indexOf('/>', mountIdx);
  assert.ok(closeIdx > mountIdx, '<WitnessGallery mount has no self-closing />');
  const block = src.slice(mountIdx, closeIdx + 2);
  assert.match(block, /pairs=\{allPairs\}/);
  assert.match(block, /uLabels=\{uLabels\}/);
  assert.match(block, /group=\{group\}/);
  assert.match(block, /onHoverSigma=\{setCertHoverRows\}/);
  assert.match(block, /onHoverPi=\{setCertHoverLabels\}/);
  assert.match(block, /onSwitchToDirectedTriangle=\{onSwitchToDirectedTriangle\}/);
});

test('SigmaLoop — accepts onSwitchToDirectedTriangle prop on the public component', () => {
  const src = read(SIGMA_LOOP_PATH);
  // Public default export accepts the prop.
  const exportLine = src.match(/export default function SigmaLoop\(\{[^}]+\}\)/);
  assert.ok(exportLine, 'SigmaLoop default export signature not found');
  assert.match(exportLine[0], /onSwitchToDirectedTriangle/);
  // Inner function also threads the prop.
  const innerLine = src.match(/function SigmaLoopInner\(\{[^}]+\}\)/);
  assert.ok(innerLine, 'SigmaLoopInner signature not found');
  assert.match(innerLine[0], /onSwitchToDirectedTriangle/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 9. App-level wiring — handleSelect on the triangle preset
// ─────────────────────────────────────────────────────────────────────────────
test('SymmetryAwareEinsumContractionsApp — wires SigmaLoop onSwitchToDirectedTriangle to the triangle preset', () => {
  const src = read(APP_PATH);
  // The handler appears next to the SigmaLoop mount and resolves the preset
  // by id ('triangle'), then calls handleSelect.
  assert.match(src, /onSwitchToDirectedTriangle=\{/);
  assert.match(src, /EXAMPLES\.findIndex\(\(ex\)\s*=>\s*ex\.id === 'triangle'\)/);
  assert.match(src, /handleSelect\(idx\)/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 10. Accessibility
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — section uses role="region" + aria-label', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /role="region"/);
  assert.match(src, /aria-label="Witness gallery — accepted vs rejected/);
});

test('WitnessGallery — every field row is keyboard-focusable (tabIndex=0)', () => {
  const src = read(GALLERY_PATH);
  assert.match(src, /tabIndex=\{0\}/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 11. No raw notation hex literals
// ─────────────────────────────────────────────────────────────────────────────
test('WitnessGallery — no raw notation hex literals', () => {
  const src = read(GALLERY_PATH);
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
      `Raw notation hex ${hex} must not appear in WitnessGallery source — use a CSS variable instead`,
    );
  }
});
