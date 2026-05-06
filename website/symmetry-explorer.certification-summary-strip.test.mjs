/**
 * Tests for CertificationSummaryStrip (C27) — source-grep style.
 *
 * Mirrors symmetry-explorer.certification-card.test.mjs and other source-grep
 * suites. These tests parse the JSX source as text and assert structural
 * properties — no DOM, no React renderer required.
 *
 * Coverage (V3.1 §27):
 *   1. Component file exists and exports default React component
 *   2. Renders all 7 V3.1 verbatim labels (candidate sigma moves,
 *      accepted witnesses, identity-only, rejected, |G_pt|, H, components)
 *   3. Each pill has tabIndex=0 and role="button"
 *   4. Components pill has hover handler writing to component bus
 *      (setActiveComponentId)
 *   5. App imports + mounts CertificationSummaryStrip
 *   6. App passes the seven live props (candidateMoves, accepted, identityOnly,
 *      rejected, gPtSize, hSize, componentsCount, setActiveComponentId)
 *   7. No raw notation hex literals in the strip source — TOKEN map only
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

const STRIP_PATH =
  'components/symmetry-aware-einsum-contractions/components/CertificationSummaryStrip.jsx';
const APP_PATH =
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Component file exists and exports default React component
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationSummaryStrip — component file exists and exports default function', () => {
  const src = read(STRIP_PATH);
  assert.match(src, /export default function CertificationSummaryStrip/);
});

test('CertificationSummaryStrip — accepts 7 metric props plus setActiveComponentId', () => {
  const src = read(STRIP_PATH);
  const start = src.indexOf('export default function CertificationSummaryStrip(');
  assert.ok(start > -1, 'default export signature not found');
  // Capture up to the destructuring close + the next ') {' for the body open.
  const end = src.indexOf(') {', start);
  const sig = src.slice(start, end);
  assert.match(sig, /candidateMoves/);
  assert.match(sig, /accepted/);
  assert.match(sig, /identityOnly/);
  assert.match(sig, /rejected/);
  assert.match(sig, /gPtSize/);
  assert.match(sig, /hSize/);
  assert.match(sig, /componentsCount/);
  assert.match(sig, /setActiveComponentId/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. V3.1 verbatim labels — all seven must appear literally in source
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationSummaryStrip — renders V3.1 "candidate sigma moves" label', () => {
  const src = read(STRIP_PATH);
  assert.match(src, /candidate sigma moves/);
});

test('CertificationSummaryStrip — renders V3.1 "accepted witnesses" label', () => {
  const src = read(STRIP_PATH);
  assert.match(src, /accepted witnesses/);
});

test('CertificationSummaryStrip — renders V3.1 "identity-only" label', () => {
  const src = read(STRIP_PATH);
  assert.match(src, /identity-only/);
});

test('CertificationSummaryStrip — renders V3.1 "rejected" label', () => {
  const src = read(STRIP_PATH);
  // The rejected label appears as a verbatim PILL_LABELS entry — guard
  // against drift to "rejected pairs" or "rejected witnesses".
  assert.match(src, /rejected:\s*'rejected'/);
});

test('CertificationSummaryStrip — renders V3.1 "|G_pt|" label', () => {
  const src = read(STRIP_PATH);
  assert.match(src, /\|G_pt\|/);
});

test('CertificationSummaryStrip — renders V3.1 "H" label (column-action stabilizer)', () => {
  const src = read(STRIP_PATH);
  // H is short enough that we anchor on the PILL_LABELS map entry.
  assert.match(src, /hSize:\s*'H'/);
});

test('CertificationSummaryStrip — renders V3.1 "components" label', () => {
  const src = read(STRIP_PATH);
  assert.match(src, /components:\s*'components'/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Each pill has tabIndex=0 and role="button"
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationSummaryStrip — MetricPill renders tabIndex={0} and role="button"', () => {
  const src = read(STRIP_PATH);
  // The MetricPill body declares tabIndex and role; both must be present.
  assert.match(src, /tabIndex=\{0\}/);
  assert.match(src, /role="button"/);
});

test('CertificationSummaryStrip — MetricPill has aria-label prop threaded into element', () => {
  const src = read(STRIP_PATH);
  // The shared MetricPill component should set aria-label from props.
  assert.match(src, /aria-label=\{ariaLabel\}/);
});

test('CertificationSummaryStrip — every pill mount supplies an ariaLabel prop', () => {
  const src = read(STRIP_PATH);
  // Count <MetricPill mounts vs ariaLabel= props inside them. Each pill
  // must wire aria-label so screen readers can describe the live value.
  const pillMounts = (src.match(/<MetricPill\b/g) || []).length;
  const ariaProps  = (src.match(/ariaLabel=/g) || []).length;
  assert.equal(pillMounts, 7, 'expected exactly 7 MetricPill mounts');
  assert.ok(ariaProps >= 7, `expected at least 7 ariaLabel= props, found ${ariaProps}`);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Components pill has hover handler writing to component bus
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationSummaryStrip — components pill writes to setActiveComponentId on hover', () => {
  const src = read(STRIP_PATH);
  // The onCompsHover handler must call setActiveComponentId, and the pill
  // must use it as its onHover prop.
  assert.match(src, /onCompsHover\s*=\s*useCallback\([\s\S]*?setActiveComponentId\(/);
  assert.match(src, /onCompsLeave\s*=\s*useCallback\([\s\S]*?setActiveComponentId\(null\)/);
  // The components pill mount uses the comps handlers.
  const compsIdx = src.indexOf('cert-summary-pill-components');
  assert.ok(compsIdx > -1, 'components pill testId not found');
  const blockStart = src.lastIndexOf('<MetricPill', compsIdx);
  const blockEnd   = src.indexOf('/>', compsIdx);
  assert.ok(blockStart > -1 && blockEnd > blockStart, 'components MetricPill block not isolated');
  const block = src.slice(blockStart, blockEnd);
  assert.match(block, /onHover=\{onCompsHover\}/);
  assert.match(block, /onLeave=\{onCompsLeave\}/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. App imports + mounts CertificationSummaryStrip
// ─────────────────────────────────────────────────────────────────────────────
test('App — imports CertificationSummaryStrip from sibling components folder', () => {
  const src = read(APP_PATH);
  assert.match(
    src,
    /import\s+CertificationSummaryStrip\s+from\s+['"]\.\/components\/CertificationSummaryStrip\.jsx['"]/,
  );
});

test('App — mounts <CertificationSummaryStrip ...> in the JSX tree', () => {
  const src = read(APP_PATH);
  assert.match(src, /<CertificationSummaryStrip/);
});

test('App — passes the seven live props plus setActiveComponentId to the strip', () => {
  const src = read(APP_PATH);
  const mountIdx = src.indexOf('<CertificationSummaryStrip');
  assert.ok(mountIdx > -1, '<CertificationSummaryStrip mount not found');
  const closeIdx = src.indexOf('/>', mountIdx);
  assert.ok(closeIdx > mountIdx, '<CertificationSummaryStrip mount has no self-closing />');
  const mountBlock = src.slice(mountIdx, closeIdx + 2);
  assert.match(mountBlock, /candidateMoves=\{/);
  assert.match(mountBlock, /accepted=\{/);
  assert.match(mountBlock, /identityOnly=\{/);
  assert.match(mountBlock, /rejected=\{/);
  assert.match(mountBlock, /gPtSize=\{/);
  assert.match(mountBlock, /hSize=\{/);
  assert.match(mountBlock, /componentsCount=\{/);
  assert.match(mountBlock, /setActiveComponentId=\{setActiveComponentId\}/);
});

test('App — CertificationSummaryStrip receives derived outputActionSize for hSize', () => {
  const src = read(APP_PATH);
  assert.match(src, /import\s+\{\s*restrictStabilizerToPositions\s*\}\s+from\s+['"]\.\/engine\/outputOrbit\.js['"]/);
  assert.match(src, /const outputActionSize = useMemo\(/);

  const mountIdx = src.indexOf('<CertificationSummaryStrip');
  assert.ok(mountIdx > -1, '<CertificationSummaryStrip mount not found');
  const closeIdx = src.indexOf('/>', mountIdx);
  const mountBlock = src.slice(mountIdx, closeIdx + 2);
  assert.match(mountBlock, /hSize=\{outputActionSize\}/);
  assert.doesNotMatch(mountBlock, /hSize=\{group\?\.fullElements\?\.length \?\? 1\}/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. No raw notation hex literals — TOKEN map only
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationSummaryStrip — no raw notation hex literals (TOKEN map only)', () => {
  const src = read(STRIP_PATH);
  const FORBIDDEN_NOTATION_HEXES = [
    '#F0524D', // coral
    '#64748B', // ein-w (summed)
    '#4A7CFF', // ein-v / info
    '#FA9E33', // warning
    '#23B761', // success
    '#292C2D', // gray-900
    '#5D5F60', // gray-600
    '#888B8D', // gray-500
    '#AAACAD', // gray-400
    '#D9DCDC', // gray-200
    '#F1F3F5', // gray-100
    '#FEF2F1', // coral-light
  ];
  for (const hex of FORBIDDEN_NOTATION_HEXES) {
    assert.equal(
      src.includes(hex),
      false,
      `Raw notation hex ${hex} must not appear in CertificationSummaryStrip source — use a CSS variable via TOKEN`,
    );
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Bonus: hover targets — strip writes to upstream highlight selectors so
// the user can see which panel each metric came from. The set of selectors
// is part of the V3.1 §27 contract; assert the wiring exists.
// ─────────────────────────────────────────────────────────────────────────────
test('CertificationSummaryStrip — hover targets cover wreath / audit / generator', () => {
  const src = read(STRIP_PATH);
  assert.match(src, /'#wreath-structure'/);
  assert.match(src, /'\.witness-gallery-mount'/);
  assert.match(src, /'#generator-construction'/);
});
