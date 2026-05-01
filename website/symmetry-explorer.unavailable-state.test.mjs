/**
 * Tests for UnavailableDetailsPanel (C49) — source-grep style.
 *
 * Mirrors symmetry-explorer.tuple-pattern-meter.test.mjs and other Act-3
 * source-grep suites. These tests parse the JSX source as text and assert
 * structural properties — no DOM, no React renderer required.
 *
 * Coverage (V3.1 §49):
 *   1. Panel file exists with the expected default export
 *   2. Panel renders the V3.1 verbatim heading
 *   3. Panel renders all four V3.1 detail lines
 *   4. Panel uses live typed-partition count (engine import)
 *   5. Panel uses live brute-force count (engine import)
 *   6. "Try n = …" CTA button + appendix link surfaced
 *   7. ComponentCostView mounts the panel under each unavailable row
 *   8. ComponentCostView passes failedCondition derived from the trace
 *   9. App threads onDimensionNChange into ComponentCostView
 *  10. No raw notation hex in the panel source
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

const PANEL_PATH =
  'components/symmetry-aware-einsum-contractions/components/UnavailableDetailsPanel.jsx';
const COST_VIEW_PATH =
  'components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx';
const APP_PATH =
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Component file exists and exports a default function
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — component file exists and exports default', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /export default function UnavailableDetailsPanel/);
});

test('UnavailableDetailsPanel — accepts componentId, sizes, groupSize, failedCondition, onLowerN, currentN props', () => {
  const src = read(PANEL_PATH);
  const start = src.indexOf('export default function UnavailableDetailsPanel(');
  assert.ok(start > -1, 'default export signature not found');
  const end = src.indexOf(') {', start);
  const sig = src.slice(start, end);
  assert.match(sig, /componentId/);
  assert.match(sig, /sizes/);
  assert.match(sig, /groupSize/);
  assert.match(sig, /failedCondition/);
  assert.match(sig, /onLowerN/);
  assert.match(sig, /currentN/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. V3.1 verbatim heading
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — V3.1 verbatim heading rendered', () => {
  const src = read(PANEL_PATH);
  assert.match(
    src,
    /α unavailable under the current interactive budget/,
    'V3.1 verbatim heading must be present',
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. All four V3.1 detail lines
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — detail line 1 "No exact shortcut applies."', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /No exact shortcut applies\./);
});

test('UnavailableDetailsPanel — detail line 2 "Typed partition patterns exceed budget"', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /Typed partition patterns exceed budget/);
});

test('UnavailableDetailsPanel — detail line 3 "Corrected brute force pair touches exceed budget"', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /Corrected brute force pair touches exceed budget/);
});

test('UnavailableDetailsPanel — detail line 4 "The page reports unavailable rather than guessing."', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /The page reports unavailable rather than guessing\./);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Live typed-partition count — engine import + usage
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — imports generateTypedSetPartitions from engine', () => {
  const src = read(PANEL_PATH);
  assert.match(
    src,
    /import\s*\{\s*generateTypedSetPartitions\s*\}\s*from\s*['"]\.\.\/engine\/partition\/typedPartitions\.js['"]/,
  );
});

test('UnavailableDetailsPanel — calls generateTypedSetPartitions for live count', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /generateTypedSetPartitions\(\s*safeSizes\s*\)/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Live brute-force count — engine import + usage
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — imports bruteForceEstimate + BRUTE_FORCE_BUDGET from engine', () => {
  const src = read(PANEL_PATH);
  assert.match(
    src,
    /import\s*\{[\s\S]*?bruteForceEstimate[\s\S]*?BRUTE_FORCE_BUDGET[\s\S]*?\}\s*from\s*['"]\.\.\/engine\/budget\.js['"]/,
  );
});

test('UnavailableDetailsPanel — calls bruteForceEstimate for live count', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /bruteForceEstimate\(\s*safeSizes\s*,\s*safeGroupSize\s*\)/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. CTAs — "Try n = …" button + appendix link
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — Try n CTA button is rendered', () => {
  const src = read(PANEL_PATH);
  // The button literal contains "Try n = " followed by the suggested value.
  assert.match(src, /Try n = \{suggestedN\}/);
  // And it must be inside a <button> with onClick wired to onLowerN.
  assert.match(src, /onClick=\{\(\)\s*=>\s*onLowerN\?\.\(suggestedN\)/);
});

test('UnavailableDetailsPanel — appendix link points to #appendix-section-7 (Appendix B)', () => {
  const src = read(PANEL_PATH);
  // V3.1: B.9 is the unavailable-case leaf of the classification tree
  // (Appendix B), mounted at #appendix-section-7. The previously used
  // #appendix-section-6 was the typed-partition theorem (Appendix C),
  // which states the formula but not the budget/unavailable contract.
  assert.match(src, /href="#appendix-section-7"/);
  assert.match(src, /Read Appendix B\.9/);
  // Guard against regression to the prior C-pointing anchor.
  assert.doesNotMatch(src, /href="#appendix-section-6"/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Accessibility — role="region" + aria-label, focusable button + link
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — section uses role="region" + aria-label', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /role="region"/);
  assert.match(src, /aria-label="Unavailable count details"/);
});

test('UnavailableDetailsPanel — Try-n button carries dynamic aria-label', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /aria-label=\{`Try a smaller dimension: lower n to \$\{suggestedN\}`\}/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. ComponentCostView mounts the panel + has classify helper
// ─────────────────────────────────────────────────────────────────────────────
test('ComponentCostView — imports UnavailableDetailsPanel', () => {
  const src = read(COST_VIEW_PATH);
  assert.match(
    src,
    /import UnavailableDetailsPanel from '\.\/UnavailableDetailsPanel\.jsx'/,
  );
});

test('ComponentCostView — mounts <UnavailableDetailsPanel when actualAcc is null', () => {
  const src = read(COST_VIEW_PATH);
  // Mount appears inside the per-row JSX.
  assert.match(src, /<UnavailableDetailsPanel/);
  // Mount is gated by actualAcc === null.
  assert.match(src, /\{actualAcc === null \?/);
});

test('ComponentCostView — defines classifyFailedCondition heuristic', () => {
  const src = read(COST_VIEW_PATH);
  assert.match(src, /function classifyFailedCondition\(trace\)/);
  // The heuristic checks for partition / brute-force keywords.
  assert.match(src, /reason\.includes\(['"]partition['"]\)/);
  assert.match(src, /reason\.includes\(['"]brute-force['"]\)|reason\.includes\(['"]brute force['"]\)/);
});

test('ComponentCostView — passes failedCondition derived from comp.accumulation.trace', () => {
  const src = read(COST_VIEW_PATH);
  assert.match(src, /failedCondition=\{classifyFailedCondition\(comp\.accumulation\?\.trace\)\}/);
});

test('ComponentCostView — passes onLowerN sourced from onDimensionNChange', () => {
  const src = read(COST_VIEW_PATH);
  assert.match(
    src,
    /onLowerN=\{typeof onDimensionNChange === ['"]function['"] \? onDimensionNChange : null\}/,
  );
});

test('ComponentCostView — declares onDimensionNChange as an optional prop', () => {
  const src = read(COST_VIEW_PATH);
  // Default-export signature accepts the new prop (with default null).
  assert.match(src, /onDimensionNChange = null,/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 9. App threads onDimensionNChange into ComponentCostView
// ─────────────────────────────────────────────────────────────────────────────
test('App — passes onDimensionNChange={setDefaultSize} to ComponentCostView', () => {
  const src = read(APP_PATH);
  // Locate the ComponentCostView mount inside §3 Projection.
  const start = src.indexOf('<ComponentCostView');
  assert.ok(start > -1, '<ComponentCostView mount not found in App');
  const end = src.indexOf('/>', start);
  assert.ok(end > -1, 'self-closing /> not found after <ComponentCostView');
  const mountBlock = src.slice(start, end + 2);
  assert.match(mountBlock, /onDimensionNChange=\{setDefaultSize\}/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 10. Hex audit — no raw notation hex literals
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — no raw notation hex literals', () => {
  const src = read(PANEL_PATH);
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
      `Raw notation hex ${hex} must not appear in UnavailableDetailsPanel source — use a CSS variable instead`,
    );
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 11. Honest framing — V3.1 says "this is a feature, not an error"
// ─────────────────────────────────────────────────────────────────────────────
test('UnavailableDetailsPanel — surfaces V3.1 "feature, not an error" framing', () => {
  const src = read(PANEL_PATH);
  assert.match(src, /This is a feature, not an error/);
});
