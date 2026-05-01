/**
 * Tests for C38 Final Formula Hero polish (V3.1 §9 climax).
 *
 * Coverage:
 *  1. Caption "Multiplication-chain events and accumulation-update events are
 *     added, not multiplied." appears verbatim.
 *  2. Compact/full toggle has 2 buttons with aria-pressed.
 *  3. Hover-product-term has onMouseEnter / onMouseLeave handlers (source-grep).
 *  4. Hover-α-row — CaseBadge wraps formula for tooltip; hover handlers exist
 *     via CaseBadge passthrough.
 *  5. Click-α-row has a wrapping <a href="#appendix-section-7"> link (Appendix B).
 *  6. Hex audit — no raw hex outside the LaTeX color-helper context.
 *  7. Accessibility — toggle buttons have aria-pressed; hover target has
 *     tabIndex and role="button".
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = (...parts) => resolve(__dirname, ...parts);
const read = (...parts) => readFileSync(root(...parts), 'utf-8');

const COMPONENT_PATH = 'components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx';

// ─── 1. Caption verbatim ──────────────────────────────────────────────────────
test('C38: caption "Multiplication-chain events and accumulation-update events are added, not multiplied." appears verbatim in TotalCostView', () => {
  const src = read(COMPONENT_PATH);
  const EXPECTED = 'Multiplication-chain events and accumulation-update events are added, not multiplied.';
  assert.ok(
    src.includes(EXPECTED),
    `Expected verbatim caption not found. Looking for:\n  "${EXPECTED}"`,
  );
});

// ─── 2. Compact / full toggle has exactly 2 buttons ──────────────────────────
test('C38: compact/full toggle renders 2 labelled buttons', () => {
  const src = read(COMPONENT_PATH);
  // The toggle array must have both 'compact' and 'full' id entries
  assert.match(src, /id:\s*'compact'/);
  assert.match(src, /id:\s*'full'/);
  // Both must sit inside a role="group" aria-label block
  assert.match(src, /role="group"/);
  assert.match(src, /aria-label="Formula detail level"/);
  // Two button labels
  assert.match(src, /label:\s*'Compact'/);
  assert.match(src, /label:\s*'Full formula'/);
});

// ─── 3. Hover-product-term: onMouseEnter / onMouseLeave on the span ──────────
test('C38: hero formula top line has onMouseEnter and onMouseLeave hover handlers', () => {
  const src = read(COMPONENT_PATH);
  // The product-term hover span wraps the Burnside tooltip
  assert.match(src, /onMouseEnter=\{.*setShowBurnsideTooltip.*true/);
  assert.match(src, /onMouseLeave=\{.*setShowBurnsideTooltip.*false/);
  // The tooltip state variable exists
  assert.match(src, /showBurnsideTooltip/);
  // SimpleTooltip is rendered with BURNSIDE_PRODUCT_TOOLTIP text
  assert.match(src, /BURNSIDE_PRODUCT_TOOLTIP/);
});

// ─── 4. Hover-α-row: CaseBadge passthrough provides tooltip on hover ─────────
test('C38: each alpha-row formula is wrapped in CaseBadge for hover tooltip', () => {
  const src = read(COMPONENT_PATH);
  // FormulaRow wraps CaseBadge around Latex — CaseBadge's passthrough mode
  // provides onPointerEnter / onPointerLeave internally
  assert.match(src, /CaseBadge[\s\S]{0,200}regimeId=\{leaf\.id\}[\s\S]{0,200}Latex math=\{leaf\.formula\}/);
});

// ─── 5. Click-α-row: <a href="#appendix-section-7"> link (Appendix B) ────────
test('C38: each alpha-row has an anchor link pointing to #appendix-section-7', () => {
  const src = read(COMPONENT_PATH);
  // V3.1: Appendix B (classification cases) lives at #appendix-section-7.
  // The legacy #appendix-section-2 was Appendix D (dummy renamings) — wrong
  // content for an alpha-shortcut "full derivation" target.
  assert.match(src, /href=\{appendixHref\}/);
  assert.match(src, /appendixHref\s*=\s*['"]#appendix-section-7['"]/);
  // aria-label on the link
  assert.match(src, /aria-label=\{`\$\{shortcutLabel\}.*Appendix B/);
  // Guard against regression to the old D-pointing anchor.
  assert.doesNotMatch(src, /appendixHref\s*=\s*['"]#appendix-section-2['"]/);
});

// ─── 6. Hex audit ────────────────────────────────────────────────────────────
test('C38: TotalCostView does not introduce raw hex outside LaTeX textcolor helpers', () => {
  const src = read(COMPONENT_PATH);

  // Strip the tc() helper calls (textcolor{#...}{...}) — these are LaTeX, not CSS.
  const withoutLatexColors = src
    .replace(/\\textcolor\{#[0-9A-Fa-f]{3,8}\}/g, '')
    .replace(/textcolor\{[^}]+\}/g, '');

  // #f3c5bf is a pre-existing design accent used in the Cost Savings header
  // dividers (EditorialComparisonSpread). It pre-dates V3.1 and is not
  // introduced by the C38 polish. Allowlist it exactly as ComponentCostView's
  // test does for its own pre-existing hex.
  const ALLOWED_HEX = new Set(['#f3c5bf']);

  const bareHexMatches = (withoutLatexColors.match(/#[0-9A-Fa-f]{6}\b/g) ?? [])
    .filter((h) => !ALLOWED_HEX.has(h.toLowerCase()));

  assert.deepEqual(
    bareHexMatches,
    [],
    `Raw hex literals found in TotalCostView outside textcolor helpers: ${bareHexMatches.join(', ')}`,
  );
});

// ─── 7. Accessibility ─────────────────────────────────────────────────────────
test('C38: toggle buttons have aria-pressed; hover target has tabIndex and role="button"', () => {
  const src = read(COMPONENT_PATH);
  // Toggle buttons
  assert.match(src, /aria-pressed=\{active\}/);
  // Product-term hover span
  assert.match(src, /tabIndex=\{0\}/);
  assert.match(src, /role="button"/);
  // α-row links have aria-label
  assert.match(src, /aria-label=\{`\$\{shortcutLabel\}/);
});
