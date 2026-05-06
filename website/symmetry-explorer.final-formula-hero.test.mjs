/**
 * Tests for C38 Final Formula Hero polish (V3.1 §9 climax).
 *
 * Coverage:
 *  1. Caption "Multiplication-chain events and accumulation-update events are
 *     added, not multiplied." appears verbatim.
 *  2. Formula hero always renders the full formula; no compact/full toggle.
 *  3. Hover-total-formula has onMouseEnter / onMouseLeave handlers (source-grep).
 *  4. Hover-α-row — CaseBadge wraps formula for tooltip; hover handlers exist
 *     via CaseBadge passthrough.
 *  5. Click-α-row has a wrapping <a href="#appendix-section-7"> link (Appendix B).
 *  6. Hex audit — no raw hex outside the LaTeX color-helper context.
 *  7. Accessibility — hover target has tabIndex and role="button".
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

// ─── 2. Always render full formula, with no compact/full toggle ───────────────
test('C38: formula hero always renders the full formula without a detail toggle', () => {
  const src = read(COMPONENT_PATH);
  assert.doesNotMatch(src, /formulaMode/);
  assert.doesNotMatch(src, /setFormulaMode/);
  assert.doesNotMatch(src, /compactLine/);
  assert.doesNotMatch(src, /label:\s*'Compact'/);
  assert.doesNotMatch(src, /label:\s*'Full formula'/);
  assert.doesNotMatch(src, /aria-label="Formula detail level"/);
  assert.match(src, /<Latex display math=\{topLine\} themeOverride=\{themeOverride\}/);
});

// ─── 3. Hover-total-formula: onMouseEnter / onMouseLeave on the span ─────────
test('C38: hero formula top line has an accurate total-formula tooltip', () => {
  const src = read(COMPONENT_PATH);
  // The hover span wraps a tooltip describing both the multiplication and accumulation terms.
  assert.match(src, /onMouseEnter=\{.*setShowFormulaTooltip.*true/);
  assert.match(src, /onMouseLeave=\{.*setShowFormulaTooltip.*false/);
  // The tooltip state variable exists
  assert.match(src, /showFormulaTooltip/);
  // SimpleTooltip is rendered with total-formula tooltip text.
  assert.match(src, /TOTAL_FORMULA_TOOLTIP/);
  assert.match(src, /multiplication-chain events/);
  assert.match(src, /accumulation updates/);
  assert.doesNotMatch(src, /BURNSIDE_PRODUCT_TOOLTIP/);
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

  // Cost Savings dividers are tokenized as bg-coral-light; raw CSS hex should
  // only appear inside stripped LaTeX textcolor helpers.
  const ALLOWED_HEX = new Set();

  const bareHexMatches = (withoutLatexColors.match(/#[0-9A-Fa-f]{6}\b/g) ?? [])
    .filter((h) => !ALLOWED_HEX.has(h.toLowerCase()));

  assert.deepEqual(
    bareHexMatches,
    [],
    `Raw hex literals found in TotalCostView outside textcolor helpers: ${bareHexMatches.join(', ')}`,
  );
});

// ─── 7. Accessibility ─────────────────────────────────────────────────────────
test('C38: hover target has tabIndex and role="button"', () => {
  const src = read(COMPONENT_PATH);
  assert.doesNotMatch(src, /aria-pressed=\{active\}/);
  // Product-term hover span
  assert.match(src, /tabIndex=\{0\}/);
  assert.match(src, /role="button"/);
  // α-row links have aria-label
  assert.match(src, /aria-label=\{`\$\{shortcutLabel\}/);
});
