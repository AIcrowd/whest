/**
 * Tests for TwoQuotientSchematic (C16) — source-grep style.
 *
 * These tests parse the JSX source as a string and assert structural
 * properties. They run in Node.js via the native test runner (no DOM).
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

const COMPONENT_PATH = 'components/symmetry-aware-einsum-contractions/components/TwoQuotientSchematic.jsx';
const STORIES_PATH   = 'components/symmetry-aware-einsum-contractions/components/TwoQuotientSchematic.stories.jsx';
const APP_PATH       = 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Component file exists and exports a default function
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — component file exists and exports default', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /export default function TwoQuotientSchematic/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Component renders an SVG element
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — renders <svg', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /<svg\b/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. All 4 box labels are present (X, X/G_pt, Y, Y/H)
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — has box label "X"', () => {
  const src = read(COMPONENT_PATH);
  // Box component receives label="X" for the full assignment space
  assert.match(src, /label="X"/);
});

test('TwoQuotientSchematic — has box label "X/G_pt"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /label="X\/G_pt"/);
});

test('TwoQuotientSchematic — has box label "Y"', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /label="Y"/);
});

test('TwoQuotientSchematic — has Y/H label (trivial or nontrivial)', () => {
  const src = read(COMPONENT_PATH);
  // Either the literal string 'Y/H' or 'Y/H = Y' (trivial H case)
  assert.match(src, /Y\/H/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Arrow labels: G_pt, H, π_V (or unicode / text equivalents)
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — has G_pt arrow label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /G_pt/);
});

test('TwoQuotientSchematic — has H arrow label', () => {
  const src = read(COMPONENT_PATH);
  // The label="H" prop on the HArrow for the bottom row
  assert.match(src, /label="H"/);
});

test('TwoQuotientSchematic — has π_V / pi_V projection label', () => {
  const src = read(COMPONENT_PATH);
  // Either πᵥ (unicode), π_V, or pi_V in source
  assert.ok(
    /π_V|πᵥ|pi_V|pi-V/.test(src),
    'Expected pi_V / πᵥ label in TwoQuotientSchematic source'
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Example-toggle has 3 buttons (one per preset)
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — example toggle has 3 preset entries', () => {
  const src = read(COMPONENT_PATH);
  // PRESET_ORDER drives the toggle; check it has exactly 3 keys
  assert.match(src, /PRESET_ORDER\s*=\s*\[/);
  const match = src.match(/PRESET_ORDER\s*=\s*\[([^\]]+)\]/);
  assert.ok(match, 'PRESET_ORDER array not found');
  const entries = match[1].split(',').map((s) => s.trim()).filter(Boolean);
  assert.equal(entries.length, 3, `Expected 3 presets, got ${entries.length}`);
});

test('TwoQuotientSchematic — renders <button> elements (not div) for the toggle', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /<button\b/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. prefers-reduced-motion respected
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — uses prefers-reduced-motion / matchMedia', () => {
  const src = read(COMPONENT_PATH);
  assert.ok(
    /prefers-reduced-motion|matchMedia/.test(src),
    'Expected prefers-reduced-motion or matchMedia in TwoQuotientSchematic source'
  );
});

test('TwoQuotientSchematic — reduced-motion guard applied to transitions', () => {
  const src = read(COMPONENT_PATH);
  // reducedMotion flag is threaded into transition: 'none' paths
  assert.match(src, /reducedMotion/);
  assert.match(src, /transition.*none/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Hex audit — only design-system tokens (no raw hex)
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — no raw hex colors outside design-system tokens', () => {
  const src = read(COMPONENT_PATH);
  const hexes = src.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  // All hex refs must be inside the TOKEN map block (design-system definitions)
  // or inside a comment. We check by ensuring no hex appears outside the TOKEN
  // constant definition section.
  //
  // Strategy: strip the TOKEN block, then check remaining source for hex.
  const withoutTokenBlock = src.replace(/const TOKEN\s*=\s*\{[\s\S]*?\};/, '');
  const remainingHexes = withoutTokenBlock.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  assert.deepEqual(
    remainingHexes,
    [],
    `Raw hex outside TOKEN block: ${remainingHexes.join(', ')}`,
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. Accessibility — ARIA labels on interactive elements
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — has aria-label on the SVG', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /role="img"/);
  assert.match(src, /aria-label="Two-quotient schematic diagram"/);
});

test('TwoQuotientSchematic — toggle button group has aria-label', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /aria-label="Select example preset"/);
});

test('TwoQuotientSchematic — interactive arrow labels have aria-label', () => {
  const src = read(COMPONENT_PATH);
  // aria-label uses a template literal: `Hover to highlight: ${labelHover}`
  assert.match(src, /aria-label=\{`Hover to highlight:/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 9. Three hover states wired
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic — hover states: gpt, h, pi all wired', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /startHover\('gpt'\)/);
  assert.match(src, /startHover\('h'\)/);
  assert.match(src, /startHover\('pi'\)/);
});

// ─────────────────────────────────────────────────────────────────────────────
// 10. App mounts TwoQuotientSchematic in §4 rows-cols section
// ─────────────────────────────────────────────────────────────────────────────
test('App — imports TwoQuotientSchematic', () => {
  const src = read(APP_PATH);
  assert.match(src, /import TwoQuotientSchematic from '\.\/components\/TwoQuotientSchematic\.jsx'/);
});

test('App — mounts <TwoQuotientSchematic /> in §4 rows-cols section', () => {
  const src = read(APP_PATH);
  assert.match(src, /<TwoQuotientSchematic\s*\/>/);
  // Verify it appears after the §4 comment
  const idx4 = src.indexOf('§4 Rows and Columns');
  const idxMount = src.indexOf('<TwoQuotientSchematic');
  assert.ok(idx4 > -1, '§4 Rows and Columns comment not found in App');
  assert.ok(idxMount > -1, '<TwoQuotientSchematic mount not found in App');
  assert.ok(idxMount > idx4, '<TwoQuotientSchematic must appear after §4 comment');
});

// ─────────────────────────────────────────────────────────────────────────────
// 11. Stories file: ≥3 named exports
// ─────────────────────────────────────────────────────────────────────────────
test('TwoQuotientSchematic stories — file exists and has ≥3 named exports', () => {
  const src = read(STORIES_PATH);
  const exports = src.match(/^export const \w+/gm) ?? [];
  assert.ok(
    exports.length >= 3,
    `Expected ≥3 named story exports, found ${exports.length}: ${exports.join(', ')}`,
  );
});

test('TwoQuotientSchematic stories — includes CrossS2Default story', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const CrossS2Default/);
});

test('TwoQuotientSchematic stories — includes BilinearTrace story', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const BilinearTrace/);
});

test('TwoQuotientSchematic stories — includes TripleOuterAllVisible story', () => {
  const src = read(STORIES_PATH);
  assert.match(src, /export const TripleOuterAllVisible/);
});
