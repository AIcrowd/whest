/**
 * Tests for TwoQuotientSchematic (C16) — source-grep style.
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
const STORIES_PATH = 'components/symmetry-aware-einsum-contractions/components/TwoQuotientSchematic.stories.jsx';
const APP_PATH = 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

test('TwoQuotientSchematic — component file exists and exports default', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /export default function TwoQuotientSchematic\(\{ current = null \}\)/);
});

test('TwoQuotientSchematic — renders a DOM lane schematic', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /<figure/);
  assert.match(src, /function QuotientLane/);
  assert.match(src, /function ReachRelationPanel/);
  assert.match(src, /Row quotient/);
  assert.match(src, /Column quotient/);
  assert.match(src, /Projection reach/);
});

test('TwoQuotientSchematic — defaults to current selected preset plus non-duplicate references', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /const PRESET_ORDER\s*=\s*\['crossS2', 'bilinearTrace', 'tripleOuter'\]/);
  assert.match(src, /buildCurrentView\(current\)/);
  assert.match(src, /referenceId: normalizePresetId\(presetName\)/);
  assert.match(src, /tabLabel: `Current: \$\{presetName\}`/);
  assert.match(src, /\.filter\(\(id\) => id !== currentView\.referenceId\)/);
});

test('TwoQuotientSchematic — reference examples preserve the three pedagogical cases', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /Cross S₂/);
  assert.match(src, /Bilinear trace/);
  assert.match(src, /Triple outer/);
  assert.match(src, /H is trivial/);
  assert.match(src, /H is nontrivial/);
  assert.match(src, /Projection drops nothing/);
});

test('TwoQuotientSchematic — has reader-facing row and column labels', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /Full product assignments/);
  assert.match(src, /Product-orbit rows/);
  assert.match(src, /Visible output assignments/);
  assert.match(src, /Stored-output columns/);
  assert.match(src, /Projection is not a third quotient/);
  assert.match(src, /marks the filled row-column cells counted by α/);
});

test('TwoQuotientSchematic — current summary includes n, M, |Y/H|, α, H, and branching state', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /hStatus = hSize > 1 \? 'H nontrivial' : 'H trivial'/);
  assert.match(src, /projectionStatus = branchRows > 0 \? 'projection branches' : 'projection functional'/);
  assert.match(src, /\{ label: 'n', value: formatNumber\(dimensionN\) \}/);
  assert.match(src, /\{ label: 'M', value: `\$\{formatNumber\(rowCount\)\} rows` \}/);
  assert.match(src, /\{ label: '\|Y\/H\|', value: `\$\{formatNumber\(columnCount\)\} columns` \}/);
  assert.match(src, /\{ label: 'α', value: `\$\{formatNumber\(alpha\)\} filled cells` \}/);
});

test('TwoQuotientSchematic — hover states for G_pt, H, and pi are still wired', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /startHover\('gpt'\)/);
  assert.match(src, /startHover\('h'\)/);
  assert.match(src, /startHover\('pi'\)/);
  assert.match(src, /aria-label=\{`Highlight \$\{label\}`\}/);
  assert.match(src, /aria-label="Highlight projection reach relation"/);
});

test('TwoQuotientSchematic — uses design-system token recipe, no raw hex', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /var\(--coral\)/);
  assert.match(src, /var\(--coral-light\)/);
  assert.match(src, /var\(--gray-200\)/);
  assert.match(src, /rounded-\[var\(--radius-lg\)\]/);
  const hexes = src.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  assert.deepEqual(hexes, [], `Raw hex in schematic source: ${hexes.join(', ')}`);
});

test('TwoQuotientSchematic — uses prefers-reduced-motion / matchMedia', () => {
  const src = read(COMPONENT_PATH);
  assert.ok(/prefers-reduced-motion|matchMedia/.test(src));
  assert.match(src, /reducedMotion/);
  assert.match(src, /transition.*none/);
});

test('App — computes and passes current preset data into the schematic', () => {
  const src = read(APP_PATH);
  assert.match(src, /const twoQuotientCurrent = useMemo/);
  assert.match(src, /branchRows/);
  assert.match(src, /outputKeys\.size/);
  assert.match(src, /dimensionN/);
  assert.match(src, /<TwoQuotientSchematic current=\{twoQuotientCurrent\} \/>/);
});

test('App — renders Section 4 intro as block-aware two-column prose', () => {
  const src = read(APP_PATH);
  assert.match(src, /blocks=\{EXPLORER_ACTS\[3\]\.introBlocks\}/);
  assert.doesNotMatch(src, /blocks=\{EXPLORER_ACTS\[3\]\.introBlocks\}[\s\S]{0,120}columns="one"/);
  assert.match(src, /max-w-\[1180px\]/);
});

test('TwoQuotientSchematic stories — file exists and covers current + references', () => {
  const src = read(STORIES_PATH);
  const exports = src.match(/^export const \w+/gm) ?? [];
  assert.ok(exports.length >= 4, `Expected ≥4 named story exports, found ${exports.length}`);
  assert.match(src, /export const CurrentPreset/);
  assert.match(src, /export const CrossS2Default/);
  assert.match(src, /export const BilinearTrace/);
  assert.match(src, /export const TripleOuterAllVisible/);
  assert.match(src, /CURRENT_SAMPLE/);
});
