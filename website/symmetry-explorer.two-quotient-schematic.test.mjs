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

test('TwoQuotientSchematic — renders a DOM microscope plus formal quotient summary', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /<figure/);
  assert.match(src, /function OneRowMicroscope/);
  assert.match(src, /function FormalQuotientSummary/);
  assert.match(src, /One row microscope/);
  assert.match(src, /One product row/);
  assert.match(src, /Each pill is one dense assignment/);
  // The Rows / Columns / Reach formulas now render through <Latex>;
  // their JSX strings carry escaped backslashes for LaTeX commands.
  assert.match(src, /X \\\\to X\/G_\{\\\\mathrm\{pt\}\}/);
  assert.match(src, /Y \\\\to Y\/H/);
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

test('TwoQuotientSchematic — explains single assignments, samples, rows, and columns', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /one choice of values for all labels/);
  assert.match(src, /not the whole space/);
  assert.match(src, /Two example tuples from X/);
  assert.match(src, /One row O/);
  assert.match(src, /Which output cells get filled\?/);
  assert.match(src, /Projection marks which row-column cells are filled/);
  assert.match(src, /This product row contributes/);
  assert.match(src, /A\[0,1\] \* B\[0\] = A\[1,0\] \* B\[0\]/);
});

test('TwoQuotientSchematic — current summary includes n, M, |Y/H|, α, dense X, H, and branching state', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /hStatus = hSize > 1 \? 'H nontrivial' : 'H trivial'/);
  assert.match(src, /projectionStatus = branchRows > 0 \? 'projection branches' : 'projection functional'/);
  assert.match(src, /denseAssignmentCount/);
  assert.match(src, /labelOrder/);
  assert.match(src, /visibleLabels/);
  assert.match(src, /\{ label: 'n', value: formatNumber\(dimensionN\) \}/);
  assert.match(src, /\{ label: 'M', value: `\$\{formatNumber\(rowCount\)\} rows` \}/);
  assert.match(src, /\{ label: '\|Y\/H\|', value: `\$\{formatNumber\(columnCount\)\} columns` \}/);
  assert.match(src, /\{ label: 'α', value: `\$\{formatNumber\(alpha\)\} filled cells` \}/);
});

test('TwoQuotientSchematic — microscope chooses a useful row and explains the flow', () => {
  const src = read(COMPONENT_PATH);
  assert.match(src, /microscopeRowIndex = orbitRows\.findIndex/);
  assert.match(src, /outputCount/);
  assert.match(src, /same product/);
  assert.match(src, /project to V/);
  assert.match(src, /Filled output cells/);
  assert.match(src, /The row fills/);
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

test('TwoQuotientSchematic — removes the older lane UI in favor of one microscope flow', () => {
  const src = read(COMPONENT_PATH);
  assert.doesNotMatch(src, /function QuotientLane/);
  assert.doesNotMatch(src, /function ReachRelationPanel/);
  assert.doesNotMatch(src, /Record reached Q columns/);
});

test('App — computes and passes current preset data into the schematic', () => {
  const src = read(APP_PATH);
  assert.match(src, /const twoQuotientCurrent = useMemo/);
  assert.match(src, /branchRows/);
  assert.match(src, /outputKeys\.size/);
  assert.match(src, /dimensionN/);
  assert.match(src, /labelOrder/);
  assert.match(src, /visibleLabels/);
  assert.match(src, /denseAssignmentCount/);
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
