import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  resolve(__dirname, 'components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx'),
  'utf-8',
);

test('DecisionLadder exports a default React component', () => {
  assert.match(source, /export default function DecisionLadder/);
});

test('DecisionLadder uses ReactFlow for graph rendering', () => {
  assert.match(source, /from ['"]@xyflow\/react['"]/);
  assert.match(source, /ReactFlow/);
});

test('DecisionLadder sources shape + regime layout from specs', () => {
  assert.match(source, /SHAPE_SPEC/);
  assert.match(source, /REGIME_SPEC/);
});

test('DecisionLadder supports an activeRegimeId prop for highlighting', () => {
  assert.match(source, /activeRegimeId/);
});

test('DecisionLadder renders a flow-chart with source / question / leaf node types', () => {
  assert.match(source, /function SourceNode/);
  assert.match(source, /function QuestionNode/);
  assert.match(source, /function LeafNode/);
});

test('DecisionLadder defines yes/no edge styling', () => {
  assert.match(source, /EDGE_YES/);
  assert.match(source, /EDGE_NO/);
});

test('DecisionLadder enables scroll-zoom on the ReactFlow', () => {
  assert.match(source, /addEventListener\('wheel',\s*handleWheel,\s*\{\s*passive:\s*false\s*\}\)/);
  assert.match(source, /event\.ctrlKey\s*\|\|\s*event\.metaKey/);
  assert.match(source, /setViewport\(\{/);
});

test('DecisionLadder intuition copy uses editorial italic example names instead of texttt preset ids', () => {
  assert.match(source, /\*bilinear-trace\*/);
  assert.match(source, /\*four-A-grid\*/);
  assert.doesNotMatch(source, /\\texttt\{bilinear-trace\}/);
  assert.doesNotMatch(source, /\\texttt\{four-A-grid\}/);
});

test('DecisionLadder stage bands use a shared neutral frame and stage-family accents from the theme', () => {
  assert.match(source, /const accent = explorerThemeColor\(getActiveExplorerThemeId\(\), isStage1 \? 'caseAllVisible' : 'caseSingleton'\)/);
  assert.match(source, /background:\s*'#FFFFFF'/);
  assert.match(source, /border:\s*`1\.5px solid \$\{explorerThemeColor\(getActiveExplorerThemeId\(\), 'border'\)\}`/);
  assert.match(source, /label: 'Stage 1 · Structure'/);
  assert.match(source, /label: 'Stage 2 · Symmetry'/);
  assert.doesNotMatch(source, /linear-gradient\(180deg, rgba\(34,197,94,0\.07\)/);
  assert.doesNotMatch(source, /linear-gradient\(180deg, rgba\(139,92,246,0\.08\)/);
  assert.doesNotMatch(source, /rgba\(240,82,77,0\.22\)/);
});

test('DecisionLadder leaf selection uses a separated same-color outer ring while spotlight keeps the stronger halo', () => {
  assert.match(source, /const bg = data\.color \?\? '#94A3B8';/);
  assert.match(source, /const darkSurface = isDarkColor\(bg\);/);
  assert.match(source, /const ACTIVE_RING = bg;/);
  assert.match(source, /const baseBorderColor = darkSurface \? mixWithWhite\(bg, 0\.18\) : mixWithBlack\(bg, 0\.16\);/);
  assert.match(source, /const borderColor = data\.active[\s\S]*\? ACTIVE_RING[\s\S]*data\.spotlight[\s\S]*\? SPOTLIGHT_RING[\s\S]*: baseBorderColor/);
  assert.match(source, /text-\[13px\]/);
  assert.match(source, /0 0 0 3px #FFFFFF,\s*0 0 0 5px \$\{ACTIVE_RING\}/);
  assert.match(source, /0 0 0 9px \$\{SPOTLIGHT_RING\}26/);
  assert.match(source, /borderWidth: data\.spotlight \? 3 : 2/);
  assert.match(source, /const textColor = darkSurface \? '#F8FAFC' : '#132228';/);
  assert.match(source, /color: textColor/);
  assert.match(source, /const presentation = getRegimePresentation\(leafId\);/);
  assert.match(source, /function labelForLeaf\(leafId, fallback\)/);
  assert.match(source, /text: labelForLeaf\(leafId,\s*presentation\?\.label \?\? spec\.label\)/);
  assert.match(source, /color: presentation\?\.color \?\? spec\.color/);
  assert.doesNotMatch(source, /const bg = '#334155';/);
  assert.doesNotMatch(source, /borderWidth: data\.spotlight \? 3 : data\.active \? 3 : 2/);
  assert.doesNotMatch(source, /inset 0 0 0 2px/);
});
