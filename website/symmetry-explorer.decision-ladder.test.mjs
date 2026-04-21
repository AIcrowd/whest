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

test('DecisionLadder stage bands use the editorial white/coral-light surface treatment', () => {
  assert.match(source, /const accent = isStage1 \? '#5D5F60' : '#F0524D'/);
  assert.match(source, /background:\s*isStage1\s*\?\s*'#FFFFFF'\s*:\s*'#FEF2F1'/);
  assert.match(source, /border:\s*`1\.5px solid \$\{isStage1 \? '#D9DCDC' : 'rgba\(240,82,77,0\.22\)'\}`/);
  assert.match(source, /label: 'Stage 1 · Structure'/);
  assert.match(source, /label: 'Stage 2 · Symmetry'/);
  assert.doesNotMatch(source, /linear-gradient\(180deg, rgba\(34,197,94,0\.07\)/);
  assert.doesNotMatch(source, /linear-gradient\(180deg, rgba\(139,92,246,0\.08\)/);
});
