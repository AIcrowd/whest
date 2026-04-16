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
  // Either explicitly enabled or left at the default (both acceptable;
  // we just want to ensure it isn't disabled).
  assert.doesNotMatch(source, /zoomOnScroll\s*=\s*\{\s*false\s*\}/);
});
