/**
 * C04 Einsum Formula Card — V3.1 migration tests
 *
 * Covers:
 *   1. AlgorithmAtAGlance accepts onHoveredLabelsChange prop
 *   2. AlgorithmAtAGlance fires onHoveredLabelsChange(new Set([...])) on label hover
 *   3. AlgorithmAtAGlance carries the V3.1 visible-label tooltip text
 *   4. AlgorithmAtAGlance carries the V3.1 summed-label tooltip text
 *   5. AlgorithmAtAGlance carries the V3.1 declared-symmetry tooltip text
 *   6. AlgorithmAtAGlance uses FormulaHighlighted from StickyBar for the formula line
 *   7. OrbitRepMatrix accepts hoveredLabels prop and derives axis highlights
 *   8. App wires hoveredLabels + onHoveredLabelsChange to AlgorithmAtAGlance
 *   9. BranchingDemo threads hoveredLabels to OrbitRepMatrix
 *  10. Hex audit — AlgorithmAtAGlance uses design-system tokens (no raw hex)
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const COMPONENTS = 'components/symmetry-aware-einsum-contractions/components';

const readComponent = (name) =>
  readFileSync(resolve(__dirname, COMPONENTS, name), 'utf-8');

const readApp = () =>
  readFileSync(
    resolve(__dirname, 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx'),
    'utf-8',
  );

// ─── 1. AlgorithmAtAGlance accepts onHoveredLabelsChange prop ────────────────

test('AlgorithmAtAGlance accepts onHoveredLabelsChange prop', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // The exported default function must declare onHoveredLabelsChange
  assert.match(
    src,
    /onHoveredLabelsChange/,
    'AlgorithmAtAGlance.jsx must reference onHoveredLabelsChange',
  );
  // It also must accept hoveredLabels (the read side of the bus)
  assert.match(
    src,
    /hoveredLabels/,
    'AlgorithmAtAGlance.jsx must reference hoveredLabels',
  );
  // Both must appear in the default export's prop destructuring
  assert.match(
    src,
    /export default function AlgorithmAtAGlance\(\{[\s\S]*?onHoveredLabelsChange/,
    'onHoveredLabelsChange must be in the exported component signature',
  );
});

// ─── 2. Formula fires onHoveredLabelsChange(new Set([...])) on label hover ───

test('AlgorithmAtAGlance fires onHoveredLabelsChange(new Set([...])) on label hover', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // The component must pass onHoveredLabelsChange down to a hover-enabled child.
  // FormulaHighlighted (from StickyBar) calls onHoveredLabelsChange(new Set([ch]))
  // internally — so we check that AlgorithmAtAGlance passes the handler through.
  assert.match(
    src,
    /onHoveredLabelsChange={onHoveredLabelsChange}/,
    'onHoveredLabelsChange must be forwarded to a child component',
  );
  // LabelChipList also fires it directly via onMouseEnter
  assert.match(
    src,
    /onHoveredLabelsChange\(new Set\(\[/,
    'LabelChipList must call onHoveredLabelsChange(new Set([ch])) directly',
  );
});

// ─── 3. V3.1 visible-label tooltip text ─────────────────────────────────────

test('AlgorithmAtAGlance has V3.1 visible-label tooltip text', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  assert.match(
    src,
    /Visible\/output label\. It survives as an axis of the result\./,
    'Visible-label tooltip must match V3.1 registries.md §4 exact string',
  );
});

// ─── 4. V3.1 summed-label tooltip text ───────────────────────────────────────

test('AlgorithmAtAGlance has V3.1 summed-label tooltip text', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  assert.match(
    src,
    /Summed label\. The evaluator loops over this label and accumulates it away\./,
    'Summed-label tooltip must match V3.1 registries.md §4 exact string',
  );
});

// ─── 5. V3.1 declared-symmetry tooltip text ──────────────────────────────────

test('AlgorithmAtAGlance has V3.1 declared-symmetry tooltip text', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  assert.match(
    src,
    /Declared operand symmetry\. This creates candidate product symmetries, but they still need certification\./,
    'Declared-symmetry tooltip must match V3.1 registries.md §4 exact string',
  );
});

// ─── 6. AlgorithmAtAGlance uses FormulaHighlighted from StickyBar ────────────

test('AlgorithmAtAGlance imports and uses FormulaHighlighted from StickyBar', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // Import
  assert.match(
    src,
    /import \{ FormulaHighlighted \} from '\.\/StickyBar\.jsx'/,
    'FormulaHighlighted must be imported from StickyBar.jsx',
  );
  // Usage — the label-hover-bus wired formula in the formula card
  assert.match(
    src,
    /<FormulaHighlighted/,
    'FormulaHighlighted must be used in the JSX',
  );
  assert.match(
    src,
    /hoveredLabels=\{hoveredLabels\}/,
    'hoveredLabels prop must be forwarded to FormulaHighlighted',
  );
});

// ─── 7. OrbitRepMatrix accepts hoveredLabels and derives axis highlights ──────

test('OrbitRepMatrix accepts hoveredLabels prop and applies axis-label highlights', () => {
  const src = readComponent('branchingViews/OrbitRepMatrix.jsx');
  // Prop accepted
  assert.match(
    src,
    /hoveredLabels\s*=\s*null/,
    'OrbitRepMatrix must accept hoveredLabels with null default',
  );
  // Y-axis highlight logic
  assert.match(
    src,
    /yAxisHighlighted/,
    'OrbitRepMatrix must compute yAxisHighlighted from hoveredLabels',
  );
  // X-axis highlight logic
  assert.match(
    src,
    /xAxisHighlighted/,
    'OrbitRepMatrix must compute xAxisHighlighted from hoveredLabels',
  );
  // data-testid on axis labels for testability
  assert.match(
    src,
    /data-testid="orbit-rep-matrix-y-axis-label"/,
    'Y-axis label must have a testid for downstream tests',
  );
  assert.match(
    src,
    /data-testid="orbit-rep-matrix-x-axis-label"/,
    'X-axis label must have a testid for downstream tests',
  );
});

// ─── 8. App wires hoveredLabels + onHoveredLabelsChange to AlgorithmAtAGlance ─

test('App passes hoveredLabels and onHoveredLabelsChange to AlgorithmAtAGlance', () => {
  const src = readApp();
  // Both props passed at the callsite
  assert.match(
    src,
    /<AlgorithmAtAGlance[\s\S]*?hoveredLabels=\{hoveredLabelSet\}/,
    'App must pass hoveredLabels={hoveredLabelSet} to AlgorithmAtAGlance',
  );
  assert.match(
    src,
    /<AlgorithmAtAGlance[\s\S]*?onHoveredLabelsChange=\{setStripHoveredLabels\}/,
    'App must pass onHoveredLabelsChange={setStripHoveredLabels} to AlgorithmAtAGlance',
  );
});

// ─── 9. BranchingDemo threads hoveredLabels to OrbitRepMatrix ─────────────────

test('BranchingDemo threads hoveredLabels prop down to OrbitRepMatrix', () => {
  const bdSrc = readComponent('BranchingDemo.jsx');
  // BranchingDemo accepts it
  assert.match(
    bdSrc,
    /hoveredLabels\s*=\s*null/,
    'BranchingDemo must accept hoveredLabels with null default',
  );
  // BranchingDemo passes it to OrbitRepMatrix
  assert.match(
    bdSrc,
    /hoveredLabels=\{hoveredLabels\}/,
    'BranchingDemo must forward hoveredLabels to OrbitRepMatrix',
  );
  // ComponentCostView also threads it
  const cvSrc = readComponent('ComponentCostView.jsx');
  assert.match(
    cvSrc,
    /hoveredLabels\s*=\s*null/,
    'ComponentCostView must accept hoveredLabels with null default',
  );
  assert.match(
    cvSrc,
    /hoveredLabels=\{hoveredLabels\}/,
    'ComponentCostView must forward hoveredLabels to BranchingDemo',
  );
});

// ─── 10. Hex audit — AlgorithmAtAGlance only uses design-system hex tokens ────

test('AlgorithmAtAGlance does not introduce raw hex colors outside design tokens', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // Strip comments and string literals that are LaTeX (textcolor args) before checking.
  // The only allowed raw hex in this file is inside textcolor{} LaTeX commands
  // (which are generated dynamically from explorerThemeColor, not hardcoded).
  // We check there are no hardcoded 6-digit hex strings outside of dynamic vars.
  const noLatex = src.replace(/\\\\textcolor\{[^}]*\}/g, '');
  // Look for hardcoded hex — but allow the coral token hex that OrbitRepMatrix uses
  // (that lives in OrbitRepMatrix.jsx, not here). AlgorithmAtAGlance must have none.
  const hexMatches = noLatex.match(/#[0-9A-Fa-f]{6}\b/g) ?? [];
  assert.deepEqual(
    hexMatches,
    [],
    `AlgorithmAtAGlance.jsx must not contain raw hex colors; found: ${hexMatches.join(', ')}`,
  );
});
