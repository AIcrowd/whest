// website/symmetry-explorer.tooltip-registry.test.mjs
//
// Source-grep tests asserting V3.1 Parts 6–8 prescribed strings appear in
// their respective component files. These cover:
//   - Part 6: O→Q hover card, wreath/sigma-loop modal, component summary tooltips,
//             classification-tree tooltip structure
//   - Part 8: Builder copy (dimension knob title)
//
// Each test reads the component source as plain text and asserts the exact V3.1
// phrase is present. No JSX runtime or DOM needed.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = (...parts) => resolve(__dirname, ...parts);
const read = (...parts) => readFileSync(root(...parts), 'utf-8');

const COMPONENTS = 'components/symmetry-aware-einsum-contractions/components';

// ── Part 6: O→Q hover card ────────────────────────────────────────────────

test('OrbitDetailCard: eyebrow reads "Worked example · O → Q" (V3.1 Part 6)', () => {
  const src = read(COMPONENTS, 'branchingViews/OrbitDetailCard.jsx');
  assert.match(
    src,
    /Worked example · O → Q/,
    'OrbitDetailCard must contain the V3.1 hover-card eyebrow "Worked example · O → Q"',
  );
});

test('OrbitDetailCard: branching caption uses V3.1 §13 "stored-output updates / contributes N to alpha" phrase (Part 6)', () => {
  const src = read(COMPONENTS, 'branchingViews/OrbitDetailCard.jsx');
  // V3.1 §13 (supersedes Part 6): "1 representative product → N stored-output
  // updates. This row contributes N to alpha." for the branching case.
  assert.ok(
    src.includes('stored-output update'),
    'OrbitDetailCard must contain the V3.1 §13 branching caption phrase "stored-output update(s)"',
  );
  assert.ok(
    src.includes('This row contributes'),
    'OrbitDetailCard must contain the V3.1 §13 branching caption phrase "This row contributes ... to alpha"',
  );
});

test('OrbitDetailCard: non-branching caption matches V3.1 §13 "1 representative product → 1 stored-output update." (Part 6)', () => {
  const src = read(COMPONENTS, 'branchingViews/OrbitDetailCard.jsx');
  // V3.1 §13 (supersedes Part 6 empty-cell wording): non-branching rows
  // (and empty-cell hover) now read "1 representative product → 1 stored-output
  // update. This row contributes 1 to alpha."
  assert.ok(
    src.includes('1 representative product → 1 stored-output update.'),
    'OrbitDetailCard must contain the V3.1 §13 non-branching caption verbatim',
  );
  assert.ok(
    src.includes('This row contributes 1 to alpha.'),
    'OrbitDetailCard must contain the V3.1 §13 non-branching tail "This row contributes 1 to alpha."',
  );
});

// ── Part 6: Wreath and sigma-loop modal ───────────────────────────────────

test('WreathStructureView: modal title reads "All row moves in the candidate wreath space" (V3.1 Part 6)', () => {
  const src = read(COMPONENTS, 'WreathStructureView.jsx');
  assert.match(
    src,
    /All row moves in the candidate wreath space/,
    'WreathStructureView modal title must be the V3.1 phrase',
  );
});

test('WreathStructureView: modal body reads V3.1 "candidate space and the matching relabeling test" (Part 6)', () => {
  const src = read(COMPONENTS, 'WreathStructureView.jsx');
  assert.match(
    src,
    /This modal shows row moves from the candidate space and the matching relabeling test that decides whether each move is accepted into the detected pointwise product group\./,
    'WreathStructureView modal body must match V3.1 Part 6 modal body verbatim',
  );
});

test('WreathStructureView: visible prose contains "Candidate row moves" (V3.1 Part 6 wreath visible prose)', () => {
  const src = read(COMPONENTS, 'WreathStructureView.jsx');
  assert.match(
    src,
    /Candidate row moves come from/,
    'WreathStructureView visible prose must start with "Candidate row moves come from"',
  );
});

test('WreathStructureView: outcome label uses "✓ kept in G_pt" (V3.1 Part 6 outcome labels)', () => {
  const src = read(COMPONENTS, 'WreathStructureView.jsx');
  assert.match(
    src,
    /✓ kept in G_pt/,
    'WreathStructureView outcome label must read "✓ kept in G_pt" per V3.1',
  );
});

// ── Part 6: Classification-tree tooltip structure (CaseBadge) ─────────────

test('CaseBadge: tooltip section header reads "Applies when:" (V3.1 Part 6 tooltip template)', () => {
  const src = read(COMPONENTS, 'CaseBadge.jsx');
  assert.match(
    src,
    /Applies when:/,
    'CaseBadge must use "Applies when:" per V3.1 classification-tree tooltip template',
  );
});

test('CaseBadge: tooltip contains "Counts: filled O → Q cells for this component" (V3.1 Part 6)', () => {
  const src = read(COMPONENTS, 'CaseBadge.jsx');
  assert.match(
    src,
    /Counts: filled O → Q cells for this component/,
    'CaseBadge must contain the V3.1 "Counts:" line from the tooltip template',
  );
});

test('CaseBadge: long regime tooltips are capped and positioned without translateY clipping', () => {
  const src = read(COMPONENTS, 'CaseBadge.jsx');
  assert.match(src, /maxHeight:\s*tooltipPos\.maxHeight/);
  assert.match(src, /overflowY:\s*'auto'/);
  assert.match(src, /transform:\s*'translateX\(-50%\)'/);
  assert.match(src, /const tooltipWidth = Math\.min\(\s*TOOLTIP_WIDTH,\s*vw - \(VIEWPORT_PADDING \* 2\)\s*\)/);
  assert.match(src, /tooltipWidth \/ 2 \+ VIEWPORT_PADDING/);
  assert.match(src, /className="pointer-events-auto fixed z-\[9999\]/);
  assert.match(src, /role="tooltip"/);
  assert.match(src, /onPointerLeave=\{scheduleClose\}/);
  assert.doesNotMatch(src, /translateX\(-50%\) translateY\(-100%\)/);
});

// ── Part 6: Component summary title-attribute tooltips (ComponentCostView) ─

test('ComponentCostView: Dense tuple title uses V3.1 "full assignment-space product of n_ell" (Part 6)', () => {
  const src = read(COMPONENTS, 'ComponentCostView.jsx');
  assert.match(
    src,
    /Dense tuple count = .+, the full assignment-space product of n_ell before any symmetry collapse\./,
    'ComponentCostView M_a Dense tuple title must match V3.1 Part 6 template',
  );
});

test('ComponentCostView: Dense baseline title matches V3.1 without trailing count (Part 6)', () => {
  const src = read(COMPONENTS, 'ComponentCostView.jsx');
  assert.match(
    src,
    /Dense baseline: one update per full assignment before quotienting by the pointwise group\./,
    'ComponentCostView alpha_a Dense baseline title must match V3.1 Part 6 verbatim',
  );
  // Must NOT include the old parenthetical that V3.1 dropped
  assert.doesNotMatch(
    src,
    /Dense baseline: one update per full assignment before quotienting by the pointwise group \(/,
    'ComponentCostView Dense baseline must not have trailing "(X total assignments)" parenthetical',
  );
});

test('ComponentCostView: Mult savings title uses "M_a" not "Mₐ" (V3.1 Part 6)', () => {
  const src = read(COMPONENTS, 'ComponentCostView.jsx');
  assert.match(
    src,
    /Mult savings: dense M_a would be /,
    'ComponentCostView Mult savings title must use M_a per V3.1 Part 6',
  );
});

test('ComponentCostView: Acc savings title uses "alpha_a" not "αₐ" (V3.1 Part 6)', () => {
  const src = read(COMPONENTS, 'ComponentCostView.jsx');
  assert.match(
    src,
    /Acc savings: dense alpha_a would be /,
    'ComponentCostView Acc savings title must use alpha_a per V3.1 Part 6',
  );
});

// ── Part 8: Builder UI copy ───────────────────────────────────────────────

test('ExampleChooser: dimension knob title matches V3.1 Part 8 "Per-label dimension. This changes..." (Part 8)', () => {
  const src = read(COMPONENTS, 'ExampleChooser.jsx');
  assert.match(
    src,
    /Per-label dimension\. This changes assignment-grid sizes and count values, but not the structural classification path unless domain compatibility changes\./,
    'ExampleChooser dimension knob title must match V3.1 Part 8 verbatim',
  );
});
