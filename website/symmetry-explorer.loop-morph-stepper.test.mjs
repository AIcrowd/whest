// website/symmetry-explorer.loop-morph-stepper.test.mjs
//
// Source-grep coverage for V3.1 §7 — C07 LoopMorphStepper (NEW).
//
// LoopMorphStepper is the dense → representative loop morph: two pseudocode
// columns plus a 5-step stepper that walks the reader through the
// transformation (dense assignments → group → choose representative →
// project → accumulate). These tests pin the contract that survives future
// refactoring; behavior is exercised by the Storybook stories.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const COMPONENT_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/LoopMorphStepper.jsx',
);
const APP_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx',
);
const STORIES_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/LoopMorphStepper.stories.jsx',
);

test('LoopMorphStepper.jsx exists and exports default', () => {
  assert.ok(existsSync(COMPONENT_PATH), 'expected LoopMorphStepper.jsx to exist');
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Default export of the LoopMorphStepper component.
  assert.match(src, /export default LoopMorphStepper/);
  // Component is declared as a function so the prop / behavior contract is
  // greppable.
  assert.match(src, /function LoopMorphStepper\(/);
});

test('LoopMorphStepper renders both dense and representative pseudocode strings', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Dense-side pseudocode — verbatim from the V3.1 §7 spec.
  assert.match(src, /for full_assignment in X:/);
  assert.match(src, /R\[project\(full_assignment\)\] \+= product_at\(full_assignment\)/);
  // Representative-side pseudocode — verbatim from the V3.1 §7 spec.
  assert.match(src, /for rep in RepSet:/);
  assert.match(src, /base_val = product_at\(rep\)/);
  assert.match(src, /for out_rep in Outs\(rep\):/);
  assert.match(src, /R\[out_rep\] \+= coeff\(rep, out_rep\) \* base_val/);
});

test('LoopMorphStepper exposes Prev / Next stepper controls', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Prev / Next button rendering.
  assert.match(src, /data-testid="loop-morph-stepper-prev"/);
  assert.match(src, /data-testid="loop-morph-stepper-next"/);
  // Visible button labels (verbatim).
  assert.match(src, /← Prev/);
  assert.match(src, /Next →/);
  // Buttons wired to step advance/retreat helpers.
  assert.match(src, /onClick=\{goPrev\}/);
  assert.match(src, /onClick=\{goNext\}/);
  // aria-labels on the stepper buttons so SR users get a clear label.
  assert.match(src, /aria-label="Previous step"/);
  assert.match(src, /aria-label="Next step"/);
});

test('LoopMorphStepper declares 5 steps in a STEPS array', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // STEPS array literal exists.
  assert.match(src, /const STEPS = \[/);
  // N_STEPS derived from STEPS — pins the count to 5 in a single source.
  assert.match(src, /const N_STEPS = STEPS\.length/);
  assert.match(src, /\/\/ 5/);
  // Step ids — verbatim from the V3.1 §7 spec, one per stage.
  assert.match(src, /id: 'dense'/);
  assert.match(src, /id: 'group'/);
  assert.match(src, /id: 'choose'/);
  assert.match(src, /id: 'project'/);
  assert.match(src, /id: 'accumulate'/);
  // Each step carries a caption describing the transformation.
  assert.match(src, /caption:/);
});

test('LoopMorphStepper has aria-label on the stepper region and aria-current on the active dot', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // The dot list (tablist) carries an aria-label so SR users hear the role.
  assert.match(src, /aria-label="Loop morph stepper position"/);
  // Each individual dot stamps aria-current when it is the active step.
  assert.match(src, /aria-current=\{isCurrent \? 'step' : undefined\}/);
  // Each dot also exposes aria-selected so the tablist semantics are honored.
  assert.match(src, /aria-selected=\{isCurrent\}/);
  // Step indicator is rendered with a stable testid for harnessing.
  assert.match(src, /data-testid="loop-morph-stepper-dots"/);
  assert.match(src, /data-testid="loop-morph-stepper-dot"/);
});

test('LoopMorphStepper gates motion behind prefers-reduced-motion', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Reduced-motion hook mirrors the TwoQuotientSchematic pattern.
  assert.match(src, /usePrefersReducedMotion/);
  // matchMedia query is queried against the prefers-reduced-motion preference.
  assert.match(src, /matchMedia\('\(prefers-reduced-motion: reduce\)'\)/);
  // When reduced, the morph transition collapses to instant (transition: none).
  assert.match(src, /reducedMotion \? 'none'/);
});

test('LoopMorphStepper wires arrow-key navigation', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // Arrow keys advance / retreat the stepper.
  assert.match(src, /e\.key === 'ArrowLeft'/);
  assert.match(src, /e\.key === 'ArrowRight'/);
  // Home / End jump to first / last.
  assert.match(src, /e\.key === 'Home'/);
  assert.match(src, /e\.key === 'End'/);
  // Container has an onKeyDown handler so the keys fire while focus is inside.
  assert.match(src, /onKeyDown=\{handleKeyDown\}/);
});

test('LoopMorphStepper exposes hover tokens for RepSet, Outs(rep), and coeff', () => {
  const src = readFileSync(COMPONENT_PATH, 'utf-8');
  // HOVER_TOKENS map lists the three identifiers from the V3.1 §7 spec.
  assert.match(src, /HOVER_TOKENS/);
  assert.match(src, /RepSet:/);
  assert.match(src, /Outs:/);
  assert.match(src, /coeff:/);
  // Each token exposes a stable testid + data-token-id so harness tests can
  // target a specific identifier.
  assert.match(src, /data-testid="loop-morph-stepper-token"/);
  assert.match(src, /data-testid="loop-morph-stepper-hover-strip"/);
  assert.match(src, /data-token-id=/);
  // Hover events are wired (mouse + focus parity for keyboard users).
  assert.match(src, /onMouseEnter=\{\(\) => onTokenHover\(p\.tokenId\)\}/);
  assert.match(src, /onFocus=\{\(\) => onTokenHover\(p\.tokenId\)\}/);
});

test('App no longer mounts LoopMorphStepper in §3 Projection', () => {
  const src = readFileSync(APP_PATH, 'utf-8');
  assert.doesNotMatch(src, /import LoopMorphStepper from '\.\/components\/LoopMorphStepper\.jsx';/);
  assert.doesNotMatch(src, /<LoopMorphStepper\b/);
});

test('LoopMorphStepper stories file declares ≥3 stories', () => {
  assert.ok(existsSync(STORIES_PATH), 'expected LoopMorphStepper.stories.jsx to exist');
  const src = readFileSync(STORIES_PATH, 'utf-8');
  // Default-export Storybook meta with the right title prefix.
  assert.match(src, /title:\s*'Section3\/LoopMorphStepper'/);
  // Count `export const X = {` blocks → must be at least 3.
  const matches = src.match(/export const \w+\s*=\s*\{/g) ?? [];
  assert.ok(matches.length >= 3, `expected ≥3 stories, got ${matches.length}`);
});
