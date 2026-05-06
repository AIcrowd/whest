// website/symmetry-explorer.burnside-card.test.mjs
//
// Source-grep coverage for V3.1 §10 — C10 Burnside Card on
// MultiplicationCostCard. The Burnside Table makes the |Fix(g)| sum
// concrete: one row per element of G_a, with cycle-decomposition,
// fixed-assignments factor, and the M_a = (1/|G_a|) Σ_g |Fix(g)|
// average rendered below.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC_PATH = resolve(
  __dirname,
  'components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx',
);
const SRC = readFileSync(SRC_PATH, 'utf-8');

test('BurnsideTable renders the three V3.1 column headers', () => {
  // V3.1 §10 mandates these exact column labels:
  // | Group element | Label cycles | Fixed assignments | Contribution |
  assert.match(SRC, />Group element</);
  assert.match(SRC, />Label cycles</);
  assert.match(SRC, />Fixed assignments</);
  assert.match(SRC, />Contribution</);
});

test('BurnsideTable renders the M_a Burnside-average formula', () => {
  // The formula must include `M_a = (1/|G_a|) Σ_g |Fix(g)|` form, with
  // the LaTeX \frac{1}{|G_a|} normalisation factor and the Fix(g) sum.
  assert.match(SRC, /M_a\s*\\;=\\;/);
  assert.match(SRC, /\\frac\{1\}\{\|G_a\|\}/);
  assert.match(SRC, /\\sum_\{g \\in G_a\}/);
  assert.match(SRC, /\|\\mathrm\{Fix\}\(g\)\|/);
});

test('BurnsideTable surfaces the V3.1 verbatim tooltip', () => {
  // The polish spec requires this exact phrasing:
  // "Burnside counts products, not updates."
  assert.match(SRC, /Burnside counts products, not updates\./);
  // It must be reachable both as a tooltip (title) and for screen readers
  // (aria-label) so the contrast against α (which counts updates) lands
  // for every input modality.
  assert.match(
    SRC,
    /title="Burnside counts products, not updates\."/,
  );
  assert.match(
    SRC,
    /aria-label="Burnside counts products, not updates\."/,
  );
});

test('BurnsideTable rows wire onMouseEnter/onFocus to the hoveredLabels bus', () => {
  // Each row must write the row's cycle labels to the shared
  // hoveredLabels bus on hover/focus, and clear it on leave/blur.
  assert.match(SRC, /onMouseEnter=\{\(\) => writeHover\(new Set\(row\.flatLabels\)\)\}/);
  assert.match(SRC, /onMouseLeave=\{\(\) => writeHover\(null\)\}/);
  assert.match(SRC, /onFocus=\{\(\) => writeHover\(new Set\(row\.flatLabels\)\)\}/);
  assert.match(SRC, /onBlur=\{\(\) => writeHover\(null\)\}/);
  // The card must accept onHoveredLabelsChange as a prop and forward it.
  assert.match(SRC, /onHoveredLabelsChange\s*=\s*null/);
  assert.match(SRC, /onHoveredLabelsChange=\{onHoveredLabelsChange\}/);
});

test('BurnsideTable rows are keyboard-focusable', () => {
  // Each <tr> must be tabbable so the hover/focus bus also fires for
  // keyboard users. The dismiss icon next to the table title also
  // carries tabIndex=0 so its tooltip is keyboard-reachable.
  const trBlock = SRC.match(/<tr[\s\S]*?key=\{`burnside-row-[\s\S]*?<\/tr>/);
  assert.ok(trBlock, 'expected a Burnside <tr> block');
  assert.match(trBlock[0], /tabIndex=\{0\}/);
});

test('BurnsideTable exposes a stable data-testid for integration tests', () => {
  // The table has a data-testid keyed off the component index so harness
  // tests can target a specific component's Burnside table.
  assert.match(SRC, /data-testid=\{testId\}/);
  assert.match(SRC, /burnside-table-\$\{testIdSuffix\}/);
  assert.match(SRC, /testIdSuffix=\{String\(i\)\}/);
});

test('BurnsideTable computes |Fix(g)| as a product over cycles of n_{label}', () => {
  // The Burnside sum's per-element contribution is ∏_{cycle c} n_{label_in_c}.
  // The card derives both a symbolic form ("n_{i} \cdot n_{k}") and a
  // numeric form (using comp.sizes); both must be present in the source.
  assert.match(SRC, /g\.fullCyclicForm\(\)/);
  assert.match(SRC, /n_\{\$\{lbl\}\}/);
  assert.match(SRC, /sizes\[c\[0\]\]/);
});
