import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const SIGMA_LOOP_SRC = readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/SigmaLoop.jsx', import.meta.url),
  'utf8',
);
const SIGMA_LOOP_STYLES = readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/styles.css', import.meta.url),
  'utf8',
);
const APP_SRC = readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
  'utf8',
);

test('SigmaLoop renders its intro through the shared inline-math path so sigma and pi colorize', () => {
  assert.match(SIGMA_LOOP_SRC, /InlineMathText/);
  assert.match(SIGMA_LOOP_SRC, /notationLatex\('sigma_row_move'\)/);
  assert.match(SIGMA_LOOP_SRC, /notationLatex\('pi_relabeling'\)/);
  // The intro paragraph also carries an `order-1` flexbox class so the
  // sibling animation panel can be visually promoted to render directly
  // beneath it (see SigmaLoop.jsx — the parent .sigma-loop is now a flex
  // column to support that visual ordering).
  assert.match(SIGMA_LOOP_SRC, /className="explorer-support-prose mb-2 order-1"/);
  assert.doesNotMatch(SIGMA_LOOP_SRC, /Each σ is a wreath element; each accepted pair shows a row move together with its matching relabeling π\./);
});

test('sigma-loop panel title uses the shared inline-math path so sigma and pi colorize in the heading', () => {
  assert.match(APP_SRC, /InlineMathText/);
  assert.match(APP_SRC, /notationLatex\('sigma_row_move'\)/);
  assert.match(APP_SRC, /notationLatex\('pi_relabeling'\)/);
  assert.doesNotMatch(APP_SRC, />\s*σ-Loop &amp; π Detection\s*</);
});

test('sigma-loop compact chips render latex in inherited control color', () => {
  assert.match(SIGMA_LOOP_SRC, /<Latex math=\{String\.raw`\\sigma`\} inheritColor \/>/);
  assert.match(SIGMA_LOOP_SRC, /<Latex math=\{String\.raw`\\pi`\} inheritColor \/>/);
  assert.match(SIGMA_LOOP_SRC, /pair-chip pair-valid/);
  assert.match(SIGMA_LOOP_SRC, /pair-chip pair-invalid/);
});

// Per user feedback ("we do not show all the identified Sigmas") the σ list
// is now fully inlined — both valid-pair and rejected-pair chips render in
// `.pair-chips-grid` blocks. The earlier `.valid-toggle` and
// `.rejected-toggle` overflow buttons (which gated extra valid pairs behind
// "▸ N more" and rejected σ's behind a modal trigger) were removed.
test('valid sigma-pair chips use the primary coral accent (overflow toggles removed)', () => {
  assert.match(SIGMA_LOOP_STYLES, /\.pair-chip\.pair-valid \{[^}]*var\(--coral\)/s);
  assert.match(SIGMA_LOOP_STYLES, /\.pair-pi \{[^}]*var\(--coral\)/s);
  assert.doesNotMatch(SIGMA_LOOP_STYLES, /\.pair-chip\.pair-valid \{[^}]*var\(--success\)/s);
  assert.doesNotMatch(SIGMA_LOOP_STYLES, /\.pair-pi \{[^}]*var\(--success\)/s);
  // The bulk-overflow toggle elements no longer ship in the JSX.
  assert.doesNotMatch(SIGMA_LOOP_SRC, /valid-toggle/,
    'The "▸ N more (σ, π) pairs" toggle was removed; all valid pairs render inline.');
  assert.doesNotMatch(SIGMA_LOOP_SRC, /rejected-toggle/,
    'The "▸ N rejected σ\'s" toggle was removed; all rejected σ\'s render inline.');
});
