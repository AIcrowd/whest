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

test('AlgorithmAtAGlance renders above the Act 1 setup section in the app', () => {
  const appSrc = readFileSync(
    resolve(__dirname, 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx'),
    'utf-8',
  );
  const preambleIdx = appSrc.indexOf('<AlgorithmAtAGlance');
  const setupIdx = appSrc.indexOf('id={EXPLORER_ACTS[0].id}');
  assert.ok(preambleIdx >= 0, 'AlgorithmAtAGlance must be rendered');
  assert.ok(setupIdx >= 0, 'Act 1 setup section must still be rendered');
  assert.ok(
    preambleIdx < setupIdx,
    'AlgorithmAtAGlance must render above the Act 1 setup section',
  );
});

test('Preamble lays Einsum notation and the code side-by-side on wide viewports', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // The two-column grid responsive class is the structural contract.
  assert.match(src, /lg:grid-cols-2/);
  // Both columns are present.
  assert.match(src, /EinsumIntroColumn/);
  assert.match(src, /MentalFrameworkColumn/);
  // MentalFrameworkCode is still imported (it is the code card).
  assert.match(src, /import MentalFrameworkCode/);
});

test('Einsum column uses a color-coded KaTeX formula for a non-trivial chain', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // KaTeX \textcolor is the symbolic way to colour label tokens.
  assert.match(src, /\\textcolor\{/);
  // Summed labels are coloured in coral (--primary).
  assert.match(src, /#F0524D/);
  assert.match(src, /\\sum_\{/);
  // The example is substantially larger than a 2-operand matmul — 4 operands
  // A, B, C, D with 5 labels (i, j, k, l, m) so the reader sees why cost
  // explodes without symmetry.
  assert.match(src, /CHAIN_FORMULA/);
  assert.match(src, /A\[/);
  assert.match(src, /B\[/);
  assert.match(src, /C\[/);
  assert.match(src, /D\[/);
  // Legend introduces "summed" and "free" by colour.
  assert.match(src, /summed/);
  assert.match(src, /free/);
  // No hand-SVG figure remains in this file.
  assert.doesNotMatch(src, /<svg/);
});

test('Einsum column transitions to symmetry at the bottom', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  assert.match(src, /Where symmetry enters/);
  // The transition introduces G, orbits, and ties back to μ and α.
  assert.match(src, /group G/i);
  assert.match(src, /orbit/i);
  assert.match(src, /\\mu/);
  assert.match(src, /\\alpha/);
});

test('μ / α definitions live in the code step headers — no separate callouts in the preamble', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // μ and α still appear somewhere in the preamble (in the worked example).
  assert.match(src, /\\mu/);
  assert.match(src, /\\alpha/);
  // But the old standalone "Two kinds of work" NarrativeCallout pair must be gone.
  assert.doesNotMatch(src, /Two kinds of work/);
  assert.doesNotMatch(src, /NarrativeCallout/);
});

test('MentalFrameworkCode renders a short, natural-reading pseudocode (not the 20-line teaching model)', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  // Local LINES constant defines the compact pseudocode; the long-form
  // buildMentalModelLines() is no longer used by the preamble card.
  assert.match(src, /const LINES = \[/);
  assert.doesNotMatch(src, /buildMentalModelLines/);
  // The two core lines of the algorithm.
  assert.match(src, /for rep in RepSet:/);
  assert.match(src, /for out in Outs\(rep\)/);
  // Accumulation line now surfaces the symmetry coefficient so the reader
  // sees that one product can land with different weights in different bins.
  assert.match(src, /R\[out\] \+= coeff\(rep, out\) \* base_val/);
  // Syntax colouring still reuses the shared tokenizer so the palette stays
  // consistent with the rest of the site.
  assert.match(src, /tokenizePseudocodeLine/);
});

test('MentalFrameworkCode opens with the slogan above the code', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  // Slogan is the whole algorithm in two sentences; placed above the code
  // grid so the reader grasps the shape before diving into Python.
  assert.match(src, /Compute each distinct product ONCE/);
  assert.match(src, /Spread it to every output cell it contributes to/);
  const sloganIdx = src.indexOf('Compute each distinct product ONCE');
  const gridIdx = src.indexOf('aria-label="Symmetry-aware contraction pseudocode"');
  assert.ok(sloganIdx < gridIdx, 'slogan must render before the code grid');
});

test('MentalFrameworkCode uses inline annotation rows at the correct indent', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  // The AnnotationRow renderer exists and the two annotation entries sit in
  // LINES at indent 4 (inside `for rep`) and indent 8 (inside `for out`).
  assert.match(src, /function AnnotationRow/);
  assert.match(src, /kind:\s*'annotation',\s*step:\s*'mult',\s*indent:\s*4/);
  assert.match(src, /kind:\s*'annotation',\s*step:\s*'acc',\s*indent:\s*8/);
  // The old above-code StepPanel component is gone.
  assert.doesNotMatch(src, /<StepPanel/);
  // Code rows no longer carry coloured left-rules (grouping is carried by
  // the inline annotation rows instead).
  assert.doesNotMatch(src, /border-l-\[3px\]/);
});

test('MentalFrameworkCode step metadata names the Feynman-style labels and colors', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  // Step 1 · multiply once (coral) and Step 2 · accumulate many (amber).
  // μ and α themselves are introduced in the Counting Convention block
  // below the code, so the inline step labels stay terse.
  assert.match(src, /kicker:\s*'Step 1'/);
  assert.match(src, /kicker:\s*'Step 2'/);
  assert.match(src, /label:\s*'multiply once'/);
  assert.match(src, /label:\s*'accumulate many'/);
  assert.match(src, /color:\s*'text-primary'/);
  assert.match(src, /color:\s*'text-amber-700'/);
  // Heavy vertical-bar glyph marks each annotation row.
  assert.match(src, /┃/);
});

test('MentalFrameworkCode renders the Counting Convention panel introducing μ and α', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  assert.match(src, /Counting convention/);
  assert.match(src, /Multiplication Cost \(μ\)/);
  assert.match(src, /Accumulation Cost \(α\)/);
  // Anchors into the code above — names both lines it talks about.
  assert.match(src, /base_val/);
  assert.match(src, /R\[out\] \+= coeff/);
  // Panel sits inside the figure as its own bottom band with a soft
  // top border; each column flows at natural height (items-start on the
  // parent grid), so no stretch-to-match CSS is needed here.
  assert.match(src, /border-t border-stone-200\/70/);
});

test('MentalFrameworkCode uses Feynman-friendly comments for RepSet, Outs(rep) and coeff', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  // Three separate one-line comments, in plain language.
  assert.match(src, /# RepSet\s+— unique input tuples to multiply/);
  assert.match(src, /# Outs\(rep\)\s+— unique output bins this product lands in/);
  assert.match(src, /# coeff\s+— how many orbit copies land on that bin/);
  // No jargon-heavy "representatives we walk" phrasing anywhere.
  assert.doesNotMatch(src, /representatives we walk/);
});

test('Preamble no longer embeds a worked example — the payoff lives inside the explorer', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // The old "A worked example" block is gone.
  assert.doesNotMatch(src, /A worked example/);
  // The specific numeric payoff for Tr(A·A) has been removed from the preamble.
  assert.doesNotMatch(src, /Tr\(A·A\)/);
  assert.doesNotMatch(src, /ExplorerMetricCard/);
});

test('Obsolete preamble components (SVG figure, margin bar) are deleted', () => {
  assert.throws(
    () => readFileSync(resolve(__dirname, COMPONENTS, 'EinsumIntroFigure.jsx')),
    /ENOENT/,
    'EinsumIntroFigure.jsx must be removed — replaced by color-coded KaTeX formula',
  );
  assert.throws(
    () => readFileSync(resolve(__dirname, COMPONENTS, 'CodeMarginBar.jsx')),
    /ENOENT/,
    'CodeMarginBar.jsx must be removed — replaced by step-header rows + row tints',
  );
});

test('Orphan PseudocodeRail component and modal state helper are still gone', () => {
  assert.throws(
    () => readFileSync(resolve(__dirname, COMPONENTS, 'PseudocodeRail.jsx')),
    /ENOENT/,
  );
  assert.throws(
    () =>
      readFileSync(
        resolve(
          __dirname,
          'components/symmetry-aware-einsum-contractions/lib/mentalModelState.js',
        ),
      ),
    /ENOENT/,
  );
});
