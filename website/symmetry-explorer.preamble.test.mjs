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
const readMainContent = (name) =>
  readFileSync(
    resolve(__dirname, 'components/symmetry-aware-einsum-contractions/content/main', name),
    'utf-8',
  );

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
  assert.match(src, /A tensor operation, written as one equation/);
  assert.doesNotMatch(src, /A tensor operation, written as one formula/);
  // Both columns are present.
  assert.match(src, /EinsumIntroColumn/);
  assert.match(src, /MentalFrameworkColumn/);
  // MentalFrameworkCode is still imported (it is the code card).
  assert.match(src, /import MentalFrameworkCode/);
  assert.match(src, /<MentalFrameworkColumn example=\{example\} \/>/);
  assert.match(src, /<MentalFrameworkCode example=\{example\} \/>/);
});

test('Einsum column uses a parametric exact-einsum lead-in followed by a color-coded expanded form', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  assert.match(src, /buildSection1ExampleView/);
  assert.match(src, /buildSection1ExampleView\(example,\s*\{/);
  assert.match(src, /freeLabelColor:\s*explorerThemeColor\(explorerThemeId,\s*'hero'\)/);
  assert.match(src, /summedLabelColor:\s*explorerThemeColor\(explorerThemeId,\s*'summedSide'\)/);
  assert.match(src, /view\.exactEinsumText/);
  assert.match(src, /Latex display math=\{view\.expandedEquationLatex\}/);
  assert.match(src, /mt-2\.5 flex justify-center text-\[19px\]/);
  assert.match(src, /operandCount/);
  assert.match(src, /labelCount/);
  assert.match(src, /The summed labels/);
  assert.match(src, /survive as the axes of \$R\$/);
  assert.match(src, /Declared symmetries:/);
  assert.match(src, /The dense direct grid has \$\$\{view\.denseGridScalingLatex\}\$ assignments before symmetry is used\./);
  assert.doesNotMatch(src, /DENSE_SCALING/);
  assert.doesNotMatch(src, /What this fixes/);
  assert.doesNotMatch(src, /CHAIN_FORMULA/);
  assert.doesNotMatch(src, /Exact einsum/);
  assert.doesNotMatch(src, /Example — the selected contraction/);
  assert.match(src, /rounded-2xl border border-stone-200 bg-white px-5 py-6/);
  assert.doesNotMatch(src, /A\[/);
  assert.doesNotMatch(src, /B\[/);
  assert.doesNotMatch(src, /C\[/);
  assert.doesNotMatch(src, /D\[/);
  assert.ok(
    src.indexOf('view.expandedEquationLatex') < src.indexOf('The summed labels'),
    'expanded contraction should render before the prose explanation',
  );
  assert.ok(
    src.indexOf('The summed labels') < src.indexOf('<ColorLegend freeLabelColor={freeLabelColor} summedLabelColor={summedLabelColor} />'),
    'legend should stay inside the same einsum surface after the explanatory prose',
  );
  assert.match(src, /textAlign: 'justify'/);
});

test('Preamble label legend uses the same theme roles as the interaction graph', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');

  assert.match(src, /const explorerThemeId = getActiveExplorerThemeId\(\)/);
  assert.match(src, /const freeLabelColor = explorerThemeColor\(explorerThemeId, 'hero'\)/);
  assert.match(src, /const summedLabelColor = explorerThemeColor\(explorerThemeId, 'summedSide'\)/);
  assert.match(src, /backgroundColor: summedLabelColor/);
  assert.match(src, /backgroundColor: freeLabelColor/);
  assert.doesNotMatch(src, /backgroundColor: notationColor\('w_summed'\)/);
});

test('Einsum column transitions to symmetry at the bottom', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  assert.match(src, /import EditorialCallout/);
  assert.match(src, /Where symmetry enters/);
  assert.match(src, /title="Not every product is distinct"/);
  assert.match(src, /mainPreamble\.slots\.calloutBodyBeforeGroup/);
  assert.match(src, /mainPreamble\.slots\.calloutBodyBetweenGroupAndOrbits/);
  assert.match(src, /mainPreamble\.slots\.calloutBodyAfterOrbits/);
  assert.match(src, /mainPreamble\.slots\.calloutFooter/);
});

test('AlgorithmAtAGlance sources its narrative prose from the content registry', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  const preambleSrc = readMainContent('preamble.ts');

  assert.match(src, /import\s+\{\s*default as renderProseBlocks\s*\}\s+from\s+'\.\.\/content\/renderProseBlocks\.jsx'/);
  assert.match(src, /import\s+mainPreamble\s+from\s+'\.\.\/content\/main\/preamble\.ts'/);
  assert.match(src, /function renderSingleProseBlock\(blocks = \[\], keyPrefix = 'main-prose-block'\)/);
  assert.match(src, /renderSingleProseBlock\(mainPreamble\.slots\.calloutFooter,\s*'symmetry-callout-footer'\)/);
  assert.match(src, /mainPreamble\.slots\.einsumIntroBeforeSummed/);
  assert.match(src, /mainPreamble\.slots\.mentalFrameworkIntroBeforeRepSet/);
  assert.match(src, /mainPreamble\.slots\.handoffBeforeSectionLink/);
  assert.match(preambleSrc, /einsumIntroBeforeSummed/);
  assert.match(preambleSrc, /mentalFrameworkIntroBeforeRepSet/);
  assert.match(preambleSrc, /calloutBodyBeforeGroup/);
  assert.match(preambleSrc, /handoffBeforeSectionLink/);
  assert.doesNotMatch(src, /function renderEinsumIntro/);
  assert.doesNotMatch(src, /function renderMentalFrameworkIntro/);
  assert.doesNotMatch(src, /function renderCalloutBody/);
  assert.doesNotMatch(src, /\.replace\('Section 1'/);
  assert.doesNotMatch(src, /\.split\('summed over'\)/);
  assert.doesNotMatch(src, /\.split\('RepSet'\)/);
  assert.doesNotMatch(src, /\.split\('group \$G\$'\)/);
  assert.doesNotMatch(src, /Every index label that appears on an input but not on the output is/);
  assert.doesNotMatch(src, /If several operands are identical or individually symmetric/);
  assert.doesNotMatch(src, /The rest of this page shows how the explorer detects the symmetry group/);
});

test('main-page section references are rendered as links, not plain text labels', () => {
  const preambleSrc = readComponent('AlgorithmAtAGlance.jsx');

  assert.match(preambleSrc, /<SectionReferenceLink href="#setup">Section 1<\/SectionReferenceLink>/);
  assert.doesNotMatch(preambleSrc, /Start with <strong className="font-semibold">Section 1<\/strong> below/);
});

test('μ / α definitions live in the code step headers — no separate callouts in the preamble', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // μ and α still appear somewhere in the preamble via registry-backed footer prose.
  assert.match(src, /mainPreamble\.slots\.calloutFooter/);
  // But the old standalone "Two kinds of work" NarrativeCallout pair must be gone.
  assert.doesNotMatch(src, /Two kinds of work/);
  assert.match(src, /EditorialCallout/);
  assert.doesNotMatch(src, /NarrativeCallout/);
});

test('MentalFrameworkCode renders a short, natural-reading pseudocode (not the 20-line teaching model)', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  // Local BASE_LINES define the compact pseudocode and buildLines(example)
  // parametrizes the inline comments from the current contraction.
  assert.match(src, /const BASE_LINES = \[/);
  assert.match(src, /function buildLines\(example\)/);
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
  // Slogan is now a compact two-part editorial cue above the code grid.
  assert.match(src, /Multiply Once/);
  assert.match(src, /Accumulate Many/);
  const sloganIdx = src.indexOf('Multiply Once');
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
  // Step 1 · multiply once and Step 2 · accumulate many both use coral.
  // μ and α themselves are introduced in the Counting Convention block
  // below the code, so the inline step labels stay terse.
  assert.match(src, /kicker:\s*'Step 1'/);
  assert.match(src, /kicker:\s*'Step 2'/);
  assert.match(src, /label:\s*'multiply once'/);
  assert.match(src, /label:\s*'accumulate many'/);
  assert.match(src, /color:\s*'text-\[#ef5a4c\]'/);
  // Heavy vertical-bar glyph marks each annotation row.
  assert.match(src, /┃/);
});

test('MentalFrameworkCode renders the Counting Convention panel introducing μ and α', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  assert.match(src, /Counting convention/);
  assert.match(src, /The number of representative products is/);
  assert.match(src, /<Latex math=\{notationLatex\('m_total'\)\} \/>/);
  assert.match(src, /<Latex[\s\S]*notationLatex\('mu_total'\)[\s\S]*notationLatex\('k_operands'\)[\s\S]*notationLatex\('m_total'\)/);
  assert.match(src, /<Latex math=\{notationLatex\('alpha_total'\)\} \/>/);
  assert.match(src, /<Latex math=\{String\.raw`\$\{notationLatex\('mu_total'\)\} \+ \$\{notationLatex\('alpha_total'\)\}`\} \/>/);
  // Anchors into the code above — names both lines it talks about.
  assert.match(src, /base_val/);
  assert.match(src, /R\[out\] \+= coeff/);
  // Panel sits inside the figure as its own bottom band with a soft
  // top border; mt-auto anchors it to the bottom of the figure so that
  // when the parent column stretches to match the left side's height
  // (items-stretch on the grid), the Counting Convention band stays
  // glued to the bottom instead of floating in the middle.
  assert.match(src, /border-t border-stone-200\/70/);
  assert.match(src, /mt-auto border-t border-stone-200\/70/);
  assert.match(src, /bg-gray-50/);
});

test('Preamble columns are forced to equal height so both bottoms align', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  // items-stretch on the parent grid makes both columns the same height;
  // the shorter side grows a spacer / stretched figure to fill.
  assert.match(src, /items-stretch/);
  assert.doesNotMatch(src, /items-start/);
  // Left column is a flex column so the callout can be anchored at the
  // bottom via a flex-1 spacer above it.
  assert.match(src, /flex h-full flex-col/);
  // Right column's figure wrapper is flex-1 so the MentalFrameworkCode
  // figure fills the remaining vertical space.
  assert.match(src, /mt-6 flex flex-1 flex-col/);
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

test('MentalFramework column renders RepSet, Outs(rep), and coeff(rep, out) as soft inline-code chips', () => {
  const src = readComponent('AlgorithmAtAGlance.jsx');
  assert.match(src, /className="explorer-inline-code">RepSet<\/code>/);
  assert.match(src, /className="explorer-inline-code">Outs\(rep\)<\/code>/);
  assert.match(src, /className="explorer-inline-code">coeff\(rep, out\)<\/code>/);
  assert.doesNotMatch(src, /<code className="font-mono">RepSet<\/code>/);
});

test('MentalFrameworkCode parameterizes inline comments from the active example', () => {
  const src = readComponent('MentalFrameworkCode.jsx');
  assert.match(src, /function normalizeExampleForPseudocode/);
  assert.match(src, /function buildBaseValueComment/);
  assert.match(src, /function buildReduceComment/);
  assert.match(src, /base_val = product_at\(rep\)\$\{baseValueComment\}/);
  assert.match(src, /R\[out\] \+= coeff\(rep, out\) \* base_val\$\{reduceComment\}/);
  assert.match(src, /across all contracted indices/);
  assert.match(src, /# R\[/);
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
