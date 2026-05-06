import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

function countMatches(source, pattern) {
  const matches = source.match(pattern);
  return matches ? matches.length : 0;
}

test('symmetry explorer acts use prose-first intros and output framing', () => {
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );
  const introSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/SectionIntroProse.jsx', import.meta.url),
    'utf8',
  );
  const stylesSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/styles.css', import.meta.url),
    'utf8',
  );
  assert.match(appSource, /import SectionIntroProse from '\.\/components\/SectionIntroProse\.jsx';/);
  // V3.1 topology: 10 sections. Section 9 now folds its prose into
  // TotalCostView so the component recap can lead the section body.
  // The regex [0-9] covers all 10 acts (indices 0-9).
  // Section 9 (.introParagraphs) was already folded into TotalCostView; the
  // post-§10 user feedback retired the heavyweight ExplorerSectionCard
  // chrome for §10 too (Appendix Transition is now a lightweight gray-50
  // strip that visually parallels the in-card "Appendix note" block at
  // the end of §9). EXPLORER_ACTS[9].heading is still rendered (just no
  // longer as a section-card title prop), and .introParagraphs are
  // intentionally dropped from §10 since they duplicated the prose
  // already in the §9 appendix-note button.
  assert.ok(countMatches(appSource, /EXPLORER_ACTS\[[0-9]\]\.introParagraphs/g) >= 7);
  assert.match(appSource, /EXPLORER_ACTS\[9\]\.heading/,
    'EXPLORER_ACTS[9].heading must still be rendered somewhere in the §10 region');
  assert.match(introSource, /md:grid-cols-2/);
  assert.match(introSource, /balancedColumns/);
  assert.match(appSource, /<SectionIntroProse paragraphs=\{EXPLORER_ACTS\[0\]\.introParagraphs\} balancedColumns \/>/);
  assert.match(appSource, /function ProjectionIntroProse\(\{ paragraphs \}\)/);
  assert.match(appSource, /const PROJECTION_ALPHA_FORMULA = String\.raw`\\alpha = \\#\\\{/);
  assert.match(appSource, /<ProjectionIntroProse paragraphs=\{EXPLORER_ACTS\[2\]\.introParagraphs\} \/>/);
  assert.match(appSource, /projection-example-math overflow-x-auto whitespace-nowrap text-\[18px\]/);
  assert.match(appSource, /const PROJECTION_MATRIX_BRIDGE = 'The matrix below is just this incidence test drawn out/);
  assert.match(stylesSource, /\.projection-example-math \.katex-display/);
  assert.match(stylesSource, /margin: 0\.28rem 0/);
  assert.match(stylesSource, /font-size: 1\.12em/);
  assert.match(appSource, /Worked example — <span className="font-semibold">Cross S2<\/span>/);
  assert.match(appSource, /CROSS_S2_CONTRACTION_FORMULA/);
  assert.match(appSource, /CROSS_S2_OPERANDS_FORMULA/);
  assert.match(appSource, /CROSS_S2_ORBIT_FORMULA/);
  assert.match(appSource, /CROSS_S2_PRODUCT_EQUALITY_FORMULA/);
  assert.match(appSource, /CROSS_S2_PROJECTION_DESTINATIONS_FORMULA/);
  assert.match(appSource, /\\begin\{aligned\}A\[0,1\]B\[0\] &= 2 \\cdot 5 = 10\\\\ A\[1,0\]B\[0\] &= 2 \\cdot 5 = 10\\end\{aligned\}/);
  assert.match(appSource, /\(0,1,0\) \\mapsto R\[0,0\]/);
  assert.match(introSource, /textAlign:\s*'justify'/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[8\]\.supportingSentence/);
  assert.doesNotMatch(appSource, /<SectionIntroProse paragraphs=\{EXPLORER_ACTS\[8\]\.introParagraphs\} \/>/);
  assert.match(appSource, /<TotalCostView[\s\S]*componentCosts=\{componentCosts\}/);
  assert.equal(countMatches(appSource, /label="Interpretation"/g), 0);
  assert.equal(countMatches(appSource, /label="Approach"/g), 0);
  // "What this produces" callout in sections that have it.
  assert.ok(countMatches(appSource, /label="What this produces"/g) >= 8);
  assert.match(appSource, /Scope of the calculation/);
  assert.equal(countMatches(appSource, /tone="preamble"/g), 2);
  assert.equal(countMatches(appSource, /label="Candidate, not proof"/g), 0);
  assert.equal(countMatches(appSource, /label="What the model accepts"/g), 0);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[8\]\.why/);
  assert.doesNotMatch(appSource, /onOpenModalSection=\{/);
});

test('masthead lands the result and appendix transition exposes an A-E map', () => {
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );

  assert.match(appSource, /An interactive paper/);
  assert.match(appSource, /Counting symmetry-aware einsums/);
  assert.match(appSource, /Symmetry can turn many label assignments into one representative product/);
  assert.match(appSource, /reuse is only half the cost/);
  assert.match(appSource, /multiplication-chain work \$\\mu\$ and filled \$O \\to Q\$ updates \$\\alpha\$/);
  assert.match(appSource, /so \$\\mathrm\{Total\}=\\mu\+\\alpha\$\.`\}<\/InlineMathText>/);
  assert.doesNotMatch(appSource, /<Latex math=\{String\.raw`G_\{\\text\{pt\}\}`\} \/> product rows/);
  assert.match(appSource, /APPENDIX_MAP/);
  assert.match(appSource, /letter: 'A', title: 'Product-side certification', hash: '#appendix-section-1'/);
  assert.match(appSource, /letter: 'B', title: 'Classification-tree cases', hash: '#appendix-section-7'/);
  assert.match(appSource, /letter: 'C', title: 'Typed partition theorem', hash: '#appendix-section-6'/);
  assert.match(appSource, /letter: 'D', title: 'Completed-expression formal symmetry', hash: '#appendix-section-4'/);
  assert.match(appSource, /letter: 'E', title: 'Scope, assumptions, and exactness', hash: '#appendix-section-8'/);
  // Letter cards stop click propagation so a click inside one of them only
  // deep-links to its sub-section instead of also triggering the outer
  // appendix-note click-anywhere handler (which opens the appendix at the
  // top). Both code paths are atomically captured here.
  assert.match(appSource, /onClick=\{\(e\) => \{ e\.stopPropagation\(\); openAppendix\(item\.hash\); \}\}/);
});

test('"What this produces" callout uses shared vertical centering on desktop', () => {
  const calloutSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/NarrativeCallout.jsx', import.meta.url),
    'utf8',
  );

  assert.match(calloutSource, /sm:items-center/);
  assert.match(calloutSource, /sm:flex sm:items-center/);
  assert.doesNotMatch(calloutSource, /sm:items-start/);
});

test('preamble callouts collapse their top spacing when the label is omitted', () => {
  const calloutSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/NarrativeCallout.jsx', import.meta.url),
    'utf8',
  );

  assert.match(calloutSource, /const hasHeader = Boolean\(label \|\| title\);/);
  assert.match(calloutSource, /className=\{hasHeader \? 'mt-2 space-y-3' : 'space-y-3'\}/);
});

test('article route still wires the appendix modal while the main page stays appendix-free', () => {
  const appSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx', import.meta.url),
    'utf8',
  );
  const einsumGlanceCopy = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/content/main/einsumGlance.js', import.meta.url),
    'utf8',
  );

  assert.match(appSource, /<ExpressionLevelModal/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*example=\{analysisExample\}/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*onSelectPreset=\{handleSelect\}/);
  assert.match(appSource, /const APPENDIX_ROOT_HASH = '#appendix';/);
  assert.match(appSource, /const APPENDIX_SECTION_HASH_PREFIX = '#appendix-section-';/);
  assert.match(appSource, /function isAppendixHash\(hash = ''\)/);
  assert.match(appSource, /function scrollToHashTarget\(hash\)/);
  assert.match(appSource, /const openAppendix = useCallback\(\(hash = APPENDIX_ROOT_HASH\) =>/);
  assert.match(appSource, /const closeAppendix = useCallback\(\(\) =>/);
  assert.match(appSource, /window\.addEventListener\('hashchange', syncAppendixFromHash\)/);
  assert.match(appSource, /window\.history\.replaceState\(null, '', hash\)/);
  assert.match(appSource, /window\.history\.replaceState\(null, '', fallbackHash\)/);
  assert.match(appSource, /onClick=\{\(\) => openAppendix\(\)\}/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*onClose=\{closeAppendix\}/);
  assert.match(appSource, /Is this the full symmetry of the final expression\?/);
  // V4 rewrote the appendix-button paragraph to introduce H = Stab_{G_pt}(V)|_V
  // alongside G_pt and to use the new G_f = H × ∏_d S(W_d) formula.
  assert.match(appSource, /The cost above uses .*notationLatex\('g_pointwise'\).* on product assignments/);
  assert.match(appSource, /notationLatex\('h_output'\)/);
  assert.match(appSource, /String\.raw`H = \\mathrm\{Stab\}_\{G_\{\\text\{pt\}\}\}\(V\)\|_V`/);
  assert.match(appSource, /String\.raw`G_\{\\text\{f\}\} = H \\times \\prod_d S\(W_d\)`/);
  assert.match(appSource, /Its dummy-label factor acts after summation/);
  assert.match(appSource, /accepted explicit-index einsum language uses lowercase[\s\S]*explicit outputs[\s\S]*forbids[\s\S]*duplicate output labels/);
  assert.doesNotMatch(einsumGlanceCopy, /restricted explicit-index einsum language/);
  assert.doesNotMatch(appSource, /VERBATIM, AUDIT-VERIFIED/);
  assert.doesNotMatch(appSource, /REVIEW_RESPONSE\.md §5/);
  assert.doesNotMatch(appSource, /AUDIT\.md/);
  assert.doesNotMatch(appSource, /empirically verified on 22 presets \+ 543 σ-checks/);
});

test('route wrapper uses a branded centered loading shell instead of plain text', () => {
  const routeSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/index.tsx', import.meta.url),
    'utf8',
  );

  assert.match(routeSource, /import \{ Loader2 \} from 'lucide-react';/);
  assert.match(routeSource, /function SymmetryAwareEinsumContractionsLoading\(/);
  assert.match(routeSource, /className="relative flex h-88 w-88 items-center justify-center"/);
  assert.match(routeSource, /className="flopscope-wordmark[^"]*text-\[38px\][^"]*sm:text-\[48px\]/);
  assert.match(routeSource, /Loader2[\s\S]*animate-spin/);
  assert.match(routeSource, /strokeWidth=\{1\.05\}/);
  assert.match(routeSource, /min-h-\[100svh\]/);
  assert.doesNotMatch(routeSource, /radial-gradient/);
  assert.doesNotMatch(routeSource, /Loading Symmetry Aware Einsum Contractions/);
});

test('sticky bar wordmark relies on Next Link basePath handling instead of pre-prefixing the home href', () => {
  const stickyBarSource = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/components/StickyBar.jsx', import.meta.url),
    'utf8',
  );

  assert.match(stickyBarSource, /import Link from 'next\/link';/);
  assert.match(stickyBarSource, /href="\/"/);
  assert.doesNotMatch(stickyBarSource, /href=\{withBasePath\('\/'\)\}/);
});
