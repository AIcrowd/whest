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
  assert.match(appSource, /import SectionIntroProse from '\.\/components\/SectionIntroProse\.jsx';/);
  // After the Partition Counting section was inserted, intros now appear in
  // 5 acts: §1 setup, §2 structure, §3 proof, §4 decompose, §5 partition counting.
  // §6 cost-savings still has no intro (it goes straight into TotalCostView).
  assert.equal(countMatches(appSource, /EXPLORER_ACTS\[[0-5]\]\.introParagraphs/g), 5);
  assert.match(appSource, /title={EXPLORER_ACTS\[5\]\.heading}/);
  assert.match(appSource, /description={<InlineMathText>{EXPLORER_ACTS\[5\]\.question}<\/InlineMathText>}/);
  assert.match(introSource, /md:grid-cols-2/);
  assert.match(introSource, /textAlign:\s*'justify'/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[5\]\.supportingSentence/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[5\]\.introParagraphs/);
  assert.equal(countMatches(appSource, /label="Interpretation"/g), 0);
  assert.equal(countMatches(appSource, /label="Approach"/g), 0);
  // 5 intro-bearing acts × "What this produces" callout each.
  assert.equal(countMatches(appSource, /label="What this produces"/g), 5);
  assert.match(appSource, /Scope of the calculation/);
  assert.equal(countMatches(appSource, /tone="preamble"/g), 2);
  assert.equal(countMatches(appSource, /label="Candidate, not proof"/g), 0);
  assert.equal(countMatches(appSource, /label="What the model accepts"/g), 0);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[5\]\.why/);
  assert.doesNotMatch(appSource, /onOpenModalSection=\{/);
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
  const section1Copy = fs.readFileSync(
    new URL('./components/symmetry-aware-einsum-contractions/content/main/section1.js', import.meta.url),
    'utf8',
  );

  assert.match(appSource, /<ExpressionLevelModal/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*example=\{example\}/);
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
  assert.doesNotMatch(section1Copy, /restricted explicit-index einsum language/);
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
  assert.match(routeSource, /className="whest-wordmark[^"]*text-\[38px\][^"]*sm:text-\[48px\]/);
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
