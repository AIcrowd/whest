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
  assert.equal(countMatches(appSource, /EXPLORER_ACTS\[[0-4]\]\.introParagraphs/g), 4);
  assert.match(appSource, /title={EXPLORER_ACTS\[4\]\.heading}/);
  assert.match(appSource, /description={<InlineMathText>{EXPLORER_ACTS\[4\]\.question}<\/InlineMathText>}/);
  assert.match(introSource, /md:grid-cols-2/);
  assert.match(introSource, /textAlign:\s*'justify'/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.supportingSentence/);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.introParagraphs/);
  assert.equal(countMatches(appSource, /label="Interpretation"/g), 0);
  assert.equal(countMatches(appSource, /label="Approach"/g), 0);
  assert.equal(countMatches(appSource, /label="What this produces"/g), 4);
  assert.match(appSource, /Scope of the calculation/);
  assert.equal(countMatches(appSource, /tone="preamble"/g), 2);
  assert.equal(countMatches(appSource, /label="Candidate, not proof"/g), 0);
  assert.equal(countMatches(appSource, /label="What the model accepts"/g), 0);
  assert.doesNotMatch(appSource, /EXPLORER_ACTS\[4\]\.why/);
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

  assert.match(appSource, /<ExpressionLevelModal/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*example=\{example\}/);
  assert.match(appSource, /<ExpressionLevelModal[\s\S]*onSelectPreset=\{handleSelect\}/);
  assert.match(appSource, /Is this the full symmetry of the final expression\?/);
  assert.match(appSource, /The cost above uses .*notationLatex\('g_pointwise'\).* for accumulation/);
  assert.match(appSource, /The fully summed expression can have a larger label-renaming formal symmetry/);
  assert.match(appSource, /String\.raw`G_\{\\text\{f\}\} = G_\{\\mathrm\{out\}\} \\times \\prod_d S\(W_d\)`/);
  assert.match(appSource, /where each <Latex math=\{String\.raw`W_d`\} \/> is a same-domain block of summed labels/);
  assert.match(appSource, /notationLatex\('g_output'\)/);
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
