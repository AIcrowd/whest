import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('RegimeTrace animates rows with staggered delay', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/RegimeTrace.jsx');
  assert.match(src, /animationDelay/);
  assert.match(src, /animate-trace-in/);
});

test('styles.css defines the trace-in keyframes', () => {
  const src = read('components/symmetry-aware-einsum-contractions/styles.css');
  assert.match(src, /@keyframes trace-in/);
  assert.match(src, /\.animate-trace-in/);
});

test('FormulaPopover exports a default function and reads REGIME_SPEC', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/FormulaPopover.jsx');
  assert.match(src, /export default function FormulaPopover/);
  assert.match(src, /REGIME_SPEC/);
});

test('BipartiteGraph accepts a highlightedLabels prop', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx');
  assert.match(src, /highlightedLabels/);
  assert.match(src, /getActiveExplorerThemeId/);
  assert.match(src, /explorerThemeColor/);
  assert.match(src, /getExplorerThemeOperandPalette/);
  assert.match(src, /const V_COLOR = explorerThemeColor\(explorerThemeId, 'hero'\)/);
  assert.match(src, /const W_COLOR = explorerThemeColor\(explorerThemeId, 'summedSide'\)/);
  assert.match(src, /const U_FALLBACK_COLOR = explorerThemeColor\(explorerThemeId, 'heroMuted'\)/);
  assert.match(src, /const HIGHLIGHT_COLOR = explorerThemeColor\(explorerThemeId, 'heroMuted'\)/);
});

test('BipartiteGraph renders notation headers as math, not plain text pills', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx');
  assert.match(src, /import PanZoomCanvas/);
  assert.match(src, /<PanZoomCanvas/);
  assert.match(src, /ariaLabel="Bipartite graph \(zoomable\)"/);
  assert.match(src, /import Latex/);
  assert.match(src, /foreignObject/);
  assert.match(src, /notationLatex\('v_free'\)/);
  assert.match(src, /notationLatex\('w_summed'\)/);
  assert.match(src, /notationLatex\('u_axis_classes'\)/);
  assert.match(src, /notationLatex\('l_labels'\)/);
  assert.doesNotMatch(src, /text=\{notationText\('v_free'\)\}/);
  assert.doesNotMatch(src, /text=\{notationText\('w_summed'\)\}/);
});

test('BipartiteGraph leaves the three operand-group boxes borderless', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx');
  assert.match(src, /width=\{lbFullW\} height=\{bottom - top\}\s*rx=\{14\} fill=\{colors\.fill\}/);
  assert.doesNotMatch(src, /width=\{lbFullW\} height=\{bottom - top\}[\s\S]*stroke=\{colors\.stroke\}/);
});

test('BipartiteGraph group labels render without gray badge backgrounds', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/BipartiteGraph.jsx');
  assert.match(src, /function LabelBadge\(\{ x, y, text, color \}\)/);
  assert.match(src, /<text x=\{x\} y=\{y - 4\}/);
  assert.match(src, /function MathLabelBadge\(\{ x, y, math, color, width \}\)/);
  assert.match(src, /<foreignObject x=\{x\} y=\{y - 18\} width=\{width\} height=\{18\}>/);
  assert.doesNotMatch(src, /fill="#F8F9F9"/);
  assert.doesNotMatch(src, /<rect x=\{x\} y=\{y - 10\} width=\{w\} height=\{18\} rx=\{4\} fill=\{bg\}/);
});

test('IncidenceMatrix exposes notation-aware row and column legends', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/IncidenceMatrix.jsx');
  assert.match(src, /import Latex/);
  assert.match(src, /notationLatex\('u_axis_classes'\)/);
  assert.match(src, /notationLatex\('v_free'\)/);
  assert.match(src, /notationLatex\('w_summed'\)/);
  assert.match(src, /getActiveExplorerThemeId/);
  assert.match(src, /getExplorerThemeFingerprintPalette/);
  assert.match(src, /defaultLabelColor = explorerThemeColor\(explorerThemeId,\s*'muted'\)/);
  assert.match(src, /rows:/i);
  assert.match(src, /columns:/i);
});

test('IncidenceMatrix visual accents use coral/slate chrome instead of noir math ink roles', () => {
  const src = read('components/symmetry-aware-einsum-contractions/styles.css');
  assert.match(src, /\.inc-col-header\.inc-col-v \{ color: var\(--coral\); \}/);
  assert.match(src, /\.inc-row-label\.inc-col-v \{ color: var\(--coral\); \}/);
  assert.match(src, /\.inc-fp-label\.inc-col-v \{ color: var\(--coral\); \}/);
  assert.match(src, /\.inc-cell\.inc-cell-active \{\s*background: color-mix\(in oklab, var\(--coral-hover\) 10%, transparent\);\s*color: var\(--coral-hover\);/);
  assert.match(src, /\.inc-fp-item:has\(\.inc-col-v\) \{ border-color: color-mix\(in oklab, var\(--coral\) 30%, transparent\); \}/);
});

test('Section 2 graph and matrix wrappers no longer draw outer card borders', () => {
  const src = read('components/symmetry-aware-einsum-contractions/styles.css');
  const graphBlock = src.match(/\.graph-container\s*\{[^}]*\}/)?.[0] ?? '';
  const matrixBlock = src.match(/\.matrix-wrapper\s*\{[^}]*\}/)?.[0] ?? '';
  const fingerprintsBlock = src.match(/\.fingerprints\s*\{[^}]*\}/)?.[0] ?? '';
  assert.match(graphBlock, /background: var\(--white\);/);
  assert.match(graphBlock, /padding: 16px;/);
  assert.doesNotMatch(graphBlock, /border:/);
  assert.match(matrixBlock, /background: var\(--white\);/);
  assert.match(matrixBlock, /padding: 16px;/);
  assert.doesNotMatch(matrixBlock, /border:/);
  assert.match(fingerprintsBlock, /background: var\(--white\);/);
  assert.match(fingerprintsBlock, /padding: 16px;/);
  assert.doesNotMatch(fingerprintsBlock, /border:/);
});

test('Major two-column explorer layouts use the shared faint center divider treatment', () => {
  const proseSrc = read('components/symmetry-aware-einsum-contractions/components/SectionIntroProse.jsx');
  const appSrc = read('components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  const atAGlanceSrc = read('components/symmetry-aware-einsum-contractions/components/AlgorithmAtAGlance.jsx');
  const componentCostSrc = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  const stylesSrc = read('components/symmetry-aware-einsum-contractions/styles.css');
  assert.match(proseSrc, /editorial-two-col-divider-md grid gap-x-8 gap-y-4 md:grid-cols-2/);
  assert.match(appSrc, /editorial-two-col-divider-md mt-6 grid grid-cols-1 gap-6 md:grid-cols-2/);
  assert.match(appSrc, /editorial-two-col-divider-lg mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2/);
  assert.match(atAGlanceSrc, /editorial-two-col-divider-lg grid items-stretch gap-8 lg:grid-cols-2 lg:gap-10/);
  assert.match(componentCostSrc, /editorial-two-col-divider-lg editorial-two-col-divider-lg-inset border-y border-gray-100 py-6 grid gap-6 lg:grid-cols-2/);
  assert.match(stylesSrc, /\.editorial-two-col-divider-md::before/);
  assert.match(stylesSrc, /\.editorial-two-col-divider-lg::before/);
  assert.match(stylesSrc, /\.editorial-two-col-divider-lg-inset::before/);
  assert.match(stylesSrc, /background: var\(--gray-100\);/);
});

test('ExplorerSubsectionHeader uses the shared 12px coral kicker register', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ExplorerSubsectionHeader.jsx');
  assert.match(src, /text-\[12px\]/);
  assert.match(src, /tracking-\[0\.14em\]/);
  assert.doesNotMatch(src, /text-\[11px\]/);
});

test('Shared support prose utility uses serif 15px editorial body styling', () => {
  const src = read('components/symmetry-aware-einsum-contractions/styles.css');
  const block = src.match(/\.explorer-support-prose\s*\{[^}]*\}/)?.[0] ?? '';
  assert.match(block, /font-family: var\(--font-paper-serif\);/);
  assert.match(block, /font-size: 15px;/);
  assert.match(block, /line-height: 1\.72;/);
  assert.match(block, /color: var\(--gray-700\);/);
  assert.match(block, /text-align: justify;/);
});

test('Explorer editorial accent is scoped and consumed by NarrativeCallout', () => {
  const stylesSrc = read('components/symmetry-aware-einsum-contractions/styles.css');
  const calloutSrc = read('components/symmetry-aware-einsum-contractions/components/NarrativeCallout.jsx');
  assert.match(stylesSrc, /--explorer-editorial-accent: var\(--editorial-accent\);/);
  assert.match(calloutSrc, /if \(tone === 'preamble'\)/);
  assert.match(calloutSrc, /rounded-2xl border border-primary\/20 bg-accent\/40 px-5 py-5/);
  assert.match(calloutSrc, /font-heading text-base font-semibold text-foreground/);
  assert.match(
    calloutSrc,
    /border-\[color:color-mix\(in_oklab,var\(--explorer-editorial-accent\)_28%,var\(--explorer-border\)\)\]/,
  );
  assert.match(
    calloutSrc,
    /bg-\[color:color-mix\(in_oklab,var\(--explorer-editorial-accent\)_10%,var\(--explorer-surface\)\)\]/,
  );
  assert.match(calloutSrc, /text-\[var\(--explorer-editorial-accent\)\]/);
});
