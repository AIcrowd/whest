import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const BOUNDARY = 'components/symmetry-aware-einsum-contractions/components/AnalysisLoadingBoundary.jsx';
const APP = 'components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx';

test('AnalysisLoadingBoundary provides local skeleton states with reduced-motion support', () => {
  const src = read(BOUNDARY);
  assert.match(src, /export default function AnalysisLoadingBoundary/);
  assert.match(src, /data-analysis-loading-boundary/);
  assert.match(src, /aria-busy=\{isLoading \? 'true' : undefined\}/);
  assert.match(src, /role="status"/);
  assert.match(src, /Preparing \{label\}/);
  assert.match(src, /motion-reduce:animate-none/);
  assert.match(src, /absolute inset-0 z-10 flex flex-col/);
  assert.match(src, /mt-4 grid min-h-0 flex-1/);
  assert.match(src, /variant === 'matrix'/);
  assert.match(src, /variant === 'graph'/);
  assert.match(src, /variant === 'compact'/);
  assert.match(src, /CardsSkeleton/);
});

test('App renders only a screen-reader global status and delegates visible loading to local boundaries', () => {
  const src = read(APP);
  assert.match(src, /import AnalysisLoadingBoundary from '\.\/components\/AnalysisLoadingBoundary\.jsx'/);
  assert.match(src, /const localAnalysisLoading = analysisUpdating && Boolean\(pendingSelection\)/);
  assert.match(src, /className="sr-only"[\s\S]*data-analysis-updating="true"/);
  assert.doesNotMatch(src, /Showing the previous analysis until the new one is ready/);
});

test('App wraps the expensive interactive components in AnalysisLoadingBoundary', () => {
  const src = read(APP);
  const expectedLabels = [
    'bipartite graph',
    'incidence matrix',
    'O → Q matrix',
    'rows and columns schematic',
    'label interaction graph',
    'component accounting',
    'wreath structure',
    'sigma loop',
    'generator construction',
    'certification summary',
    'shortcut decision ladder',
    'naive alpha meter',
    'tuple pattern meter',
    'typed partition counter',
    'total cost view',
  ];

  for (const label of expectedLabels) {
    assert.match(
      src,
      new RegExp(`<AnalysisLoadingBoundary[^>]*isLoading=\\{localAnalysisLoading\\}[^>]*label="${label}"`),
      `missing local loading boundary for ${label}`,
    );
  }
});
