import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('OrbitDetailCard exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.match(src, /export default (?:function |const |memo\()?OrbitDetailCard/);
});

test('OrbitDetailCard uses flipPosition for viewport-edge handling', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.match(src, /from '\.\/floatingPosition\.js'/);
  assert.match(src, /flipPosition\(/);
});

test('OrbitDetailCard renders consolidated 4-zone layout (header, projection, equation+ledger, other-reached summary)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  // Branching caption now lives near the header, not as a bottom paragraph.
  assert.match(src, /data-testid="orbit-detail-branching-caption"/);
  // Mini row preview is gone (redundant with the canvas underneath).
  assert.doesNotMatch(src, /data-testid="worked-example-row-preview"/);
  // Projection sketch survives.
  assert.match(src, /data-testid="worked-example-projection"/);
  assert.match(src, /data-testid="orbit-detail-coefficient-note"/);
  // Full-detail blocks stay in the component for modal inline mode.
  assert.match(src, /data-testid="orbit-detail-this-q-ledger"/);
  assert.match(src, /data-testid="orbit-detail-other-reached-summary"/);
});

test('OrbitDetailCard reserves equation and ledgers for expanded inline mode', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.match(src, /const showFullDetail = mode === 'inline';/);
  assert.match(src, /\{showFullDetail && expressionInfo && canonicalEquationLatex/);
  assert.match(src, /\{showFullDetail && filled && expressionInfo && contributing\.length > 0/);
  assert.match(src, /\{showFullDetail && otherReached\.length > 0/);
});

test('OrbitDetailCard keeps coefficient as a note, not a confusing toggle mode', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.doesNotMatch(src, /data-testid="orbit-detail-coefficient-toggle"/);
  assert.doesNotMatch(src, /data-coefficient-row="true"/);
  assert.doesNotMatch(src, /coefficientView/);
  assert.match(src, /Multiple members landing in the same/);
  assert.match(src, /change the coefficient, not the number of update events/);
});

test('OrbitDetailCard has no close button in the hover card chrome', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.doesNotMatch(src, /data-action="clear-pin"/);
  assert.doesNotMatch(src, /× close/);
});

test('OrbitDetailCard supports two modes: floating (default) and inline (modal) — only floating uses position:fixed', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.match(src, /mode\s*=\s*['"]floating['"]/);
  // Inline mode skips fixed positioning + flip math.
  assert.match(src, /mode\s*===?\s*['"]floating['"]/);
});

test('OrbitDetailCard listens for Escape and IntersectionObserver auto-dismiss when in floating mode', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.match(src, /['"]keydown['"]/);
  assert.match(src, /Escape/);
  assert.match(src, /IntersectionObserver/);
});

test('OrbitDetailCard imports Latex + the orbitRepMatrixLayout helpers it actually consumes', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  // Latex for the einsum equation.
  assert.match(src, /import Latex from '\.\.\/Latex\.jsx'/);
  // labelledTuple + tupleKey for tuple display + projection-match keys.
  assert.match(src, /from '\.\/orbitRepMatrixLayout\.js'/);
  assert.match(src, /labelledTuple/);
  assert.match(src, /tupleKey/);
});

test('OrbitDetailCard wrapper uses p-5 padding for breathing room', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  // The card outer div uses p-5 (Tailwind: 1.25rem = 20px) instead of p-4.
  assert.match(src, /data-testid="orbit-detail-card"[\s\S]{0,200}className="p-5"/);
});

test('OrbitDetailCard constrains overflow in floating and inline layouts', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  assert.match(src, /maxWidth:\s*`calc\(100vw -/);
  assert.match(src, /overflowX:\s*'hidden'/);
  assert.match(src, /grid-cols-\[minmax\(0,1fr\)_auto_minmax\(0,1fr\)\]/);
  assert.match(src, /break-words/);
});

test('OrbitDetailCard uses no raw hex outside design tokens', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/OrbitDetailCard.jsx');
  const allowed = new Set([
    '#FFFFFF', '#1F2526', '#D9DCDC', '#F8F9F9', '#ECEFEF',
    '#F0524D', '#FEF2F1', '#B23E3A', '#4A7CFF', '#64748B', '#9AA0A0', '#F4F6F6',
  ]);
  const hexes = src.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  for (const h of hexes) {
    assert.ok(allowed.has(h.toUpperCase()), `disallowed hex ${h} — use a design token`);
  }
});
