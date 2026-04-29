import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('MatrixHoverTooltip exports a default component using forwardRef', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/MatrixHoverTooltip.jsx');
  assert.match(src, /export default (?:function |const )?MatrixHoverTooltip|export default (?:React\.)?forwardRef/);
  assert.match(src, /forwardRef/);
});

test('MatrixHoverTooltip uses position:fixed and updates DOM via ref (no useState for hover content)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/MatrixHoverTooltip.jsx');
  // Position fixed so it stays glued to viewport (under the cursor).
  assert.match(src, /position:\s*['"]fixed['"]/);
  // Hover content updates imperatively via textContent, not setState.
  assert.match(src, /textContent\s*=/);
  // useState is not used for the tooltip's body/position (perf-critical).
  // We allow useState for `visible` only because that toggles display.
  const useStateCalls = src.match(/useState\b/g) ?? [];
  // One in the import, one in the component body.
  assert.equal(useStateCalls.length, 2, 'exactly one useState call expected (just `visible`)');
  assert.match(src, /\[visible,\s*setVisible\]\s*=\s*useState\(/);
});

test('MatrixHoverTooltip renders empty by default and exposes update + hide via ref', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/MatrixHoverTooltip.jsx');
  // Imperative API: ref.current.update(content, position) and ref.current.hide().
  assert.match(src, /update[\s(:]/);
  assert.match(src, /hide[\s(:]/);
});

test('MatrixHoverTooltip uses no raw hex outside design tokens', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/branchingViews/MatrixHoverTooltip.jsx');
  const allowed = new Set([
    '#FFFFFF', '#1F2526', '#D9DCDC', '#F8F9F9', '#ECEFEF',
    '#F0524D', '#FEF2F1', '#B23E3A', '#4A7CFF', '#64748B', '#9AA0A0', '#F4F6F6',
  ]);
  const hexes = src.match(/#[0-9A-Fa-f]{3,6}\b/g) ?? [];
  for (const h of hexes) {
    assert.ok(allowed.has(h.toUpperCase()), `disallowed hex ${h} — use a design token`);
  }
});
