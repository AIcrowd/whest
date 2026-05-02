import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) { return readFileSync(resolve(__dirname, rel), 'utf-8'); }

test('rowsCols intro uses concept-first blocks with display equations', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/rowsCols.js');
  const introMatch = src.match(/intro:\s*\[([\s\S]*?)\],\s*produces:/);
  assert.ok(introMatch, 'intro slot should be present');
  const paragraphCount = (introMatch[1].match(/p\(/g) ?? []).length;
  const headingCount = (introMatch[1].match(/h\(/g) ?? []).length;
  const equationCount = (introMatch[1].match(/eq\(/g) ?? []).length;
  assert.equal(paragraphCount, 7, `expected 7 paragraphs in intro, got ${paragraphCount}`);
  assert.equal(headingCount, 3, `expected 3 section labels in intro, got ${headingCount}`);
  assert.equal(equationCount, 3, `expected 3 standalone equations in intro, got ${equationCount}`);
});

test('rowsCols intro defines rows, columns, and reach without inline long alpha formula', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/rowsCols.js');
  assert.match(src, /The \$O \\\\to Q\$ matrix has two quotients/);
  assert.match(src, /h\('Rows: product orbits', 1\)/);
  assert.match(src, /h\('Columns: stored output representatives', 2\)/);
  assert.match(src, /h\('Projection: filled cells', 'full'\)/);
  assert.match(src, /eq\('\\\\displaystyle O \\\\in X\/G_\{\\\\text\{pt\}\}', 1, false, 'Row quotient'\)/);
  assert.match(src, /eq\('\\\\displaystyle H = \\\\mathrm\{Stab\}_\{G_\{\\\\text\{pt\}\}\}\(V\)\|_V,\\\\qquad Q \\\\in Y\/H', 2, false, 'Column quotient'\)/);
  assert.match(src, /eq\('\\\\displaystyle \\\\alpha = \\\\#\\\\\{\(O,Q\) \\\\in X\/G_\{\\\\text\{pt\}\} \\\\times Y\/H : \\\\pi_V\(O\) \\\\cap Q \\\\neq \\\\varnothing\\\\\}', 'full', true, 'Accumulation count'\)/);
  assert.match(src, /Thus \$M\$ counts product-orbit rows, while \$\\\\alpha\$ counts filled \$O \\\\to Q\$ cells/);
  assert.doesNotMatch(src, /direct update count is \$\\\\alpha =/);
  assert.doesNotMatch(src, /A product orbit may contain many full assignments\. When those assignments are projected to the output labels/);
  assert.doesNotMatch(src, /Counting product orbits alone is therefore not enough: a single product orbit can update multiple stored output representatives/);
});
