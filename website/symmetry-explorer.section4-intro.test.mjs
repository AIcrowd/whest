import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) { return readFileSync(resolve(__dirname, rel), 'utf-8'); }

test('rowsCols intro carries 3 edited paragraphs without repeated projection prose', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/rowsCols.js');
  const introMatch = src.match(/intro:\s*\[([\s\S]*?)\],\s*produces:/);
  assert.ok(introMatch, 'intro slot should be present');
  const paragraphCount = (introMatch[1].match(/p\(/g) ?? []).length;
  assert.equal(paragraphCount, 3, `expected 3 paragraphs in intro, got ${paragraphCount}`);
});

test('rowsCols intro keeps the row/column definitions and one crisp bridge forward', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/rowsCols.js');
  // ¶1 + ¶2 — core rows/columns content.
  assert.match(src, /detected pointwise group \$G_\{\\\\text\{pt\}\}\$ acts on the full assignment grid \$X\$/);
  assert.match(src, /Accumulation is the part that is easy to over-compress/);
  assert.match(src, /Enumerating every concrete assignment is correct but wasteful/);
  assert.match(src, /the remaining sections explain when this \$O \\\\to Q\$ reach relation factors/);
  assert.doesNotMatch(src, /A product orbit may contain many full assignments\. When those assignments are projected to the output labels/);
  assert.doesNotMatch(src, /Counting product orbits alone is therefore not enough: a single product orbit can update multiple stored output representatives/);
});
