import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
function read(rel) { return readFileSync(resolve(__dirname, rel), 'utf-8'); }

test('rowsCols intro carries 5 paragraphs (the 3 projection-phenomenon paragraphs are here)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/rowsCols.js');
  const introMatch = src.match(/intro:\s*\[([\s\S]*?)\],\s*produces:/);
  assert.ok(introMatch, 'intro slot should be present');
  const paragraphCount = (introMatch[1].match(/p\(/g) ?? []).length;
  assert.equal(paragraphCount, 5, `expected 5 paragraphs in intro, got ${paragraphCount}`);
});

test('rowsCols intro keeps ¶1 + ¶2 verbatim and includes ¶3 + ¶4 + ¶5 (the branching-projection prose)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/content/main/rowsCols.js');
  // ¶1 + ¶2 — core rows/columns content.
  assert.match(src, /detected pointwise group \$G_\{\\\\text\{pt\}\}\$ acts on the full assignment grid \$X\$/);
  assert.match(src, /Accumulation is the part that is easy to over-compress/);
  // ¶3 + ¶4 + ¶5 — projection branching explanation.
  assert.match(src, /A product-orbit representative can contain many full index assignments/);
  assert.match(src, /A product orbit may contain many full assignments\. When those assignments are projected to the output labels/);
  assert.match(src, /Counting product orbits alone is therefore not enough: a single product orbit can update multiple stored output representatives/);
});
