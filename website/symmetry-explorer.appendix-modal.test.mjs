import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL('./components/symmetry-aware-einsum-contractions/components/ExpressionLevelModal.jsx', import.meta.url),
  'utf8',
);

test('appendix modal uses paper-appendix section titles and removes audit provenance copy', () => {
  assert.match(source, /Appendix/);
  assert.match(source, /The distinction/);
  assert.match(source, /How the formal group is built/);
  assert.match(source, /Why Burnside on the formal group overcounts/);
  assert.match(source, /Storage-aware savings/);

  assert.doesNotMatch(source, /VERBATIM, AUDIT-VERIFIED/);
  assert.doesNotMatch(source, /REVIEW_RESPONSE\.md §5/);
  assert.doesNotMatch(source, /AUDIT\.md/);
  assert.doesNotMatch(source, /empirically verified on 22 presets \+ 543 σ-checks/);
});
