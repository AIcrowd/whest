import test from 'node:test';
import assert from 'node:assert/strict';

import {resolveOpRef, scanMarkdownForBareOps} from './scripts/check-op-links.mjs';

test('resolveOpRef normalizes aliases to canonical labels', () => {
  const refs = {
    abs: {canonical_name: 'absolute', href: '/docs/api/ops/absolute', label: '`we.absolute`'},
  };
  const resolved = resolveOpRef('abs', refs);

  assert.equal(resolved.href, '/docs/api/ops/absolute');
  assert.equal(resolved.label, 'we.absolute');
});

test('scanMarkdownForBareOps flags prose mentions but ignores fenced code', () => {
  const refs = {
    absolute: {canonical_name: 'absolute', href: '/docs/api/ops/absolute', label: '`we.absolute`'},
  };
  const problems = scanMarkdownForBareOps(
    ['Use we.absolute in prose.', '```python\ny = we.absolute(x)\n```'].join('\n'),
    refs,
  );

  assert.deepEqual(problems, ['we.absolute']);
});
