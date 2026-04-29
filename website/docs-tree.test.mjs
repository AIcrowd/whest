import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  STANDALONE_SYMMETRY_AWARE_EINSUM_URL,
  injectSymmetryAwareEinsumContractionsLink,
} from './lib/docsTree.js';

test('injectSymmetryAwareEinsumContractionsLink replaces the old embedded docs item with an external launch item', () => {
  const tree = {
    type: 'root',
    name: 'Documentation',
    children: [
      {
        type: 'folder',
        name: 'Understanding Flopscope',
        children: [
          { type: 'page', name: 'How Flopscope Works', url: '/docs/understanding/how-flopscope-works' },
          { type: 'page', name: 'Symmetry Explorer', url: '/docs/understanding/symmetry-explorer' },
        ],
      },
    ],
  };

  const updated = injectSymmetryAwareEinsumContractionsLink(tree);
  const folder = updated.children[0];

  assert.equal(folder.children[1].name, 'Symmetry Aware Einsum Contractions');
  assert.equal(folder.children[1].url, STANDALONE_SYMMETRY_AWARE_EINSUM_URL);
  assert.equal(folder.children[1].external, true);
  assert.equal(folder.children.some((item) => item.url === '/docs/understanding/symmetry-explorer'), false);
});

test('standalone symmetry explorer links use Next-aware routing so basePath is preserved on static deploys', () => {
  const mdxSource = fs.readFileSync(new URL('./components/mdx.tsx', import.meta.url), 'utf8');
  const sidebarSource = fs.readFileSync(new URL('./components/docs/DocsSidebarItem.jsx', import.meta.url), 'utf8');

  assert.match(mdxSource, /import Link from 'next\/link'/);
  assert.match(mdxSource, /href=\{STANDALONE_SYMMETRY_AWARE_EINSUM_URL\}/);
  assert.match(sidebarSource, /external=\{isStandaloneLaunch \? false : item\.external\}/);
  assert.match(sidebarSource, /target=\{isStandaloneLaunch \? '_blank' : undefined\}/);
});
