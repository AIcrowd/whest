import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  extractSameOriginLinksFromHtml,
  outputPathForHref,
  findBrokenSameOriginLinks,
} from './gh-pages-link-audit.mjs';

test('extractSameOriginLinksFromHtml ignores external and fragment-only links', () => {
  const html = `
    <a href="/flopscope/docs/api/numpy/angle/">angle</a>
    <a href="/flopscope/docs/api/numpy/angle/#notes">angle notes</a>
    <a href="https://aicrowd.github.io/flopscope/docs/api/numpy/absolute/">absolute same origin</a>
    <a href="//aicrowd.github.io/flopscope/docs/api/numpy/protocol-relative/">protocol relative same origin</a>
    <a href="./local/">local</a>
    <a href="../sibling/">sibling</a>
    <a href="#local-fragment">fragment</a>
    <a href="mailto:test@example.com">mail</a>
    <a href="https://numpy.org/doc/stable/">numpy</a>
    <a href="/flopscopex/docs/api/numpy/not-in-base/">wrong base</a>
    <a href="/flopscope/docs/api/numpy/arctan2/">arctan2</a>
  `;

  const links = extractSameOriginLinksFromHtml({
    html,
    page: 'docs/api/numpy/angle/index.html',
    deployBasePath: '/flopscope',
  });

  assert.deepEqual(links, [
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '../sibling/',
      text: 'sibling',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: './local/',
      text: 'local',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '//aicrowd.github.io/flopscope/docs/api/numpy/protocol-relative/',
      text: 'protocol relative same origin',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '/flopscope/docs/api/numpy/angle/',
      text: 'angle',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '/flopscope/docs/api/numpy/angle/#notes',
      text: 'angle notes',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '/flopscope/docs/api/numpy/arctan2/',
      text: 'arctan2',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '/flopscopex/docs/api/numpy/not-in-base/',
      text: 'wrong base',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: 'https://aicrowd.github.io/flopscope/docs/api/numpy/absolute/',
      text: 'absolute same origin',
    },
  ]);
});

test('outputPathForHref resolves directory routes and static assets under the deploy base path', () => {
  const outRoot = '/tmp/site/out';

  assert.equal(
    outputPathForHref({
      href: '/flopscope/docs/api/numpy/angle/',
      outRoot,
      deployBasePath: '/flopscope',
    }),
    path.join(outRoot, 'docs/api/numpy/angle/index.html'),
  );

  assert.equal(
    outputPathForHref({
      href: '/flopscope/ops.json',
      outRoot,
      deployBasePath: '/flopscope',
    }),
    path.join(outRoot, 'ops.json'),
  );

  assert.equal(
    outputPathForHref({
      href: '/flopscope/docs/api/numpy/angle/?view=full#notes',
      outRoot,
      deployBasePath: '/flopscope',
    }),
    path.join(outRoot, 'docs/api/numpy/angle/index.html'),
  );

  assert.equal(
    outputPathForHref({
      href: '../arctan2/?view=full#notes',
      page: 'docs/api/numpy/angle/index.html',
      outRoot,
      deployBasePath: '/flopscope',
    }),
    path.join(outRoot, 'docs/api/numpy/arctan2/index.html'),
  );

  assert.equal(
    outputPathForHref({
      href: '/flopscope/docs/api/numpy/v1.0',
      outRoot,
      deployBasePath: '/flopscope',
    }),
    path.join(outRoot, 'docs/api/numpy/v1.0/index.html'),
  );

  assert.equal(
    outputPathForHref({
      href: '/flopscopex/docs/api/numpy/angle/',
      outRoot,
      deployBasePath: '/flopscope',
    }),
    null,
  );
});

test('findBrokenSameOriginLinks reports missing exported targets', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'gh-pages-link-audit-'));
  const outRoot = path.join(root, 'out');

  await mkdir(path.join(outRoot, 'docs/api/numpy/angle'), { recursive: true });
  await mkdir(path.join(outRoot, 'docs/api/numpy/arctan2'), { recursive: true });
  await mkdir(path.join(outRoot, 'docs/api/numpy/absolute'), { recursive: true });
  await mkdir(path.join(outRoot, 'docs/api/numpy/protocol-relative'), { recursive: true });
  await writeFile(
    path.join(outRoot, 'docs/api/numpy/angle/index.html'),
    [
      '<a href="https://aicrowd.github.io/flopscope/docs/api/numpy/absolute/">absolute ok</a>',
      '<a href="//aicrowd.github.io/flopscope/docs/api/numpy/protocol-relative/">protocol relative ok</a>',
      '<a href="/flopscope/docs/api/numpy/arctan2/">ok</a>',
      '<a href="../relative-ok/">relative ok</a>',
      '<a href="../../../../../escaped/">relative escapes base</a>',
      '<a href="/flopscope/docs/api/numpy/missing/">broken</a>',
      '<a href="/flopscopex/docs/api/numpy/wrong-base/">wrong base</a>',
    ].join(''),
  );
  await writeFile(path.join(outRoot, 'docs/api/numpy/arctan2/index.html'), '<p>ok</p>');
  await writeFile(path.join(outRoot, 'docs/api/numpy/absolute/index.html'), '<p>ok</p>');
  await writeFile(path.join(outRoot, 'docs/api/numpy/protocol-relative/index.html'), '<p>ok</p>');
  await mkdir(path.join(outRoot, 'docs/api/numpy/relative-ok'), { recursive: true });
  await writeFile(path.join(outRoot, 'docs/api/numpy/relative-ok/index.html'), '<p>ok</p>');

  const result = await findBrokenSameOriginLinks({
    outRoot,
    deployBasePath: '/flopscope',
  });

  assert.equal(result.links.length, 7);
  assert.deepEqual(result.brokenTargets, [
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '../../../../../escaped/',
      text: 'relative escapes base',
      target: 'missing-base-path',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '/flopscope/docs/api/numpy/missing/',
      text: 'broken',
      target: 'docs/api/numpy/missing/index.html',
    },
    {
      page: 'docs/api/numpy/angle/index.html',
      href: '/flopscopex/docs/api/numpy/wrong-base/',
      text: 'wrong base',
      target: 'missing-base-path',
    },
  ]);
});
