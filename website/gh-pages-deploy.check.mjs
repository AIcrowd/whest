import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const outRoot = path.join(websiteRoot, 'out');
const DEPLOY_BASE_PATH = '/whest';

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('for-agents links use normal static paths for machine-readable artifacts', async () => {
  const source = await readSource('content/docs/api/for-agents.mdx');

  assert.doesNotMatch(source, /pathname:\/\//);
  assert.match(source, /<StaticFileLink href="\/llms\.txt">llms\.txt<\/StaticFileLink>/);
  assert.match(source, /<StaticFileLink href="\/llms-full\.txt">llms-full\.txt<\/StaticFileLink>/);
  assert.match(source, /<StaticFileLink href="\/ops\.json">ops\.json<\/StaticFileLink>/);
  assert.match(source, /\/api-data\/ops\/<slug>\.json/);
});

test('docs route stays GH Pages-safe and avoids runtime filesystem reads', async () => {
  const source = await readSource('app/docs/[[...slug]]/page.tsx');
  const opSource = await readSource('app/docs/api/ops/[slug]/page.tsx');

  assert.doesNotMatch(source, /node:fs|node:fs\/promises|from 'fs'|from "fs"/);
  assert.doesNotMatch(source, /process\.cwd\(/);
  assert.doesNotMatch(opSource, /node:fs|node:fs\/promises|from 'fs'|from "fs"/);
  assert.doesNotMatch(opSource, /process\.cwd\(/);
});

test('build-generated public artifacts exist and are non-empty', async () => {
  const llmsPath = path.join(websiteRoot, 'public', 'llms.txt');
  const llmsFullPath = path.join(websiteRoot, 'public', 'llms-full.txt');
  const opsIndexPath = path.join(websiteRoot, 'public', 'ops.json');
  const sampleDetailPath = path.join(websiteRoot, 'public', 'api-data', 'ops', 'einsum.json');
  const outSampleDetailPath = path.join(websiteRoot, 'out', 'api-data', 'ops', 'einsum.json');
  const outNoJekyllPath = path.join(websiteRoot, 'out', '.nojekyll');

  await access(llmsPath);
  await access(llmsFullPath);
  await access(opsIndexPath);
  await access(sampleDetailPath);
  await access(outSampleDetailPath);
  await access(outNoJekyllPath);

  const llms = await readFile(llmsPath, 'utf8');
  const llmsFull = await readFile(llmsFullPath, 'utf8');
  const sampleDetail = await readFile(sampleDetailPath, 'utf8');

  assert.match(llms, /^# whest/m);
  assert.ok(llms.length > 100);
  assert.ok(llmsFull.length > 1000);
  assert.match(sampleDetail, /"schema_version": 1/);
  assert.match(sampleDetail, /"detail_json_href": "\/api-data\/ops\/einsum\.json"/);
});

async function collectHtmlFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectHtmlFiles(fullPath)));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.html')) {
      files.push(fullPath);
    }
  }

  return files;
}

function outputPathForHref(href) {
  if (!href.startsWith('/')) return null;
  if (!href.startsWith(DEPLOY_BASE_PATH)) return null;

  const relative = href.slice(DEPLOY_BASE_PATH.length) || '/';
  if (!relative.endsWith('/') && path.extname(relative)) {
    return path.join(outRoot, relative.replace(/^\//, ''));
  }

  const withoutQuery = relative.split('#', 1)[0].split('?', 1)[0];
  const normalized = withoutQuery === '/' ? '' : withoutQuery.replace(/^\//, '');
  return path.join(outRoot, normalized, 'index.html');
}

test('exported site has no broken same-origin HTML links or hardcoded root logo paths', async () => {
  const htmlFiles = await collectHtmlFiles(outRoot);
  const brokenTargets = [];

  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, 'utf8');
    const dom = new JSDOM(html);
    const anchors = [...dom.window.document.querySelectorAll('a[href]')];

    assert.doesNotMatch(
      html,
      /https:\/\/aicrowd\.github\.io\/logo\.png|href="\/logo\.png"|src="\/logo\.png"/,
      `hardcoded root logo path found in ${path.relative(outRoot, htmlFile)}`,
    );

    for (const anchor of anchors) {
      const href = anchor.getAttribute('href');
      if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('http')) {
        continue;
      }
      if (href.startsWith('/') && !href.startsWith(DEPLOY_BASE_PATH)) {
        brokenTargets.push({
          page: path.relative(outRoot, htmlFile),
          href,
          target: 'missing-base-path',
        });
        continue;
      }
      const outputTarget = outputPathForHref(href);
      if (!outputTarget) continue;

      try {
        await access(outputTarget);
      } catch {
        brokenTargets.push({
          page: path.relative(outRoot, htmlFile),
          href,
          target: path.relative(outRoot, outputTarget),
        });
      }
    }
  }

  assert.deepEqual(brokenTargets, []);
});
