import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  collectHtmlFiles,
  DEFAULT_ALLOWED_ORIGINS,
  DEFAULT_DEPLOY_BASE_PATH,
  inspectLinksFromHtml,
} from './gh-pages-link-audit.mjs';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const outRoot = path.join(websiteRoot, 'out');

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

  assert.doesNotMatch(source, /node:fs|node:fs\/promises|from 'fs'|from "fs"/);
  assert.doesNotMatch(source, /process\.cwd\(/);
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

  assert.match(llms, /^# flopscope/m);
  assert.ok(llms.length > 100);
  assert.ok(llmsFull.length > 1000);
  assert.match(sampleDetail, /"schema_version": 1/);
  assert.match(sampleDetail, /"detail_json_href": "\/api-data\/ops\/einsum\.json"/);
});

test('exported site avoids hardcoded root logo paths and root-relative links without deploy base path', async () => {
  const htmlFiles = await collectHtmlFiles(outRoot);
  const missingBasePathLinks = [];

  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, 'utf8');
    const page = path.relative(outRoot, htmlFile);

    assert.doesNotMatch(
      html,
      /https:\/\/aicrowd\.github\.io\/logo\.png|href="\/logo\.png"|src="\/logo\.png"/,
      `hardcoded root logo path found in ${path.relative(outRoot, htmlFile)}`,
    );

    const { brokenTargets } = inspectLinksFromHtml({
      html,
      page,
      deployBasePath: DEFAULT_DEPLOY_BASE_PATH,
      allowedOrigins: DEFAULT_ALLOWED_ORIGINS,
    });

    missingBasePathLinks.push(
      ...brokenTargets
        .filter((item) => item.target === 'missing-base-path')
        .map(({ page: brokenPage, href }) => ({ page: brokenPage, href })),
    );
  }

  assert.deepEqual(missingBasePathLinks, []);
});
