import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('for-agents links use normal static paths for machine-readable artifacts', async () => {
  const source = await readSource('content/docs/api/for-agents.mdx');

  assert.doesNotMatch(source, /pathname:\/\//);
  assert.match(source, /\.\.\/\.\.\/\.\.\/llms\.txt/);
  assert.match(source, /\.\.\/\.\.\/\.\.\/llms-full\.txt/);
  assert.match(source, /\.\.\/\.\.\/\.\.\/ops\.json/);
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
