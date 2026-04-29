import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_ALLOWED_ORIGINS,
  DEFAULT_DEPLOY_BASE_PATH,
  extractSameOriginLinksFromOutRoot,
} from './gh-pages-link-audit.mjs';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const outRoot = path.join(websiteRoot, 'out');
const artifactRoot = path.join(websiteRoot, '.generated', 'link-audit');
const manifestPath = path.join(artifactRoot, 'exported-links.json');

const links = await extractSameOriginLinksFromOutRoot({
  outRoot,
  deployBasePath: DEFAULT_DEPLOY_BASE_PATH,
  allowedOrigins: DEFAULT_ALLOWED_ORIGINS,
});

const manifest = {
  generatedAt: new Date().toISOString(),
  deployBasePath: DEFAULT_DEPLOY_BASE_PATH,
  outRoot: 'out',
  linkCount: links.length,
  links,
};

await mkdir(artifactRoot, { recursive: true });
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(`wrote ${path.relative(websiteRoot, manifestPath)}`);
