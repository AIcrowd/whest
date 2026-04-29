import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_ALLOWED_ORIGINS,
  DEFAULT_DEPLOY_BASE_PATH,
  findBrokenSameOriginLinks,
} from './gh-pages-link-audit.mjs';

function formatBrokenLinksReport({ generatedAt, linkCount, brokenTargets }) {
  const lines = [
    '# Deploy link audit',
    '',
    `- Generated at: ${generatedAt}`,
    `- Deploy base path: \`${DEFAULT_DEPLOY_BASE_PATH}\``,
    '- Export root: `out`',
    `- Same-origin links checked: ${linkCount}`,
    `- Broken internal targets: ${brokenTargets.length}`,
    '',
    `## Status: ${brokenTargets.length === 0 ? 'PASS' : 'FAIL'}`,
    '',
  ];

  if (brokenTargets.length === 0) {
    lines.push('No broken same-origin internal links found.', '');
    return lines.join('\n');
  }

  lines.push('## Broken links', '');

  for (const brokenTarget of brokenTargets) {
    lines.push(
      `- page \`${brokenTarget.page}\` href \`${brokenTarget.href}\` target \`${brokenTarget.target}\` text ${JSON.stringify(brokenTarget.text)}`,
    );
  }

  lines.push('');
  return lines.join('\n');
}

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const outRoot = path.join(websiteRoot, 'out');
const artifactRoot = path.join(websiteRoot, '.generated', 'link-audit');
const brokenLinksPath = path.join(artifactRoot, 'broken-links.json');
const reportPath = path.join(artifactRoot, 'report.md');

const result = await findBrokenSameOriginLinks({
  outRoot,
  deployBasePath: DEFAULT_DEPLOY_BASE_PATH,
  allowedOrigins: DEFAULT_ALLOWED_ORIGINS,
});

const generatedAt = new Date().toISOString();
const brokenLinksArtifact = {
  generatedAt,
  deployBasePath: DEFAULT_DEPLOY_BASE_PATH,
  outRoot: 'out',
  linkCount: result.links.length,
  brokenCount: result.brokenTargets.length,
  brokenTargets: result.brokenTargets,
};

const report = formatBrokenLinksReport({
  generatedAt,
  linkCount: result.links.length,
  brokenTargets: result.brokenTargets,
});

await mkdir(artifactRoot, { recursive: true });
await writeFile(brokenLinksPath, `${JSON.stringify(brokenLinksArtifact, null, 2)}\n`);
await writeFile(reportPath, report);

console.log(`wrote ${path.relative(websiteRoot, brokenLinksPath)}`);
console.log(`wrote ${path.relative(websiteRoot, reportPath)}`);

if (result.brokenTargets.length > 0) {
  console.error(`deploy link audit found ${result.brokenTargets.length} broken internal target(s)`);
  process.exitCode = 1;
} else {
  console.log(`deploy link audit passed (${result.links.length} same-origin links checked)`);
}
