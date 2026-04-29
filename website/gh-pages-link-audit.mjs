import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { JSDOM } from 'jsdom';

export const DEFAULT_DEPLOY_BASE_PATH = '/flopscope';
export const DEFAULT_ALLOWED_ORIGINS = [
  'https://aicrowd.github.io',
  'https://flopscope.dev',
];
const STATIC_FILE_EXTENSIONS = new Set([
  '.css',
  '.gif',
  '.html',
  '.ico',
  '.jpeg',
  '.jpg',
  '.js',
  '.json',
  '.map',
  '.md',
  '.pdf',
  '.png',
  '.svg',
  '.txt',
  '.webp',
  '.xml',
]);

function normalizeDeployBasePath(deployBasePath) {
  if (!deployBasePath || deployBasePath === '/') return '';
  return deployBasePath.endsWith('/') ? deployBasePath.slice(0, -1) : deployBasePath;
}

function stripQueryAndHash(value) {
  return value.replace(/[?#].*$/, '');
}

function normalizeAllowedOrigins(allowedOrigins) {
  if (!allowedOrigins || allowedOrigins.length === 0) {
    return new Set(DEFAULT_ALLOWED_ORIGINS);
  }
  return new Set(allowedOrigins);
}

function shouldIgnoreHref(href) {
  return (
    !href ||
    href.startsWith('#') ||
    href.startsWith('mailto:') ||
    href.startsWith('tel:') ||
    href.startsWith('javascript:')
  );
}

function pagePathForOutputFile(page) {
  const normalizedPage = (page || 'index.html').replace(/\\/g, '/');
  if (normalizedPage === 'index.html') return '/';
  if (normalizedPage.endsWith('/index.html')) {
    return `/${normalizedPage.slice(0, -'index.html'.length)}`;
  }
  return `/${normalizedPage}`;
}

function isWithinDeployBasePath(pathname, deployBasePath) {
  const normalizedBasePath = normalizeDeployBasePath(deployBasePath);
  if (normalizedBasePath === '') return pathname.startsWith('/');
  return pathname === normalizedBasePath || pathname.startsWith(`${normalizedBasePath}/`);
}

function linkText(anchor) {
  return anchor.textContent?.replace(/\s+/g, ' ').trim() ?? '';
}

function resolveHref({ href, page, deployBasePath }) {
  const basePath = normalizeDeployBasePath(deployBasePath);
  const pagePath = pagePathForOutputFile(page);
  const currentUrl = new URL(`${basePath}${pagePath}`, 'https://flopscope.dev');
  return new URL(href, currentUrl);
}

function relativePathFromResolvedHref({ resolvedHref, deployBasePath }) {
  const normalizedBasePath = normalizeDeployBasePath(deployBasePath);
  return normalizedBasePath === ''
    ? resolvedHref.pathname
    : resolvedHref.pathname.slice(normalizedBasePath.length) || '/';
}

function shouldTreatAsStaticFile(relativePath) {
  const extension = path.extname(relativePath);
  return STATIC_FILE_EXTENSIONS.has(extension);
}

function compareLinkRecords(a, b) {
  return (
    a.page.localeCompare(b.page) ||
    a.href.localeCompare(b.href) ||
    (a.target ?? '').localeCompare(b.target ?? '') ||
    a.text.localeCompare(b.text)
  );
}

function classifyHref({
  href,
  page,
  deployBasePath,
  allowedOrigins,
}) {
  if (shouldIgnoreHref(href)) return { ignored: true };

  const resolvedHref = resolveHref({ href, page, deployBasePath });
  const normalizedAllowedOrigins = normalizeAllowedOrigins(allowedOrigins);
  const sameOrigin = normalizedAllowedOrigins.has(resolvedHref.origin);

  if (!sameOrigin) {
    return { ignored: true };
  }

  const inDeployBasePath = isWithinDeployBasePath(
    stripQueryAndHash(resolvedHref.pathname),
    deployBasePath,
  );

  return {
    ignored: false,
    sameOrigin,
    inDeployBasePath,
    resolvedHref,
    relativePath: inDeployBasePath
      ? relativePathFromResolvedHref({ resolvedHref, deployBasePath })
      : null,
  };
}

export async function collectHtmlFiles(dir) {
  const entries = (await readdir(dir, { withFileTypes: true })).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
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

  return files.sort((a, b) => a.localeCompare(b));
}

export function inspectLinksFromHtml({
  html,
  page,
  deployBasePath,
  allowedOrigins,
}) {
  const dom = new JSDOM(html);
  const anchors = [...dom.window.document.querySelectorAll('a[href]')];
  const links = [];
  const brokenTargets = [];

  for (const anchor of anchors) {
    const href = anchor.getAttribute('href')?.trim();
    const classification = classifyHref({
      href,
      page,
      deployBasePath,
      allowedOrigins,
    });
    if (classification.ignored) continue;

    const record = {
      page,
      href,
      text: linkText(anchor),
    };

    links.push(record);

    if (!classification.inDeployBasePath) {
      brokenTargets.push({
        ...record,
        target: 'missing-base-path',
      });
      continue;
    }
  }

  links.sort(compareLinkRecords);
  brokenTargets.sort(compareLinkRecords);
  return { links, brokenTargets };
}

export function extractSameOriginLinksFromHtml({
  html,
  page,
  deployBasePath,
  allowedOrigins,
}) {
  return inspectLinksFromHtml({
    html,
    page,
    deployBasePath,
    allowedOrigins,
  }).links;
}

export async function extractSameOriginLinksFromOutRoot({
  outRoot,
  deployBasePath,
  allowedOrigins,
}) {
  const htmlFiles = await collectHtmlFiles(outRoot);
  const links = [];

  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, 'utf8');
    links.push(
      ...extractSameOriginLinksFromHtml({
        html,
        page: path.relative(outRoot, htmlFile),
        deployBasePath,
        allowedOrigins,
      }),
    );
  }

  return links.sort(compareLinkRecords);
}

export function outputPathForHref({
  href,
  page,
  outRoot,
  deployBasePath,
  allowedOrigins,
}) {
  const classification = classifyHref({
    href,
    page,
    deployBasePath,
    allowedOrigins,
  });
  if (classification.ignored || !classification.inDeployBasePath) return null;

  const relative = classification.relativePath;
  if (!relative.endsWith('/') && shouldTreatAsStaticFile(relative)) {
    return path.join(outRoot, relative.replace(/^\//, ''));
  }

  const normalizedRoute =
    relative === '/' ? '' : relative.replace(/^\//, '').replace(/\/$/, '');
  return path.join(outRoot, normalizedRoute, 'index.html');
}

export async function findBrokenSameOriginLinks({
  outRoot,
  deployBasePath,
  allowedOrigins,
}) {
  const htmlFiles = await collectHtmlFiles(outRoot);
  const links = [];
  const brokenTargets = [];

  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, 'utf8');
    const page = path.relative(outRoot, htmlFile);
    const inspected = inspectLinksFromHtml({
      html,
      page,
      deployBasePath,
      allowedOrigins,
    });

    links.push(...inspected.links);
    brokenTargets.push(...inspected.brokenTargets);

    for (const link of inspected.links) {
      const outputTarget = outputPathForHref({
        href: link.href,
        page,
        outRoot,
        deployBasePath,
        allowedOrigins,
      });
      if (!outputTarget) continue;

      try {
        await access(outputTarget);
      } catch {
        brokenTargets.push({
          ...link,
          target: path.relative(outRoot, outputTarget),
        });
      }
    }
  }

  links.sort(compareLinkRecords);
  brokenTargets.sort(compareLinkRecords);
  return { links, brokenTargets };
}
