#!/usr/bin/env node
/**
 * Post-build script: generates llms.txt and llms-full.txt from Next.js static export.
 *
 * Reads HTML files from out/, extracts text content, and produces:
 *   out/llms.txt       - index with section headers and links
 *   out/llms-full.txt  - all page content concatenated
 *
 * Run: node scripts/generate-llmstxt.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(__dirname, '..', 'out');
const SITE_URL = 'https://aicrowd.github.io';
const BASE_PATH = '/whest';

/**
 * Section configuration mirroring the Docusaurus llmstxt plugin.
 * Each key is a section header; value is {slug: description}.
 */
const SECTIONS = {
  'Getting Started': {
    'getting-started/installation': 'Install whest and verify setup',
    'getting-started/first-budget': 'Create your first FLOP budget',
  },
  'How-To Guides': {
    'how-to/migrate-from-numpy': 'Migrate existing NumPy code to whest',
    'how-to/use-einsum': 'Einstein summation patterns and costs',
    'how-to/exploit-symmetry': 'Save FLOPs via symmetry detection',
    'how-to/use-linalg': 'Linear algebra operations and costs',
    'how-to/use-fft': 'FFT operations, real vs complex, and costs',
    'how-to/plan-your-budget': 'Query FLOP costs before executing',
    'how-to/calibrate-weights': 'Measure per-operation FLOP weights with perf counters',
    'how-to/debug-budget-overruns': 'Diagnose and fix budget overruns',
  },
  'Concepts': {
    'concepts/flop-counting-model': 'How FLOP costs are computed analytically',
    'concepts/operation-categories': 'Free vs counted vs blocked operations',
    'concepts/numpy-compatibility-testing': 'NumPy compatibility test suite coverage',
  },
  'Explanation': {
    'explanation/subgraph-symmetry': 'Algorithm walkthrough for subgraph symmetry detection',
    'explanation/symmetry-explorer': 'Interactive visualization of einsum symmetry detection',
  },
  'Architecture': {
    'architecture/client-server': 'Client-server architecture for sandboxed execution',
    'architecture/docker': 'Running whest with Docker and Docker Compose',
  },
  'Development': {
    'development/contributing': 'Repository layout, local workflows, and generated-doc rules',
  },
  'API Reference': {
    'api': 'Complete API reference for all 482 operations',
  },
  'Reference': {
    'reference/for-agents': 'Guide for AI coding assistants',
    'reference/operation-audit': 'Complete 482-operation inventory with costs',
    'reference/empirical-weights': 'FLOP weight calibration results from hardware perf counters',
    'reference/cheat-sheet': 'Quick FLOP cost reference for all operations',
  },
  'Troubleshooting': {
    'troubleshooting/common-errors': 'Common errors and fixes',
  },
  'Changelog': {
    'changelog': 'Release history and breaking changes',
  },
};

/**
 * Extract plain text from an HTML file, stripping nav/script/style/header/footer.
 */
function extractText(htmlPath) {
  const html = fs.readFileSync(htmlPath, 'utf-8');
  const dom = new JSDOM(html);
  const doc = dom.window.document;

  // Remove elements that are not content
  const removeSelectors = [
    'script',
    'style',
    'nav',
    'header',
    'footer',
    // Docusaurus selectors (kept for compatibility)
    '.navbar',
    '.footer',
    '.pagination-nav',
    '.table-of-contents',
    '.theme-doc-toc-mobile',
    '.theme-doc-toc-desktop',
    '.theme-doc-breadcrumbs',
    '.theme-doc-sidebar-container',
    // Fumadocs selectors
    '[data-sidebar]',
    '[data-toc]',
    '.fd-toc',
    '.fd-sidebar',
    '.fd-nav',
    '[role="navigation"]',
  ];

  for (const sel of removeSelectors) {
    for (const el of doc.querySelectorAll(sel)) {
      el.remove();
    }
  }

  // Try to get the main article content first
  const article =
    doc.querySelector('article') ||
    doc.querySelector('.markdown') ||
    doc.querySelector('main') ||
    doc.body;

  const text = (article.textContent || '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return text;
}

/**
 * Resolve a doc slug to its HTML file path in the build directory.
 * Fumadocs puts doc pages under docs/ prefix, so we check both
 * <outDir>/docs/<slug>/index.html and <outDir>/<slug>/index.html.
 */
function resolveHtmlPath(slug) {
  // Fumadocs routes: try docs/<slug>/index.html first
  const docsIndexPath = path.join(OUT_DIR, 'docs', slug, 'index.html');
  if (fs.existsSync(docsIndexPath)) {
    return docsIndexPath;
  }
  // Try <outDir>/<slug>/index.html
  const indexPath = path.join(OUT_DIR, slug, 'index.html');
  if (fs.existsSync(indexPath)) {
    return indexPath;
  }
  // Try <outDir>/<slug>.html
  const directPath = path.join(OUT_DIR, slug + '.html');
  if (fs.existsSync(directPath)) {
    return directPath;
  }
  return null;
}

// --- Main ---

const siteBase = `${SITE_URL}${BASE_PATH}/`;

const llmsTxtLines = [];
const llmsFullLines = [];

// Header
llmsTxtLines.push('# whest');
llmsTxtLines.push('');
llmsTxtLines.push('> NumPy-compatible math primitives with FLOP counting');
llmsTxtLines.push('');

let found = 0;
let missing = 0;

for (const [section, pages] of Object.entries(SECTIONS)) {
  llmsTxtLines.push(`## ${section}`);
  llmsTxtLines.push('');

  llmsFullLines.push('='.repeat(60));
  llmsFullLines.push(section);
  llmsFullLines.push('='.repeat(60));
  llmsFullLines.push('');

  for (const [slug, description] of Object.entries(pages)) {
    // Determine the actual URL based on where the HTML was found
    const htmlPath = resolveHtmlPath(slug);
    const isDocsRoute = htmlPath && htmlPath.includes(path.join('out', 'docs'));
    const urlSlug = isDocsRoute ? `docs/${slug}` : slug;
    const pageUrl = `${siteBase}${urlSlug}`;
    llmsTxtLines.push(`- [${description}](${pageUrl})`);

    if (htmlPath) {
      const text = extractText(htmlPath);
      llmsFullLines.push(`--- ${slug} ---`);
      llmsFullLines.push(`URL: ${pageUrl}`);
      llmsFullLines.push('');
      llmsFullLines.push(text);
      llmsFullLines.push('');
      found++;
    } else {
      llmsFullLines.push(`--- ${slug} ---`);
      llmsFullLines.push(`URL: ${pageUrl}`);
      llmsFullLines.push('');
      llmsFullLines.push('[Page not found in build output]');
      llmsFullLines.push('');
      missing++;
    }
  }

  llmsTxtLines.push('');
}

// Write output
fs.mkdirSync(OUT_DIR, { recursive: true });

const llmsTxtPath = path.join(OUT_DIR, 'llms.txt');
const llmsFullPath = path.join(OUT_DIR, 'llms-full.txt');

fs.writeFileSync(llmsTxtPath, llmsTxtLines.join('\n'), 'utf-8');
fs.writeFileSync(llmsFullPath, llmsFullLines.join('\n'), 'utf-8');

console.log(`[generate-llmstxt] ${llmsTxtPath} (${found} pages found, ${missing} missing)`);
console.log(`[generate-llmstxt] ${llmsFullPath}`);
