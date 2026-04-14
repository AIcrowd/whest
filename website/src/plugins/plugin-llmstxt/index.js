const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

/**
 * Section configuration mirroring the mkdocs.yml llmstxt-md config.
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
    '.navbar',
    '.footer',
    '.pagination-nav',
    '.table-of-contents',
    '.theme-doc-toc-mobile',
    '.theme-doc-toc-desktop',
    '.theme-doc-breadcrumbs',
    '.theme-doc-sidebar-container',
  ];

  for (const sel of removeSelectors) {
    for (const el of doc.querySelectorAll(sel)) {
      el.remove();
    }
  }

  // Try to get the main article content first
  const article = doc.querySelector('article') || doc.querySelector('.markdown') || doc.querySelector('main') || doc.body;
  const text = (article.textContent || '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return text;
}

/**
 * Resolve a doc slug to its HTML file path in the build directory.
 * Docusaurus generates paths like: <outDir>/<slug>/index.html or <outDir>/<slug>.html
 */
function resolveHtmlPath(outDir, slug) {
  // Try <outDir>/<slug>/index.html first (most common for Docusaurus)
  const indexPath = path.join(outDir, slug, 'index.html');
  if (fs.existsSync(indexPath)) {
    return indexPath;
  }
  // Try <outDir>/<slug>.html
  const directPath = path.join(outDir, slug + '.html');
  if (fs.existsSync(directPath)) {
    return directPath;
  }
  return null;
}

module.exports = function pluginLlmsTxt(_context, _options) {
  return {
    name: 'plugin-llmstxt',

    async postBuild({siteConfig, outDir}) {
      const {url, baseUrl} = siteConfig;
      const siteBase = url + baseUrl;

      const llmsTxtLines = [];
      const llmsFullLines = [];

      // Header for llms.txt
      llmsTxtLines.push('# whest');
      llmsTxtLines.push('');
      llmsTxtLines.push('> NumPy-compatible math primitives with FLOP counting');
      llmsTxtLines.push('');

      for (const [section, pages] of Object.entries(SECTIONS)) {
        llmsTxtLines.push(`## ${section}`);
        llmsTxtLines.push('');

        llmsFullLines.push(`${'='.repeat(60)}`);
        llmsFullLines.push(`${section}`);
        llmsFullLines.push(`${'='.repeat(60)}`);
        llmsFullLines.push('');

        for (const [slug, description] of Object.entries(pages)) {
          const pageUrl = `${siteBase}${slug}`;
          llmsTxtLines.push(`- [${description}](${pageUrl})`);

          const htmlPath = resolveHtmlPath(outDir, slug);
          if (htmlPath) {
            const text = extractText(htmlPath);
            llmsFullLines.push(`--- ${slug} ---`);
            llmsFullLines.push(`URL: ${pageUrl}`);
            llmsFullLines.push('');
            llmsFullLines.push(text);
            llmsFullLines.push('');
          } else {
            llmsFullLines.push(`--- ${slug} ---`);
            llmsFullLines.push(`URL: ${pageUrl}`);
            llmsFullLines.push('');
            llmsFullLines.push('[Page not found in build output]');
            llmsFullLines.push('');
          }
        }

        llmsTxtLines.push('');
      }

      const llmsTxtPath = path.join(outDir, 'llms.txt');
      const llmsFullPath = path.join(outDir, 'llms-full.txt');

      // Ensure output directory exists
      fs.mkdirSync(outDir, {recursive: true});

      fs.writeFileSync(llmsTxtPath, llmsTxtLines.join('\n'), 'utf-8');
      fs.writeFileSync(llmsFullPath, llmsFullLines.join('\n'), 'utf-8');

      console.log(`[plugin-llmstxt] Generated ${llmsTxtPath}`);
      console.log(`[plugin-llmstxt] Generated ${llmsFullPath}`);
    },
  };
};
