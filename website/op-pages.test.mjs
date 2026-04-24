import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import ts from 'typescript';
import vm from 'node:vm';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const nodeRequire = createRequire(import.meta.url);

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

async function loadTsModule(relativePath, overrides = {}) {
  const source = await readSource(relativePath);
  const {outputText} = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
    fileName: path.basename(relativePath),
  });

  const module = {exports: {}};
  const context = {
    module,
    exports: module.exports,
    require(specifier) {
      if (specifier in overrides) return overrides[specifier];
      return nodeRequire(specifier);
    },
  };

  const script = new vm.Script(
    `(function (exports, require, module, __filename, __dirname) {${outputText}\n})`,
    {filename: `${relativePath}.test.cjs`},
  );
  script.runInNewContext(context)(
    module.exports,
    context.require,
    module,
    relativePath,
    path.dirname(path.join(websiteRoot, relativePath)),
  );

  return module.exports;
}

function collectText(node) {
  if (typeof node === 'string') return node;
  if (!node || typeof node !== 'object') return '';
  const children = node.props?.children;
  if (Array.isArray(children)) return children.map(collectText).join('');
  return collectText(children);
}

function flattenElements(node) {
  if (!node || typeof node !== 'object') return [];
  if (typeof node.type === 'function') {
    return flattenElements(node.type(node.props));
  }
  const children = node.props?.children;
  const childNodes = Array.isArray(children)
    ? children.flatMap(flattenElements)
    : flattenElements(children);
  return [node, ...childNodes];
}

test('/docs/api table links rows to standalone op pages', async () => {
  const [source, operationDataSource] = await Promise.all([
    readSource('components/api-reference/index.tsx'),
    readSource('components/api-reference/operation-data.ts'),
  ]);

  assert.match(source, /OperationCostIndex/);
  assert.match(operationDataSource, /detail_href/);
  assert.match(
    operationDataSource,
    /href:\s*op\.blocked \? undefined : op\.detail_href \?\? ref\?\.href/,
  );
  assert.doesNotMatch(source, /\/docs\/api\/ops\//);
});

test('canonical api route is present and avoids runtime filesystem access', async () => {
  const source = await readSource('app/docs/api/[[...slug]]/page.tsx');

  assert.match(source, /generateStaticParams/);
  assert.match(source, /source\.getPage/);
  assert.match(source, /publicApiRoutes/);
  assert.doesNotMatch(source, /node:fs|node:fs\/promises|from 'fs'|from "fs"/);
  assert.doesNotMatch(source, /process\.cwd\(/);
});

test('canonical api route prefers authored docs pages over colliding generated routes', async () => {
  const loaderCalls = [];
  const sourceCalls = [];
  const authoredPage = {
    data: {
      title: 'stats namespace',
      description: 'Authored namespace page.',
      toc: [],
      full: false,
      body() {
        return {
          type: 'authored-body',
          props: {children: 'Authored stats namespace'},
        };
      },
    },
  };

  const module = await loadTsModule('app/docs/api/[[...slug]]/page.tsx', {
    '@/.generated/public-api-routes.json': {
      __esModule: true,
      default: {
        stats: {
          kind: 'symbol',
          slug: 'stats-generated',
          href: '/docs/api/stats/',
          canonical_name: 'stats.generated',
        },
      },
    },
    '@/.generated/op-doc-imports': {opDocImports: {}},
    '@/.generated/symbol-doc-imports': {
      symbolDocImports: {
        'stats-generated': async () => {
          loaderCalls.push('generated');
          return {
            default: {
              display_name: 'generated stats',
              summary: 'Generated stats page.',
              import_path: 'we.stats.generated',
            },
          };
        },
      },
    },
    '@/components/api-reference/OperationDocNav': {
      __esModule: true,
      default() {
        return {type: 'nav', props: {}};
      },
    },
    '@/components/api-reference/OperationDocPage': {
      __esModule: true,
      default() {
        return {type: 'operation-doc', props: {}};
      },
    },
    '@/components/api-reference/PublicApiSymbolPage': {
      __esModule: true,
      default() {
        return {type: 'symbol-doc', props: {}};
      },
    },
    '@/components/mdx': {
      getMDXComponents() {
        return {};
      },
    },
    '@/lib/source': {
      source: {
        getPage(slug) {
          sourceCalls.push([...slug]);
          return authoredPage;
        },
        generateParams() {
          return [];
        },
      },
    },
    'fumadocs-ui/layouts/docs/page': {
      DocsBody(props) {
        return {type: 'DocsBody', props};
      },
      DocsDescription(props) {
        return {type: 'DocsDescription', props};
      },
      DocsPage(props) {
        return {type: 'DocsPage', props};
      },
      DocsTitle(props) {
        return {type: 'DocsTitle', props};
      },
    },
    'fumadocs-ui/mdx': {
      createRelativeLink() {
        return function RelativeLink(props) {
          return {type: 'a', props};
        };
      },
    },
    'lucide-react': {
      ChevronRight() {
        return {type: 'ChevronRight', props: {}};
      },
    },
    'next/link': {
      __esModule: true,
      default(props) {
        return {type: 'a', props};
      },
    },
    'next/navigation': {
      notFound() {
        throw new Error('notFound');
      },
    },
  });

  const tree = await module.default({
    params: Promise.resolve({slug: ['stats']}),
  });
  const elements = flattenElements(tree);

  assert.deepEqual(sourceCalls, [['api', 'stats']]);
  assert.deepEqual(loaderCalls, []);
  assert.equal(
    elements.some(
      (element) =>
        element.type === 'DocsTitle' &&
        collectText(element) === 'stats namespace',
    ),
    true,
  );
  assert.equal(
    elements.some(
      (element) =>
        element.type === 'authored-body' &&
        collectText(element).includes('Authored stats namespace'),
    ),
    true,
  );
});

test('generated route inventory reserves authored namespace landing pages for the docs tree', async () => {
  const [routesSource, metaSource] = await Promise.all([
    readSource('.generated/public-api-routes.json'),
    readSource('content/docs/api/meta.json'),
  ]);
  const routes = JSON.parse(routesSource);
  const meta = JSON.parse(metaSource);

  for (const namespace of ['random', 'stats', 'flops', 'testing']) {
    assert.equal(meta.pages.includes(namespace), true, namespace);
    assert.equal(routes[namespace], undefined, namespace);
  }

  for (const pathKey of [
    'random/sample',
    'stats/norm',
    'stats/norm/pdf',
    'flops/einsum-cost',
    'testing/assert-allclose',
  ]) {
    assert.equal(typeof routes[pathKey]?.href, 'string', pathKey);
  }
});

test('legacy op route redirects to canonical api paths', async () => {
  const source = await readSource('app/docs/api/ops/[slug]/page.tsx');

  assert.match(source, /legacyOpRedirects/);
  assert.match(source, /redirect\(/);
  assert.doesNotMatch(source, /OperationDocPage/);
});

test('canonical api pages use dedicated renderers for op and symbol records', async () => {
  const pageSource = await readSource('app/docs/api/[[...slug]]/page.tsx');
  const symbolPageSource = await readSource('components/api-reference/PublicApiSymbolPage.tsx');

  assert.match(pageSource, /OperationDocHeader/);
  assert.match(pageSource, /PublicApiSymbolPage/);
  assert.match(symbolPageSource, /OperationDocBody/);
  assert.match(symbolPageSource, /OperationDocSignature/);
});

test('api docs tree reserves the dedicated operation-cost-index child page', async () => {
  const [routesSource, metaSource] = await Promise.all([
    readSource('.generated/public-api-routes.json'),
    readSource('content/docs/api/meta.json'),
  ]);
  const routes = JSON.parse(routesSource);
  const meta = JSON.parse(metaSource);

  assert.equal(meta.pages.includes('operation-cost-index'), true);
  assert.equal(routes['operation-cost-index'], undefined);
});

test('api runtime code blocks use shared whest themes and preserve shiki backgrounds', async () => {
  const bodySource = await readSource('components/api-reference/OperationDocBody.tsx');
  const exampleSource = await readSource('components/api-reference/OperationDocExample.tsx');
  const themeSource = await readSource('components/api-reference/runtime-shiki-themes.ts');

  assert.match(bodySource, /from '\.\/runtime-shiki-themes'/);
  assert.match(exampleSource, /from '\.\/runtime-shiki-themes'/);
  assert.match(bodySource, /ServerCodeBlock/);
  assert.match(exampleSource, /ServerCodeBlock/);
  assert.doesNotMatch(bodySource, /github-light|github-dark/);
  assert.doesNotMatch(exampleSource, /github-light|github-dark/);
  assert.match(bodySource, /keepBackground/);
  assert.match(exampleSource, /keepBackground/);
  assert.match(themeSource, /whestLight/);
  assert.match(themeSource, /whestDark/);
  assert.match(themeSource, /editor\.background': '#232628'/);
});

test('api dark-mode styling is explicitly defined for shared chrome', async () => {
  const cssSource = await readSource('components/api-reference/styles.module.css');
  const navSource = await readSource('components/api-reference/OperationDocNav.tsx');
  const globalSource = await readSource('app/global.css');

  assert.match(cssSource, /:global\(\.dark\)\s+\.apiReference/);
  assert.match(cssSource, /--api-link-color/);
  assert.match(cssSource, /--api-nav-card-border/);
  assert.match(cssSource, /--api-chip-counted-bg/);
  assert.match(cssSource, /--api-summary-fg/);
  assert.match(cssSource, /--api-signature-ink/);
  assert.match(cssSource, /:global\(\.dark\)\s+\.docHeader/);
  assert.match(cssSource, /\.apiCodeFigure/);
  assert.match(cssSource, /\.apiCodeViewport/);
  assert.match(cssSource, /\.apiCodeFigure\s+:global\(pre\)/);
  assert.match(cssSource, /\.apiCodeFigure\s+:global\(\.line\)/);
  assert.match(cssSource, /color:\s*var\(--shiki-dark/);
  assert.match(navSource, /styles\.docNav/);
  assert.match(navSource, /styles\.docNavCard/);
  assert.match(globalSource, /\.dark article > div\.prose :not\(pre\) > code/);
  assert.match(globalSource, /border-color:\s*rgba\(232,\s*234,\s*235,\s*0\.26\)/);
});

test('api billing presentation is unified around a single cost field', async () => {
  const tableSource = await readSource('components/api-reference/ApiReference.tsx');
  const rowSource = await readSource('components/api-reference/OperationRow.tsx');
  const overlaySource = await readSource('components/api-reference/OperationDocOverlay.tsx');
  const billedCostSource = await readSource('components/api-reference/BilledCost.tsx');

  assert.match(tableSource, /<th>Cost<\/th>/);
  assert.doesNotMatch(tableSource, /<th>Weight<\/th>/);
  assert.doesNotMatch(tableSource, /<th>Cost Formula<\/th>/);
  assert.match(tableSource, /colSpan=\{4\}/);

  assert.match(overlaySource, /Cost/);
  assert.doesNotMatch(overlaySource, /Weight/);
  assert.doesNotMatch(overlaySource, /Cost Formula/);

  assert.match(rowSource, /BilledCost/);
  assert.match(overlaySource, /BilledCost/);
  assert.match(billedCostSource, /weight !== 1/);
  assert.match(billedCostSource, /display_type === 'free'/);
  assert.match(billedCostSource, /display_type === 'blocked'/);
  assert.match(billedCostSource, /Weight multiplier/);
  assert.match(billedCostSource, /formatWeight/);
  assert.match(billedCostSource, /stripMathDelimiters/);
  assert.match(billedCostSource, /op\.cost_formula/);
  assert.match(billedCostSource, />0</);
  assert.match(billedCostSource, /&mdash;|\\u2014/);
});
