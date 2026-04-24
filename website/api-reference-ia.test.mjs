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

async function readJson(relativePath) {
  return JSON.parse(await readSource(relativePath));
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

function directChildImports(symbolRecords, namespaceImportPath) {
  const directDepth = namespaceImportPath.split('.').length + 1;
  return Object.values(symbolRecords)
    .map((record) => record.import_path)
    .filter((importPath) => importPath.startsWith(`${namespaceImportPath}.`))
    .filter((importPath) => importPath.split('.').length === directDepth)
    .sort();
}

function pathKeyFromHref(href) {
  return href.replace(/^\/docs\/api\//, '').replace(/\/$/, '');
}

async function loadActualPublicApiDataModule() {
  const [opsJson, opRefsJson, publicApiSymbolsJson] = await Promise.all([
    readJson('public/ops.json'),
    readJson('.generated/op-refs.json'),
    readJson('.generated/public-api-symbols.json'),
  ]);
  const operationDataModule = await loadTsModule(
    'components/api-reference/operation-data.ts',
    {
      '../../public/ops.json': {__esModule: true, default: opsJson},
      '../../.generated/op-refs.json': {__esModule: true, default: opRefsJson},
    },
  );
  const publicApiDataModule = await loadTsModule(
    'components/api-reference/public-api-data.ts',
    {
      '../../.generated/public-api-symbols.json': {
        __esModule: true,
        default: publicApiSymbolsJson,
      },
      './operation-data': {
        getPublicApiOperations: operationDataModule.getPublicApiOperations,
      },
    },
  );
  return {publicApiDataModule, publicApiSymbolsJson};
}

test('api reference boundaries isolate the operation cost index and public api data loader', async () => {
  const [
    indexSource,
    apiReferenceSource,
    operationCostIndexSource,
    operationDataSource,
    publicApiDataSource,
  ] = await Promise.all([
    readSource('components/api-reference/index.tsx'),
    readSource('components/api-reference/ApiReference.tsx'),
    readSource('components/api-reference/OperationCostIndex.tsx'),
    readSource('components/api-reference/operation-data.ts'),
    readSource('components/api-reference/public-api-data.ts'),
  ]);

  assert.match(indexSource, /import OperationCostIndex from '\.\/OperationCostIndex'/);
  assert.match(indexSource, /import \{getPublicApiOperations\} from '\.\/operation-data'/);
  assert.match(indexSource, /const operations = getPublicApiOperations\(\)/);
  assert.doesNotMatch(indexSource, /opsData|opRefsJson|publicApiSymbolsJson|getPublicApiData/);

  assert.match(apiReferenceSource, /OperationCostIndex/);
  assert.doesNotMatch(apiReferenceSource, /SymbolGrid|symbols:\s*PublicApiSymbol\[]|next\/link/);

  assert.match(operationCostIndexSource, /FilterBar/);
  assert.match(operationCostIndexSource, /OperationRow/);
  assert.doesNotMatch(operationCostIndexSource, /SymbolGrid|next\/link/);

  assert.match(operationDataSource, /opsData/);
  assert.match(operationDataSource, /opRefsJson/);
  assert.match(operationDataSource, /export function getPublicApiOperations/);
  assert.doesNotMatch(operationDataSource, /publicApiSymbolsJson/);

  assert.doesNotMatch(publicApiDataSource, /opsData/);
  assert.doesNotMatch(publicApiDataSource, /opRefsJson/);
  assert.match(publicApiDataSource, /publicApiSymbolsJson/);
  assert.match(publicApiDataSource, /export function getPublicApiData/);
  assert.match(publicApiDataSource, /getPublicApiOperations/);
  assert.match(publicApiDataSource, /namespaces/);
  assert.match(publicApiDataSource, /highlightedEntries/);
});

test('operation data loader resolves canonical metadata without loading symbol data', async () => {
  const operationDataSource = await readSource(
    'components/api-reference/operation-data.ts',
  );
  const {outputText} = ts.transpileModule(operationDataSource, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
    fileName: 'operation-data.ts',
  });

  const opsFixture = {
    operations: [
      {
        name: 'we.alias_op',
        module: 'numpy.core',
        whest_ref: 'we.alias_op',
        numpy_ref: 'np.alias_op',
        category: 'counted_matrix',
        cost_formula: 'n',
        cost_formula_latex: '$n$',
        free: false,
        blocked: false,
        status: 'ok',
        notes: 'alias op',
        weight: 2,
      },
      {
        name: 'numpy.linalg.canonical_op',
        module: 'numpy.linalg',
        whest_ref: 'we.canonical_op',
        numpy_ref: 'np.linalg.canonical_op',
        category: 'free',
        cost_formula: '0',
        cost_formula_latex: '$0$',
        free: true,
        blocked: false,
        status: 'ok',
        notes: 'canonical op',
        weight: 1,
      },
      {
        name: 'we.blocked_op',
        module: 'numpy.random',
        whest_ref: 'we.blocked_op',
        numpy_ref: 'np.blocked_op',
        category: 'blacklisted',
        cost_formula: 'blocked',
        cost_formula_latex: '$-$',
        free: false,
        blocked: true,
        status: 'blocked',
        notes: 'blocked op',
        weight: 1,
        detail_href: '/docs/api/blocked-op/',
      },
    ],
  };
  const opRefsFixture = {
    'we.alias_op': {
      canonical_name: 'numpy.linalg.canonical_op',
      href: '/docs/api/alias-op/',
    },
    'we.blocked_op': {
      canonical_name: 'we.blocked_op',
      href: '/docs/api/blocked-op/',
    },
  };

  const module = {exports: {}};
  const context = {
    module,
    exports: module.exports,
    require(specifier) {
      if (specifier === '../../public/ops.json') {
        return {__esModule: true, default: opsFixture};
      }
      if (specifier === '../../.generated/op-refs.json') {
        return {__esModule: true, default: opRefsFixture};
      }
      throw new Error(`Unexpected dependency: ${specifier}`);
    },
  };

  const script = new vm.Script(
    `(function (exports, require, module, __filename, __dirname) {${outputText}\n})`,
    {filename: 'operation-data.test.cjs'},
  );
  script.runInNewContext(context)(
    module.exports,
    context.require,
    module,
    'operation-data.ts',
    websiteRoot,
  );

  const operations = module.exports.getPublicApiOperations();
  assert.equal(operations.length, 3);

  const aliasOp = operations.find((operation) => operation.name === 'we.alias_op');
  assert.deepEqual(
    {
      area: aliasOp.area,
      displayType: aliasOp.display_type,
      href: aliasOp.href,
    },
    {
      area: 'linalg',
      displayType: 'free',
      href: '/docs/api/alias-op/',
    },
  );

  const blockedOp = operations.find(
    (operation) => operation.name === 'we.blocked_op',
  );
  assert.equal(blockedOp.area, 'random');
  assert.equal(blockedOp.display_type, 'blocked');
  assert.equal(blockedOp.href, undefined);
});

test('api landing page is authored as a hub-first page with the cost index lower on the page', async () => {
  const [mdxSource, mdxComponentsSource, hubSource, highlightsSource] =
    await Promise.all([
      readSource('content/docs/api/index.mdx'),
      readSource('components/mdx.tsx'),
      readSource('components/api-reference/ApiNamespaceHub.tsx'),
      readSource('components/api-reference/ApiEntryHighlights.tsx'),
    ]);

  assert.match(mdxSource, /<ApiNamespaceHub\s*\/>/);
  assert.match(mdxSource, /<ApiEntryHighlights\s*\/>/);
  assert.doesNotMatch(mdxSource, /<OperationCostIndex/);
  assert.ok(
    mdxSource.indexOf('<ApiNamespaceHub />') <
      mdxSource.indexOf('<ApiEntryHighlights />'),
  );
  assert.doesNotMatch(mdxSource, /<ApiReference\s*\/>/);
  assert.match(mdxSource, /operation-cost-index/);

  assert.match(mdxComponentsSource, /ApiNamespaceHub/);
  assert.match(mdxComponentsSource, /ApiEntryHighlights/);
  assert.match(mdxComponentsSource, /OperationCostIndex/);

  assert.match(hubSource, /getHubNamespaces/);
  assert.doesNotMatch(hubSource, /rootNamespace|child_namespaces/);

  assert.match(highlightsSource, /getHubEntryHighlights/);
  assert.match(highlightsSource, /getRelatedOperationAreas/);
  assert.doesNotMatch(highlightsSource, /entry\.kind === 'namespace'/);
});

test('namespace hub renders only direct root namespaces and links only concrete entry pages', async () => {
  const fixture = [
    {
      import_path: 'we.linalg',
      href: '/docs/api/linalg/',
      summary: 'Linear algebra operations.',
      highlighted_entries: [
        {
          kind: 'operation',
          title: 'we.matmul',
          href: '/docs/api/matmul/',
        },
      ],
    },
    {
      import_path: 'we.random',
      href: '/docs/api/random/',
      summary: 'Random generation.',
      highlighted_entries: [],
    },
  ];

  const linkCalls = [];
  const styles = new Proxy({}, {get: (_target, key) => String(key)});
  const module = await loadTsModule('components/api-reference/ApiNamespaceHub.tsx', {
    './public-api-data': {
      getHubNamespaces() {
        return fixture;
      },
      getHubNamespacePreviewLinks(importPath) {
        return importPath === 'we.linalg'
          ? [
              {
                kind: 'operation',
                title: 'we.matmul',
                href: '/docs/api/matmul/',
              },
            ]
          : [];
      },
    },
    './styles.module.css': {__esModule: true, default: styles},
    'next/link': {
      __esModule: true,
      default(props) {
        linkCalls.push(props.href);
        return {
          type: 'a',
          props,
        };
      },
    },
  });

  const tree = module.default();
  const elements = flattenElements(tree);
  const namespaceTitles = elements
    .filter((element) => element.props?.className === 'hubNamespaceTitleLink')
    .map((element) => collectText(element));

  assert.deepEqual(namespaceTitles, ['linalg', 'random']);
  assert.deepEqual(linkCalls, [
    '/docs/api/linalg/',
    '/docs/api/matmul/',
    '/docs/api/random/',
  ]);
});

test('entry highlights leave namespace items unlinked while linking concrete destinations', async () => {
  const fixture = [
    {
      kind: 'operation',
      title: 'we.matmul',
      href: '/docs/api/matmul/',
      description: 'Operation summary.',
      namespace_import_path: 'we.linalg',
    },
  ];
  const relatedAreas = [
    {
      import_path: 'we.fft',
      href: '/docs/api/fft/',
      summary: 'FFT operations.',
      preview_links: [
        {
          kind: 'operation',
          title: 'we.fft.fft',
          href: '/docs/api/fft/fft/',
          description: 'FFT leaf page.',
        },
      ],
    },
  ];

  const linkCalls = [];
  const styles = new Proxy({}, {get: (_target, key) => String(key)});
  const module = await loadTsModule('components/api-reference/ApiEntryHighlights.tsx', {
    './public-api-data': {
      getHubEntryHighlights() {
        return fixture;
      },
      getRelatedOperationAreas() {
        return relatedAreas;
      },
    },
    './styles.module.css': {__esModule: true, default: styles},
    'next/link': {
      __esModule: true,
      default(props) {
        linkCalls.push(props.href);
        return {
          type: 'a',
          props,
        };
      },
    },
  });

  const tree = module.default();
  const articles = flattenElements(tree).filter((element) => element.type === 'article');

  assert.equal(articles.length, 2);
  assert.equal(collectText(articles[0]).includes('we.matmul'), true);
  assert.equal(collectText(articles[1]).includes('fft'), true);
  assert.deepEqual(linkCalls, ['/docs/api/matmul/', '/docs/api/fft/fft/']);
});

test('namespace landing pages are authored, registered in MDX, and listed in the docs tree', async () => {
  const [
    mdxComponentsSource,
    namespaceListSource,
    randomPage,
    statsPage,
    flopsPage,
    testingPage,
    metaSource,
  ] = await Promise.all([
    readSource('components/mdx.tsx'),
    readSource('components/api-reference/NamespaceSymbolList.tsx'),
    readSource('content/docs/api/random.mdx'),
    readSource('content/docs/api/stats.mdx'),
    readSource('content/docs/api/flops.mdx'),
    readSource('content/docs/api/testing.mdx'),
    readSource('content/docs/api/meta.json'),
  ]);

  assert.match(mdxComponentsSource, /NamespaceSymbolList/);
  assert.match(namespaceListSource, /getNamespaceSectionData/);
  assert.match(namespaceListSource, /namespaceMembers/);

  assert.match(randomPage, /<NamespaceSymbolList namespace="random" \/>/);
  assert.match(statsPage, /<NamespaceSymbolList namespace="stats" \/>/);
  assert.match(flopsPage, /<NamespaceSymbolList namespace="flops" \/>/);
  assert.match(testingPage, /<NamespaceSymbolList namespace="testing" \/>/);

  const meta = JSON.parse(metaSource);
  assert.deepEqual(meta.pages, [
    'index',
    'operation-cost-index',
    'random',
    'stats',
    'flops',
    'testing',
    'for-agents',
  ]);
});

test('generated namespace inventories stay aligned with the authored docs IA', async () => {
  const [
    {publicApiDataModule, publicApiSymbolsJson},
    publicApiRoutes,
  ] = await Promise.all([
    loadActualPublicApiDataModule(),
    readJson('.generated/public-api-routes.json'),
  ]);
  const {getNamespaceSectionData, getPublicApiData} = publicApiDataModule;
  const publicApiData = getPublicApiData();
  const rootNamespace = publicApiData.namespaces.find(
    (namespace) => namespace.import_path === 'we',
  );

  assert.ok(rootNamespace);
  for (const namespace of ['random', 'stats', 'flops', 'testing']) {
    assert.equal(
      rootNamespace.child_namespaces.includes(`we.${namespace}`),
      true,
      namespace,
    );
  }

  const expectedSectionTitles = {
    random: ['Sampling routines', 'State and configuration helpers'],
    stats: ['Distribution objects'],
    flops: [
      'Core estimator primitives',
      'Linear algebra estimators',
      'FFT and signal estimators',
      'Polynomial helpers',
    ],
    testing: ['Assertion helpers'],
  };
  const expectedImportsByNamespace = {
    random: [
      ...new Set([
        ...directChildImports(publicApiSymbolsJson, 'we.random'),
        ...publicApiData.operations
          .filter((operation) => operation.area === 'random' && operation.href)
          .map((operation) => operation.whest_ref),
      ]),
    ].sort(),
    stats: directChildImports(publicApiSymbolsJson, 'we.stats'),
    flops: directChildImports(publicApiSymbolsJson, 'we.flops'),
    testing: directChildImports(publicApiSymbolsJson, 'we.testing'),
  };

  for (const namespace of ['random', 'stats', 'flops', 'testing']) {
    const sectionData = getNamespaceSectionData(namespace);
    const entries = sectionData.sections.flatMap((section) => section.entries);
    const sectionTitles = Array.from(
      sectionData.sections,
      (section) => section.title,
    );
    const importPaths = Array.from(entries, (entry) => entry.import_path);

    assert.deepEqual(
      sectionTitles,
      expectedSectionTitles[namespace],
      namespace,
    );
    assert.equal(new Set(importPaths).size, importPaths.length, namespace);
    assert.deepEqual(
      [...importPaths].sort(),
      expectedImportsByNamespace[namespace],
      namespace,
    );

    for (const entry of entries) {
      assert.equal(entry.import_path.startsWith(`we.${namespace}.`), true);
      const route = publicApiRoutes[pathKeyFromHref(entry.href)];
      assert.ok(route, entry.href);
      assert.equal(route.href, entry.href);

      for (const member of entry.members) {
        const memberRoute = publicApiRoutes[pathKeyFromHref(member.href)];
        assert.ok(memberRoute, member.href);
        assert.equal(memberRoute.href, member.href);
      }
    }
  }

  const randomSections = new Map(
    getNamespaceSectionData('random').sections.map((section) => [
      section.title,
      section.entries.map((entry) => entry.import_path),
    ]),
  );
  assert.equal(
    randomSections
      .get('State and configuration helpers')
      .includes('we.random.RandomState'),
    true,
  );
  assert.equal(
    randomSections
      .get('State and configuration helpers')
      .includes('we.random.SeedSequence'),
    true,
  );
  assert.equal(
    randomSections
      .get('State and configuration helpers')
      .includes('we.random.symmetric'),
    true,
  );
  assert.equal(
    randomSections.get('Sampling routines').includes('we.random.sample'),
    true,
  );
  assert.equal(
    randomSections.get('State and configuration helpers').includes('we.random.sample'),
    false,
  );
});

test('namespace symbol list renders grouped entries and member links from shared section data', async () => {
  const linkCalls = [];
  const styles = new Proxy({}, {get: (_target, key) => String(key)});
  const module = await loadTsModule(
    'components/api-reference/NamespaceSymbolList.tsx',
    {
      './public-api-data': {
        getNamespaceSectionData() {
          return {
            sections: [
              {
                title: 'Distribution objects',
                description: 'Canonical object pages first.',
                entries: [
                  {
                    href: '/docs/api/stats/norm/',
                    import_path: 'we.stats.norm',
                    summary: 'Normal distribution.',
                    members: [
                      {
                        href: '/docs/api/stats/norm/pdf/',
                        label: 'we.stats.norm.pdf',
                      },
                    ],
                  },
                ],
              },
            ],
          };
        },
      },
      './styles.module.css': {__esModule: true, default: styles},
      'next/link': {
        __esModule: true,
        default(props) {
          linkCalls.push(props.href);
          return {
            type: 'a',
            props,
          };
        },
      },
    },
  );

  const tree = module.default({namespace: 'stats'});
  const elements = flattenElements(tree);

  assert.equal(
    elements.some(
      (element) =>
        element.props?.className === 'subsectionTitle' &&
        collectText(element) === 'Distribution objects',
    ),
    true,
  );
  assert.equal(
    elements.some(
      (element) =>
        element.props?.className === 'namespaceRowTitle' &&
        collectText(element).includes('we.stats.norm'),
    ),
    true,
  );
  assert.deepEqual(linkCalls, [
    '/docs/api/stats/norm/',
    '/docs/api/stats/norm/pdf/',
  ]);
});

test('random namespace section data separates samplers from helper-style entry points', async () => {
  const styles = new Proxy({}, {get: (_target, key) => String(key)});
  const module = await loadTsModule(
    'components/api-reference/public-api-data.ts',
    {
      '../../.generated/public-api-symbols.json': {
        __esModule: true,
        default: {
          randomState: {
            aliases: [],
            display_name: 'RandomState',
            href: '/docs/api/random/random-state/',
            import_path: 'we.random.RandomState',
            kind: 'class',
            members: [],
            module: 'numpy.random',
            name: 'RandomState',
            status_note: '',
            summary: 'Legacy RNG container.',
          },
          seedSequence: {
            aliases: [],
            display_name: 'SeedSequence',
            href: '/docs/api/random/seed-sequence/',
            import_path: 'we.random.SeedSequence',
            kind: 'class',
            members: [],
            module: 'numpy.random',
            name: 'SeedSequence',
            status_note: '',
            summary: 'Seed mixing helper.',
          },
          symmetric: {
            aliases: [],
            display_name: 'symmetric',
            href: '/docs/api/random/symmetric/',
            import_path: 'we.random.symmetric',
            kind: 'function',
            members: [],
            module: 'whest.random',
            name: 'symmetric',
            status_note: '',
            summary: 'Symmetry-aware sampling helper.',
          },
          sample: {
            aliases: [],
            display_name: 'sample',
            href: '/docs/api/random/sample/',
            import_path: 'we.random.sample',
            kind: 'function',
            members: [],
            module: 'numpy.random',
            name: 'sample',
            status_note: '',
            summary: 'Sample random values.',
          },
        },
      },
      './operation-data': {
        getPublicApiOperations() {
          return [
            {
              area: 'random',
              blocked: false,
              category: 'counted',
              cost_formula: 'n',
              cost_formula_latex: '$n$',
              display_type: 'counted',
              free: false,
              href: '/docs/api/random/normal/',
              module: 'numpy.random',
              name: 'we.random.normal',
              notes: 'Normal sampler.',
              numpy_ref: 'np.random.normal',
              status: 'ok',
              weight: 1,
              whest_ref: 'we.random.normal',
            },
            {
              area: 'random',
              blocked: false,
              category: 'counted',
              cost_formula: 'n',
              cost_formula_latex: '$n$',
              display_type: 'counted',
              free: false,
              href: '/docs/api/random/default-rng/',
              module: 'numpy.random',
              name: 'we.random.default_rng',
              notes: 'Generator constructor.',
              numpy_ref: 'np.random.default_rng',
              status: 'ok',
              weight: 1,
              whest_ref: 'we.random.default_rng',
            },
          ];
        },
      },
      './styles.module.css': {__esModule: true, default: styles},
    },
  );

  const {getNamespaceSectionData} = module;
  const data = getNamespaceSectionData('random');
  const byTitle = new Map(data.sections.map((section) => [section.title, section]));

  assert.deepEqual([...byTitle.keys()], [
    'Sampling routines',
    'State and configuration helpers',
  ]);
  assert.equal(byTitle.has('Namespace entry points'), false);

  const samplingNames = byTitle
    .get('Sampling routines')
    .entries.map((entry) => entry.import_path);
  const helperNames = byTitle
    .get('State and configuration helpers')
    .entries.map((entry) => entry.import_path);

  assert.equal(samplingNames.includes('we.random.sample'), true);
  assert.equal(samplingNames.includes('we.random.RandomState'), false);
  assert.equal(samplingNames.includes('we.random.SeedSequence'), false);
  assert.equal(helperNames.includes('we.random.RandomState'), true);
  assert.equal(helperNames.includes('we.random.SeedSequence'), true);
  assert.equal(helperNames.includes('we.random.symmetric'), true);
  assert.equal(helperNames.includes('we.random.default_rng'), true);
});

test('api hub data narrows to curated namespaces, concrete highlights, and related areas', async () => {
  const {publicApiDataModule} = await loadActualPublicApiDataModule();

  const hubNamespaces = publicApiDataModule.getHubNamespaces();
  const relatedAreas = publicApiDataModule.getRelatedOperationAreas();
  const highlights = publicApiDataModule.getHubEntryHighlights();

  assert.deepEqual(
    Array.from(hubNamespaces, (namespace) => namespace.import_path),
    ['we.random', 'we.stats', 'we.flops', 'we.testing'],
  );
  assert.deepEqual(
    Array.from(relatedAreas, (namespace) => namespace.import_path),
    ['we.fft', 'we.linalg'],
  );
  assert.equal(highlights.every((entry) => entry.kind !== 'namespace'), true);
  assert.deepEqual(
    Array.from(highlights, (entry) => entry.title),
    [
      'we.einsum',
      'we.random.symmetric',
      'we.stats.norm',
      'we.flops.einsum_cost',
      'we.testing.assert_allclose',
      'we.configure',
    ],
  );
});

test('api hub mdx delegates the full table to the dedicated operation-cost-index child page', async () => {
  const [hubSource, indexPageSource, meta] = await Promise.all([
    readSource('content/docs/api/index.mdx'),
    readSource('content/docs/api/operation-cost-index.mdx'),
    readJson('content/docs/api/meta.json'),
  ]);

  assert.doesNotMatch(hubSource, /<OperationCostIndex/);
  assert.match(hubSource, /\/docs\/api\/operation-cost-index\//);
  assert.match(indexPageSource, /<OperationCostIndex showHeading=\{false\} \/>/);
  assert.deepEqual(meta.pages.slice(0, 3), ['index', 'operation-cost-index', 'random']);
});
