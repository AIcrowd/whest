import publicApiSymbolsJson from '../../.generated/public-api-symbols.json';
import type {PublicApiSymbolRecord} from './op-doc-types';
import {getPublicApiOperations, type Operation} from './operation-data';

export interface PublicApiMemberLink {
  label: string;
  href: string;
}

export interface PublicApiSymbol {
  aliases: string[];
  display_name: string;
  href: string;
  import_path: string;
  kind: 'call' | 'object' | 'class';
  members: PublicApiMemberLink[];
  module_name: string;
  name: string;
  namespace_import_path: string;
  status_note?: string;
  summary: string;
}

export interface PublicApiHighlightedEntry {
  kind: 'namespace' | 'symbol' | 'operation';
  title: string;
  href: string;
  description: string;
  namespace_import_path: string;
}

export interface PublicApiNamespace {
  import_path: string;
  display_name: string;
  href: string;
  summary: string;
  operation_count: number;
  symbol_count: number;
  child_namespaces: string[];
  highlighted_entries: PublicApiHighlightedEntry[];
}

export interface PublicApiData {
  operations: Operation[];
  symbols: PublicApiSymbol[];
  namespaces: PublicApiNamespace[];
  highlightedEntries: PublicApiHighlightedEntry[];
}

export type ApiNamespaceKey = 'random' | 'stats' | 'flops' | 'testing';

export interface NamespaceSectionEntry {
  href: string;
  import_path: string;
  summary: string;
  members: PublicApiMemberLink[];
}

export interface NamespaceSection {
  title: string;
  description: string;
  entries: NamespaceSectionEntry[];
}

export interface NamespaceSectionData {
  sections: NamespaceSection[];
}

export interface RelatedOperationArea extends PublicApiNamespace {
  preview_links: PublicApiHighlightedEntry[];
}

const HUB_NAMESPACE_IMPORT_PATHS = [
  'we.random',
  'we.stats',
  'we.flops',
  'we.testing',
] as const;

const RELATED_OPERATION_AREA_IMPORT_PATHS = ['we.fft', 'we.linalg'] as const;

const HUB_ENTRY_POINT_NAMES = [
  'we.einsum',
  'we.random.symmetric',
  'we.stats.norm',
  'we.flops.einsum_cost',
  'we.testing.assert_allclose',
  'we.configure',
] as const;

const HUB_NAMESPACE_PREVIEW_NAMES = {
  'we.random': ['we.random.normal', 'we.random.uniform', 'we.random.symmetric'],
  'we.stats': ['we.stats.norm', 'we.stats.uniform', 'we.stats.lognorm'],
  'we.flops': [
    'we.flops.einsum_cost',
    'we.flops.matrix_power_cost',
    'we.flops.fft_cost',
  ],
  'we.testing': [
    'we.testing.assert_allclose',
    'we.testing.assert_array_equal',
  ],
} as const;

const RELATED_OPERATION_AREA_PREVIEW_NAMES = {
  'we.fft': ['we.fft.fft', 'we.fft.fftfreq'],
  'we.linalg': ['we.linalg.cholesky', 'we.linalg.svd'],
} as const;

type NamespaceAccumulator = {
  import_path: string;
  display_name: string;
  href: string;
  summary: string;
  operation_count: number;
  symbol_count: number;
  child_namespaces: Set<string>;
  direct_operations: Operation[];
  direct_symbols: PublicApiSymbol[];
  member_highlights: PublicApiHighlightedEntry[];
};

export function getPublicApiData(): PublicApiData {
  const publicApiSymbols =
    publicApiSymbolsJson as Record<string, PublicApiSymbolRecord>;
  const operations = getPublicApiOperations();

  const symbols: PublicApiSymbol[] = Object.values(publicApiSymbols)
    .map((symbol) => {
      const namespaceImportPath = namespaceForSymbol(symbol.import_path);
      return {
        aliases: symbol.aliases,
        display_name: symbol.display_name,
        href: symbol.href,
        import_path: symbol.import_path,
        kind: normalizeSymbolKind(symbol.kind),
        members: (symbol.members ?? []).map((member) => ({
          href: member.href,
          label: member.label,
        })),
        module_name: symbol.module,
        name: symbol.name,
        namespace_import_path: namespaceImportPath,
        status_note: symbol.status_note || undefined,
        summary: symbol.summary,
      };
    })
    .sort((left, right) => left.import_path.localeCompare(right.import_path));

  const namespaces = buildNamespaces(operations, symbols);
  const highlightedEntries =
    namespaces.find((namespace) => namespace.import_path === 'we')
      ?.highlighted_entries ?? [];

  return {
    operations,
    symbols,
    namespaces,
    highlightedEntries,
  };
}

export function getHubNamespaces(): PublicApiNamespace[] {
  const {namespaces} = getPublicApiData();
  return HUB_NAMESPACE_IMPORT_PATHS.map((importPath) =>
    namespaces.find((namespace) => namespace.import_path === importPath),
  ).filter((namespace): namespace is PublicApiNamespace => Boolean(namespace));
}

export function getHubEntryHighlights(): PublicApiHighlightedEntry[] {
  const {operations, symbols} = getPublicApiData();
  return HUB_ENTRY_POINT_NAMES.map((name) =>
    resolveConcretePublicApiEntry(name, symbols, operations),
  ).filter(
    (entry): entry is PublicApiHighlightedEntry =>
      Boolean(entry) && entry.kind !== 'namespace',
  );
}

export function getHubNamespacePreviewLinks(
  importPath: (typeof HUB_NAMESPACE_IMPORT_PATHS)[number],
): PublicApiHighlightedEntry[] {
  const {operations, symbols} = getPublicApiData();
  return HUB_NAMESPACE_PREVIEW_NAMES[importPath]
    .map((name) => resolveConcretePublicApiEntry(name, symbols, operations))
    .filter(
      (entry): entry is PublicApiHighlightedEntry =>
        Boolean(entry) && entry.kind !== 'namespace',
    );
}

export function getRelatedOperationAreas(): RelatedOperationArea[] {
  const {namespaces, operations, symbols} = getPublicApiData();
  return RELATED_OPERATION_AREA_IMPORT_PATHS.map((importPath) => {
    const namespace = namespaces.find(
      (candidate) => candidate.import_path === importPath,
    );
    if (!namespace) return null;
    const previewLinks =
      RELATED_OPERATION_AREA_PREVIEW_NAMES[
        importPath as keyof typeof RELATED_OPERATION_AREA_PREVIEW_NAMES
      ]
        .map((name) => resolveConcretePublicApiEntry(name, symbols, operations))
        .filter(
          (entry): entry is PublicApiHighlightedEntry =>
            Boolean(entry) && entry.kind !== 'namespace',
        );

    return {
      ...namespace,
      preview_links: previewLinks,
    };
  }).filter((namespace): namespace is RelatedOperationArea => Boolean(namespace));
}

export function getNamespaceSectionData(
  namespace: ApiNamespaceKey,
): NamespaceSectionData {
  const {symbols} = getPublicApiData();
  const namespaceSymbols = getNamespaceSymbols(symbols, `we.${namespace}`);

  if (namespace === 'stats') {
    return {
      sections: [
        {
          title: 'Distribution objects',
          description:
            'Start with the canonical distribution objects, then use their child links to drill into generated method pages.',
          entries: namespaceSymbols,
        },
      ],
    };
  }

  if (namespace === 'random') {
    const helperNames = new Set([
      'we.random.RandomState',
      'we.random.SeedSequence',
      'we.random.default_rng',
      'we.random.get_state',
      'we.random.seed',
      'we.random.set_state',
      'we.random.symmetric',
    ]);
    const randomOperations = getOperationEntriesForArea('random');
    const namespaceEntries = namespaceSymbols.map((entry) => ({
      ...entry,
      members: [],
    }));
    const helperEntries = namespaceEntries.filter((entry) =>
      helperNames.has(entry.import_path),
    );
    const samplingEntries = namespaceEntries.filter(
      (entry) => !helperNames.has(entry.import_path),
    );

    return {
      sections: [
        {
          title: 'Sampling routines',
          description:
            'Canonical counted leaf pages for the random distributions and samplers exposed under `we.random`, plus direct sampler entry points.',
          entries: [
            ...samplingEntries,
            ...randomOperations.filter(
              (entry) => !helperNames.has(entry.import_path),
            ),
          ],
        },
        {
          title: 'State and configuration helpers',
          description:
            'Generator construction, seeding, legacy state management, and symmetry-aware helper utilities.',
          entries: [
            ...helperEntries,
            ...randomOperations.filter((entry) =>
              helperNames.has(entry.import_path),
            ),
          ],
        },
      ],
    };
  }

  if (namespace === 'flops') {
    const flopsSymbols = namespaceSymbols;

    return {
      sections: [
        {
          title: 'Core estimator primitives',
          description:
            'General-purpose helpers for pointwise work, reductions, contractions, and other common building blocks.',
          entries: flopsSymbols.filter((entry) =>
            /(?:einsum|pointwise|reduction|trace|unwrap)_cost$/.test(
              entry.import_path,
            ),
          ),
        },
        {
          title: 'Linear algebra estimators',
          description:
            'Cost helpers for decompositions, solves, norms, and other matrix-heavy routines.',
          entries: flopsSymbols.filter((entry) =>
            /(?:cholesky|cond|det|eig|eigh|eigvals|eigvalsh|inv|lstsq|matrix_norm|matrix_power|matrix_rank|multi_dot|norm|pinv|qr|slogdet|solve|svd|svdvals|tensorinv|tensorsolve|vector_norm)_cost$/.test(
              entry.import_path,
            ),
          ),
        },
        {
          title: 'FFT and signal estimators',
          description:
            'Helpers for Fourier transforms, spectral routines, and window-generation costs.',
          entries: flopsSymbols.filter((entry) =>
            /(?:bartlett|blackman|fft|fftn|hamming|hanning|hfft|kaiser|rfft|rfftn)_cost$/.test(
              entry.import_path,
            ),
          ),
        },
        {
          title: 'Polynomial helpers',
          description:
            'Cost formulas for polynomial arithmetic, fitting, evaluation, and root finding.',
          entries: flopsSymbols.filter((entry) =>
            /\.(?:poly[a-z]*|roots)_cost$/.test(entry.import_path),
          ),
        },
      ],
    };
  }

  return {
    sections: [
      {
        title: 'Assertion helpers',
        description:
          'Testing utilities re-exported on the public API for array and tolerance checks.',
        entries: namespaceSymbols,
      },
    ],
  };
}

function buildNamespaces(
  operations: Operation[],
  symbols: PublicApiSymbol[],
): PublicApiNamespace[] {
  const namespaceMap = new Map<string, NamespaceAccumulator>();

  const ensureNamespace = (importPath: string): NamespaceAccumulator => {
    const existing = namespaceMap.get(importPath);
    if (existing) return existing;

    const namespace: NamespaceAccumulator = {
      import_path: importPath,
      display_name: importPath,
      href: hrefForNamespace(importPath),
      summary: summaryForNamespace(importPath),
      operation_count: 0,
      symbol_count: 0,
      child_namespaces: new Set<string>(),
      direct_operations: [],
      direct_symbols: [],
      member_highlights: [],
    };

    namespaceMap.set(importPath, namespace);

    if (importPath !== 'we') {
      const parentPath = parentNamespace(importPath);
      ensureNamespace(parentPath).child_namespaces.add(importPath);
    }

    return namespace;
  };

  ensureNamespace('we');

  for (const operation of operations) {
    const namespace = ensureNamespace(namespaceForArea(operation.area));
    namespace.operation_count += 1;
    namespace.direct_operations.push(operation);
  }

  for (const symbol of symbols) {
    const namespacePath = symbol.namespace_import_path;
    ensureNamespace(namespacePath);
    const namespace = ensureNamespace(namespacePath);
    namespace.symbol_count += 1;
    namespace.direct_symbols.push(symbol);

    if (symbol.members.length > 0) {
      const symbolNamespace = ensureNamespace(symbol.import_path);
      for (const member of symbol.members) {
        symbolNamespace.member_highlights.push({
          kind: 'symbol',
          title: member.label,
          href: member.href,
          description: `Child API on ${symbol.display_name}.`,
          namespace_import_path: symbolNamespace.import_path,
        });
      }
    }
  }

  return Array.from(namespaceMap.values())
    .map((namespace) => ({
      child_namespaces: Array.from(namespace.child_namespaces).sort(),
      display_name: namespace.display_name,
      highlighted_entries: buildHighlightedEntries(namespace, namespaceMap),
      href: namespace.href,
      import_path: namespace.import_path,
      operation_count: namespace.operation_count,
      summary: namespace.summary,
      symbol_count: namespace.symbol_count,
    }))
    .sort((left, right) => left.import_path.localeCompare(right.import_path));
}

function resolveConcretePublicApiEntry(
  name: string,
  symbols: PublicApiSymbol[],
  operations: Operation[],
): PublicApiHighlightedEntry | null {
  const symbol = symbols.find((entry) => entry.import_path === name);
  if (symbol) {
    return {
      kind: 'symbol',
      title: symbol.import_path,
      href: symbol.href,
      description: symbol.summary,
      namespace_import_path: symbol.namespace_import_path,
    };
  }

  const operation = operations.find(
    (entry) => entry.whest_ref === name || entry.name === name,
  );
  if (!operation?.href) return null;

  return {
    kind: 'operation',
    title: operation.whest_ref,
    href: operation.href,
    description: operation.notes,
    namespace_import_path: namespaceForSymbol(operation.whest_ref),
  };
}

function getNamespaceSymbols(
  symbols: PublicApiSymbol[],
  namespaceImportPath: string,
): NamespaceSectionEntry[] {
  return symbols
    .filter((symbol) => symbol.namespace_import_path === namespaceImportPath)
    .sort((left, right) => left.import_path.localeCompare(right.import_path))
    .map((symbol) => ({
      href: symbol.href,
      import_path: symbol.import_path,
      summary: symbol.summary,
      members: symbol.members,
    }));
}

function getOperationEntriesForArea(
  area: Operation['area'],
): NamespaceSectionEntry[] {
  return getPublicApiOperations()
    .filter((operation) => operation.area === area && Boolean(operation.href))
    .sort((left, right) => left.whest_ref.localeCompare(right.whest_ref))
    .map((operation) => ({
      href: operation.href!,
      import_path: operation.whest_ref,
      summary: operation.notes || `Cost formula: ${operation.cost_formula}.`,
      members: [],
    }));
}

function buildHighlightedEntries(
  namespace: NamespaceAccumulator,
  namespaceMap: Map<string, NamespaceAccumulator>,
): PublicApiHighlightedEntry[] {
  const namespaceHighlights = Array.from(namespace.child_namespaces)
    .sort((left, right) => left.localeCompare(right))
    .map((importPath) => {
      const child = namespaceMap.get(importPath);
      return {
        kind: 'namespace' as const,
        title: importPath,
        href: hrefForNamespace(importPath),
        description:
          child?.summary ?? `Public API namespace for ${importPath}.`,
        namespace_import_path: namespace.import_path,
      };
    });

  const symbolHighlights = [...namespace.direct_symbols]
    .sort((left, right) => left.import_path.localeCompare(right.import_path))
    .map((symbol) => ({
      kind: 'symbol' as const,
      title: symbol.display_name,
      href: symbol.href,
      description: symbol.summary,
      namespace_import_path: namespace.import_path,
    }));

  const operationHighlights = [...namespace.direct_operations]
    .sort((left, right) => left.name.localeCompare(right.name))
    .filter((operation) => Boolean(operation.href))
    .map((operation) => ({
      kind: 'operation' as const,
      title: operation.whest_ref,
      href: operation.href!,
      description: operation.notes || operation.cost_formula,
      namespace_import_path: namespace.import_path,
    }));

  return [
    ...namespaceHighlights,
    ...namespace.member_highlights,
    ...symbolHighlights,
    ...operationHighlights,
  ].slice(0, 6);
}

function normalizeSymbolKind(
  kind: string,
): PublicApiSymbol['kind'] {
  if (kind === 'object' || kind === 'class') return kind;
  return 'call';
}

function namespaceForSymbol(importPath: string): string {
  const lastDot = importPath.lastIndexOf('.');
  if (lastDot <= 'we'.length) return 'we';
  return importPath.slice(0, lastDot);
}

function namespaceForArea(area: Operation['area']): string {
  if (area === 'core') return 'we';
  return `we.${area}`;
}

function parentNamespace(importPath: string): string {
  const lastDot = importPath.lastIndexOf('.');
  if (lastDot <= 'we'.length) return 'we';
  return importPath.slice(0, lastDot);
}

function hrefForNamespace(importPath: string): string {
  if (importPath === 'we') return '/docs/api/';
  return `/docs/api/${importPath.slice(3).replaceAll('.', '/')}/`;
}

function summaryForNamespace(importPath: string): string {
  if (importPath === 'we') {
    return 'Top-level public API surface for operations, helpers, and types.';
  }
  if (importPath === 'we.random') {
    return 'Random-number generation and sampling APIs.';
  }
  if (importPath === 'we.stats') {
    return 'Statistical distributions and related helper APIs.';
  }
  if (importPath === 'we.linalg') {
    return 'Linear algebra operations in the public API surface.';
  }
  if (importPath === 'we.fft') {
    return 'Fast Fourier transform operations in the public API surface.';
  }
  return `Public API namespace for ${importPath}.`;
}
