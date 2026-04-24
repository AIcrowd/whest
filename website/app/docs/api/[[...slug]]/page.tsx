import publicApiRoutesJson from '@/.generated/public-api-routes.json';
import { opDocImports } from '@/.generated/op-doc-imports';
import { symbolDocImports } from '@/.generated/symbol-doc-imports';
import OperationDocNav from '@/components/api-reference/OperationDocNav';
import OperationDocPage from '@/components/api-reference/OperationDocPage';
import PublicApiSymbolPage from '@/components/api-reference/PublicApiSymbolPage';
import type {
  OperationDocRecord,
  PublicApiSymbolRecord,
} from '@/components/api-reference/op-doc-types';
import { getMDXComponents } from '@/components/mdx';
import { source } from '@/lib/source';
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
} from 'fumadocs-ui/layouts/docs/page';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { ChevronRight } from 'lucide-react';
import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';

type PublicApiRouteEntry = {
  kind: 'op' | 'symbol';
  slug: string;
  href: string;
  canonical_name: string;
};

const publicApiRoutes = publicApiRoutesJson as Record<string, PublicApiRouteEntry>;

async function loadGeneratedApiDoc(slug: string[]) {
  const pathKey = slug.join('/');
  const route = publicApiRoutes[pathKey];
  if (!route) return null;

  if (route.kind === 'op') {
    const loader = opDocImports[route.slug];
    if (!loader) return null;
    const mod = await loader();
    return { kind: 'op' as const, record: mod.default as OperationDocRecord };
  }

  const loader = symbolDocImports[route.slug];
  if (!loader) return null;
  const mod = await loader();
  return { kind: 'symbol' as const, record: mod.default as PublicApiSymbolRecord };
}

function formatFullNumpyRef(numpyRef: string) {
  return numpyRef.replace(/^np\./, 'numpy.');
}

function getTopicForOperation(op: OperationDocRecord) {
  if (op.area === 'linalg') {
    return { label: 'Linear algebra', href: '/docs/api' };
  }
  if (op.area === 'fft') {
    return { label: 'Discrete Fourier Transform', href: '/docs/api' };
  }
  if (op.area === 'random') {
    return { label: 'Random sampling', href: '/docs/api' };
  }
  if (
    /^(histogram|histogram_bin_edges|bincount|digitize|percentile|nanpercentile|quantile|nanquantile|median|nanmedian|mean|nanmean|average|std|nanstd|var|nanvar|corrcoef|cov)\b/.test(
      op.name,
    )
  ) {
    return { label: 'Statistics', href: '/docs/api' };
  }

  return { label: 'Array routines', href: '/docs/api' };
}

function ApiBreadcrumb({
  items,
}: {
  items: Array<{ label: string; href?: string }>;
}) {
  return (
    <div className="flex items-center gap-1.5 text-sm text-fd-muted-foreground">
      {items.map((item, index) => {
        const className =
          index === items.length - 1
            ? 'truncate text-fd-primary font-medium'
            : 'truncate';

        return (
          <div key={`${item.href ?? 'label'}-${item.label}`} className="contents">
            {index > 0 ? <ChevronRight className="size-3.5 shrink-0" /> : null}
            {item.href ? (
              <Link
                href={item.href}
                className={`${className} transition-opacity hover:opacity-80`}
              >
                {item.label}
              </Link>
            ) : (
              <span className={className}>{item.label}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function OperationDocBreadcrumb({ op }: { op: OperationDocRecord }) {
  const topic = getTopicForOperation(op);
  return (
    <ApiBreadcrumb
      items={[
        { label: 'Public API', href: '/docs/api' },
        topic,
        { label: formatFullNumpyRef(op.numpy_ref) },
      ]}
    />
  );
}

function SymbolDocBreadcrumb({ symbol }: { symbol: PublicApiSymbolRecord }) {
  const segments = symbol.import_path.replace(/^we\./, '').split('.');
  const items: Array<{ label: string; href?: string }> = [
    { label: 'Public API', href: '/docs/api' },
  ];

  let runningHref = '/docs/api';
  for (const segment of segments.slice(0, -1)) {
    runningHref += `/${segment.replace(/_/g, '-')}`;
    items.push({ label: segment, href: `${runningHref}/` });
  }
  items.push({ label: symbol.import_path });

  return <ApiBreadcrumb items={items} />;
}

function renderDocsSourcePage(page: ReturnType<typeof source.getPage>) {
  if (!page) notFound();

  const MDX = page.data.body;
  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      <DocsTitle
        className="text-[2.5rem] font-semibold leading-[1.1] tracking-[-0.02em]"
        style={{ fontVariationSettings: "'opsz' 72" }}
      >
        {page.data.title}
      </DocsTitle>
      <DocsDescription>{page.data.description}</DocsDescription>
      <DocsBody>
        <MDX
          components={getMDXComponents({
            a: createRelativeLink(source, page),
          })}
        />
      </DocsBody>
    </DocsPage>
  );
}

export default async function ApiPage(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const { slug = [] } = await props.params;
  const docsPage = source.getPage(['api', ...slug]);
  if (docsPage) {
    return renderDocsSourcePage(docsPage);
  }

  const apiDoc = await loadGeneratedApiDoc(slug);
  if (!apiDoc) notFound();

  if (apiDoc.kind === 'op') {
    const op = apiDoc.record;
    return (
      <DocsPage
        breadcrumb={{ component: <OperationDocBreadcrumb op={op} /> }}
        footer={
          op.previous || op.next
            ? {
                component: (
                  <OperationDocNav previous={op.previous} next={op.next} />
                ),
              }
            : { enabled: false }
        }
      >
        <DocsTitle>{op.whest_ref}</DocsTitle>
        <DocsBody>
          {/* OperationDocPage composes OperationDocHeader, overlay metadata, and the structured doc body. */}
          <OperationDocPage op={op} />
        </DocsBody>
      </DocsPage>
    );
  }

  const symbol = apiDoc.record;
  return (
    <DocsPage breadcrumb={{ component: <SymbolDocBreadcrumb symbol={symbol} /> }}>
      <DocsTitle>{symbol.display_name}</DocsTitle>
      <DocsDescription>{symbol.summary}</DocsDescription>
      <DocsBody>
        <PublicApiSymbolPage symbol={symbol} />
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  const docsParams = source
    .generateParams()
    .filter((params) => Array.isArray(params.slug) && params.slug[0] === 'api')
    .map((params) => ({ slug: (params.slug ?? []).slice(1) }));

  const apiParams = Object.keys(publicApiRoutes).map((path) => ({
    slug: path.split('/').filter(Boolean),
  }));

  const deduped = new Map<string, { slug: string[] }>();
  for (const params of [...docsParams, ...apiParams]) {
    deduped.set((params.slug ?? []).join('/'), { slug: params.slug ?? [] });
  }
  return [...deduped.values()];
}

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}): Promise<Metadata> {
  const { slug = [] } = await props.params;
  const docsPage = source.getPage(['api', ...slug]);
  if (docsPage) {
    return {
      title: docsPage.data.title,
      description: docsPage.data.description,
    };
  }

  const apiDoc = await loadGeneratedApiDoc(slug);
  if (!apiDoc) notFound();

  if (apiDoc.kind === 'op') {
    return {
      title: apiDoc.record.whest_ref,
      description: apiDoc.record.summary || apiDoc.record.notes,
    };
  }

  return {
    title: apiDoc.record.display_name,
    description: apiDoc.record.summary,
  };
}
