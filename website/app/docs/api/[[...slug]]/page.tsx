import { source } from '@/lib/source';
import { publicApiLeafImports } from '@/.generated/public-api-leaf-imports';
import publicApiRouteMap from '@/.generated/public-api-route-map.json';
import { getMDXComponents } from '@/components/mdx';
import OperationDocNav from '@/components/api-reference/OperationDocNav';
import OperationDocPage from '@/components/api-reference/OperationDocPage';
import styles from '@/components/api-reference/styles.module.css';
import type { ApiDocRecord } from '@/components/api-reference/op-doc-types';
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
} from 'fumadocs-ui/layouts/docs/page';
import type { Metadata } from 'next';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { notFound } from 'next/navigation';

type RouteRecord = {
  canonical_name: string;
  href: string;
  import_path: string;
  slug: string;
  kind: string;
  module: string;
  callable: boolean;
};

type RouteMap = {
  schema_version: number;
  routes: Record<string, RouteRecord>;
};

function canonicalHrefForSlug(slug: string[]): string {
  const suffix = slug.length ? `${slug.join('/')}/` : '';
  return `/docs/api/${suffix}`;
}

async function loadGeneratedDoc(importPath: string): Promise<ApiDocRecord | null> {
  const loader = (publicApiLeafImports as Record<
    string,
    (() => Promise<{ default: unknown }>) | undefined
  >)[importPath];
  if (!loader) {
    return null;
  }
  const module = await loader();
  return module.default as ApiDocRecord;
}

function TitleParts({ name }: { name: string }) {
  const lastDot = name.lastIndexOf('.');
  if (lastDot < 0) {
    return <span className={styles.docTitleFunction}>{name}</span>;
  }
  return (
    <>
      <span className={styles.docTitleNamespace}>{name.slice(0, lastDot + 1)}</span>
      <span className={styles.docTitleFunction}>{name.slice(lastDot + 1)}</span>
    </>
  );
}

export default async function ApiPage(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const slug = params.slug ?? [];
  const authoredPage = source.getPage(['api', ...slug]);

  if (authoredPage) {
    const MDX = authoredPage.data.body;
    return (
      <DocsPage toc={authoredPage.data.toc} full={authoredPage.data.full}>
        <DocsTitle
          className="text-[2.5rem] font-semibold leading-[1.1] tracking-[-0.02em]"
          style={{ fontVariationSettings: "'opsz' 72" }}
        >
          {authoredPage.data.title}
        </DocsTitle>
        <DocsDescription>{authoredPage.data.description}</DocsDescription>
        <DocsBody>
          <MDX
            components={getMDXComponents({
              a: createRelativeLink(source, authoredPage),
            })}
          />
        </DocsBody>
      </DocsPage>
    );
  }

  const routeMap = publicApiRouteMap as RouteMap;
  const route = routeMap.routes[canonicalHrefForSlug(slug)];
  if (!route) notFound();

  const doc = await loadGeneratedDoc(route.import_path);
  if (!doc) notFound();

  return (
    <DocsPage
      footer={
        doc.previous || doc.next
          ? {
              component: <OperationDocNav previous={doc.previous} next={doc.next} />,
            }
          : { enabled: false }
      }
    >
      <DocsTitle>
        <TitleParts name={doc.display_name ?? doc.import_path ?? doc.flopscope_ref} />
      </DocsTitle>
      <DocsBody>
        <OperationDocPage op={doc} />
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  const authored = source
    .generateParams()
    .filter((params) => params.slug?.[0] === 'api')
    .map((params) => ({ slug: params.slug.slice(1) }));

  const generated = Object.keys((publicApiRouteMap as RouteMap).routes).map((href) => ({
    slug: href.replace(/^\/docs\/api\/?/, '').replace(/\/$/, '').split('/').filter(Boolean),
  }));

  const seen = new Set<string>();
  return [...authored, ...generated].filter((params) => {
    const key = (params.slug ?? []).join('/');
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}): Promise<Metadata> {
  const params = await props.params;
  const slug = params.slug ?? [];
  const authoredPage = source.getPage(['api', ...slug]);
  if (authoredPage) {
    return {
      title: authoredPage.data.title,
      description: authoredPage.data.description,
    };
  }

  const route = (publicApiRouteMap as RouteMap).routes[canonicalHrefForSlug(slug)];
  if (!route) return {};
  const doc = await loadGeneratedDoc(route.import_path);
  if (!doc) return {};
  return {
    title: doc.display_name ?? doc.import_path ?? doc.flopscope_ref,
    description: doc.summary,
  };
}
