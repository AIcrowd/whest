import fs from 'node:fs/promises';
import path from 'node:path';
import { source } from '@/lib/source';
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
} from 'fumadocs-ui/layouts/docs/page';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { getMDXComponents } from '@/components/mdx';
import type { Metadata } from 'next';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { ChevronRight } from 'lucide-react';
import OperationDocNav from '@/components/api-reference/OperationDocNav';
import type { OperationNavLink } from '@/components/api-reference/op-doc-types';

type OpBreadcrumbMeta = {
  name: string;
  area: 'core' | 'linalg' | 'fft' | 'random' | 'stats';
  numpy_ref: string;
  previous?: OperationNavLink | null;
  next?: OperationNavLink | null;
};

function isOperationSlug(slug?: string[]): slug is [string, string, string] {
  return Boolean(slug && slug.length === 3 && slug[0] === 'api' && slug[1] === 'ops');
}

async function loadOpBreadcrumbMeta(name: string): Promise<OpBreadcrumbMeta | null> {
  const slug = name.replaceAll('.', '-');
  const candidatePaths = [
    path.join(process.cwd(), '.generated', 'ops', `${slug}.json`),
    path.join(process.cwd(), 'website', '.generated', 'ops', `${slug}.json`),
  ];

  for (const candidate of candidatePaths) {
    try {
      const raw = await fs.readFile(candidate, 'utf8');
      const parsed = JSON.parse(raw) as OpBreadcrumbMeta;
      return {
        name: parsed.name,
        area: parsed.area,
        numpy_ref: parsed.numpy_ref,
        previous: parsed.previous ?? null,
        next: parsed.next ?? null,
      };
    } catch {
      // Try the next candidate path.
    }
  }

  return null;
}

function formatFullNumpyRef(numpyRef: string) {
  return numpyRef.replace(/^np\./, 'numpy.');
}

function getTopicForOperation(op: OpBreadcrumbMeta) {
  if (op.area === 'linalg') {
    return { label: 'Linear algebra', href: '/docs/api/linalg' };
  }
  if (op.area === 'fft') {
    return { label: 'Discrete Fourier Transform', href: '/docs/api/fft' };
  }
  if (op.area === 'random') {
    return { label: 'Random sampling', href: '/docs/api/random' };
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

function OpBreadcrumb({ op }: { op: OpBreadcrumbMeta }) {
  const topic = getTopicForOperation(op);
  const items: Array<{ label: string; href?: string }> = [
    { label: 'NumPy reference', href: '/docs/api' },
    { label: 'Routines and objects by topic', href: '/docs/api/ops' },
    topic,
    { label: formatFullNumpyRef(op.numpy_ref) },
  ];

  return (
    <div className="flex items-center gap-1.5 text-sm text-fd-muted-foreground">
      {items.map((item, index) => {
        const className = index === items.length - 1 ? 'truncate text-fd-primary font-medium' : 'truncate';

        return (
          <div key={item.label} className="contents">
            {index > 0 ? <ChevronRight className="size-3.5 shrink-0" /> : null}
            {item.href ? (
              <Link href={item.href} className={`${className} transition-opacity hover:opacity-80`}>
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

export default async function Page(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();
  const opMeta = isOperationSlug(params.slug)
    ? await loadOpBreadcrumbMeta(params.slug[2])
    : null;

  const MDX = page.data.body;

  return (
    <DocsPage
      toc={page.data.toc}
      full={page.data.full}
      breadcrumb={opMeta ? { component: <OpBreadcrumb op={opMeta} /> } : undefined}
      footer={
        opMeta
          ? {
              enabled: Boolean(opMeta.previous || opMeta.next),
              component: <OperationDocNav previous={opMeta.previous} next={opMeta.next} />,
            }
          : undefined
      }
    >
      <DocsTitle>{page.data.title}</DocsTitle>
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

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}): Promise<Metadata> {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  return {
    title: page.data.title,
    description: page.data.description,
  };
}
