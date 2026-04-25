import { opDocImports, opDocSlugs } from '@/.generated/op-doc-imports';
import OperationDocNav from '@/components/api-reference/OperationDocNav';
import OperationDocPage from '@/components/api-reference/OperationDocPage';
import type { OperationDocRecord } from '@/components/api-reference/op-doc-types';
import styles from '@/components/api-reference/styles.module.css';
import { DocsBody, DocsPage, DocsTitle } from 'fumadocs-ui/layouts/docs/page';
import { ChevronRight } from 'lucide-react';
import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';

async function loadOpDoc(slug: string): Promise<OperationDocRecord | null> {
  const loader = opDocImports[slug];
  if (!loader) {
    return null;
  }
  const module = await loader();
  return module.default as OperationDocRecord;
}

function formatFullNumpyRef(numpyRef: string) {
  return numpyRef.replace(/^np\./, 'numpy.');
}

// Expand the short JAX-style alias used in op records (e.g. ``fnp.einsum``,
// ``flops.configure``) into the full dotted module path used as the page H1
// (``flopscope.numpy.einsum``, ``flopscope.configure``). The alias form is
// still shown in the signature block so users see the import they would
// actually write.
function expandFlopscopeRef(ref: string): string {
  if (ref.startsWith('fnp.')) {
    return `flopscope.numpy.${ref.slice('fnp.'.length)}`;
  }
  if (ref.startsWith('flops.')) {
    return `flopscope.${ref.slice('flops.'.length)}`;
  }
  if (ref.startsWith('flopscope.')) {
    return ref;
  }
  return `flopscope.${ref}`;
}

function OperationTitle({ flopscopeRef }: { flopscopeRef: string }) {
  const fullRef = expandFlopscopeRef(flopscopeRef);
  const lastDot = fullRef.lastIndexOf('.');
  if (lastDot < 0) {
    return <span className={styles.docTitleFunction}>{fullRef}</span>;
  }
  return (
    <>
      <span className={styles.docTitleNamespace}>
        {fullRef.slice(0, lastDot + 1)}
      </span>
      <span className={styles.docTitleFunction}>{fullRef.slice(lastDot + 1)}</span>
    </>
  );
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

function OperationDocBreadcrumb({ op }: { op: OperationDocRecord }) {
  const topic = getTopicForOperation(op);
  const items: Array<{ label: string; href?: string }> = [
    { label: 'NumPy reference', href: '/docs/api' },
    { label: 'Routines and objects by topic', href: '/docs/api' },
    topic,
    { label: formatFullNumpyRef(op.numpy_ref) },
  ];

  return (
    <div className="flex items-center gap-1.5 text-sm text-fd-muted-foreground">
      {items.map((item, index) => {
        const className =
          index === items.length - 1
            ? 'truncate text-fd-primary font-medium'
            : 'truncate';

        return (
          <div key={item.label} className="contents">
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

export default async function OperationPage(props: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await props.params;
  const op = await loadOpDoc(slug);
  if (!op) notFound();

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
      <DocsTitle>
        <OperationTitle flopscopeRef={op.flopscope_ref} />
      </DocsTitle>
      <DocsBody>
        <OperationDocPage op={op} />
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  return opDocSlugs.map((slug) => ({ slug }));
}

export async function generateMetadata(props: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await props.params;
  const op = await loadOpDoc(slug);
  if (!op) notFound();

  return {
    title: op.flopscope_ref,
    description: op.summary || op.notes,
  };
}
