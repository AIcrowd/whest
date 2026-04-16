import { opDocImports, opDocSlugs } from '@/.generated/op-doc-imports';
import OperationDocPage from '@/components/api-reference/OperationDocPage';
import type { OpDocPayload } from '@/components/api-reference/op-doc-types';
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
} from 'fumadocs-ui/layouts/docs/page';
import type { Metadata } from 'next';
import { notFound } from 'next/navigation';

async function loadOpDoc(slug: string): Promise<OpDocPayload | null> {
  const loader = opDocImports[slug];
  if (!loader) {
    return null;
  }
  const module = await loader();
  return module.default;
}

export default async function OperationPage(props: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await props.params;
  const doc = await loadOpDoc(slug);
  if (!doc) notFound();

  return (
    <DocsPage>
      <DocsTitle>{doc.op.whest_ref}</DocsTitle>
      <DocsDescription>{doc.op.summary || doc.op.notes}</DocsDescription>
      <DocsBody>
        <OperationDocPage doc={doc} />
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
  const doc = await loadOpDoc(slug);
  if (!doc) notFound();

  return {
    title: doc.op.whest_ref,
    description: doc.op.summary || doc.op.notes,
  };
}
