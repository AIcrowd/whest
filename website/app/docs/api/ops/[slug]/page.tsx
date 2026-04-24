import legacyOpRedirectsJson from '@/.generated/legacy-op-redirects.json';
import { notFound, redirect } from 'next/navigation';

const legacyOpRedirects = legacyOpRedirectsJson as Record<string, string>;

export default async function LegacyOperationPage(props: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await props.params;
  const href = legacyOpRedirects[slug];
  if (!href) notFound();

  redirect(href);
}

export async function generateStaticParams() {
  return Object.keys(legacyOpRedirects).map((slug) => ({ slug }));
}
