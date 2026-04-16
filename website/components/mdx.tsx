import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps } from 'react';
import Link from 'next/link';
import { Accordions, Accordion } from 'fumadocs-ui/components/accordion';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import ApiReference from './api-reference';
import SortableTable from './shared/SortableTable';
import Mermaid from './mermaid';
import RichOutputPre from './rich-output-pre';
import { isRichOutputBlock } from './rich-output.mjs';
import StaticFileLink from './static-file-link';
import { STANDALONE_SYMMETRY_AWARE_EINSUM_URL } from '@/lib/docsTree';

function DocsPre(props: ComponentProps<'pre'> & { title?: string }) {
  if (isRichOutputBlock(props)) {
    return <RichOutputPre {...props} />;
  }

  return (
    <CodeBlock {...props}>
      <Pre>{props.children}</Pre>
    </CodeBlock>
  );
}

function SymmetryExplorerStandaloneHandoff() {
  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-muted/30 p-5">
      <p className="m-0 text-sm text-foreground">
        <code>Symmetry Explorer</code> now lives as the standalone interactive tool{' '}
        <strong>Symmetry Aware Einsum Contractions</strong>.
      </p>
      <p className="mt-2 text-sm text-muted-foreground">Open it in a new tab:</p>
      <Link
        href={STANDALONE_SYMMETRY_AWARE_EINSUM_URL}
        target="_blank"
        rel="noreferrer noopener"
        className="mt-4 inline-flex items-center rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground no-underline transition hover:bg-primary/90"
      >
        Launch Symmetry Aware Einsum Contractions
      </Link>
    </div>
  );
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    pre: DocsPre,
    ApiReference,
    SymmetryExplorer: SymmetryExplorerStandaloneHandoff,
    SortableTable,
    Mermaid,
    Accordions,
    Accordion,
    StaticFileLink,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
