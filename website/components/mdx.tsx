import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps } from 'react';
import { Accordions, Accordion } from 'fumadocs-ui/components/accordion';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import ApiReference from './api-reference';
import OperationDocPage from './api-reference/OperationDocPage';
import OpRef from './op-ref';
import SymmetryExplorer from './symmetry-explorer';
import SortableTable from './shared/SortableTable';
import Mermaid from './mermaid';
import RichOutputPre from './rich-output-pre';
import { isRichOutputBlock } from './rich-output.mjs';

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

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    pre: DocsPre,
    ApiReference,
    OperationDocPage,
    OpRef,
    SymmetryExplorer,
    SortableTable,
    Mermaid,
    Accordions,
    Accordion,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
