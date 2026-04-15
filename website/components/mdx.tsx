import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { Accordions, Accordion } from 'fumadocs-ui/components/accordion';
import ApiReference from './api-reference';
import SymmetryExplorer from './symmetry-explorer';
import SortableTable from './shared/SortableTable';
import Mermaid from './mermaid';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    ApiReference,
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
