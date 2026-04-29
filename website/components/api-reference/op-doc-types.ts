export interface DocField {
  name: string;
  type: string;
  body: string[];
}

export interface DocLink {
  label: string;
  target: string;
  href?: string;
  external_url?: string;
  description?: string;
  description_inline?: DocInlineNode[];
  role?: string;
  original_target?: string;
  unresolved?: boolean;
}

export interface DocInlineNode {
  kind:
    | 'text'
    | 'code'
    | 'math'
    | 'emphasis'
    | 'strong'
    | 'link'
    | 'role_reference';
  text?: string;
  latex?: string;
  role?: string;
  target?: string;
  original_target?: string;
  display_text?: string;
  suppress_link?: boolean;
  explicit_title?: boolean;
  unresolved?: boolean;
  href?: string;
  external_url?: string;
}

export interface DocFieldNode {
  type: 'field_list';
  name: string;
  data_type?: string;
  inline: DocInlineNode[];
  body_blocks?: DocFieldBodyBlock[];
}

export interface DocLinkListNode {
  type: 'link_list';
  links: DocLink[];
}

export interface DocLine {
  kind: 'input' | 'output';
  prompt?: '>>>' | '...';
  text: string;
}

export interface DocDirectiveBlock {
  type: 'directive_block';
  directive: string;
  version?: string;
  argument_inline: DocInlineNode[];
  options: Array<{name: string; value: string}>;
  content_blocks: DocNestedBlock[];
  supported: boolean;
  raw_source: string;
}

export interface DocParagraphBlock {
  type: 'paragraph' | 'text';
  inline: DocInlineNode[];
}

export interface DocMathBlock {
  type: 'math_block';
  formulas: string[];
}

export type DocNestedBlock =
  | DocParagraphBlock
  | DocDirectiveBlock
  | DocDefinitionListBlock
  | DocLiteralBlock
  | DocMathBlock
  | DocListBlock
  | DocRawBlock;

export interface DocDefinitionListBlock {
  type: 'definition_list';
  items: {
    term_inline: DocInlineNode[];
    blocks: DocNestedBlock[];
  }[];
}

export type DocFieldBodyBlock =
  | DocParagraphBlock
  | DocMathBlock
  | DocDefinitionListBlock
  | DocDirectiveBlock
  | DocListBlock
  | DocRawBlock;

export interface DocFieldListBlock {
  type: 'field_list';
  title: 'Parameters' | 'Returns';
  items: DocFieldNode[];
}

export interface DocListBlock {
  type: 'list';
  ordered: boolean;
  items: {
    blocks: DocNestedBlock[];
  }[];
}

export interface DocLiteralBlock {
  type: 'literal_block';
  text: string;
  language?: string;
}

export interface DocDoctestBlock {
  type: 'doctest_block';
  lines: DocLine[];
}

export interface DocRawBlock {
  type: 'raw_block';
  raw_kind: string;
  raw_text: string;
}

export type DocBlock =
  | DocParagraphBlock
  | DocDefinitionListBlock
  | DocFieldListBlock
  | DocLinkListNode
  | DocListBlock
  | DocDirectiveBlock
  | DocLiteralBlock
  | DocMathBlock
  | DocDoctestBlock
  | DocRawBlock;

export interface DocSection {
  title: string;
  blocks: DocBlock[];
}

export interface DocExample {
  code: string;
  output: string;
  source: string;
}

export interface OperationNavLink {
  href: string;
  label: string;
}

export interface OperationDocRecord {
  name: string;
  canonical_name: string;
  slug: string;
  module: string;
  flopscope_ref: string;
  numpy_ref: string;
  signature: string;
  summary: string;
  area: string;
  display_type: string;
  weight: number;
  aliases: string[];
  notes: string;
  cost_formula: string;
  cost_formula_latex: string;
  provenance_label?: string;
  provenance_url?: string;
  flopscope_source_url?: string;
  upstream_source_url?: string;
  parameters: DocField[];
  returns: DocField[];
  see_also: DocLink[];
  notes_sections: string[];
  example?: DocExample | null;
  body_sections?: DocSection[];
  doc_coverage?: Record<string, unknown>;
  previous?: OperationNavLink | null;
  next?: OperationNavLink | null;
}

export interface RelatedGuideLink {
  title: string;
  href: string;
}

export interface PublicApiSymbolRecord {
  name: string;
  canonical_name: string;
  slug: string;
  href: string;
  kind: string;
  module: string;
  import_path: string;
  display_name: string;
  summary: string;
  signature: string;
  aliases: string[];
  source_url?: string;
  upstream_source_url?: string;
  related_guides?: RelatedGuideLink[];
  body_sections?: DocSection[];
}
