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
  whest_ref: string;
  numpy_ref: string;
  signature: string;
  summary: string;
  area: 'core' | 'linalg' | 'fft' | 'random' | 'stats';
  display_type: 'counted' | 'custom' | 'free' | 'blocked';
  weight: number;
  aliases: string[];
  notes: string;
  cost_formula: string;
  cost_formula_latex: string;
  provenance_label?: string;
  provenance_url?: string;
  whest_source_url?: string;
  upstream_source_url?: string;
  parameters: DocField[];
  returns: DocField[];
  see_also: DocLink[];
  notes_sections: string[];
  example?: DocExample | null;
  previous?: OperationNavLink | null;
  next?: OperationNavLink | null;
}
