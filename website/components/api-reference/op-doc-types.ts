export interface OpDocBlock {
  type: string;
  text?: string;
  code?: string;
  language?: string;
  name?: string;
  body?: string;
  raw?: string;
}

export interface OpDocFieldItem {
  name: string;
  type: string;
  desc_blocks: OpDocBlock[];
}

export interface OpDocSection {
  kind: string;
  title: string;
  blocks?: OpDocBlock[];
  items?: OpDocFieldItem[];
}

export interface OpDocPayload {
  schema_version: number;
  slug: string;
  detail_href: string;
  detail_json_href: string;
  source: {
    whest: string | null;
    numpy: string | null;
  };
  op: {
    name: string;
    module: string;
    whest_ref: string;
    numpy_ref: string;
    category: string;
    status: string;
    weight: number;
    cost_formula: string;
    cost_formula_latex: string;
    notes: string;
    summary: string;
    signature: string;
  };
  docs: {
    sections: OpDocSection[];
  };
}
