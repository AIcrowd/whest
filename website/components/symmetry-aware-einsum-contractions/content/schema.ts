export type ProseBlock =
  | { kind: 'paragraph'; text: string }
  | { kind: 'callout'; text: string }
  | { kind: 'label'; text: string }
  | { kind: 'caption'; text: string };

export type SectionCopy = {
  title?: string;
  deck?: string;
  blocks?: ProseBlock[];
  slots?: Record<string, ProseBlock[]>;
};
