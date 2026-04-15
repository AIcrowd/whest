import { Terminal } from 'lucide-react';
import { Fragment, type ComponentProps } from 'react';
import { CodeBlock } from 'fumadocs-ui/components/codeblock';
import {
  decodeAnsiEscapes,
  extractRichOutputText,
  parseAnsiRichText,
} from './rich-output.mjs';

const COLOR_MAP = {
  cyan: '#32e6ff',
  green: '#29ff6a',
  white: '#f8fafc',
  yellow: '#facc15',
  red: '#fb7185',
  blue: '#60a5fa',
  magenta: '#f472b6',
  black: '#020617',
} as const;

type PreProps = ComponentProps<'pre'> & {
  title?: string;
};

type RichSegment = {
  text: string;
  color: keyof typeof COLOR_MAP | null;
  bold: boolean;
  dim: boolean;
};

export default function RichOutputPre({ children }: PreProps) {
  const text = decodeAnsiEscapes(extractRichOutputText(children)).trimEnd();
  const lines = parseAnsiRichText(text) as RichSegment[][];

  return (
    <CodeBlock
      title={'\u200B'}
      icon={<Terminal className="size-3.5" />}
      className="my-4 overflow-hidden rounded-xl border border-[#163542] bg-[#02060a] text-[#f8fafc] shadow-sm not-prose [&>div:first-child]:border-[#14303c] [&>div:first-child]:bg-[#071018] [&>div:first-child]:text-slate-200"
      viewportProps={{
        className:
          'bg-[#02060a] px-4 py-4 text-[#f8fafc] selection:bg-slate-700/80 selection:text-white',
      }}
    >
      <pre className="m-0 min-w-full whitespace-pre font-mono text-[0.8125rem] leading-6">
        {lines.map((line, lineIndex) => (
          <Fragment key={lineIndex}>
            {line.map((segment, segmentIndex) => (
              <span
                key={`${lineIndex}-${segmentIndex}`}
                style={{
                  color: segment.color ? COLOR_MAP[segment.color] : undefined,
                  fontWeight: segment.bold ? 700 : undefined,
                  opacity: segment.dim ? 0.72 : undefined,
                }}
              >
                {segment.text}
              </span>
            ))}
            {lineIndex < lines.length - 1 ? '\n' : null}
          </Fragment>
        ))}
      </pre>
    </CodeBlock>
  );
}
