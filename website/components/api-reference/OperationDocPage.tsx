import Link from 'next/link';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import type { ReactNode } from 'react';
import type {
  OpDocBlock,
  OpDocFieldItem,
  OpDocPayload,
  OpDocSection,
} from './op-doc-types';

function renderInlineCode(text: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`)/g).filter(Boolean);
  return parts.map((part, idx) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={`${part}-${idx}`}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function renderBlock(block: OpDocBlock, key: string): ReactNode {
  if (block.type === 'code') {
    return (
      <CodeBlock key={key}>
        <Pre>{block.code}</Pre>
      </CodeBlock>
    );
  }

  if (block.type === 'directive') {
    return (
      <div
        key={key}
        className="not-prose rounded-lg border border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground"
      >
        <div className="font-semibold text-foreground">{block.name}</div>
        {block.body ? (
          <pre className="mt-2 whitespace-pre-wrap font-mono text-xs leading-6">
            {block.body}
          </pre>
        ) : null}
      </div>
    );
  }

  return (
    <p key={key} className="whitespace-pre-line leading-7 text-foreground">
      {renderInlineCode(block.text ?? '')}
    </p>
  );
}

function renderFieldItems(items: OpDocFieldItem[]): ReactNode {
  return (
    <div className="space-y-6 pl-4 sm:pl-8">
      {items.map((item) => (
        <div key={`${item.name}-${item.type}`}>
          <div className="text-sm leading-7">
            <span className="font-semibold text-foreground">{item.name}</span>
            {item.type ? (
              <>
                {' '}
                <span className="text-muted-foreground">: </span>
                <span className="font-medium text-muted-foreground">{item.type}</span>
              </>
            ) : null}
          </div>
          <div className="mt-1 space-y-3 pl-4 sm:pl-6">
            {item.desc_blocks.map((block, idx) =>
              renderBlock(block, `${item.name}-${idx}`),
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function renderSection(section: OpDocSection): ReactNode {
  const barKinds = new Set(['parameters', 'returns', 'notes']);

  return (
    <section key={section.kind} id={section.kind} className="mt-8">
      <h2
        className={
          barKinds.has(section.kind)
            ? 'mb-5 bg-muted px-3 py-1 text-sm font-semibold uppercase tracking-[0.08em] text-foreground'
            : 'mb-4 text-2xl font-semibold tracking-tight text-foreground'
        }
      >
        {section.title}
      </h2>
      {section.items?.length ? renderFieldItems(section.items) : null}
      {section.blocks?.length ? (
        <div className="space-y-4">{section.blocks.map((block, idx) => renderBlock(block, `${section.kind}-${idx}`))}</div>
      ) : null}
    </section>
  );
}

function formatWeight(weight: number): string {
  return Number.isInteger(weight) ? `${weight}×` : `${weight}×`;
}

export default function OperationDocPage({
  doc,
}: {
  doc: OpDocPayload;
}) {
  return (
    <div className="space-y-8">
      <div className="not-prose space-y-3">
        <div className="text-lg font-semibold tracking-tight text-foreground">
          {doc.op.signature}
        </div>
        <div className="flex flex-wrap items-center gap-4 text-sm">
          {doc.source.whest ? (
            <Link className="font-medium text-primary underline-offset-4 hover:underline" href={doc.source.whest}>
              whest source
            </Link>
          ) : null}
          {doc.source.numpy ? (
            <Link className="font-medium text-primary underline-offset-4 hover:underline" href={doc.source.numpy}>
              numpy source
            </Link>
          ) : null}
        </div>
        <div className="grid gap-4 rounded-xl border border-border bg-muted/20 p-4 text-sm sm:grid-cols-2 xl:grid-cols-4">
          <div>
            <div className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">Module</div>
            <div className="mt-1 text-foreground">{doc.op.module}</div>
          </div>
          <div>
            <div className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">Type</div>
            <div className="mt-1 text-foreground">{doc.op.category}</div>
          </div>
          <div>
            <div className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">Weight</div>
            <div className="mt-1 text-foreground">{formatWeight(doc.op.weight)}</div>
          </div>
          <div>
            <div className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">NumPy Ref</div>
            <div className="mt-1 text-foreground">{doc.op.numpy_ref}</div>
          </div>
          <div className="sm:col-span-2 xl:col-span-4">
            <div className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">Cost Formula</div>
            <div className="mt-1 text-foreground">{doc.op.cost_formula_latex || doc.op.cost_formula}</div>
          </div>
          {doc.op.notes ? (
            <div className="sm:col-span-2 xl:col-span-4">
              <div className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">Whest Context</div>
              <div className="mt-1 text-foreground">{doc.op.notes}</div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="space-y-2">
        {doc.docs.sections.map((section) => renderSection(section))}
      </div>
    </div>
  );
}
