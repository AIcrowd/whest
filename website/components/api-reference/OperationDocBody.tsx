import {ServerCodeBlock} from 'fumadocs-ui/components/codeblock.rsc';
import {Fragment, type ReactNode} from 'react';
import Latex from '../shared/Latex';
import OperationDocFieldList from './OperationDocFieldList';
import OperationDocLink from './OperationDocLink';
import OperationDocSectionHeading from './OperationDocSectionHeading';
import OperationDocSignature from './OperationDocSignature';
import {apiCodeThemes} from './runtime-shiki-themes';
import styles from './styles.module.css';
import type {
  DocBlock,
  DocDirectiveBlock,
  DocDefinitionListBlock,
  DocFieldBodyBlock,
  DocListBlock,
  DocMathBlock,
  DocNestedBlock,
  DocFieldListBlock,
  DocInlineNode,
  DocLine,
  DocLinkListNode,
  DocRawBlock,
  DocSection,
} from './op-doc-types';

function renderInline(nodes: DocInlineNode[]): ReactNode[] {
  return nodes.flatMap((node, index) => [
    <span key={`${node.kind}-${index}`}>{renderInlineNode(node)}</span>,
  ]);
}

function renderInlineNode(node: DocInlineNode): ReactNode {
  switch (node.kind) {
    case 'math':
      return <Latex math={node.latex ?? ''} />;
    case 'code':
      return <code>{node.text ?? ''}</code>;
    case 'emphasis':
      return <em className={styles.docInlineEmphasis}>{node.text ?? ''}</em>;
    case 'strong':
      return <strong>{node.text ?? ''}</strong>;
    case 'link': {
      if (!node.href && !node.external_url) {
        return node.text ?? '';
      }

      return (
        <OperationDocLink
          label={node.text ?? node.href ?? ''}
          href={node.href}
          externalUrl={node.external_url}
        />
      );
    }
    case 'role_reference': {
      const label = node.display_text ?? node.text ?? node.target ?? '';
      if (node.suppress_link || (!node.href && !node.external_url)) {
        return label;
      }

      return (
        <OperationDocLink
          label={label}
          href={node.href}
          externalUrl={node.external_url}
        />
      );
    }
    default:
      return node.text ?? '';
  }
}

function renderFieldList(block: DocFieldListBlock): ReactNode {
  return (
    <OperationDocFieldList
      items={block.items.map((item, index) => ({
        key: `${block.title}-${index}`,
        name: item.name,
        type: item.data_type,
        body: renderFieldBody(item.body_blocks, item.inline),
      }))}
    />
  );
}

function renderFieldBody(bodyBlocks: DocFieldBodyBlock[] | undefined, fallbackInline: DocInlineNode[]): ReactNode {
  if (!bodyBlocks || bodyBlocks.length === 0) {
    return renderInline(fallbackInline);
  }

  return bodyBlocks.map((block, index) => {
    if (block.type === 'paragraph') {
      return (
        <p key={`field-block-${index}`} className={styles.docFieldParagraph}>
          {renderInline(block.inline)}
        </p>
      );
    }

    return <div key={`field-block-${index}`}>{renderNestedBlock(block)}</div>;
  });
}

function renderNestedDefinitionList(block: DocDefinitionListBlock): ReactNode {
  return (
    <dl className={styles.docSubDefinitionList}>
      {block.items.map((item, index) => (
        <div key={`subdef-${index}`} className={styles.docSubDefinitionItem}>
          <dt className={styles.docSubDefinitionTerm}>{renderInline(item.term_inline)}</dt>
          <dd className={styles.docSubDefinitionBody}>
            {item.blocks.map((nestedBlock, nestedIndex) => (
              <div key={`subdef-${index}-${nestedIndex}`}>{renderNestedBlock(nestedBlock)}</div>
            ))}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function renderMathBlock(block: DocMathBlock, compact = false): ReactNode {
  return (
    <div className={compact ? styles.docNestedMathBlock : styles.docMathBlock}>
      {block.formulas.map((formula, index) => (
        <Latex key={`math-${index}`} math={formula} display />
      ))}
    </div>
  );
}

function renderNestedBlock(block: DocNestedBlock): ReactNode {
  switch (block.type) {
    case 'paragraph':
    case 'text':
      return <p className={styles.docFieldParagraph}>{renderInline(block.inline)}</p>;
    case 'directive_block':
      return renderDirective(block);
    case 'definition_list':
      return renderNestedDefinitionList(block);
    case 'list':
      return renderList(block, true);
    case 'literal_block':
      return (
        <pre className={styles.docLiteralBlock}>
          <code>{block.text}</code>
        </pre>
      );
    case 'math_block':
      return renderMathBlock(block, true);
    case 'raw_block':
      return renderRawBlock(block);
    default:
      return null;
  }
}

function renderDirective(directive: DocDirectiveBlock): ReactNode {
  const labels = {
    versionchanged: 'Changed in version',
    versionadded: 'Added in version',
    deprecated: 'Deprecated since',
    note: 'Note',
    warning: 'Warning',
    plot: 'Plot Source',
  } as const;

  const label = labels[directive.directive as keyof typeof labels] ?? directive.directive;

  if (directive.directive === 'plot') {
    return (
      <div className={styles.docDirective}>
        <strong>{label}.</strong>
        {directive.content_blocks.map((contentBlock, index) => (
          <div key={`plot-${index}`} className={styles.docDirectiveBody}>
            {renderNestedBlock(contentBlock)}
          </div>
        ))}
      </div>
    );
  }

  if (directive.directive === 'note' || directive.directive === 'warning') {
    return (
      <div className={styles.docDirective}>
        <strong>{label}.</strong>{' '}
        {directive.argument_inline.length > 0 ? renderInline(directive.argument_inline) : null}
        {directive.content_blocks.length > 0 ? (
          <div className={styles.docDirectiveBody}>
            {directive.content_blocks.map((contentBlock, index) => (
              <div key={`${directive.directive}-${index}`}>{renderNestedBlock(contentBlock)}</div>
            ))}
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className={styles.docDirective}>
      <strong>
        {label} {directive.version}.
      </strong>{' '}
      {renderInline(directive.argument_inline)}
    </div>
  );
}

function renderList(block: DocListBlock, nested = false) {
  const Tag = block.ordered ? 'ol' : 'ul';
  return (
    <Tag className={nested ? styles.docNestedList : styles.docLinkList}>
      {block.items.map((item, index) => (
        <li key={`item-${index}`}>
          {item.blocks.map((nestedBlock, nestedIndex) => (
            <div key={`item-${index}-${nestedIndex}`}>{renderNestedBlock(nestedBlock)}</div>
          ))}
        </li>
      ))}
    </Tag>
  );
}

function renderLinks(block: DocLinkListNode): ReactNode {
  return (
    <ul className={styles.docLinkList}>
      {block.links.map((link, index) => (
        <li key={`${link.target}-${index}`}>
          <OperationDocLink
            label={link.href ? (link.label.startsWith('we.') ? link.label : `we.${link.label}`) : link.label}
            href={link.href}
            externalUrl={link.external_url}
          />
          {link.description_inline && link.description_inline.length > 0 ? (
            <span className={styles.docLinkDescription}>
              {' '}
              {renderInline(link.description_inline)}
            </span>
          ) : link.description ? (
            <span className={styles.docLinkDescription}> {link.description}</span>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

function renderRawBlock(block: DocRawBlock): ReactNode {
  return (
    <div className={styles.docRawBlock}>
      <strong>{block.raw_kind}</strong>
      <pre className={styles.docLiteralBlock}>
        <code>{block.raw_text}</code>
      </pre>
    </div>
  );
}

async function renderDoctest(block: DocBlock & {lines: DocLine[]}): Promise<ReactNode> {
  const transcript = block.lines
    .map((line) => (line.kind === 'input' ? `${line.prompt ?? ''} ${line.text}`.trim() : line.text))
    .join('\n');
  return renderHighlightedCode(transcript, 'python');
}

async function renderHighlightedCode(code: string, lang: string): Promise<ReactNode> {
  return (
    <ServerCodeBlock
      code={code}
      lang={lang}
      themes={apiCodeThemes}
      codeblock={{
        allowCopy: false,
        keepBackground: true,
        className: styles.apiCodeFigure,
        viewportProps: {className: styles.apiCodeViewport},
      }}
    />
  );
}

async function renderLiteralBlock(block: DocBlock & {text: string; language?: string}): Promise<ReactNode> {
  if (block.language) {
    return renderHighlightedCode(block.text, block.language);
  }

  return (
    <pre className={styles.docLiteralBlock}>
      <code>{block.text}</code>
    </pre>
  );
}

function renderBlock(block: DocBlock): ReactNode {
  switch (block.type) {
    case 'paragraph':
      return <p className={styles.docParagraph}>{renderInline(block.inline)}</p>;
    case 'field_list':
      return renderFieldList(block);
    case 'directive_block':
      return renderDirective(block);
    case 'definition_list':
      return renderNestedDefinitionList(block);
    case 'math_block':
      return renderMathBlock(block);
    case 'literal_block':
      return null;
    case 'list':
      return renderList(block);
    case 'link_list':
      return renderLinks(block);
    case 'raw_block':
      return renderRawBlock(block);
    case 'doctest_block':
    default:
      return null;
  }
}

function normalizeText(value: string) {
  return value.trim().replace(/\s+/g, ' ');
}

export default async function OperationDocBody({
  sections,
  headerSummary,
  signature,
  whestSourceUrl,
  upstreamSourceUrl,
}: {
  sections: DocSection[];
  headerSummary?: string;
  signature?: string;
  whestSourceUrl?: string;
  upstreamSourceUrl?: string;
}): Promise<ReactNode> {
  const visibleSections = sections.filter((section) => {
    if (section.title !== 'Summary') {
      return true;
    }

    const summaryText = section.blocks
      .flatMap((block) => {
        if (block.type !== 'paragraph' && block.type !== 'text') {
          return [];
        }

        return block.inline
          .map((node) => node.text ?? node.display_text ?? node.target ?? node.latex ?? '')
          .join('')
          .trim();
      })
      .filter(Boolean)
      .join(' ')
      .trim();

    if (!summaryText) {
      return false;
    }

    if (!headerSummary) {
      return true;
    }

    return normalizeText(summaryText) !== normalizeText(headerSummary);
  });

  const signatureInsertIndex =
    signature && visibleSections.length > 0
      ? Math.max(
          0,
          visibleSections.findIndex((section) => section.title === 'Parameters'),
        )
      : -1;

  const renderedSections = await Promise.all(
    visibleSections.map(async (section, sectionIndex) => {
      const renderedBlocks = await Promise.all(
        section.blocks.map(async (block, blockIndex) => {
          const key = `${section.title}-${blockIndex}`;
          if (block.type === 'doctest_block') {
            return <div key={key}>{await renderDoctest(block)}</div>;
          }
          if (block.type === 'literal_block') {
            return <div key={key}>{await renderLiteralBlock(block)}</div>;
          }
          return <div key={key}>{renderBlock(block)}</div>;
        }),
      );

      return (
        <Fragment key={section.title}>
          {sectionIndex === signatureInsertIndex ? (
            <OperationDocSignature
              signature={signature}
              whestSourceUrl={whestSourceUrl}
              upstreamSourceUrl={upstreamSourceUrl}
            />
          ) : null}
          <section className={styles.docSection}>
            <OperationDocSectionHeading title={section.title} />
            {renderedBlocks}
          </section>
        </Fragment>
      );
    }),
  );

  if (renderedSections.length === 0 && signature) {
    return (
      <OperationDocSignature
        signature={signature}
        whestSourceUrl={whestSourceUrl}
        upstreamSourceUrl={upstreamSourceUrl}
      />
    );
  }

  return <>{renderedSections}</>;
}
