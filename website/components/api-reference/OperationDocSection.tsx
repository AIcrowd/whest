import type {ReactNode} from 'react';
import OperationDocLink from './OperationDocLink';
import OperationDocFieldList from './OperationDocFieldList';
import OperationDocSectionHeading from './OperationDocSectionHeading';
import Latex from '../shared/Latex';
import styles from './styles.module.css';
import type {DocField, DocLink} from './op-doc-types';

type OperationDocSectionTitle =
  | 'Parameters'
  | 'Returns'
  | 'See also'
  | 'Members'
  | 'Related guides'
  | 'Notes';

type OperationDocSectionProps =
  | {title: 'Parameters' | 'Returns'; fields: DocField[]; links?: never; paragraphs?: never}
  | {
      title: 'See also' | 'Members' | 'Related guides';
      fields?: never;
      links: DocLink[];
      paragraphs?: never;
    }
  | {title: 'Notes'; fields?: never; links?: never; paragraphs: string[]};

const INLINE_PATTERN = /:math:`([^`]+)`|``([^`]+)``|`([^`]+)`|\*([^*]+)\*/g;

function renderInlineMarkup(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  for (let keyIndex = 0; (match = INLINE_PATTERN.exec(text)) !== null; keyIndex += 1) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    if (match[1]) {
      nodes.push(<Latex key={`math-${keyIndex}`} math={match[1]} />);
    } else if (match[2] || match[3]) {
      nodes.push(
        <code key={`code-${keyIndex}`}>
          {match[2] ?? match[3]}
        </code>,
      );
    } else if (match[4]) {
      nodes.push(
        <em key={`em-${keyIndex}`} className={styles.docInlineEmphasis}>
          {match[4]}
        </em>,
      );
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function joinBody(lines: string[]) {
  return lines.join(' ');
}

function parseDirective(paragraph: string) {
  const match = paragraph.match(/^\.\.\s+(versionchanged|versionadded|deprecated)::\s*(.+)$/);
  if (!match) {
    return null;
  }

  const [, directive, content] = match;
  const labels = {
    versionchanged: 'Changed in version',
    versionadded: 'Added in version',
    deprecated: 'Deprecated since',
  } as const;

  const [version, ...rest] = content.split(/\s+/);
  return {
    label: labels[directive as keyof typeof labels],
    version,
    detail: rest.join(' '),
  };
}

function renderNotes(paragraphs: string[]) {
  const rendered: ReactNode[] = [];

  for (let index = 0; index < paragraphs.length; index += 1) {
    const paragraph = paragraphs[index];
    const directive = parseDirective(paragraph);

    if (directive) {
      rendered.push(
        <div key={`directive-${index}`} className={styles.docDirective}>
          <strong>{directive.label} {directive.version}.</strong>{' '}
          {renderInlineMarkup(directive.detail)}
        </div>,
      );
      continue;
    }

    if (paragraph.endsWith('::') && paragraphs[index + 1]) {
      rendered.push(
        <p key={`note-${index}`} className={styles.docParagraph}>
          {renderInlineMarkup(paragraph.slice(0, -1))}
        </p>,
      );
      rendered.push(
        <pre key={`literal-${index + 1}`} className={styles.docLiteralBlock}>
          <code>{paragraphs[index + 1]}</code>
        </pre>,
      );
      index += 1;
      continue;
    }

    rendered.push(
      <p key={`note-${index}`} className={styles.docParagraph}>
        {renderInlineMarkup(paragraph)}
      </p>,
    );
  }

  return rendered;
}

function renderLink(link: DocLink, prefixWhestLabel: boolean) {
  const label =
    link.href && prefixWhestLabel
      ? link.label.startsWith('we.')
        ? link.label
        : `we.${link.label}`
      : link.label;

  return (
    <OperationDocLink
      label={link.href ? label : link.label}
      href={link.href}
      externalUrl={link.external_url}
      muted={!link.href && !link.external_url}
    />
  );
}

export default function OperationDocSection(props: OperationDocSectionProps) {
  const {title} = props;
  const isFieldSection = title === 'Parameters' || title === 'Returns';

  if (isFieldSection && props.fields.length === 0) {
    return null;
  }

  if (
    (title === 'See also' ||
      title === 'Members' ||
      title === 'Related guides') &&
    props.links.length === 0
  ) {
    return null;
  }

  if (title === 'Notes' && props.paragraphs.length === 0) {
    return null;
  }

  return (
    <section className={styles.docSection}>
      <OperationDocSectionHeading title={title} />
      {isFieldSection ? (
        <OperationDocFieldList
          items={props.fields.map((field) => ({
            key: `${title}-${field.name}`,
            name: field.name,
            type: field.type,
            body: renderInlineMarkup(joinBody(field.body)),
          }))}
        />
      ) : null}

      {title === 'See also' ||
      title === 'Members' ||
      title === 'Related guides' ? (
        <ul className={styles.docLinkList}>
          {props.links.map((link) => (
            <li key={`${link.target}-${link.external_url ?? ''}`}>
              {renderLink(link, title !== 'Related guides')}
              {link.description ? (
                <span className={styles.docLinkDescription}> {renderInlineMarkup(link.description)}</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}

      {title === 'Notes' ? renderNotes(props.paragraphs) : null}
    </section>
  );
}
