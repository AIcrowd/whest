import OperationDocLink from './OperationDocLink';
import styles from './styles.module.css';
import type {DocField, DocLink} from './op-doc-types';

type OperationDocSectionTitle = 'Parameters' | 'Returns' | 'See also' | 'Notes';

type OperationDocSectionProps =
  | {title: 'Parameters' | 'Returns'; fields: DocField[]; links?: never; paragraphs?: never}
  | {title: 'See also'; fields?: never; links: DocLink[]; paragraphs?: never}
  | {title: 'Notes'; fields?: never; links?: never; paragraphs: string[]};

function joinBody(lines: string[]) {
  return lines.join(' ');
}

function renderLink(link: DocLink) {
  const label = link.label.startsWith('we.') ? link.label : `we.${link.label}`;

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

  if (title === 'See also' && props.links.length === 0) {
    return null;
  }

  if (title === 'Notes' && props.paragraphs.length === 0) {
    return null;
  }

  return (
    <section className={styles.docSection}>
      <h2>{title}</h2>
      {isFieldSection ? (
        <dl className={styles.docDefinitionList}>
          {props.fields.map((field) => (
            <div key={`${title}-${field.name}`} className={styles.docDefinitionItem}>
              <dt className={styles.docDefinitionTerm}>
                <code>{field.name}</code>
                {field.type ? <span className={styles.docDefinitionType}>{field.type}</span> : null}
              </dt>
              <dd className={styles.docDefinitionBody}>{joinBody(field.body)}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      {title === 'See also' ? (
        <ul className={styles.docLinkList}>
          {props.links.map((link) => (
            <li key={`${link.target}-${link.external_url ?? ''}`}>
              {renderLink(link)}
              {link.description ? <span className={styles.docLinkDescription}> {link.description}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}

      {title === 'Notes'
        ? props.paragraphs.map((paragraph, index) => (
            <p key={`note-${index}`} className={styles.docParagraph}>
              {paragraph}
            </p>
          ))
        : null}
    </section>
  );
}
