import styles from './styles.module.css';
import type {OperationDocRecord} from './op-doc-types';

function SourceAction({href, label}: {href?: string; label: string}) {
  if (!href) {
    return null;
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={styles.docAction}
    >
      {label}
    </a>
  );
}

export default function OperationDocHeader({op}: {op: OperationDocRecord}) {
  return (
    <header className={styles.docHeader}>
      <p className={styles.docEyebrow}>Operation Reference</p>
      <h1 className={styles.docTitle}>{op.whest_ref}</h1>
      <p className={styles.docSignature}>
        <code>{op.signature}</code>
      </p>
      {op.summary ? <p className={styles.docSummary}>{op.summary}</p> : null}
      {op.provenance_label && op.provenance_url ? (
        <p className={styles.docProvenance}>
          {op.provenance_label}{' '}
          <a href={op.provenance_url} target="_blank" rel="noreferrer">
            {op.numpy_ref}
          </a>
        </p>
      ) : null}
      <div className={styles.docActions}>
        <SourceAction href={op.whest_source_url} label="whest source" />
        <SourceAction href={op.upstream_source_url} label="show source" />
      </div>
    </header>
  );
}
