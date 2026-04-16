import styles from './styles.module.css';
import type {OperationDocRecord} from './op-doc-types';

export default function OperationDocHeader({op}: {op: OperationDocRecord}) {
  return (
    <header className={styles.docHeader}>
      {op.summary ? <p className={styles.docSummary}>{op.summary}</p> : null}
      <div className={styles.docMetaLine}>
        {op.provenance_label && op.provenance_url ? (
          <p className={styles.docProvenance}>
            {op.provenance_label}{' '}
            <a href={op.provenance_url} target="_blank" rel="noreferrer">
              {op.numpy_ref}
            </a>
          </p>
        ) : null}
      </div>
    </header>
  );
}
