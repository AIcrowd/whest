import styles from './styles.module.css';

export default function OperationDocHeader({
  summary,
  provenanceLabel,
  provenanceUrl,
  provenanceRef,
}: {
  summary?: string | null;
  provenanceLabel?: string | null;
  provenanceUrl?: string | null;
  provenanceRef?: string | null;
}) {
  return (
    <header className={styles.docHeader}>
      {summary ? <p className={styles.docSummary}>{summary}</p> : null}
      <div className={styles.docMetaLine}>
        {provenanceLabel && provenanceUrl && provenanceRef ? (
          <p className={styles.docProvenance}>
            {provenanceLabel}{' '}
            <a href={provenanceUrl} target="_blank" rel="noreferrer">
              {provenanceRef}
            </a>
          </p>
        ) : null}
      </div>
    </header>
  );
}
