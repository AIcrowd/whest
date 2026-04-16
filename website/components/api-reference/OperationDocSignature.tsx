import styles from './styles.module.css';

function parseSignature(signature: string) {
  const match = signature.match(/^([^([]+)(.*)$/);
  if (!match) {
    return {
      namespace: '',
      functionName: signature,
      remainder: '',
    };
  }

  const name = match[1];
  const lastDot = name.lastIndexOf('.');

  return {
    namespace: lastDot >= 0 ? name.slice(0, lastDot + 1) : '',
    functionName: lastDot >= 0 ? name.slice(lastDot + 1) : name,
    remainder: match[2] ?? '',
  };
}

export default function OperationDocSignature({
  signature,
  whestSourceUrl,
  upstreamSourceUrl,
}: {
  signature?: string;
  whestSourceUrl?: string;
  upstreamSourceUrl?: string;
}) {
  if (!signature) {
    return null;
  }

  const parsed = parseSignature(signature);

  return (
    <div className={styles.docSignatureBlock}>
      <p className={styles.docSignature}>
        {parsed.namespace ? (
          <span className={styles.docSignatureNamespace}>{parsed.namespace}</span>
        ) : null}
        <span className={styles.docSignatureFunction}>{parsed.functionName}</span>
        <span className={styles.docSignatureParams}>{parsed.remainder}</span>
        {(whestSourceUrl || upstreamSourceUrl) ? (
          <span className={styles.docSignatureActions}>
            {whestSourceUrl ? (
              <a
                href={whestSourceUrl}
                target="_blank"
                rel="noreferrer"
                className={styles.docSourceLink}
              >
                [whest source]
              </a>
            ) : null}
            {upstreamSourceUrl ? (
              <a
                href={upstreamSourceUrl}
                target="_blank"
                rel="noreferrer"
                className={styles.docSourceLink}
              >
                [numpy source]
              </a>
            ) : null}
          </span>
        ) : null}
      </p>
    </div>
  );
}
