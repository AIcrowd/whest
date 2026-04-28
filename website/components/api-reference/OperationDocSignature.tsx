import {Fragment} from 'react';
import {parseSignature, type ParsedParameter} from './parseSignature';
import styles from './styles.module.css';

/**
 * Render one parameter with italic name and distinct default.
 *
 * Python type hints (``: 'str'``, ``: '_np.ndarray'``) are intentionally
 * suppressed to match numpy.org's reference rendering — the signature is
 * meant to be a quick visual skim of the call shape, and full type info
 * is available in the Parameters block below. Types are still parsed
 * (and available on the ``ParsedParameter`` object) for any callers
 * that want to render them.
 */
function ParameterToken({param}: {param: ParsedParameter}) {
  return (
    <>
      {param.prefix ? (
        <span className={styles.docSignatureStar}>{param.prefix}</span>
      ) : null}
      <span className={styles.docSignatureParam}>{param.name}</span>
      {param.default !== undefined ? (
        <>
          <span className={styles.docSignaturePunct}>=</span>
          <span className={styles.docSignatureDefault}>{param.default}</span>
        </>
      ) : null}
    </>
  );
}

export default function OperationDocSignature({
  signature,
  flopscopeSourceUrl,
  upstreamSourceUrl,
}: {
  signature?: string;
  flopscopeSourceUrl?: string;
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
        {parsed.parameters ? (
          <>
            <span className={styles.docSignaturePunct}>(</span>
            {parsed.parameters.map((param, idx) => (
              <Fragment key={idx}>
                {idx > 0 ? <span className={styles.docSignaturePunct}>, </span> : null}
                <ParameterToken param={param} />
              </Fragment>
            ))}
            <span className={styles.docSignaturePunct}>)</span>
            {/* Return type intentionally suppressed to match numpy.org's
                reference rendering. Available on ``parsed.returnType`` if
                needed elsewhere. */}
          </>
        ) : (
          <span className={styles.docSignatureParams}>{parsed.remainder}</span>
        )}
        {(flopscopeSourceUrl || upstreamSourceUrl) ? (
          <span className={styles.docSignatureActions}>
            {flopscopeSourceUrl ? (
              <a
                href={flopscopeSourceUrl}
                target="_blank"
                rel="noreferrer"
                className={styles.docSourceLink}
              >
                [flopscope source]
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
