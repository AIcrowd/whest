import BilledCost from './BilledCost';
import styles from './styles.module.css';
import type {ApiDocRecord} from './op-doc-types';

// Paper-register metadata panel: no card chrome, no "Quick Info" heading.
// Kickers + chips/values flow as part of the page directly under the
// operation signature — matches the rest of the docs where structured
// data sits in the prose flow without widget framing.
export default function OperationDocOverlay({op}: {op: ApiDocRecord}) {
  const hasOperationChrome =
    !!op.area &&
    !!op.display_type &&
    !!op.numpy_ref &&
    (typeof op.weight === 'number' ||
      !!op.cost_formula ||
      !!op.cost_formula_latex ||
      !!op.operation);

  if (!hasOperationChrome) {
    return null;
  }

  const aliases = op.aliases.length
    ? op.aliases
        .map((alias) =>
          alias.startsWith('fnp.') ||
          alias.startsWith('flops.') ||
          alias.startsWith('flopscope.')
            ? alias
            : `fnp.${alias}`,
        )
        .join(', ')
    : null;

  const billedOp = {
    weight: op.weight ?? 1,
    cost_formula: op.cost_formula ?? op.operation?.cost_formula ?? '',
    cost_formula_latex:
      op.cost_formula_latex ?? op.operation?.cost_formula_latex ?? '',
    display_type: op.display_type ?? 'counted',
    free: op.operation?.free ?? false,
    blocked: op.operation?.blocked ?? false,
  };

  return (
    <section className={styles.docOverlay} aria-label="Operation summary">
      <div className={styles.docMetaStrip}>
        <div className={styles.docMetaPair}>
          <span className={styles.docMetaLabel}>Area</span>
          <span className={`${styles.areaChip} ${styles[`area--${op.area}`]}`}>{op.area}</span>
        </div>
        <div className={styles.docMetaPair}>
          <span className={styles.docMetaLabel}>Type</span>
          <span className={`${styles.typeChip} ${styles[`type--${op.display_type}`]}`}>
            {op.display_type}
          </span>
        </div>
        <div className={styles.docMetaPair}>
          <span className={styles.docMetaLabel}>
            {op.upstream_ref_label ?? 'NumPy Ref'}
          </span>
          <span className={styles.docMetaValue}>
            {op.provenance_url ? (
              <a href={op.provenance_url} target="_blank" rel="noreferrer">
                {op.provenance_ref ?? op.numpy_ref}
              </a>
            ) : (
              op.provenance_ref ?? op.numpy_ref
            )}
          </span>
        </div>
      </div>

      {aliases ? (
        <div className={styles.docMetaRow}>
          <span className={styles.docMetaLabel}>Aliases</span>
          <span className={styles.docMetaValue}>{aliases}</span>
        </div>
      ) : null}

      <div className={styles.docMetaRow}>
        <span className={styles.docMetaLabel}>Cost</span>
        <div className={styles.docFormula}>
          <BilledCost op={billedOp} />
        </div>
      </div>

      {op.notes ? (
        <div className={styles.docMetaRow}>
          <span className={styles.docMetaLabel}>Flopscope Context</span>
          <p className={styles.docBodyText}>{op.notes}</p>
        </div>
      ) : null}
    </section>
  );
}
