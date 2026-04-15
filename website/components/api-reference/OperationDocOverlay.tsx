import Latex, {stripMathDelimiters} from '../shared/Latex';
import styles from './styles.module.css';
import type {OperationDocRecord} from './op-doc-types';

function renderFormula(value: string) {
  const trimmed = value.trim();
  if (!trimmed) {
    return <span>&mdash;</span>;
  }

  if (trimmed.startsWith('$') && trimmed.endsWith('$')) {
    return <Latex math={stripMathDelimiters(trimmed)} display />;
  }

  return <span>{trimmed}</span>;
}

export default function OperationDocOverlay({op}: {op: OperationDocRecord}) {
  const aliases = op.aliases.length ? op.aliases.map((alias) => `we.${alias}`).join(', ') : 'None';

  return (
    <section className={styles.docOverlay} aria-label="whest overlay">
      <div className={styles.docMetaGrid}>
        <div className={styles.docMetaCard}>
          <span className={styles.docMetaLabel}>Area</span>
          <span className={`${styles.areaChip} ${styles[`area--${op.area}`]}`}>{op.area}</span>
        </div>
        <div className={styles.docMetaCard}>
          <span className={styles.docMetaLabel}>Type</span>
          <span className={`${styles.typeChip} ${styles[`type--${op.display_type}`]}`}>
            {op.display_type}
          </span>
        </div>
        <div className={styles.docMetaCard}>
          <span className={styles.docMetaLabel}>Weight</span>
          <span className={styles.docMetaValue}>{op.weight.toFixed(1)}&times;</span>
        </div>
        <div className={styles.docMetaCard}>
          <span className={styles.docMetaLabel}>NumPy Ref</span>
          <span className={styles.docMetaValue}>{op.numpy_ref}</span>
        </div>
      </div>

      <div className={styles.docDenseGrid}>
        <div className={styles.docDenseItem}>
          <span className={styles.docMetaLabel}>Canonical Name</span>
          <span className={styles.docMetaValue}>{op.name}</span>
        </div>
        <div className={styles.docDenseItem}>
          <span className={styles.docMetaLabel}>Aliases</span>
          <span className={styles.docMetaValue}>{aliases}</span>
        </div>
        <div className={`${styles.docDenseItem} ${styles.docDenseItemWide}`}>
          <span className={styles.docMetaLabel}>Cost Formula</span>
          <div className={styles.docFormula}>{renderFormula(op.cost_formula_latex)}</div>
        </div>
        {op.notes ? (
          <div className={`${styles.docDenseItem} ${styles.docDenseItemWide}`}>
            <span className={styles.docMetaLabel}>Whest Notes</span>
            <p className={styles.docBodyText}>{op.notes}</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
