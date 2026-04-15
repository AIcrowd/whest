import opDocsJson from '../../.generated/op-docs.json';
import Latex, {stripMathDelimiters} from '../shared/Latex';
import styles from './styles.module.css';

type OperationDocRecord = (typeof opDocsJson)[keyof typeof opDocsJson];

const opDocs = opDocsJson as Record<string, OperationDocRecord>;

function stripCodeTicks(value: string): string {
  return value.replace(/`/g, '');
}

function renderFormula(value: string, display = false) {
  const trimmed = value.trim();
  if (!trimmed) {
    return <span>&mdash;</span>;
  }

  if (trimmed.startsWith('$') && trimmed.endsWith('$')) {
    return <Latex math={stripMathDelimiters(trimmed)} display={display} />;
  }

  return <span>{trimmed}</span>;
}

export default function OperationDocPage({name}: {name: string}) {
  const op = opDocs[name];

  if (!op) {
    throw new Error(`Unknown operation doc: ${name}`);
  }

  const whestRef = stripCodeTicks(op.whest_ref);
  const numpyRef = stripCodeTicks(op.numpy_ref);
  const aliases = op.aliases.length ? op.aliases.map((alias) => `we.${alias}`).join(', ') : 'None';

  return (
    <>
      <p className={styles.docEyebrow}>Operation Reference</p>
      <h1>
        <code>{whestRef}</code>
      </h1>
      <p className={styles.docSignature}>
        <code>{op.signature}</code>
      </p>

      <h2>Quick Info</h2>
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
          <code>{numpyRef}</code>
        </div>
      </div>

      <h2>Whest-Specific Info</h2>
      <div className={styles.docDenseGrid}>
        <div className={styles.docDenseItem}>
          <span className={styles.docMetaLabel}>Canonical Name</span>
          <code>{op.name}</code>
        </div>
        <div className={styles.docDenseItem}>
          <span className={styles.docMetaLabel}>Aliases</span>
          <code>{aliases}</code>
        </div>
        <div className={`${styles.docDenseItem} ${styles.docDenseItemWide}`}>
          <span className={styles.docMetaLabel}>Cost Formula</span>
          <div className={styles.docFormula}>{renderFormula(op.cost_formula_latex, true)}</div>
        </div>
        <div className={`${styles.docDenseItem} ${styles.docDenseItemWide}`}>
          <span className={styles.docMetaLabel}>Notes</span>
          <p className={styles.docBodyText}>{op.notes || 'No whest-specific notes available yet.'}</p>
        </div>
      </div>

      <h2>API Docs</h2>
      {op.api_docs_html ? (
        <div dangerouslySetInnerHTML={{__html: op.api_docs_html}} />
      ) : (
        <p className={styles.docPlaceholder}>
          No API docs available yet. Live docstring extraction lands in the next generator slice.
        </p>
      )}

      <h2>Whest Examples</h2>
      {op.whest_examples_html ? (
        <div dangerouslySetInnerHTML={{__html: op.whest_examples_html}} />
      ) : (
        <p className={styles.docPlaceholder}>
          No examples available yet. Whest-owned examples will be added and tracked separately.
        </p>
      )}
    </>
  );
}
