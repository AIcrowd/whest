import React from 'react';
import CostBadge from './CostBadge';
import styles from './styles.module.css';

export interface Operation {
  name: string;
  module: string;
  whest_ref: string;
  numpy_ref: string;
  category: string;
  cost_formula: string;
  cost_formula_latex: string;
  free: boolean;
  blocked: boolean;
  status: string;
  notes: string;
}

interface OperationRowProps {
  op: Operation;
  expanded: boolean;
  onToggle: () => void;
}

export default function OperationRow({op, expanded, onToggle}: OperationRowProps): React.ReactElement {
  return (
    <>
      <tr
        className={`${styles.opRow} ${expanded ? styles.opRowExpanded : ''}`}
        onClick={onToggle}
      >
        <td className={styles.opName}>
          <code>{op.whest_ref}</code>
        </td>
        <td className={styles.opModule}>{op.module}</td>
        <td>
          <CostBadge free={op.free} blocked={op.blocked} />
        </td>
        <td className={styles.opFormula}>
          <code>{op.cost_formula || '\u2014'}</code>
        </td>
        <td className={styles.opArrow}>
          <span className={`${styles.arrow} ${expanded ? styles.arrowOpen : ''}`}>
            &#9654;
          </span>
        </td>
      </tr>
      {expanded && (
        <tr className={styles.detailRow}>
          <td colSpan={5}>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>NumPy ref</span>
                <code>{op.numpy_ref}</code>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Category</span>
                <span>{op.category}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Status</span>
                <span className={`${styles.statusChip} ${styles[`status--${op.status}`]}`}>
                  {op.status}
                </span>
              </div>
              {op.notes && (
                <div className={`${styles.detailItem} ${styles.detailItemFull}`}>
                  <span className={styles.detailLabel}>Notes</span>
                  <span>{op.notes}</span>
                </div>
              )}
              {op.cost_formula && (
                <div className={`${styles.detailItem} ${styles.detailItemFull}`}>
                  <span className={styles.detailLabel}>Cost formula</span>
                  <code>{op.cost_formula}</code>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
