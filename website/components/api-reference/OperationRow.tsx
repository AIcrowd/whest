'use client';

import React from 'react';
import Link from 'next/link';
import Latex, {stripMathDelimiters} from '../shared/Latex';
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
  weight: number;
  area: 'core' | 'linalg' | 'fft' | 'random' | 'stats';
  display_type: 'counted' | 'custom' | 'free' | 'blocked';
  href?: string;
}

function renderFormula(op: Operation) {
  const latex = op.cost_formula_latex.trim();
  if (latex.startsWith('$') && latex.endsWith('$')) {
    return <Latex math={stripMathDelimiters(latex)} />;
  }

  return <span>{op.cost_formula || '\u2014'}</span>;
}

export default function OperationRow({op}: {op: Operation}): React.ReactElement {
  const opLabel = <code>{op.whest_ref}</code>;

  return (
    <tr className={styles.opRow}>
      <td className={styles.opName}>
        {op.href ? <Link href={op.href}>{opLabel}</Link> : opLabel}
      </td>
      <td>
        <span className={`${styles.areaChip} ${styles[`area--${op.area}`]}`}>{op.area}</span>
      </td>
      <td>
        <span className={`${styles.typeChip} ${styles[`type--${op.display_type}`]}`}>
          {op.display_type}
        </span>
      </td>
      <td className={styles.opWeight}>{op.weight.toFixed(1)}&times;</td>
      <td className={styles.opFormula}>{renderFormula(op)}</td>
    </tr>
  );
}
