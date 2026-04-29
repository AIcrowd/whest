'use client';

import React from 'react';
import Link from 'next/link';
import BilledCost from './BilledCost';
import styles from './styles.module.css';

export interface Operation {
  name: string;
  module: string;
  flopscope_ref: string;
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

export default function OperationRow({op}: {op: Operation}): React.ReactElement {
  const opLabel = <span className={styles.opLink}>{op.flopscope_ref}</span>;

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
      <td className={styles.opCost}>
        <BilledCost op={op} className={styles.opBilledCost} />
      </td>
    </tr>
  );
}
