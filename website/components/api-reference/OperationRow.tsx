'use client';

import React from 'react';
import Link from 'next/link';
import CostBadge from './CostBadge';
import styles from './styles.module.css';

export interface Operation {
  name: string;
  slug: string;
  detail_href: string;
  detail_json_href: string;
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
  summary: string;
  weight?: number;
}

interface OperationRowProps {
  op: Operation;
}

export default function OperationRow({op}: OperationRowProps): React.ReactElement {
  return (
    <tr className={styles.opRow}>
      <td className={styles.opName}>
        <Link href={op.detail_href} className={styles.opLink}>
          {op.whest_ref}
        </Link>
      </td>
      <td className={styles.opModule}>{op.module}</td>
      <td>
        <CostBadge free={op.free} blocked={op.blocked} />
      </td>
      <td className={styles.opFormula}>
        <code>{op.cost_formula || '\u2014'}</code>
      </td>
      <td className={styles.opArrow}>
        <Link href={op.detail_href} className={styles.detailLink}>
          Details
        </Link>
      </td>
    </tr>
  );
}
