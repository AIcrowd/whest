import React from 'react';
import Latex, {stripMathDelimiters} from '../shared/Latex';
import styles from './styles.module.css';

export interface CostDisplayOperation {
  weight: number;
  cost_formula: string;
  cost_formula_latex: string;
  display_type: string;
  free?: boolean;
  blocked?: boolean;
}

function weightBucket(weight: number) {
  if (weight >= 16) return 'weight--16';
  if (weight >= 4) return 'weight--4';
  if (weight >= 2) return 'weight--2';
  return 'weight--1';
}

function formatWeight(weight: number) {
  return Number.isInteger(weight) ? `${weight}×` : `${weight.toFixed(1)}×`;
}

function renderFormulaContent(op: CostDisplayOperation) {
  const latex = op.cost_formula_latex.trim();
  if (latex.startsWith('$') && latex.endsWith('$')) {
    return <Latex math={stripMathDelimiters(latex)} />;
  }

  const plain = op.cost_formula.trim();
  if (plain) {
    return <span>{plain}</span>;
  }

  if (latex) {
    return <span>{latex}</span>;
  }

  return <span>&mdash;</span>;
}

export default function BilledCost({
  op,
  className,
}: {
  op: CostDisplayOperation;
  className?: string;
}): React.ReactElement {
  if (op.blocked || op.display_type === 'blocked') {
    return <span className={`${styles.billedCost} ${styles.billedCostBlocked}`.trim()}>&mdash;</span>;
  }

  if (op.free || op.display_type === 'free') {
    return (
      <span className={`${styles.billedCost} ${className ?? ''}`.trim()}>
        <span className={styles.billedCostExpression}>0</span>
      </span>
    );
  }

  const showWeight = op.weight !== 1;
  const weightLabel = formatWeight(op.weight);

  return (
    <span className={`${styles.billedCost} ${className ?? ''}`.trim()}>
      {showWeight ? (
        <span
          className={`${styles.weightChip} ${styles[weightBucket(op.weight)]} ${styles.costWeightFactor}`}
          title={`Weight multiplier ${weightLabel}`}
        >
          {weightLabel}
        </span>
      ) : null}
      <span className={styles.billedCostExpression}>{renderFormulaContent(op)}</span>
    </span>
  );
}
