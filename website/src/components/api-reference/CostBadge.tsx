import React from 'react';
import styles from './styles.module.css';

type CostType = 'free' | 'blocked' | 'counted';

interface CostBadgeProps {
  free: boolean;
  blocked: boolean;
}

export default function CostBadge({free, blocked}: CostBadgeProps): React.ReactElement {
  let type: CostType;
  let label: string;

  if (blocked) {
    type = 'blocked';
    label = 'Blocked';
  } else if (free) {
    type = 'free';
    label = 'Free';
  } else {
    type = 'counted';
    label = 'Counted';
  }

  return (
    <span className={`${styles.badge} ${styles[`badge--${type}`]}`}>
      {label}
    </span>
  );
}
