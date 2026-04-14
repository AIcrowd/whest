import React from 'react';
import styles from './styles.module.css';

export type CostFilter = 'all' | 'counted' | 'free' | 'blocked';

interface FilterBarProps {
  search: string;
  onSearchChange: (value: string) => void;
  costFilter: CostFilter;
  onCostFilterChange: (value: CostFilter) => void;
  module: string;
  onModuleChange: (value: string) => void;
  modules: string[];
  resultCount: number;
  totalCount: number;
}

const COST_FILTERS: {label: string; value: CostFilter}[] = [
  {label: 'All', value: 'all'},
  {label: 'Counted', value: 'counted'},
  {label: 'Free', value: 'free'},
  {label: 'Blocked', value: 'blocked'},
];

export default function FilterBar({
  search,
  onSearchChange,
  costFilter,
  onCostFilterChange,
  module,
  onModuleChange,
  modules,
  resultCount,
  totalCount,
}: FilterBarProps): React.ReactElement {
  return (
    <div className={styles.filterBar}>
      <div className={styles.filterRow}>
        <input
          type="text"
          className={styles.searchInput}
          placeholder="Search by name, ref, or notes..."
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
        />
        <select
          className={styles.moduleSelect}
          value={module}
          onChange={(e) => onModuleChange(e.target.value)}
        >
          <option value="">All modules</option>
          {modules.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>
      <div className={styles.filterRow}>
        <div className={styles.toggleGroup}>
          {COST_FILTERS.map(({label, value}) => (
            <button
              key={value}
              className={`${styles.toggleBtn} ${costFilter === value ? styles.toggleBtnActive : ''}`}
              onClick={() => onCostFilterChange(value)}
            >
              {label}
            </button>
          ))}
        </div>
        <span className={styles.resultCount}>
          {resultCount} of {totalCount}
        </span>
      </div>
    </div>
  );
}
