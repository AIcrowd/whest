'use client';

import React from 'react';
import styles from './styles.module.css';

export type CostFilter = 'all' | 'counted' | 'free' | 'blocked';

interface FilterBarProps {
  search: string;
  onSearchChange: (value: string) => void;
  costFilter: CostFilter;
  onCostFilterChange: (value: CostFilter) => void;
  area: string;
  onAreaChange: (value: string) => void;
  areas: string[];
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
  area,
  onAreaChange,
  areas,
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
          className={styles.areaSelect}
          aria-label="Area"
          value={area}
          onChange={(e) => onAreaChange(e.target.value)}
        >
          <option value="">All areas</option>
          {areas.map((value) => (
            <option key={value} value={value}>{value}</option>
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
