'use client';

import React, {useMemo, useState} from 'react';
import FilterBar, {type CostFilter} from './FilterBar';
import type {Operation} from './operation-data';
import OperationRow from './OperationRow';
import styles from './styles.module.css';

export interface OperationCostIndexProps {
  operations: Operation[];
  showHeading?: boolean;
}

export default function OperationCostIndex({
  operations,
  showHeading = true,
}: OperationCostIndexProps): React.ReactElement {
  const [search, setSearch] = useState('');
  const [costFilter, setCostFilter] = useState<CostFilter>('all');
  const [areaFilter, setAreaFilter] = useState('');

  const areas = useMemo(
    () => Array.from(new Set(operations.map((op) => op.area))).sort(),
    [operations],
  );

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return operations.filter((op) => {
      if (q) {
        const haystack =
          `${op.name} ${op.whest_ref} ${op.numpy_ref} ${op.notes}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      if (costFilter === 'free' && !op.free) return false;
      if (costFilter === 'blocked' && !op.blocked) return false;
      if (costFilter === 'counted' && (op.free || op.blocked)) return false;
      if (areaFilter && op.area !== areaFilter) return false;
      return true;
    });
  }, [operations, search, costFilter, areaFilter]);

  return (
    <div className={styles.operationCostIndex}>
      {showHeading ? (
        <div className={styles.sectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Operation cost index</h2>
            <p className={styles.sectionDescription}>
              Counted operations, free helpers, and blocked calls from the
              public whest API surface.
            </p>
          </div>
          <span className={styles.sectionCount}>{filtered.length} entries</span>
        </div>
      ) : null}
      <FilterBar
        search={search}
        onSearchChange={setSearch}
        costFilter={costFilter}
        onCostFilterChange={setCostFilter}
        area={areaFilter}
        onAreaChange={setAreaFilter}
        areas={areas}
        resultCount={filtered.length}
        totalCount={operations.length}
      />
      <section className={styles.referenceSection}>
        <div className={styles.tableWrap}>
          <table className={styles.opsTable}>
            <thead>
              <tr>
                <th>Operation</th>
                <th>Area</th>
                <th>Type</th>
                <th>Weight</th>
                <th>Cost Formula</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((op) => (
                <OperationRow key={op.name} op={op} />
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={5} className={styles.noResults}>
                    No operations match your filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
