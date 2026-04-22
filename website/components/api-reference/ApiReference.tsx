'use client';

import React, {useState, useMemo} from 'react';
import FilterBar, {type CostFilter} from './FilterBar';
import OperationRow, {type Operation} from './OperationRow';
import styles from './styles.module.css';

interface ApiReferenceProps {
  operations: Operation[];
}

export default function ApiReference({operations}: ApiReferenceProps): React.ReactElement {
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
      // text search
      if (q) {
        const haystack = `${op.name} ${op.whest_ref} ${op.numpy_ref} ${op.notes}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      // cost filter
      if (costFilter === 'free' && !op.free) return false;
      if (costFilter === 'blocked' && !op.blocked) return false;
      if (costFilter === 'counted' && (op.free || op.blocked)) return false;
      // area filter
      if (areaFilter && op.area !== areaFilter) return false;
      return true;
    });
  }, [operations, search, costFilter, areaFilter]);

  return (
    <div className={styles.apiReference}>
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
      <div className={styles.tableWrap}>
        <table className={styles.opsTable}>
          <thead>
            <tr>
              <th>Operation</th>
              <th>Area</th>
              <th>Type</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((op) => (
              <OperationRow key={op.name} op={op} />
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={4} className={styles.noResults}>
                  No operations match your filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
