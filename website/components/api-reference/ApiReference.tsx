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
  const [moduleFilter, setModuleFilter] = useState('');

  const modules = useMemo(
    () => Array.from(new Set(operations.map((op) => op.module))).sort(),
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
      // module filter
      if (moduleFilter && op.module !== moduleFilter) return false;
      return true;
    });
  }, [operations, search, costFilter, moduleFilter]);

  return (
    <div className={styles.apiReference}>
      <FilterBar
        search={search}
        onSearchChange={setSearch}
        costFilter={costFilter}
        onCostFilterChange={setCostFilter}
        module={moduleFilter}
        onModuleChange={setModuleFilter}
        modules={modules}
        resultCount={filtered.length}
        totalCount={operations.length}
      />
      <div className={styles.tableWrap}>
        <table className={styles.opsTable}>
          <thead>
            <tr>
              <th>Operation</th>
              <th>Module</th>
              <th>Type</th>
              <th>Cost Formula</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((op, i) => (
              <OperationRow
                key={op.whest_ref}
                op={op}
              />
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
    </div>
  );
}
