import React, {useState, useMemo} from 'react';
import {usePluginData} from '@docusaurus/useGlobalData';
import FilterBar, {type CostFilter} from './FilterBar';
import OperationRow, {type Operation} from './OperationRow';
import styles from './styles.module.css';

interface OpsData {
  operations: Operation[];
}

export default function ApiReference(): React.ReactElement {
  const data = usePluginData('plugin-api-docs') as OpsData;
  const operations = data?.operations ?? [];

  const [search, setSearch] = useState('');
  const [costFilter, setCostFilter] = useState<CostFilter>('all');
  const [moduleFilter, setModuleFilter] = useState('');
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

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
        onSearchChange={(v) => { setSearch(v); setExpandedIdx(null); }}
        costFilter={costFilter}
        onCostFilterChange={(v) => { setCostFilter(v); setExpandedIdx(null); }}
        module={moduleFilter}
        onModuleChange={(v) => { setModuleFilter(v); setExpandedIdx(null); }}
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
                expanded={expandedIdx === i}
                onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)}
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
