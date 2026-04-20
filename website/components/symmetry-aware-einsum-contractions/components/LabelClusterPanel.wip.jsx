import React from 'react';
import RoleBadge from './RoleBadge.jsx';

function clamp(n) {
  const v = Math.max(1, Math.floor(Number(n) || 1));
  return Math.min(v, 20);
}

export default function LabelClusterPanel({
  clusters = [],
  onSizeChange,
  vLabels = [],
  onSetAll,
}) {
  const vSet = new Set(vLabels);
  if (clusters.length === 0) return null;
  return (
    <div className="rounded-md border border-gray-200 bg-white">
      <div className="flex items-center justify-between border-b border-gray-100 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Label clusters
        </span>
        {onSetAll ? (
          <button
            type="button"
            className="rounded border border-gray-300 px-2 py-0.5 text-[11px] hover:bg-gray-50"
            onClick={() => {
              const raw = typeof window !== 'undefined'
                ? window.prompt('Set all clusters to n =', '5')
                : '5';
              const n = clamp(raw);
              if (Number.isFinite(n)) onSetAll(n);
            }}
          >
            Set all n
          </button>
        ) : null}
      </div>
      <ul className="divide-y divide-gray-100">
        {clusters.map((cluster) => (
          <li key={cluster.id} className="flex items-center gap-3 px-3 py-2">
            <div className="flex flex-wrap gap-1">
              {cluster.labels.map((label) => (
                <RoleBadge key={`${cluster.id}-${label}`} role={vSet.has(label) ? 'v' : 'w'}>
                  {label}
                </RoleBadge>
              ))}
            </div>
            <span className="ml-auto inline-flex items-center gap-1 text-xs">
              <span className="text-muted-foreground">n =</span>
              <input
                type="number"
                min={1}
                max={20}
                value={cluster.size}
                onChange={(event) => {
                  const n = clamp(event.target.value);
                  if (onSizeChange) onSizeChange(cluster.id, n);
                }}
                className="w-14 rounded border border-gray-300 px-2 py-0.5 text-right"
                aria-label={`Size for cluster ${cluster.id}`}
              />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
