import React from 'react';
import Latex from './Latex.jsx';
import CaseBadge from './CaseBadge.jsx';

/**
 * Teaching card: "Calculating Multiplication Costs".
 *
 * Shows the size-aware Burnside formula that every component uses — no
 * regime-specific shortcut required. Beneath the formula, renders one
 * live row per component with its computed M (multiplication orbit count).
 */
export default function MultiplicationCostCard({ components = [] }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">
        Calculating Multiplication Costs
      </div>
      <p className="mt-2 text-sm leading-6 text-foreground">
        Every component gets the same treatment. Size-aware Burnside counts
        the number of distinct multiplication representatives by averaging
        fixed-point contributions over the group. No regime dispatch; one
        formula covers trivial, all-visible, all-summed, and every mixed
        regime.
      </p>
      <div className="mt-3 rounded-md bg-gray-50 px-3 py-2">
        <Latex
          math={String.raw`M \;=\; \frac{1}{|G|} \sum_{g \in G} \prod_{c \,\in\, \mathrm{cycles}(g)} n_c`}
          display
        />
      </div>
      {components.length > 0 && (
        <div className="mt-3 space-y-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Live for this example
          </div>
          {components.map((comp, i) => {
            const count = comp.multiplicationCount
              ?? comp.multiplication?.count
              ?? null;
            return (
              <div
                key={`mult-${i}`}
                className="flex items-center gap-2 text-xs"
              >
                <CaseBadge
                  regimeId={comp.accumulation?.regimeId ?? comp.shape ?? comp.caseType}
                  caseType={comp.caseType}
                  size="xs"
                  variant="pill"
                />
                <span className="truncate text-gray-600">
                  {comp.labels?.join(', ') || '∅'}
                </span>
                <span className="ml-auto font-mono text-gray-900">
                  M = {count != null ? count.toLocaleString() : '—'}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
