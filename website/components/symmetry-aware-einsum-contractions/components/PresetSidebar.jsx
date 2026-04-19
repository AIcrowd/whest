import { useMemo } from 'react';
import { cn } from '../lib/utils.js';
import { getPresetSummary } from '../lib/presetSelection.js';
import CaseBadge from './CaseBadge.jsx';
import ExplorerSidebarItem from './ExplorerSidebarItem.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';

const CUSTOM_IDX = -1;

export default function PresetSidebar({
  examples,
  selectedPresetIdx,
  onSelect,
  onCustom,
}) {
  const presetSummaries = useMemo(() => examples.map(getPresetSummary), [examples]);

  return (
    <aside
      aria-label="Preset examples"
      className="sticky top-20 hidden max-h-[calc(100vh-5rem)] w-[18rem] shrink-0 self-start overflow-y-auto md:block"
    >
      <div className="px-1 pb-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
        Presets
      </div>

      {/* Flat list container — one outer gray-200 border wraps the whole
          preset list; 1px gray-100 dividers separate siblings. Matches
          `.preset-list` in design-system/preview/components.html. */}
      <div className="divide-y divide-gray-100 overflow-hidden rounded-lg border border-gray-200 bg-white">
        <ExplorerSidebarItem
          as="button"
          active={selectedPresetIdx === CUSTOM_IDX}
          title="Custom"
          badge="Freeform"
          badgeClassName="bg-gray-100 text-gray-500"
          className={cn(
            'relative px-4 py-3 pl-5',
            selectedPresetIdx === CUSTOM_IDX
              ? 'bg-coral-light/50'
              : 'hover:bg-gray-50',
          )}
          onClick={onCustom}
        >
          {selectedPresetIdx === CUSTOM_IDX && (
            <span className="absolute inset-y-3 left-[2px] w-1 rounded-[2px] bg-coral" />
          )}
          <code className="mt-1 block truncate text-sm text-gray-500">Define your own contraction</code>
          <span className="mt-1 block text-sm text-gray-400">
            Keep the current builder state and switch into custom mode.
          </span>
        </ExplorerSidebarItem>

        {presetSummaries.map((summary, idx) => (
          <ExplorerSidebarItem
            key={summary.id}
            as="button"
            active={selectedPresetIdx === idx}
            title={summary.name}
            className={cn(
              'relative px-4 py-3 pl-5',
              selectedPresetIdx === idx
                ? 'bg-coral-light/50'
                : 'hover:bg-gray-50',
            )}
            onClick={() => onSelect(idx)}
          >
            {/* Active rail — always coral per reference template. Regime
                identity lives on CaseBadge below, not on the rail. */}
            {selectedPresetIdx === idx && (
              <span className="absolute inset-y-3 left-[2px] w-1 rounded-[2px] bg-coral" />
            )}
            <span className="flex flex-wrap items-center gap-2">
              {summary.regimeId && (
                <CaseBadge regimeId={summary.regimeId} size="sm" variant="pill" />
              )}
              <SymmetryBadge value={summary.expectedGroup} />
            </span>
            <code className="mt-1 block truncate text-sm text-gray-500">{summary.formula}</code>
            <span className="mt-1 block text-sm text-gray-400">{summary.description}</span>
          </ExplorerSidebarItem>
        ))}
      </div>
    </aside>
  );
}
