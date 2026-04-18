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
      <div className="px-2 pb-3 text-xs font-semibold uppercase tracking-[0.18em] text-primary/75">
        Presets
      </div>

      <div className="space-y-1">
        <ExplorerSidebarItem
          as="button"
          active={selectedPresetIdx === CUSTOM_IDX}
          title="Custom"
          badge="Freeform"
          badgeClassName="bg-gray-100 text-gray-500"
          className={cn(
            'relative px-3.5 py-3',
            selectedPresetIdx === CUSTOM_IDX
              ? 'bg-coral-light/50 ring-coral/30'
              : 'hover:bg-gray-50',
          )}
          onClick={onCustom}
        >
            <span className="absolute inset-y-3 left-0.5 w-1 rounded-full bg-coral" />
          <code className="mt-1 block truncate pl-3 text-sm text-gray-500">Define your own contraction</code>
          <span className="mt-1 block pl-3 text-sm text-gray-400">
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
              'relative px-3.5 py-3',
              selectedPresetIdx === idx
                ? 'bg-coral-light/50 ring-coral/30'
                : 'hover:bg-gray-50',
            )}
            onClick={() => onSelect(idx)}
          >
            <span
              className="absolute inset-y-3 left-0.5 w-1 rounded-full"
              style={{ backgroundColor: summary.color }}
            />
            <span className="flex flex-wrap items-center gap-2 pl-3">
              {summary.regimeId && (
                <CaseBadge regimeId={summary.regimeId} size="sm" variant="pill" />
              )}
              <SymmetryBadge value={summary.expectedGroup} />
            </span>
            <code className="mt-1 block truncate pl-3 text-sm text-gray-500">{summary.formula}</code>
            <span className="mt-1 block pl-3 text-sm text-gray-400">{summary.description}</span>
          </ExplorerSidebarItem>
        ))}
      </div>
    </aside>
  );
}
