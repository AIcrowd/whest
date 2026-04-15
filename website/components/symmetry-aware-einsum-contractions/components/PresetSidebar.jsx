import { useMemo } from 'react';
import { Badge } from '@/components/ui/badge';
import { cn } from '../lib/utils.js';
import { getPresetSummary } from '../lib/presetSelection.js';
import CaseBadge from './CaseBadge.jsx';
import ExplorerSidebarItem from './ExplorerSidebarItem.jsx';

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
      className="sticky top-20 hidden max-h-[calc(100vh-5rem)] w-[19rem] shrink-0 self-start overflow-y-auto md:block"
    >
      <div className="px-2 pb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-gray-400">
        Presets
      </div>

      <div className="space-y-1">
        <button
          type="button"
          className="block w-full text-left"
          onClick={onCustom}
        >
          <ExplorerSidebarItem
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
          >
            <span className="absolute inset-y-3 left-0.5 w-1 rounded-full bg-coral" />
            <code className="mt-1 block truncate pl-3 text-xs text-gray-500">Define your own contraction</code>
            <span className="mt-1 block pl-3 text-xs text-gray-400">
              Keep the current builder state and switch into custom mode.
            </span>
          </ExplorerSidebarItem>
        </button>

        {presetSummaries.map((summary, idx) => (
          <button
            key={summary.id}
            type="button"
            className="block w-full text-left"
            onClick={() => onSelect(idx)}
          >
            <ExplorerSidebarItem
              active={selectedPresetIdx === idx}
              title={summary.name}
              className={cn(
                'relative px-3.5 py-3',
                selectedPresetIdx === idx
                  ? 'bg-coral-light/50 ring-coral/30'
                  : 'hover:bg-gray-50',
              )}
            >
              <span
                className="absolute inset-y-3 left-0.5 w-1 rounded-full"
                style={{ backgroundColor: summary.color }}
              />
              <div className="flex items-center gap-2 pl-3">
                {summary.caseType && (
                  <CaseBadge caseType={summary.caseType} size="xs" variant="compact" interactive={false} />
                )}
                <Badge variant="outline" className="text-[10px] font-semibold uppercase tracking-[0.18em]">
                  {summary.expectedGroup}
                </Badge>
              </div>
              <code className="mt-1 block truncate pl-3 text-xs text-gray-500">{summary.formula}</code>
              <span className="mt-1 block pl-3 text-xs text-gray-400">{summary.expectedGroup}</span>
            </ExplorerSidebarItem>
          </button>
        ))}
      </div>
    </aside>
  );
}
