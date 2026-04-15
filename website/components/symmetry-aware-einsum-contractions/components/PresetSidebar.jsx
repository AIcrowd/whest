import { useMemo } from 'react';
import { cn } from '../lib/utils.js';
import { getPresetSummary } from '../lib/presetSelection.js';
import CaseBadge from './CaseBadge.jsx';

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
          className={cn(
            'group flex w-full items-start gap-3 rounded-xl px-3.5 py-3 text-left transition-colors',
            selectedPresetIdx === CUSTOM_IDX
              ? 'bg-coral-light/50 ring-1 ring-coral/30'
              : 'hover:bg-gray-50',
          )}
          onClick={onCustom}
        >
          <span className="mt-0.5 h-full min-h-10 w-1 shrink-0 rounded-full bg-coral" />
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-2">
              <span className="truncate text-sm font-medium text-gray-900">Custom</span>
              <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-500">
                Freeform
              </span>
            </span>
            <code className="mt-1 block truncate text-[11px] text-gray-500">Define your own contraction</code>
            <span className="mt-1 block text-[11px] text-gray-400">
              Keep the current builder state and switch into custom mode.
            </span>
          </span>
        </button>

        {presetSummaries.map((summary, idx) => (
          <button
            key={summary.id}
            type="button"
            className={cn(
              'group flex w-full items-start gap-3 rounded-xl px-3.5 py-3 text-left transition-colors',
              selectedPresetIdx === idx
                ? 'bg-coral-light/50 ring-1 ring-coral/30'
                : 'hover:bg-gray-50',
            )}
            onClick={() => onSelect(idx)}
          >
            <span
              className="mt-0.5 h-full min-h-10 w-1 shrink-0 rounded-full"
              style={{ backgroundColor: summary.color }}
            />
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-gray-900">{summary.name}</span>
                {summary.caseType && (
                  <CaseBadge caseType={summary.caseType} size="xs" variant="compact" interactive={false} />
                )}
              </span>
              <code className="mt-1 block truncate text-xs text-gray-500">{summary.formula}</code>
              <span className="mt-1 block text-xs text-gray-400">{summary.expectedGroup}</span>
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}
