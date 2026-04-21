import { useMemo } from 'react';
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
      className="sticky top-20 hidden max-h-[calc(100vh-5rem)] w-[18rem] shrink-0 self-start overflow-y-auto border-b border-gray-100 md:block xl:w-[20rem]"
    >
      <div className="overflow-hidden border-r border-gray-200 bg-white">
        <div className="border-b border-gray-100 px-4 py-4">
          <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
            Presets
          </div>
          <div className="mt-1 text-sm font-semibold text-gray-900">
            {examples.length} worked contractions
          </div>
        </div>

        <div className="divide-y divide-gray-100">
          <ExplorerSidebarItem
            as="button"
            active={selectedPresetIdx === CUSTOM_IDX}
            title="Custom"
            glyph="⚙"
            description="Keep the current builder state and switch into custom mode."
            formula="— build below —"
            className={selectedPresetIdx === CUSTOM_IDX ? 'bg-coral-light/50' : 'hover:bg-gray-50'}
            onClick={onCustom}
          >
            {selectedPresetIdx === CUSTOM_IDX && (
              <span className="absolute inset-y-3 left-[2px] w-1 rounded-[2px] bg-coral" />
            )}
            <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-gray-500">
              Freeform
            </span>
          </ExplorerSidebarItem>

          {presetSummaries.map((summary, idx) => (
            <ExplorerSidebarItem
              key={summary.id}
              as="button"
              active={selectedPresetIdx === idx}
              title={summary.name}
              glyph={summary.glyph}
              description={summary.description}
              formula={summary.formula}
              className={selectedPresetIdx === idx ? 'bg-coral-light/50' : 'hover:bg-gray-50'}
              onClick={() => onSelect(idx)}
            >
              {selectedPresetIdx === idx && (
                <span className="absolute inset-y-3 left-[2px] w-1 rounded-[2px] bg-coral" />
              )}
              {summary.caseIds?.map((caseId) => (
                <CaseBadge key={caseId} regimeId={caseId} size="xs" variant="pill" />
              ))}
              <SymmetryBadge
                value={summary.expectedGroup}
                className="h-5 rounded-full border-gray-200 bg-white px-2 py-0 text-[10px] font-semibold text-gray-600"
              />
            </ExplorerSidebarItem>
          ))}
        </div>
      </div>
    </aside>
  );
}
