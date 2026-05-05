import { useEffect, useMemo, useRef, useState } from 'react';
import { getPresetSummary } from '../lib/presetSelection.js';
import CaseBadge from './CaseBadge.jsx';
import ExplorerSidebarItem from './ExplorerSidebarItem.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';

const CUSTOM_IDX = -1;

// V3.1 pedagogical lens preset IDs, in the order they appear in the sidebar.
const LENS_PRESET_IDS = [
  'cross-s2',          // opening branching microscope (default)
  'triple-outer',      // all-visible sanity check
  'triangle',          // certification microscope
  'cross-c3-partial',  // partition-counting spotlight
  'bilinear-trace',    // formal-symmetry boundary
];

// V3.1 §2 spec — tags per preset for search/filter and pill rendering.
// Tag vocabulary: branching, all-visible, formal, partition, certification.
export const PRESET_TAGS = {
  'cross-s2':         ['branching', 'partition'],
  'triple-outer':     ['all-visible'],
  'triangle':         ['certification', 'formal'],
  'cross-c3-partial': ['partition', 'branching'],
  'bilinear-trace':   ['formal', 'certification'],
};

// V3.1 §2 spec — pedagogical signature shown on hover/focus per preset.
export const PRESET_MINI_PREVIEW = {
  'cross-s2':         'mini O→Q row with two filled cells',
  'triple-outer':     'row count equals filled-cell count',
  'triangle':         'accepted/rejected sigma badges',
  'cross-c3-partial': 'equality-pattern chips',
  'bilinear-trace':   'formal-symmetry mismatch',
};

function tagsForPreset(summary) {
  return PRESET_TAGS[summary.id] ?? [];
}

function miniPreviewFor(summary) {
  return PRESET_MINI_PREVIEW[summary.id] ?? null;
}

function SidebarGroupHeading({ children }) {
  return (
    <div className="border-b border-gray-100 px-4 pb-1.5 pt-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
        {children}
      </div>
    </div>
  );
}

function TagPill({ tag }) {
  return (
    <span
      data-preset-tag={tag}
      className="rounded-full border border-gray-200 bg-gray-50 px-1.5 py-0 text-[10px] font-medium uppercase tracking-[0.1em] text-gray-500"
    >
      {tag}
    </span>
  );
}

export default function PresetSidebar({
  examples,
  selectedPresetIdx,
  pendingPresetIdx = null,
  isPreparing = false,
  onSelect,
  onCustom,
}) {
  const presetSummaries = useMemo(() => examples.map(getPresetSummary), [examples]);

  // V3.1 §2: search/filter input + state.
  const [filterQuery, setFilterQuery] = useState('');

  // V3.1 §2: track previous preset id so a "← Return to {prevName}" pill can
  // appear when the active preset changes (e.g. via a §2 narrative spotlight
  // link). We treat all preset switches as eligible "spotlight" navigations
  // for now. The previous *id* (string, stable across registry edits) is what
  // we remember, so we resolve it back to a name at render time.
  const prevPresetIdRef = useRef(null);
  const [previousPresetId, setPreviousPresetId] = useState(null);
  const lastSelectedIdxRef = useRef(selectedPresetIdx);

  useEffect(() => {
    if (selectedPresetIdx !== lastSelectedIdxRef.current) {
      const priorIdx = lastSelectedIdxRef.current;
      const priorSummary = priorIdx >= 0 ? presetSummaries[priorIdx] : null;
      if (priorSummary) {
        prevPresetIdRef.current = priorSummary.id;
        setPreviousPresetId(priorSummary.id);
      } else {
        prevPresetIdRef.current = null;
        setPreviousPresetId(null);
      }
      lastSelectedIdxRef.current = selectedPresetIdx;
    }
  }, [selectedPresetIdx, presetSummaries]);

  const previousPresetEntry = useMemo(() => {
    if (!previousPresetId) return null;
    const idx = presetSummaries.findIndex((s) => s.id === previousPresetId);
    if (idx < 0) return null;
    return { summary: presetSummaries[idx], idx };
  }, [previousPresetId, presetSummaries]);

  function matchesFilter(summary) {
    const q = filterQuery.trim().toLowerCase();
    if (!q) return true;
    if (summary.name?.toLowerCase().includes(q)) return true;
    if (summary.id?.toLowerCase().includes(q)) return true;
    const tags = tagsForPreset(summary);
    return tags.some((t) => t.toLowerCase().includes(q));
  }

  const lensEntries = useMemo(
    () => presetSummaries
      .map((summary, idx) => ({ summary, idx }))
      .filter(({ summary }) => LENS_PRESET_IDS.includes(summary.id))
      .filter(({ summary }) => matchesFilter(summary))
      .sort((a, b) => LENS_PRESET_IDS.indexOf(a.summary.id) - LENS_PRESET_IDS.indexOf(b.summary.id)),
    [presetSummaries, filterQuery],
  );

  const referenceEntries = useMemo(
    () => presetSummaries
      .map((summary, idx) => ({ summary, idx }))
      .filter(({ summary }) => !LENS_PRESET_IDS.includes(summary.id))
      .filter(({ summary }) => matchesFilter(summary)),
    [presetSummaries, filterQuery],
  );

  function handleReturn() {
    if (!previousPresetEntry) return;
    onSelect(previousPresetEntry.idx);
  }

  function renderPresetItem({ summary, idx }) {
    const tags = tagsForPreset(summary);
    const previewText = miniPreviewFor(summary);
    const isActive = selectedPresetIdx === idx;
    return (
      <div key={summary.id} className="relative">
        <ExplorerSidebarItem
          as="button"
          active={isActive}
          title={summary.name}
          glyph={summary.glyph}
          description={summary.description}
          formula={summary.formula}
          className={isActive ? 'bg-coral-light/50' : 'hover:bg-gray-50'}
          onClick={() => onSelect(idx)}
          data-preset-id={summary.id}
          aria-busy={isPreparing && pendingPresetIdx === idx ? 'true' : undefined}
          data-mini-preview-text={previewText ?? undefined}
        >
          {isActive && (
            <span className="absolute inset-y-3 left-[2px] w-1 rounded-[2px] bg-coral" />
          )}
          {summary.caseIds?.map((caseId) => (
            <CaseBadge key={caseId} regimeId={caseId} size="xs" variant="pill" />
          ))}
          <SymmetryBadge
            value={summary.expectedGroup}
            className="h-5 rounded-full border-gray-200 bg-white px-2 py-0 text-[10px] font-semibold text-gray-600"
          />
          {tags.map((tag) => (
            <TagPill key={tag} tag={tag} />
          ))}
        </ExplorerSidebarItem>
      </div>
    );
  }

  return (
    <aside
      aria-label="Preset examples"
      className="sticky top-20 hidden max-h-[calc(100vh-5rem)] w-[18rem] shrink-0 self-start overflow-y-auto border-b border-gray-100 md:block xl:w-[20rem]"
    >
      <div className="overflow-hidden border-x border-gray-200 bg-white">
        <div className="border-b border-gray-100 px-4 py-4">
          <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
            Presets
          </div>
          <div className="mt-1 text-sm font-semibold text-gray-900">
            {examples.length} worked contractions
          </div>
          <div className="mt-3">
            <input
              type="search"
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              placeholder="Search presets or tags…"
              aria-label="Search presets by name or tag"
              data-preset-search="true"
              className="w-full rounded-md border border-gray-200 bg-white px-2 py-1 text-[12px] text-gray-700 placeholder:text-gray-400 focus:border-coral focus:outline-none focus:ring-1 focus:ring-coral/40"
            />
          </div>
        </div>

        {previousPresetEntry ? (
          <div className="border-b border-gray-100 px-4 py-2">
            <button
              type="button"
              data-preset-return="true"
              onClick={handleReturn}
              className="inline-flex items-center gap-1 rounded-full border border-gray-200 bg-gray-50 px-2 py-0.5 text-[11px] font-medium text-gray-600 hover:bg-gray-100"
            >
              <span aria-hidden="true">←</span>
              <span>Return to {previousPresetEntry.summary.name}</span>
            </button>
          </div>
        ) : null}

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
            aria-busy={isPreparing && pendingPresetIdx === CUSTOM_IDX ? 'true' : undefined}
          >
            {selectedPresetIdx === CUSTOM_IDX && (
              <span className="absolute inset-y-3 left-[2px] w-1 rounded-[2px] bg-coral" />
            )}
            <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-gray-500">
              Freeform
            </span>
          </ExplorerSidebarItem>
        </div>

        <div>
          <SidebarGroupHeading>Pedagogical lenses</SidebarGroupHeading>
          <div className="divide-y divide-gray-100">
            {lensEntries.map(renderPresetItem)}
          </div>
        </div>

        <div>
          <SidebarGroupHeading>Reference presets</SidebarGroupHeading>
          <div className="divide-y divide-gray-100">
            {referenceEntries.map(renderPresetItem)}
          </div>
        </div>
      </div>
    </aside>
  );
}
