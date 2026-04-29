import { useState } from 'react';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import InlineMathText from './InlineMathText.jsx';
import OrbitRepMatrix from './branchingViews/OrbitRepMatrix.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor } from '../lib/notationSystem.js';

// Curated example used when the live preset has no branching to show.
// Same scenario the retired PartitionCountingExplainer used as its
// "Why branching happens" worked example.
const CURATED_BRANCHING_EXAMPLE = {
  caption: String.raw`$R[i,j] = \sum_k T[i,j,k]$ with $T$ fully symmetric, $n = 3$`,
  // The orbit rows below are pre-computed for n=3, S_3 acting on {i,j,k},
  // V = {i,j}, H = Stab_G(V)|_V = {(),(ij)} restricted to V.
  // Total α for this scenario is 9 = 3·1 + 3·2 (verified against engine
  // brute-force in tests/output-orbit.test.mjs).
  orbitRows: [
    { repTuple: { i: 0, j: 0, k: 0 }, outputs: [{ outTuple: { i: 0, j: 0 }, coeff: 1 }], orbitSize: 1 },
    { repTuple: { i: 1, j: 1, k: 1 }, outputs: [{ outTuple: { i: 1, j: 1 }, coeff: 1 }], orbitSize: 1 },
    { repTuple: { i: 2, j: 2, k: 2 }, outputs: [{ outTuple: { i: 2, j: 2 }, coeff: 1 }], orbitSize: 1 },
    { repTuple: { i: 0, j: 0, k: 1 }, outputs: [{ outTuple: { i: 0, j: 0 }, coeff: 1 }, { outTuple: { i: 0, j: 1 }, coeff: 1 }], orbitSize: 3 },
    { repTuple: { i: 0, j: 0, k: 2 }, outputs: [{ outTuple: { i: 0, j: 0 }, coeff: 1 }, { outTuple: { i: 0, j: 2 }, coeff: 1 }], orbitSize: 3 },
    { repTuple: { i: 1, j: 1, k: 2 }, outputs: [{ outTuple: { i: 1, j: 1 }, coeff: 1 }, { outTuple: { i: 1, j: 2 }, coeff: 1 }], orbitSize: 3 },
  ],
};

const TITLE = 'Branching Demo';
const DECK = 'Branching = one product orbit reaches multiple stored output representatives. Pick an orbit, see it.';

export default function BranchingDemo({
  componentData,
  costModel,
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,
}) {
  const themeId = getActiveExplorerThemeId();
  const [useCurated, setUseCurated] = useState(false);

  if (!componentData || !costModel) return null;

  // When the user hovers/focuses the orbit selector, emit a cross-spotlight
  // payload so the masthead halos matching letters and the classification
  // tree spotlights the leaves this preset routes through. Same contract
  // as LabelInteractionGraph and DecisionLadder: { labels, leafKeys }.
  function emitHoverPayload() {
    if (!onHover) return;
    const components = componentData.components ?? [];
    const labels = components.flatMap((c) => c.labels ?? []);
    const leafKeys = components
      .flatMap((c) => [c.shape, c.accumulation?.regimeId])
      .filter(Boolean);
    onHover({ labels, leafKeys });
  }
  function clearHoverPayload() {
    if (onHover) onHover(null);
  }

  const liveOrbitRows = costModel.orbitRows ?? [];
  const liveBranches = liveOrbitRows.some((row) => (row.outputs?.length ?? 0) > 1);
  const sourceLabel = useCurated || (!liveBranches && liveOrbitRows.length > 0)
    ? 'curated'
    : 'live';
  const orbitRows = sourceLabel === 'curated' ? CURATED_BRANCHING_EXAMPLE.orbitRows : liveOrbitRows;
  const safeIdx = orbitRows.length === 0
    ? -1
    : Math.min(Math.max(0, selectedOrbitIdx >= 0 ? selectedOrbitIdx : 0), orbitRows.length - 1);

  // Live α total — sum of (orbit, rep) pairs where projection lands. The
  // matrix's filled-cell count must equal this; it doubles as a sanity
  // check.
  const liveAlpha = orbitRows.reduce((acc, row) => acc + (row.outputs?.length ?? 0), 0);

  return (
    <div id="branching-demo" className="bg-white p-4 scroll-mt-24" data-source-label={sourceLabel}>
      <ExplorerSubsectionHeader anchorId="branching-demo" labelText="Branching demo">
        {TITLE}
      </ExplorerSubsectionHeader>
      <p className="explorer-support-prose mt-2">
        {DECK}
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px]" data-testid="branching-controls">
        <label className="flex items-center gap-2">
          <span className="font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>Orbit</span>
          {/* Native <select> styled to match design-system input rule:
              gray-50 fill, 1px gray-200 border, gray-50 hover; coral focus
              ring inherited from the page's :focus-visible style. Mono
              type (matches the rest of the cost line + chip font). */}
          <select
            data-action="select-orbit"
            className="rounded border px-2 py-1 font-mono text-[11px] transition-colors hover:bg-stone-100"
            style={{
              borderColor: explorerThemeColor(themeId, 'border'),
              color: explorerThemeColor(themeId, 'body'),
              background: explorerThemeColor(themeId, 'surfaceInset'),
            }}
            value={safeIdx >= 0 ? safeIdx : 0}
            onChange={(e) => onSelectOrbit(Number(e.target.value))}
            onFocus={emitHoverPayload}
            onBlur={clearHoverPayload}
            onMouseEnter={emitHoverPayload}
            onMouseLeave={clearHoverPayload}
            disabled={orbitRows.length === 0}
          >
            {orbitRows.length === 0 && (
              <option value={0}>no orbit data</option>
            )}
            {orbitRows.map((row, idx) => {
              const reach = row.outputs?.length ?? 0;
              return (
                <option key={idx} value={idx}>
                  Orbit {idx + 1} of {orbitRows.length} · size {row.orbitSize ?? '?'} · reaches {reach} {reach === 1 ? 'rep' : 'reps'}
                </option>
              );
            })}
          </select>
        </label>

        {!liveBranches && liveOrbitRows.length > 0 && (
          // chip--coral recipe from the design system: coral-light fill,
          // coral-tinted border, coral-hover text. Sits in the chip family.
          <button
            type="button"
            data-action="toggle-curated"
            onClick={() => setUseCurated((v) => !v)}
            className="ml-auto rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors"
            style={{
              background: explorerThemeTint(themeId, 'hero', 0.08),
              borderColor: explorerThemeTint(themeId, 'hero', 0.25),
              color: explorerThemeColor(themeId, 'heroMuted'),
            }}
          >
            {useCurated ? '← back to live preset' : 'see a branching example →'}
          </button>
        )}
      </div>

      {useCurated && (
        <div className="mt-2 text-[11px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
          curated example: <InlineMathText>{CURATED_BRANCHING_EXAMPLE.caption}</InlineMathText>
        </div>
      )}

      <div className="mt-3">
        <OrbitRepMatrix
          orbitRows={orbitRows}
          selectedOrbitIdx={safeIdx}
          onSelectOrbit={onSelectOrbit}
          onHover={onHover}
        />
      </div>

      <div className="mt-3 font-mono text-[11px]" data-testid="branching-alpha-total" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        across all {orbitRows.length} orbits: α = <strong style={{ color: notationColor('alpha_total') }}>{liveAlpha}</strong>
      </div>
    </div>
  );
}
