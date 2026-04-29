import { useState } from 'react';
import InlineMathText from './InlineMathText.jsx';
import FanView from './branchingViews/FanView.jsx';
import ArcsView from './branchingViews/ArcsView.jsx';
import GridsView from './branchingViews/GridsView.jsx';
import PileBucketsView from './branchingViews/PileBucketsView.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor, notationLatex } from '../lib/notationSystem.js';

// Curated example used when the live preset has no branching to show.
// Same scenario the retired PartitionCountingExplainer used as its
// "Why branching happens" worked example.
const CURATED_BRANCHING_EXAMPLE = {
  caption: '$R[i,j] = \sum_k T[i,j,k]$ with $T$ fully symmetric, $n = 3$',
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

const TITLE = 'When projection branches, watch one orbit fan out';
const DECK = 'Branching = one product orbit reaches multiple stored output representatives. Pick a view, pick an orbit, see it.';

const INTRO_PARAGRAPHS = [
  // Relocated from content/main/section4.js intro ¶3 (verbatim).
  'A product-orbit representative can contain many full index assignments; their projections to the visible labels may land in one stored output representative or several. $\\alpha$ counts one update per stored output representative reached. The subtlety is that projection is not always a function from product orbits to output representatives: sometimes one product orbit reaches several. That is exactly why accumulation counting is harder than multiplication counting.',
  // Relocated from PartitionCountingExplainer.jsx BODY ¶1 (verbatim).
  'A product orbit may contain many full assignments. When those assignments are projected to the output labels, they may reach one stored output representative or several. Enumerating every concrete assignment is correct but can be wasteful.',
  // Relocated from PartitionCountingExplainer.jsx BODY ¶2 (verbatim).
  'Counting product orbits alone is therefore not enough: a single product orbit can update multiple stored output representatives, so the accumulation count needs an extra reach factor on top of the orbit count.',
];

const VIEW_MODES = [
  { id: 'fan',          label: 'α Fan'          },
  { id: 'arcs',         label: 'β Arcs'         },
  { id: 'grids',        label: 'γ Grids'        },
  { id: 'pile-buckets', label: 'δ Pile→buckets' },
];

function tabStyle(themeId, active) {
  return {
    background: active ? explorerThemeTint(themeId, 'hero', 0.12) : 'transparent',
    color: active ? explorerThemeColor(themeId, 'hero') : explorerThemeColor(themeId, 'body'),
    border: `1px solid ${active ? explorerThemeColor(themeId, 'hero') : explorerThemeColor(themeId, 'border')}`,
  };
}

export default function BranchingDemo({
  componentData,
  costModel,
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
}) {
  const themeId = getActiveExplorerThemeId();
  const [activeView, setActiveView] = useState('fan');
  const [useCurated, setUseCurated] = useState(false);

  if (!componentData || !costModel) return null;

  const liveOrbitRows = costModel.orbitRows ?? [];
  const liveBranches = liveOrbitRows.some((row) => (row.outputs?.length ?? 0) > 1);
  const sourceLabel = useCurated || (!liveBranches && liveOrbitRows.length > 0)
    ? 'curated'
    : 'live';
  const orbitRows = sourceLabel === 'curated' ? CURATED_BRANCHING_EXAMPLE.orbitRows : liveOrbitRows;
  const safeIdx = orbitRows.length === 0
    ? -1
    : Math.min(Math.max(0, selectedOrbitIdx >= 0 ? selectedOrbitIdx : 0), orbitRows.length - 1);
  const activeRow = safeIdx >= 0 ? orbitRows[safeIdx] : null;

  // Synthesize a normalized "orbit" view-payload from the costModel row.
  // Members are anonymous placeholders — coeff[i] copies of a member that
  // maps to output rep i. The exact tuple values aren't visualised; the
  // shape (how many members → which Q) is what the views need.
  function makeOrbitPayload(row) {
    if (!row) return null;
    const members = [];
    (row.outputs ?? []).forEach((out, repIndex) => {
      for (let j = 0; j < (out.coeff ?? 1); j += 1) {
        members.push({ repIndex });
      }
    });
    return {
      size: row.orbitSize ?? members.length,
      members,
    };
  }
  const activeOrbit = makeOrbitPayload(activeRow);
  const reachedReps = (activeRow?.outputs ?? []).map((out) => ({ weight: out.coeff ?? 1 }));
  const liveAlpha = orbitRows.reduce((acc, row) => acc + (row.outputs?.length ?? 0), 0);

  function step(delta) {
    if (orbitRows.length === 0) return;
    const next = (safeIdx + delta + orbitRows.length) % orbitRows.length;
    onSelectOrbit(next);
  }

  return (
    <section
      id="branching-demo"
      className="mx-auto w-full max-w-[var(--prose-max)] rounded-xl border bg-white px-6 py-6 shadow-sm md:px-8 md:py-7 scroll-mt-24"
      style={{ borderColor: explorerThemeColor(themeId, 'border') }}
      aria-labelledby="branching-demo-title"
    >
      <div className="text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        Branching demo
      </div>
      <h3
        id="branching-demo-title"
        className="font-heading text-[20px] font-semibold leading-tight"
        style={{ color: explorerThemeColor(themeId, 'ink') }}
      >
        {TITLE}
      </h3>
      <p className="mt-2 max-w-[70ch] font-serif text-[15px] italic leading-7" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        {DECK}
      </p>

      <div className="mt-5 space-y-4 max-w-[78ch] font-serif text-[16px] leading-[1.75]" style={{ color: explorerThemeColor(themeId, 'body') }}>
        {INTRO_PARAGRAPHS.map((paragraph, idx) => (
          <p key={idx}>
            <InlineMathText>{paragraph}</InlineMathText>
          </p>
        ))}
      </div>

      <div className="mt-6 flex items-center gap-2 text-[12px] uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        <span>View</span>
        <div className="flex flex-wrap gap-1.5">
          <button type="button" data-view-id="fan"          onClick={() => setActiveView('fan')}          className="rounded px-2.5 py-1 text-[11px] font-semibold transition-colors" style={tabStyle(themeId, activeView === 'fan')}>α Fan</button>
          <button type="button" data-view-id="arcs"         onClick={() => setActiveView('arcs')}         className="rounded px-2.5 py-1 text-[11px] font-semibold transition-colors" style={tabStyle(themeId, activeView === 'arcs')}>β Arcs</button>
          <button type="button" data-view-id="grids"        onClick={() => setActiveView('grids')}        className="rounded px-2.5 py-1 text-[11px] font-semibold transition-colors" style={tabStyle(themeId, activeView === 'grids')}>γ Grids</button>
          <button type="button" data-view-id="pile-buckets" onClick={() => setActiveView('pile-buckets')} className="rounded px-2.5 py-1 text-[11px] font-semibold transition-colors" style={tabStyle(themeId, activeView === 'pile-buckets')}>δ Pile→buckets</button>
        </div>
      </div>

      {!liveBranches && liveOrbitRows.length > 0 && (
        <div className="mt-3 flex items-center gap-2 text-[11px]" style={{ color: explorerThemeColor(themeId, 'muted') }}>
          <span>this preset has functional projection — every orbit reaches one rep, so α = M.</span>
          <button
            type="button"
            data-action="toggle-curated"
            onClick={() => setUseCurated((v) => !v)}
            className="rounded border px-2 py-1 text-[11px] font-semibold"
            style={{ borderColor: explorerThemeColor(themeId, 'hero'), color: explorerThemeColor(themeId, 'hero') }}
          >
            {useCurated ? 'back to live preset' : 'see a branching example'}
          </button>
          {useCurated && <InlineMathText>{CURATED_BRANCHING_EXAMPLE.caption}</InlineMathText>}
        </div>
      )}

      <div className="mt-3 flex items-center gap-3 text-[12px]">
        <span className="font-semibold uppercase tracking-[0.12em]" style={{ color: explorerThemeColor(themeId, 'muted') }}>Orbit</span>
        <button type="button" data-action="prev-orbit" onClick={() => step(-1)} className="rounded border px-2 py-1 text-[11px]" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>◀ prev</button>
        <span className="font-mono">orbit {orbitRows.length === 0 ? '—' : safeIdx + 1} / {orbitRows.length}</span>
        <button type="button" data-action="next-orbit" onClick={() => step(1)} className="rounded border px-2 py-1 text-[11px]" style={{ borderColor: explorerThemeColor(themeId, 'border') }}>next ▶</button>
        <span className="ml-auto font-mono" style={{ color: notationColor('h_output') }}>
          reaches <strong>{reachedReps.length}</strong> stored output reps
        </span>
      </div>

      <div className="mt-4 rounded-md border p-4" style={{ borderColor: explorerThemeColor(themeId, 'border'), background: explorerThemeColor(themeId, 'surfaceInset') }}>
        {activeView === 'fan' && <FanView orbit={activeOrbit} reachedReps={reachedReps} />}
        {activeView === 'arcs' && <ArcsView orbit={activeOrbit} reachedReps={reachedReps} />}
        {activeView === 'grids' && <GridsView orbit={activeOrbit} allOrbits={orbitRows} reachedReps={reachedReps} hClasses={[]} />}
        {activeView === 'pile-buckets' && <PileBucketsView orbit={activeOrbit} reachedReps={reachedReps} />}
      </div>

      <div className="mt-3 font-mono text-[11px]" data-testid="branching-alpha-total" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        across all {orbitRows.length} orbits: α = <strong style={{ color: notationColor('alpha_total') }}>{liveAlpha}</strong>
      </div>
    </section>
  );
}
