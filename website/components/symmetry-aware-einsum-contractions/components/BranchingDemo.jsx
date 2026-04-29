import { useMemo, useState } from 'react';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import InlineMathText from './InlineMathText.jsx';
import OrbitRepMatrix from './branchingViews/OrbitRepMatrix.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getActiveExplorerThemeId,
} from '../lib/explorerTheme.js';
import { notationColor } from '../lib/notationSystem.js';
import { restrictStabilizerToPositions } from '../engine/outputOrbit.js';

// Curated example used when the live preset has no branching to show.
// Same scenario the retired PartitionCountingExplainer used as its
// "Why branching happens" worked example.
//
// The orbit rows below are pre-computed for n=3, S_3 acting on {i,j,k},
// V = {i,j}, H = Stab_G(V)|_V = {(),(ij)} restricted to V.
// Total α for this scenario is 9 = 3·1 + 3·2 (verified against engine
// brute-force in tests/output-orbit.test.mjs).
//
// The orbitTuples + componentInfo + expressionInfo blocks let the
// matrix tooltip render the full expanded contribution form even when
// no live engine output is around to derive them from.
const CURATED_BRANCHING_EXAMPLE = {
  caption: String.raw`$R[i,j] = \sum_k T[i,j,k]$ with $T$ fully symmetric, $n = 3$`,
  // S_3 = {e, (ij), (ik), (jk), (ijk), (ikj)} on labels [i, j, k]
  // (positions 0, 1, 2).  Each permutation expressed in array form
  // perm[i] = where position i maps to.
  componentInfo: {
    labels: ['i', 'j', 'k'],
    vLabels: ['i', 'j'],
    visiblePositions: [0, 1],
    elements: [
      { arr: [0, 1, 2] }, // e
      { arr: [1, 0, 2] }, // (ij)
      { arr: [2, 1, 0] }, // (ik)
      { arr: [0, 2, 1] }, // (jk)
      { arr: [1, 2, 0] }, // (ijk)
      { arr: [2, 0, 1] }, // (ikj)
    ],
  },
  expressionInfo: {
    subscripts: ['ijk'],
    output: 'ij',
    operandNames: ['T'],
  },
  orbitRows: [
    {
      repTuple: { i: 0, j: 0, k: 0 },
      orbitTuples: [{ i: 0, j: 0, k: 0 }],
      outputs: [{ outTuple: { i: 0, j: 0 }, coeff: 1 }],
      orbitSize: 1,
    },
    {
      repTuple: { i: 1, j: 1, k: 1 },
      orbitTuples: [{ i: 1, j: 1, k: 1 }],
      outputs: [{ outTuple: { i: 1, j: 1 }, coeff: 1 }],
      orbitSize: 1,
    },
    {
      repTuple: { i: 2, j: 2, k: 2 },
      orbitTuples: [{ i: 2, j: 2, k: 2 }],
      outputs: [{ outTuple: { i: 2, j: 2 }, coeff: 1 }],
      orbitSize: 1,
    },
    {
      repTuple: { i: 0, j: 0, k: 1 },
      orbitTuples: [
        { i: 0, j: 0, k: 1 },
        { i: 1, j: 0, k: 0 },
        { i: 0, j: 1, k: 0 },
      ],
      outputs: [
        { outTuple: { i: 0, j: 0 }, coeff: 1 },
        { outTuple: { i: 0, j: 1 }, coeff: 2 },
      ],
      orbitSize: 3,
    },
    {
      repTuple: { i: 0, j: 0, k: 2 },
      orbitTuples: [
        { i: 0, j: 0, k: 2 },
        { i: 2, j: 0, k: 0 },
        { i: 0, j: 2, k: 0 },
      ],
      outputs: [
        { outTuple: { i: 0, j: 0 }, coeff: 1 },
        { outTuple: { i: 0, j: 2 }, coeff: 2 },
      ],
      orbitSize: 3,
    },
    {
      repTuple: { i: 1, j: 1, k: 2 },
      orbitTuples: [
        { i: 1, j: 1, k: 2 },
        { i: 2, j: 1, k: 1 },
        { i: 1, j: 2, k: 1 },
      ],
      outputs: [
        { outTuple: { i: 1, j: 1 }, coeff: 1 },
        { outTuple: { i: 1, j: 2 }, coeff: 2 },
      ],
      orbitSize: 3,
    },
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
  expressionInfo = null,
}) {
  const themeId = getActiveExplorerThemeId();
  const [useCurated, setUseCurated] = useState(false);

  // Derive H = Stab_G(V)|_V for the active component, and pull V labels.
  // Used by OrbitRepMatrix's tooltip to canonicalize each orbit member's
  // π_V projection so the expanded equation can show concrete tuples.
  const liveComponentInfo = useMemo(() => {
    const c = componentData?.components?.[0];
    if (!c) return null;
    const visiblePositions = c.va.map((l) => c.labels.indexOf(l));
    return {
      labels: c.labels,
      vLabels: c.va,
      visiblePositions,
      elements: c.elements,
      hElements: restrictStabilizerToPositions(c.elements ?? [], visiblePositions),
    };
  }, [componentData]);

  if (!componentData || !costModel) return null;

  // Cross-spotlight payload is emitted by OrbitRepMatrix on row hover —
  // BranchingDemo just forwards the onHover prop to the matrix.

  const liveOrbitRows = costModel.orbitRows ?? [];
  const liveBranches = liveOrbitRows.some((row) => (row.outputs?.length ?? 0) > 1);
  const sourceLabel = useCurated || (!liveBranches && liveOrbitRows.length > 0)
    ? 'curated'
    : 'live';
  const orbitRows = sourceLabel === 'curated' ? CURATED_BRANCHING_EXAMPLE.orbitRows : liveOrbitRows;
  const activeComponentInfo = sourceLabel === 'curated'
    ? {
        ...CURATED_BRANCHING_EXAMPLE.componentInfo,
        hElements: restrictStabilizerToPositions(
          CURATED_BRANCHING_EXAMPLE.componentInfo.elements,
          CURATED_BRANCHING_EXAMPLE.componentInfo.visiblePositions,
        ),
      }
    : liveComponentInfo;
  const activeExpressionInfo = sourceLabel === 'curated'
    ? CURATED_BRANCHING_EXAMPLE.expressionInfo
    : expressionInfo;
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

      {/* Orbit selection now lives inside the matrix (click any row to
          select). The curated-fallback toggle stays — it's the only way
          to see a branching example when the live preset is functional
          projection. */}
      {!liveBranches && liveOrbitRows.length > 0 && (
        <div className="mt-3 flex items-center gap-2 text-[11px]">
          <button
            type="button"
            data-action="toggle-curated"
            onClick={() => setUseCurated((v) => !v)}
            className="rounded-full border px-2.5 py-1 font-medium transition-colors"
            style={{
              background: explorerThemeTint(themeId, 'hero', 0.08),
              borderColor: explorerThemeTint(themeId, 'hero', 0.25),
              color: explorerThemeColor(themeId, 'heroMuted'),
            }}
          >
            {useCurated ? '← back to live preset' : 'see a branching example →'}
          </button>
          {useCurated && (
            <span style={{ color: explorerThemeColor(themeId, 'muted') }}>
              curated: <InlineMathText>{CURATED_BRANCHING_EXAMPLE.caption}</InlineMathText>
            </span>
          )}
        </div>
      )}

      <div className="mt-3">
        <OrbitRepMatrix
          orbitRows={orbitRows}
          selectedOrbitIdx={safeIdx}
          onSelectOrbit={onSelectOrbit}
          onHover={onHover}
          expressionInfo={activeExpressionInfo}
          componentInfo={activeComponentInfo}
        />
      </div>

      <div className="mt-3 font-mono text-[11px]" data-testid="branching-alpha-total" style={{ color: explorerThemeColor(themeId, 'muted') }}>
        across all {orbitRows.length} orbits: α = <strong style={{ color: notationColor('alpha_total') }}>{liveAlpha}</strong>
      </div>
    </div>
  );
}
