import { useState, useEffect, useRef } from 'react';
import { buildUVertexLabels } from '../engine/uVertexLabel.js';
import IncidenceMatrix from './IncidenceMatrix.jsx';

const STAGE_LABELS = ['M', 'σ(M)', 'π(σ(M))'];
const VISIBLE_VALID_PAIR_LIMIT = 5;

export default function SigmaLoop({ results, graph, matrixData, example, variableColors, group, onSelectedPairChange }) {
  const allPairs = results.filter((result) => !result.skipped);
  const validPairs = allPairs.filter((result) => result.isValid);
  const rejectedPairs = allPairs.filter((result) => !result.isValid);
  const resultsKey = results
    .map((result, index) => {
      const sigmaSignature = JSON.stringify(result.sigmaRowPerm ?? null);
      const piSignature = JSON.stringify(result.pi ?? null);
      return [
        index,
        result.skipped ? 1 : 0,
        result.isValid ? 1 : 0,
        result.piKind ?? '',
        result.reason ?? '',
        sigmaSignature,
        piSignature,
      ].join(':');
    })
    .join('|');

  return (
    <SigmaLoopInner
      key={resultsKey}
      allPairs={allPairs}
      validPairs={validPairs}
      rejectedPairs={rejectedPairs}
      graph={graph}
      matrixData={matrixData}
      example={example}
      variableColors={variableColors}
      results={results}
      group={group}
      onSelectedPairChange={onSelectedPairChange}
    />
  );
}

function SigmaLoopInner({ allPairs, validPairs, rejectedPairs, graph, matrixData, example, variableColors, results, group, onSelectedPairChange }) {
  const { uVertices, freeLabels } = graph;
  const uLabels = buildUVertexLabels(uVertices, example);
  const labels = matrixData.labels;
  const originalMatrix = matrixData.matrix;

  const initialSelectedIdx = validPairs.length > 0
    ? allPairs.indexOf(validPairs[0])
    : allPairs.length > 0
      ? 0
      : null;

  const [selectedIdx, setSelectedIdx] = useState(initialSelectedIdx);
  const [stage, setStage] = useState(0); // 0=M, 1=σ(M), 2=π(σ(M))
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(false);
  const [showRejected, setShowRejected] = useState(false);
  const [showMoreValid, setShowMoreValid] = useState(false);
  const [modalIdx, setModalIdx] = useState(null); // index into allPairs for rejected modal

  const selected = selectedIdx !== null ? allPairs[selectedIdx] : null;
  const inlineValidPairs = validPairs.slice(0, VISIBLE_VALID_PAIR_LIMIT);
  const remainingValidPairs = validPairs.slice(VISIBLE_VALID_PAIR_LIMIT);
  const provenance = group?.generatorSelection;
  const selectedPermutationKey = selected?.pi ? permutationKeyFromPi(selected.pi, group?.allLabels || labels) : null;
  const selectedCandidate = provenance?.candidatePermutations?.find((candidate) => candidate.permutationKey === selectedPermutationKey) ?? null;

  const closeAllModals = () => {
    setShowRejected(false);
    setShowMoreValid(false);
    setModalIdx(null);
  };

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        closeAllModals();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [showRejected, showMoreValid, modalIdx]);

  // Compute the three matrix stages
  const stages = computeStages(selected, originalMatrix, labels, uVertices);

  // Playback
  function stepForward() {
    setStage(s => {
      const max = selected?.isValid ? 2 : 1;
      return Math.min(s + 1, max);
    });
  }

  function reset() {
    setStage(0);
    setPlaying(false);
    playRef.current = false;
  }

  useEffect(() => {
    playRef.current = playing;
    if (!playing) return;
    const max = selected?.isValid ? 2 : 1;
    if (stage >= max) return;
    const timer = setTimeout(() => {
      if (!playRef.current) return;
      setStage(s => {
        const next = s + 1;
        if (next >= max) setPlaying(false);
        return Math.min(next, max);
      });
    }, 900);
    return () => clearTimeout(timer);
  }, [playing, stage, selected]);

  useEffect(() => {
    onSelectedPairChange?.(selectedIdx);
  }, [selectedIdx, onSelectedPairChange]);

  function handleSelectPair(idx) {
    setSelectedIdx(idx);
    // Auto-advance to σ(M) so the user immediately sees the effect
    setStage(1);
    setPlaying(false);
  }

  function handlePlay() {
    if (stage >= (selected?.isValid ? 2 : 1)) setStage(0);
    setPlaying(true);
  }

  return (
    <div className="sigma-loop">
      {/* Summary stats */}
      <div className="sigma-summary">
        <div className="sigma-stat">
          <span className="stat-num">{results.length}</span>
          <span className="stat-label">total σ's</span>
        </div>
        <div className="sigma-stat">
          <span className="stat-num">{results.filter(r => r.skipped).length}</span>
          <span className="stat-label">identity (skipped)</span>
        </div>
        <div className="sigma-stat accepted">
          <span className="stat-num">{allPairs.filter(r => r.isValid).length}</span>
          <span className="stat-label">valid π found</span>
        </div>
        <div className="sigma-stat rejected">
          <span className="stat-num">{allPairs.filter(r => !r.isValid).length}</span>
          <span className="stat-label">rejected</span>
        </div>
      </div>

      {/* Valid pairs — shown inline */}
      {validPairs.length > 0 && (
        <div className="pair-selector">
          <span className="pair-selector-label">Valid (σ, π) pairs:</span>
          <div className="pair-chips">
            {inlineValidPairs.map((r) => {
              const origIdx = allPairs.indexOf(r);
              return (
                <button key={origIdx}
                  className="pair-chip pair-valid"
                  onClick={() => handleSelectPair(origIdx)}>
                  <span className="pair-sigma">σ = {fmtSigma(r.sigmaRowPerm, uLabels)}</span>
                  <span className="pair-pi">π = {fmtPi(r.pi)}</span>
                  <span className={`pair-kind pair-kind-${r.piKind || 'identity'}`}>
                    {kindLabel(r.piKind)}
                  </span>
                </button>
              );
            })}

            {remainingValidPairs.length > 0 && (
              <button
                className="valid-toggle"
                onClick={() => setShowMoreValid(true)}
              >
                ▸ {remainingValidPairs.length} more (σ, π) pairs
              </button>
            )}
          </div>
        </div>
      )}

      {/* Rejected pairs — opens modal list */}
      {rejectedPairs.length > 0 && (
        <div className="rejected-section">
          <button
            className="rejected-toggle"
            onClick={() => setShowRejected(true)}>
            ▸ {rejectedPairs.length} rejected σ{rejectedPairs.length !== 1 ? "'s" : ''}
          </button>
        </div>
      )}

      {/* Animation panel */}
      {selected && (
        <div className="anim-panel">
          {/* Stage indicator */}
          <div className="stage-track">
            {STAGE_LABELS.map((label, i) => {
              const reachable = selected.isValid || i <= 1;
              return (
                <div key={i} className="stage-step-wrapper">
                  {i > 0 && (
                    <div className={`stage-connector ${stage >= i ? 'stage-connector-done' : ''} ${!reachable ? 'stage-connector-disabled' : ''}`}>
                      <span className="stage-connector-label">{i === 1 ? 'apply σ' : 'apply π'}</span>
                    </div>
                  )}
                  <button
                    className={`stage-dot ${stage === i ? 'stage-dot-active' : ''} ${stage > i ? 'stage-dot-done' : ''} ${!reachable ? 'stage-dot-disabled' : ''}`}
                    onClick={() => reachable && setStage(i)}
                    disabled={!reachable}>
                    {label}
                  </button>
                </div>
              );
            })}
          </div>

          {/* Playback controls */}
          <div className="stage-controls">
            <button className="ctrl-btn" onClick={reset} title="Reset">
              ↺ Reset
            </button>
            <button className="ctrl-btn ctrl-play" onClick={handlePlay}
              disabled={playing || stage >= (selected.isValid ? 2 : 1)}>
              ▶ Play
            </button>
            <button className="ctrl-btn" onClick={stepForward}
              disabled={stage >= (selected.isValid ? 2 : 1)}>
              Step →
            </button>
          </div>

          {/* Animated matrix — uses shared IncidenceMatrix */}
          {stages && (
            <div className="anim-matrix-container">
              <IncidenceMatrix
                matrix={stages[stage].matrix}
                colLabels={stages[stage].colOrder}
                uVertices={uVertices}
                example={example}
                freeLabels={freeLabels}
                variableColors={variableColors}
                rowPerm={stages[stage].rowPerm}
                colPerm={stages[stage].colPerm}
                movedRows={stages[stage].movedRows}
                movedCols={stages[stage].movedCols}
                animate={true}
                label={STAGE_LABELS[stage]}
              />
              {stage === 2 && selected.isValid && (
                <div className="recovery-badge">= M ✓</div>
              )}
              {stage >= 1 && !selected.isValid && (
                <div className="recovery-badge-below recovery-fail">
                  π not found — {selected.reason}
                </div>
              )}
            </div>
          )}

          {/* π mapping detail (when at stage 1 or 2 and valid) */}
          {selected.isValid && selected.pi && (
            <>
              <div className="pi-detail-panel">
                <h5>π mapping</h5>
                <div className="pi-arrows">
                  {Object.entries(selected.pi).map(([from, to]) => (
                    <div key={from} className={`pi-arrow ${from !== to ? 'moved' : 'fixed'}`}>
                      <span>{from}</span>
                      <span className="arrow">→</span>
                      <span>{to}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-4 space-y-3 text-sm">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Ordered active labels</div>
                    <code className="mt-1 block font-mono text-foreground">{`[${labels.join(', ')}]`}</code>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    This π induces candidate permutation {selectedCandidate ? selectedCandidate.cycleNotation : '—'} on the ordered active labels. The right panel tests whether adding it enlarges the generated subgroup.
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Modal for rejected σ — list or detail view */}
      {(showRejected || modalIdx !== null) && (
        <RejectedModal
          rejectedPairs={rejectedPairs}
          allPairs={allPairs}
          detailIdx={modalIdx}
          labels={labels}
          uVertices={uVertices}
          example={example}
          freeLabels={freeLabels}
          uLabels={uLabels}
          originalMatrix={originalMatrix}
          variableColors={variableColors}
          onSelectDetail={(origIdx) => setModalIdx(origIdx)}
          onBack={() => setModalIdx(null)}
          onClose={closeAllModals}
          fmtSigma={fmtSigma}
        />
      )}

      {showMoreValid && (
        <ValidPairsModal
          pairs={remainingValidPairs}
          allPairs={allPairs}
          uLabels={uLabels}
          onSelectPair={(origIdx) => {
            handleSelectPair(origIdx);
            setShowMoreValid(false);
          }}
          onClose={closeAllModals}
          fmtSigma={fmtSigma}
          fmtPi={fmtPi}
          kindLabel={kindLabel}
        />
      )}
    </div>
  );
}

function kindLabel(kind) {
  if (kind === 'cross') return 'cross V/W';
  if (kind === 'v-only') return 'V only';
  if (kind === 'w-only') return 'W only';
  if (kind === 'correlated') return 'correlated';
  return 'identity';
}

function kindCaption(kind) {
  if (kind === 'cross') {
    return 'This symmetry keeps one evaluation representative but can still update multiple output bins because it crosses the V/W boundary.';
  }
  if (kind === 'w-only') {
    return 'This symmetry preserves the output bin and only compresses the summed side, so it can reduce both evaluation and reduction work.';
  }
  if (kind === 'v-only') {
    return 'This symmetry acts only on output labels, so it shrinks the representative set but can still scatter into multiple output bins.';
  }
  if (kind === 'correlated') {
    return 'This symmetry acts on both V and W without crossing between them, so the full group still matters even though the V/W roles stay separate.';
  }
  return 'Identity mappings leave the tuple structure unchanged.';
}

/* ── Rejected Modal — list view + detail view ── */

function RejectedModal({
  rejectedPairs, allPairs, detailIdx,
  labels, uVertices, example, freeLabels, uLabels, originalMatrix, variableColors,
  onSelectDetail, onBack, onClose, fmtSigma,
}) {
  const pair = detailIdx !== null ? allPairs[detailIdx] : null;

  return (
    <div className="rejected-modal-overlay" onClick={onClose}>
      <div className="rejected-modal" onClick={e => e.stopPropagation()}>
        <div className="rejected-modal-header">
          <h4>{pair ? 'Rejected σ — Detail' : `${rejectedPairs.length} Rejected σ's`}</h4>
          <button className="rejected-modal-close" onClick={onClose}>✕</button>
        </div>

        {/* List view */}
        {!pair && (
          <div className="rejected-list">
            {rejectedPairs.map((r) => {
              const origIdx = allPairs.indexOf(r);
              return (
                <button key={origIdx} className="rejected-list-item"
                  onClick={() => onSelectDetail(origIdx)}>
                  <span className="rejected-list-sigma">
                    σ = {fmtSigma(r.sigmaRowPerm, uLabels)}
                  </span>
                  <span className="rejected-list-reason">
                    {r.reason || 'no valid π'}
                  </span>
                  <span className="rejected-list-arrow">→</span>
                </button>
              );
            })}
          </div>
        )}

        {/* Detail view — shows M → σ(M) using shared IncidenceMatrix */}
        {pair && (
          <RejectedDetail
            pair={pair}
            labels={labels}
            uVertices={uVertices}
            example={example}
            freeLabels={freeLabels}
            uLabels={uLabels}
            originalMatrix={originalMatrix}
            variableColors={variableColors}
            onBack={onBack}
            fmtSigma={fmtSigma}
          />
        )}
      </div>
    </div>
  );
}

function ValidPairsModal({
  pairs,
  allPairs,
  uLabels,
  onSelectPair,
  onClose,
  fmtSigma,
  fmtPi,
  kindLabel,
}) {
  return (
    <div className="rejected-modal-overlay" onClick={onClose}>
      <div className="rejected-modal" onClick={(event) => event.stopPropagation()}>
        <div className="rejected-modal-header">
          <h4>{`${pairs.length} additional valid σ${pairs.length === 1 ? '' : "'s"}`}</h4>
          <button className="rejected-modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="rejected-list">
          {pairs.map((r) => {
            const origIdx = allPairs.indexOf(r);
            return (
              <button key={origIdx} className="rejected-list-item" onClick={() => onSelectPair(origIdx)}>
                <span className="rejected-list-sigma">
                  σ = {fmtSigma(r.sigmaRowPerm, uLabels)}
                </span>
                <span className="pair-pi">
                  π = {fmtPi(r.pi)}
                </span>
                <span className={`pair-kind pair-kind-${r.piKind || 'identity'}`}>
                  {kindLabel(r.piKind)}
                </span>
                <span className="rejected-list-arrow">→</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ── Rejected Detail — M → σ(M) with animation, no π step ── */

function RejectedDetail({ pair, labels, uVertices, example, freeLabels, uLabels, originalMatrix, variableColors, onBack, fmtSigma }) {
  const [showSigma, setShowSigma] = useState(false);

  // Build moved rows set
  const movedRows = new Set();
  if (pair.sigmaRowPerm) {
    pair.sigmaRowPerm.forEach((uIdx, k) => {
      if (uIdx !== k) movedRows.add(k);
    });
  }

  return (
    <>
      <button className="rejected-back" onClick={onBack}>
        ← Back to list
      </button>
      <div className="rejected-modal-sigma">
        σ = {fmtSigma(pair.sigmaRowPerm, uLabels)}
      </div>
      <p className="rejected-modal-reason">{pair.reason || 'No valid π found'}</p>

      <div className="rejected-detail-controls">
        <button
          className={`ctrl-btn ${!showSigma ? 'ctrl-btn-active' : ''}`}
          onClick={() => setShowSigma(false)}>
          M
        </button>
        <span className="rejected-detail-arrow">→</span>
        <button
          className={`ctrl-btn ${showSigma ? 'ctrl-btn-active' : ''}`}
          onClick={() => setShowSigma(true)}>
          σ(M)
        </button>
      </div>

      <IncidenceMatrix
        matrix={showSigma ? pair.sigmaMatrix : originalMatrix}
        colLabels={labels}
        uVertices={uVertices}
        example={example}
        freeLabels={freeLabels}
        variableColors={variableColors}
        rowPerm={showSigma ? pair.sigmaRowPerm : null}
        movedRows={showSigma ? movedRows : null}
        animate={true}
        label={showSigma ? 'σ(M)' : 'M'}
        compact={true}
      />

      {/* Fingerprints are now shown by IncidenceMatrix automatically */}
    </>
  );
}

/* ── Compute stage data ── */

function computeStages(result, originalMatrix, labels, uVertices) {
  if (!result) return null;

  const identity = Array.from({ length: uVertices.length }, (_, i) => i);

  // Stage 0: Original M
  const stage0 = {
    matrix: originalMatrix,
    colOrder: [...labels],
    rowPerm: identity,
    movedRows: new Set(),
    movedCols: new Set(),
  };

  if (!result.sigmaRowPerm) {
    return [stage0, stage0, stage0];
  }

  // Stage 1: σ(M) — rows shuffled, columns same
  const movedRows1 = new Set();
  result.sigmaRowPerm.forEach((uIdx, k) => {
    if (uIdx !== k) movedRows1.add(k);
  });
  const stage1 = {
    matrix: result.sigmaMatrix,
    colOrder: [...labels],
    rowPerm: result.sigmaRowPerm,
    movedRows: movedRows1,
    movedCols: new Set(),
  };

  if (!result.isValid || !result.pi) {
    return [stage0, stage1, stage1];
  }

  // Stage 2: π(σ(M)) = M
  // Labels are the fixed coordinate system — they NEVER move.
  // σ moved row data (stage 1). π now rearranges column data within the
  // same fixed grid. Rows stay in their σ positions. The cell values
  // change to match M, proving π(σ(M)) = M.
  //
  // Concretely: π relabels columns, so column data at position `c` in σ(M)
  // gets read as being at position `π(c)`. The resulting matrix equals M.
  // We show this by keeping rows in σ order but displaying M's data
  // (read through the permuted row/column mapping).

  const movedCols2 = new Set();
  labels.forEach((lbl, i) => {
    if (result.pi[lbl] !== lbl) movedCols2.add(i);
  });

  // Build the π(σ(M)) matrix: same row order as σ(M), but column data
  // rearranged by π. Each cell [row r, col c] = σ(M)[r, π⁻¹(c)]
  const colIdxMap = {};
  labels.forEach((lbl, i) => { colIdxMap[lbl] = i; });
  // π⁻¹: if π maps a→b, then π⁻¹ maps b→a
  const piInv = {};
  Object.entries(result.pi).forEach(([k, v]) => { piInv[v] = k; });

  // colPerm: where does each column's DATA card slide to?
  // π maps label c → π(c). The data that was under column c slides to column π(c).
  // colPerm[origColIdx] = targetColIdx
  const colPerm = labels.map((lbl) => {
    const target = result.pi[lbl] || lbl;
    return colIdxMap[target];
  });

  const stage2 = {
    matrix: result.sigmaMatrix, // keep σ(M) data — cells will visually slide
    colOrder: [...labels],
    rowPerm: result.sigmaRowPerm, // rows STAY in σ positions
    movedRows: new Set(),
    movedCols: movedCols2,
    colPerm, // drives the column slide animation
  };

  return [stage0, stage1, stage2];
}

/* ── Formatting helpers ── */

function fmtSigma(sigmaRowPerm, uLabels) {
  if (!sigmaRowPerm) return 'e';
  // Build cycle notation on U-vertex labels from the row permutation
  const n = sigmaRowPerm.length;
  const visited = new Set();
  const cycles = [];
  for (let i = 0; i < n; i++) {
    if (visited.has(i) || sigmaRowPerm[i] === i) continue;
    const cycle = [];
    let cur = i;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(uLabels?.[cur] ?? `r${cur}`);
      cur = sigmaRowPerm[cur];
    }
    if (cycle.length > 1) cycles.push(cycle);
  }
  return cycles.map(c => '(' + c.join(' ') + ')').join('') || 'e';
}

function fmtPi(pi) {
  if (!pi) return '—';
  const entries = Object.entries(pi).filter(([k, v]) => k !== v);
  if (entries.length === 0) return 'e';
  const visited = new Set();
  const cycles = [];
  for (const [k] of entries) {
    if (visited.has(k)) continue;
    const cycle = [];
    let cur = k;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = pi[cur];
    }
    if (cycle.length > 1) cycles.push(cycle);
  }
  return cycles.map(c => '(' + c.join(' ') + ')').join('') || 'e';
}

function permutationKeyFromPi(pi, orderedLabels) {
  if (!pi || !orderedLabels?.length) return null;
  const indexByLabel = new Map(orderedLabels.map((label, index) => [label, index]));
  const arr = orderedLabels.map((label) => indexByLabel.get(pi[label]));
  if (arr.some((value) => value === undefined)) return null;
  return arr.join(',');
}
