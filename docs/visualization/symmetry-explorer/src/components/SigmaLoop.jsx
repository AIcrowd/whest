import { useState, useEffect, useRef, useCallback } from 'react';

const STAGE_LABELS = ['M', 'σ(M)', 'π(σ(M))'];

export default function SigmaLoop({ results, graph, matrixData, example }) {
  const { uVertices, freeLabels } = graph;
  const labels = matrixData.labels;
  const originalMatrix = matrixData.matrix;

  const [selectedIdx, setSelectedIdx] = useState(null);
  const [stage, setStage] = useState(0); // 0=M, 1=σ(M), 2=π(σ(M))
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(false);

  // Filter results
  const allPairs = results.filter(r => !r.skipped);

  // Auto-select first valid pair (only on example change)
  const resultsKey = results.length + ':' + results.filter(r => r.isValid).length;
  useEffect(() => {
    const firstValid = allPairs.findIndex(r => r.isValid);
    if (firstValid >= 0) setSelectedIdx(firstValid);
    else if (allPairs.length > 0) setSelectedIdx(0);
    else setSelectedIdx(null);
    setStage(0);
    setPlaying(false);
    playRef.current = false;
  }, [resultsKey]);

  const selected = selectedIdx !== null ? allPairs[selectedIdx] : null;

  // Compute the three matrix stages
  const stages = computeStages(selected, originalMatrix, labels, uVertices);

  // Playback
  const stepForward = useCallback(() => {
    setStage(s => {
      const max = selected?.isValid ? 2 : 1;
      return Math.min(s + 1, max);
    });
  }, [selected]);

  const reset = useCallback(() => {
    setStage(0);
    setPlaying(false);
    playRef.current = false;
  }, []);

  useEffect(() => {
    playRef.current = playing;
    if (!playing) return;
    const max = selected?.isValid ? 2 : 1;
    if (stage >= max) { setPlaying(false); return; }
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

      {/* Pair selector chips */}
      <div className="pair-selector">
        <span className="pair-selector-label">Select (σ, π) pair:</span>
        <div className="pair-chips">
          {allPairs.map((r, i) => (
            <button key={i}
              className={`pair-chip ${r.isValid ? 'pair-valid' : 'pair-invalid'} ${selectedIdx === i ? 'pair-active' : ''}`}
              onClick={() => handleSelectPair(i)}>
              <span className="pair-sigma">σ = {fmtSigma(r.sigma)}</span>
              {r.isValid && <span className="pair-pi">π = {fmtPi(r.pi)}</span>}
              {!r.isValid && <span className="pair-rejected">✗</span>}
            </button>
          ))}
        </div>
      </div>

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

          {/* Animated matrix */}
          {stages && (
            <div className="anim-matrix-container">
              <AnimatedMatrix
                stage={stages[stage]}
                stageIdx={stage}
                labels={labels}
                uVertices={uVertices}
                freeLabels={freeLabels}
                isValid={selected.isValid}
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
          {selected.isValid && selected.pi && stage >= 1 && (
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
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Animated Matrix ── */

/**
 * AnimatedMatrix — rows are positioned absolutely and animate their Y position
 * when σ shuffles them, creating a "moving cards" effect.
 *
 * Instead of rendering a table, we render a column header row + individual
 * row cards. Each card has a stable React key (the U-vertex index it represents)
 * and its `top` position is computed from the current stage's row ordering.
 */
function AnimatedMatrix({ stage, stageIdx, labels, uVertices, freeLabels, isValid }) {
  const { matrix, colOrder, rowPerm, movedRows, movedCols, colPerm } = stage;
  const numCols = colOrder.length;
  const ROW_H = 38;
  const HEADER_H = 32;
  const COL_W = 70; // px per column
  const LABEL_W = 90;
  const numRows = matrix.length;
  const containerH = HEADER_H + numRows * ROW_H + 8;
  const containerW = LABEL_W + numCols * COL_W;

  // Row positions: positionOf[uVertexIdx] = visualRow
  const positionOf = {};
  rowPerm.forEach((uIdx, visualRow) => { positionOf[uIdx] = visualRow; });

  const identity = Array.from({ length: numRows }, (_, i) => i);
  const movedUIndices = new Set();
  rowPerm.forEach((uIdx, k) => { if (uIdx !== k) movedUIndices.add(uIdx); });

  // Column positions: if colPerm exists, columns animate to new X positions
  // colPerm[origIdx] = targetIdx (where does orig column move to?)
  const colPositions = colOrder.map((_, ci) => {
    if (colPerm) return colPerm[ci]; // animated target
    return ci; // identity
  });

  return (
    <div className="anim-matrix-wrap">
      <div className="anim-matrix-label">{STAGE_LABELS[stageIdx]}</div>
      <div className="card-matrix" style={{ height: containerH, width: containerW }}>
        {/* Column headers — FIXED positions (labels are the coordinate system) */}
        {colOrder.map((lbl, ci) => {
          const isMoved = movedCols.has(ci);
          return (
            <div key={`hdr-${lbl}`}
              className={`card-col-header-abs ${isMoved ? 'col-moved' : ''} ${freeLabels.has(lbl) ? 'col-v' : 'col-w'}`}
              style={{
                transform: `translateX(${LABEL_W + ci * COL_W}px)`,
                width: COL_W,
                height: HEADER_H,
              }}>
              {lbl}
            </div>
          );
        })}

        {/* Row cards */}
        {identity.map(uIdx => {
          const visualRow = positionOf[uIdx];
          const u = uVertices[uIdx];
          const lblStr = [...u.labels].sort().join(',');
          const isMoved = movedUIndices.has(uIdx);
          const rowData = matrix[visualRow];

          return (
            <div key={uIdx}
              className={`card-row ${isMoved ? 'card-row-moved' : ''}`}
              style={{
                transform: `translateY(${HEADER_H + visualRow * ROW_H}px)`,
                height: ROW_H,
              }}>
              <div className="card-row-label" style={{ width: LABEL_W }}>
                <span className="op-tag">Op{u.opIdx}</span>·{lblStr}
              </div>
              {/* Cells — each positioned absolutely for column animation */}
              {rowData.map((val, ci) => {
                const xPos = colPositions[ci];
                const isCellMoved = movedCols.has(ci);
                return (
                  <div key={`${uIdx}-${ci}`}
                    className={`card-cell-abs ${val > 0 ? 'cell-active' : ''} ${isCellMoved ? 'cell-col-moved' : ''}`}
                    style={{
                      transform: `translateX(${LABEL_W + xPos * COL_W}px)`,
                      width: COL_W,
                      height: ROW_H,
                    }}>
                    {val}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
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

  const piSigmaMatrix = result.sigmaMatrix.map(row => {
    return labels.map(lbl => {
      const srcLbl = piInv[lbl] || lbl;
      return row[colIdxMap[srcLbl]];
    });
  });

  // colPerm: where does each column's DATA card slide to?
  // π maps label c → π(c). The data that was under column c slides to column π(c).
  // colPerm[origColIdx] = targetColIdx
  const colPerm = labels.map((lbl, i) => {
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

function fmtSigma(sigma) {
  // Label-level swaps (Source A/C: axis permutations within operands)
  if (sigma._labelSwap) {
    const ls = sigma._labelSwap;
    const entries = Object.entries(ls).filter(([k, v]) => k !== v);
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
        cur = ls[cur] ?? cur;
      }
      if (cycle.length > 1) cycles.push(cycle);
    }
    return cycles.map(c => '(' + c.join(' ') + ')').join('') || 'e';
  }
  // Operand-level swaps (Source B)
  const entries = Object.entries(sigma).filter(([k, v]) => Number(k) !== v);
  if (entries.length === 0) return 'e';
  const visited = new Set();
  const cycles = [];
  for (const [k] of entries) {
    const kn = Number(k);
    if (visited.has(kn)) continue;
    const cycle = [];
    let cur = kn;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = sigma[cur] ?? cur;
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
