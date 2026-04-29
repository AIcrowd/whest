import { memo, useEffect, useLayoutEffect, useRef } from 'react';
import Latex from '../Latex.jsx';
import { labelledTuple, tupleKey } from './orbitRepMatrixLayout.js';
import { flipPosition } from './floatingPosition.js';

// Big floating detail card. Two modes:
//   - 'floating' (default): position:fixed near the hover point, flip on
//     viewport edges, dismiss on Esc + when the matrix scrolls offscreen
//   - 'inline':  no fixed positioning, no flip math, no dismiss handlers.
//     Used inside OrbitRepMatrixModal where the modal already owns the
//     viewport-sized container + Esc.

const COLOR = {
  bg: '#FFFFFF',
  border: '#D9DCDC',
  divider: '#ECEFEF',
  coral: '#F0524D',
  coralMuted: 'rgba(240,82,77,0.85)',
  coralLight: '#FEF2F1',
  empty: '#F8F9F9',
};

// Padding from the hover point to the card edge.
const FLOAT_PADDING = 12;
// Default card dimensions used for flip math (the card uses min/max-content
// in practice; we pass a conservative max here so the flip doesn't overflow).
const CARD_W = 440;
const CARD_H = 480;
// Maximum reps before we hide the mini row preview (it stops being readable past this).
const ROW_PREVIEW_MAX_REPS = 30;

// TODO(mobile): floating mode currently has no tap-outside dismiss. Esc +
// IntersectionObserver cover desktop and scroll-out cases; touch users
// dismiss via the × button only. Plan defers mobile gestures to a
// follow-up — see plan §"Open follow-ups".

function membersProjectingTo(orbit, repTuple, componentInfo) {
  if (!orbit?.orbitTuples || !componentInfo) return [];
  const { vLabels } = componentInfo;
  if (!vLabels) return [];
  return orbit.orbitTuples.filter((m) => vLabels.every((l) => m[l] === repTuple?.[l]));
}

function canonicalEquationLatex(expressionInfo) {
  if (!expressionInfo) return '';
  const { subscripts = [], output = '', operandNames = [] } = expressionInfo;
  if (!subscripts.length || !output) return '';
  const inOps = new Set();
  subscripts.forEach((s) => [...s].forEach((c) => inOps.add(c)));
  const summed = [...inOps].filter((c) => !output.includes(c)).sort();
  const opTerms = subscripts.map((s, i) => {
    const name = operandNames[i] ?? operandNames[0] ?? 'T';
    return `${name}[${[...s].join(',')}]`;
  });
  const sumPrefix = summed.length ? `\\sum_{${summed.join(',')}}\\,` : '';
  return `R[${[...output].join(',')}] \\;=\\; ${sumPrefix}${opTerms.join(' \\cdot ')}`;
}

function OrbitDetailCard({
  hover,                     // was `pin`. { row, col, clickX, clickY } | null — when null, render nothing
  orbitRows = [],
  reps = [],
  cells = [],
  expressionInfo = null,
  componentInfo = null,
  onDismiss = () => {},      // called on Esc, scroll-out, or × close
  matrixRef = null,          // ref to the matrix outer element — used by IntersectionObserver
  mode = 'floating',         // 'floating' | 'inline'
}) {
  const cardRef = useRef(null);
  // Capture onDismiss in a ref so the three effects below don't re-run when
  // callers pass an unstable arrow. Effects call `onDismissRef.current()` instead.
  const onDismissRef = useRef(onDismiss);
  useEffect(() => { onDismissRef.current = onDismiss; }, [onDismiss]);

  // `active` collapses hover from an object reference to a boolean, so the
  // two effects below only re-run on show/hide transitions (not on every
  // cell change within the hover session).
  const active = hover !== null;

  // Esc to dismiss (floating mode only — modal mode handles its own Esc).
  useEffect(() => {
    if (mode !== 'floating' || !active) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onDismissRef.current(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [mode, active]);

  // Auto-dismiss when matrix scrolls offscreen (floating mode only).
  useEffect(() => {
    if (mode !== 'floating' || !active || !matrixRef?.current) return undefined;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.intersectionRatio < 0.05) onDismissRef.current();
    }, { threshold: 0.05 });
    observer.observe(matrixRef.current);
    return () => observer.disconnect();
  }, [mode, active, matrixRef]);

  // Compute floating position when hover changes (and on window resize).
  // Writes el.style imperatively — no React state, no double render per hover.
  useLayoutEffect(() => {
    if (mode !== 'floating' || !hover) return undefined;
    let rafId = null;
    const compute = () => {
      rafId = null;
      const el = cardRef.current;
      if (!el) return;
      const next = flipPosition({
        clickX: hover.clickX ?? 0,
        clickY: hover.clickY ?? 0,
        cardW: CARD_W,
        cardH: CARD_H,
        viewportW: window.innerWidth,
        viewportH: window.innerHeight,
        padding: FLOAT_PADDING,
      });
      el.style.left = `${next.left}px`;
      el.style.top = `${next.top}px`;
    };
    const onResize = () => {
      if (rafId !== null) return;
      rafId = requestAnimationFrame(compute);
    };
    compute(); // initial position is synchronous
    window.addEventListener('resize', onResize);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      window.removeEventListener('resize', onResize);
    };
  }, [mode, hover]);

  if (!hover) return null;

  const row = orbitRows[hover.row];
  const rep = reps[hover.col];
  const coeff = cells[hover.row]?.[hover.col] ?? null;
  const filled = coeff !== null;

  if (!row || !rep) return null;

  const allLabels = expressionInfo?.subscripts
    ? [...new Set(expressionInfo.subscripts.flatMap((s) => [...s]))]
    : Object.keys(row.repTuple ?? {});
  const vLabels = componentInfo?.vLabels ?? Object.keys(rep.tuple ?? {});

  const contributing = membersProjectingTo(row, rep.tuple, componentInfo);
  const otherReached = (row.outputs ?? [])
    .filter((o) => tupleKey(o.outTuple) !== tupleKey(rep.tuple))
    .map((o) => ({ outTuple: o.outTuple, members: membersProjectingTo(row, o.outTuple, componentInfo) }));
  const branchCount = row.outputs?.length ?? 1;
  const operandName = expressionInfo?.operandNames?.[0] ?? 'T';

  const wrapperStyle = mode === 'floating'
    ? {
        position: 'fixed',
        left: 0,                    // initial; useLayoutEffect overwrites synchronously
        top: 0,
        width: CARD_W,
        maxHeight: `min(${CARD_H}px, calc(100vh - ${2 * FLOAT_PADDING}px))`,
        overflowY: 'auto',
        zIndex: 50,
        background: COLOR.bg,
        border: `1px solid ${COLOR.border}`,
        borderRadius: 8,
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
      }
    : {
        background: COLOR.bg,
      };

  return (
    <div
      ref={cardRef}
      data-testid="orbit-detail-card"
      data-mode={mode}
      style={wrapperStyle}
      className="p-4"
    >
      {/* Eyebrow + close button */}
      <div className="flex items-baseline gap-3">
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-900">
          Worked example · O → Q
        </div>
        <button
          type="button"
          data-action="clear-pin"
          onClick={onDismiss}
          className="ml-auto text-[10px] font-medium text-gray-600 hover:text-gray-900 transition-colors"
        >
          × close
        </button>
      </div>

      {/* Tuples header */}
      <div className="mt-2 font-mono text-[11px] text-gray-900 leading-7">
        <div>
          <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mr-2 font-sans">orbit O</span>
          <strong>{labelledTuple(row.repTuple)}</strong>
          <span className="ml-2 text-gray-500 font-sans">
            · {row.orbitSize ?? '?'} member{row.orbitSize === 1 ? '' : 's'}
            {branchCount > 1 ? ` · branches to ${branchCount} reps` : ''}
          </span>
        </div>
        <div>
          <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mr-2 font-sans">stored rep Q</span>
          <strong>{labelledTuple(rep.tuple)}</strong>
        </div>
      </div>

      {/* Mini row preview — only when reps.length is small enough to be readable. */}
      {reps.length <= ROW_PREVIEW_MAX_REPS && (
        <div data-testid="worked-example-row-preview" className="mt-3 rounded p-2.5" style={{ background: COLOR.empty }}>
          <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mb-2">
            orbit O's row in the O → Q matrix
          </div>
          <div className="flex gap-[2px] h-5">
            {reps.map((r, c) => {
              const coef = cells[hover.row]?.[c] ?? null;
              const isThis = c === hover.col;
              const isOther = !isThis && coef !== null;
              return (
                <div
                  key={c}
                  className="flex-1 rounded-sm"
                  style={{
                    background: isThis ? COLOR.coral : (isOther ? COLOR.coralLight : COLOR.bg),
                    border: isOther ? '1px solid rgba(240,82,77,0.45)' : `1px solid ${COLOR.border}`,
                  }}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* π_V projection sketch */}
      {(row.orbitTuples?.length ?? 0) > 0 && (
        <div data-testid="worked-example-projection" className="mt-3 rounded p-2.5 font-mono text-[10.5px] leading-7" style={{ background: COLOR.empty }}>
          <div className="grid grid-cols-[1fr_auto_1fr] gap-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 font-sans mb-1">
            <span>members of O</span>
            <span>π<sub>V</sub></span>
            <span>destination</span>
          </div>
          {row.orbitTuples.map((m, i) => {
            const memberV = {};
            for (const l of vLabels) memberV[l] = m[l];
            const matchesThis = vLabels.every((l) => m[l] === rep.tuple[l]);
            return (
              <div key={i} className="grid grid-cols-[1fr_auto_1fr] gap-2 items-center">
                <span>{labelledTuple(m)}</span>
                <span className="text-gray-400 text-center">→</span>
                <span style={{ color: matchesThis ? COLOR.coral : COLOR.coralMuted, fontWeight: matchesThis ? 600 : 500 }}>
                  R[{vLabels.map((l) => memberV[l]).join(', ')}] {matchesThis ? '← THIS Q ●' : '← OTHER Q ●'}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Einsum equation */}
      {expressionInfo && canonicalEquationLatex(expressionInfo) && (
        <div className="mt-3 pt-2 border-t" style={{ borderColor: COLOR.divider }}>
          <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mb-1.5">einsum equation</div>
          <div className="pl-2">
            <Latex math={canonicalEquationLatex(expressionInfo)} display />
          </div>
        </div>
      )}

      {/* This Q's contributions */}
      {filled && expressionInfo && contributing.length > 0 && (
        <div className="mt-3 pt-2 border-t" style={{ borderColor: COLOR.divider }}>
          <div className="text-[9px] font-semibold uppercase tracking-[0.16em] mb-1.5" style={{ color: COLOR.coral }}>
            contribution to R[{vLabels.map((l) => rep.tuple[l]).join(', ')}] · this Q
          </div>
          <div className="pl-2 font-mono text-[11px] leading-7 text-gray-900">
            {contributing.map((m, i) => (
              <div key={i}>
                If ({allLabels.join(', ')}) = ({allLabels.map((l) => m[l]).join(', ')}), then R[{vLabels.map((l) => m[l]).join(', ')}] += {operandName}[{allLabels.map((l) => m[l]).join(', ')}]
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Other Qs (dimmed) */}
      {otherReached.length > 0 && expressionInfo && (
        <div className="mt-3 pt-2 border-t" style={{ borderColor: COLOR.divider }}>
          <div className="italic text-[10px] font-semibold tracking-[0.16em] uppercase mb-1.5" style={{ color: COLOR.coralMuted }}>
            contribution to other Qs · the orbit's other reached bins
          </div>
          {otherReached.map((entry, ei) => (
            <div key={ei} className="pl-2 font-mono text-[11px] leading-7 text-gray-700 mt-2 first:mt-0">
              <div className="text-[10px] uppercase tracking-[0.16em] text-gray-400 mb-1 font-sans">
                R[{vLabels.map((l) => entry.outTuple[l]).join(', ')}]
              </div>
              {entry.members.map((m, i) => (
                <div key={i}>
                  If ({allLabels.join(', ')}) = ({allLabels.map((l) => m[l]).join(', ')}), then R[{vLabels.map((l) => m[l]).join(', ')}] += {operandName}[{allLabels.map((l) => m[l]).join(', ')}]
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* Branching note */}
      <div className="mt-3 pt-2 border-t font-serif text-[12.5px] italic leading-7 text-gray-700" style={{ borderColor: COLOR.divider }}>
        {branchCount > 1 ? (
          <>
            <strong className="not-italic text-gray-900">Branching:</strong> this single product orbit O fills <strong className="not-italic text-gray-900">{branchCount} cells in its row</strong>. α for this orbit alone is {branchCount} — one update per filled cell, regardless of member count.
          </>
        ) : filled ? (
          <>
            <strong className="not-italic text-gray-900">{contributing.length}</strong> of {row.orbitSize} member{row.orbitSize === 1 ? '' : 's'} of this orbit project{contributing.length === 1 ? 's' : ''} to this output bin. Each filled cell adds 1 to α.
          </>
        ) : (
          <>No member of this orbit projects to this Q. The cell stays empty and contributes nothing to α.</>
        )}
      </div>
    </div>
  );
}

// Memo by structural equality on hover (row/col/clickX/clickY) + reference equality on data refs.
function detailPropsEqual(prev, next) {
  const prevH = prev.hover;
  const nextH = next.hover;
  const sameHover = prevH === nextH
    || (prevH && nextH && prevH.row === nextH.row && prevH.col === nextH.col
        && prevH.clickX === nextH.clickX && prevH.clickY === nextH.clickY);
  if (!sameHover) return false;
  return (
    prev.orbitRows === next.orbitRows
    && prev.reps === next.reps
    && prev.cells === next.cells
    && prev.expressionInfo === next.expressionInfo
    && prev.componentInfo === next.componentInfo
    && prev.mode === next.mode
  );
}

export default memo(OrbitDetailCard, detailPropsEqual);
