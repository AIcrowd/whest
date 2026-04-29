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
      {/* Zone 1 — Header: eyebrow + close + tuples + branching caption */}
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
        {/* Branching caption — small, italic, gray. Replaces the bottom paragraph. */}
        <div
          data-testid="orbit-detail-branching-caption"
          className="mt-2 text-[11px] italic font-serif text-gray-500 leading-snug"
        >
          {branchCount > 1 ? (
            <>This orbit fills <strong className="not-italic font-semibold text-gray-700">{branchCount} cells in its row</strong> — α for this orbit alone is {branchCount}.</>
          ) : filled ? (
            <><strong className="not-italic font-semibold text-gray-700">{contributing.length}</strong> of {row.orbitSize} member{row.orbitSize === 1 ? '' : 's'} project{contributing.length === 1 ? 's' : ''} to this Q. +1 to α.</>
          ) : (
            <>No member projects to this Q. The cell is empty.</>
          )}
        </div>
      </div>

      {/* Zone 2 — Projection: π_V projection sketch */}
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

      {/* Zone 3 — Equation + this-Q ledger */}
      {expressionInfo && canonicalEquationLatex(expressionInfo) && (
        <div className="mt-3 pt-2 border-t" style={{ borderColor: COLOR.divider }}>
          <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mb-1.5">einsum equation</div>
          <div className="pl-2">
            <Latex math={canonicalEquationLatex(expressionInfo)} display />
          </div>
        </div>
      )}

      {filled && expressionInfo && contributing.length > 0 && (
        <div
          data-testid="orbit-detail-this-q-ledger"
          className="mt-3 pt-2 border-t"
          style={{ borderColor: COLOR.divider }}
        >
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

      {/* Zone 4 — Other-reached summary (1-line, first 3 destinations + N more) */}
      {otherReached.length > 0 && (
        <div
          data-testid="orbit-detail-other-reached-summary"
          className="mt-3 pt-2 border-t font-mono text-[11px] leading-snug text-gray-500"
          style={{ borderColor: COLOR.divider }}
        >
          <span className="text-[9px] uppercase tracking-[0.18em] text-gray-400 font-sans mr-2">
            also reaches
          </span>
          {otherReached.slice(0, 3).map((entry, ei) => (
            <span key={ei} className="mr-2 last:mr-0">
              R[{vLabels.map((l) => entry.outTuple[l]).join(', ')}]{ei < Math.min(otherReached.length, 3) - 1 ? ',' : ''}
            </span>
          ))}
          {otherReached.length > 3 && (
            <span className="text-gray-400">+ {otherReached.length - 3} more</span>
          )}
        </div>
      )}
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
