import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { COMPONENT_COLORS as COMP_COLORS } from '../engine/componentPalette.js';
import { explorerThemeColor } from '../lib/explorerTheme.js';
import { getRegimePresentation } from './regimePresentation.js';
import { getActiveExplorerThemeId, notationColor, notationText } from '../lib/notationSystem.js';
import InlineMathText from './InlineMathText.jsx';
import RoleBadge from './RoleBadge.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';

const GRAPH_SIZE = 220;
const CENTER = GRAPH_SIZE / 2;
const ORBIT_R = 80;
const NODE_R = 14;
function compLeafId(comp) {
  return comp?.accumulation?.regimeId ?? comp?.shape ?? null;
}

function circlePos(i, total, radius) {
  const angle = (2 * Math.PI * i) / total - Math.PI / 2;
  return {
    x: CENTER + radius * Math.cos(angle),
    y: CENTER + radius * Math.sin(angle),
  };
}

function dedupEdges(edges) {
  const seen = new Set();
  return edges.filter(([a, b]) => {
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// Viewport-aware tooltip positioning. Tooltip lives in a React portal attached
// to document.body so it escapes any ancestor transforms (the PanZoomCanvas
// wrapping this SVG creates one).
function computeGraphTooltipPos(rect) {
  const tooltipW = 288;
  const tooltipH = 200;
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  let x = rect.left + rect.width / 2;
  x = Math.max(tooltipW / 2 + 8, Math.min(x, vw - tooltipW / 2 - 16));
  const roomAbove = rect.top;
  const roomBelow = vh - rect.bottom;
  if (roomAbove >= tooltipH + 12) return { x, y: rect.top - 8, flipped: false };
  if (roomBelow >= tooltipH + 12) return { x, y: rect.bottom + 8, flipped: true };
  const flipped = roomBelow > roomAbove;
  return { x, y: flipped ? rect.bottom + 8 : rect.top - 8, flipped };
}

export function LabelInteractionGraph({
  allLabels = [],
  vLabels = [],
  interactionGraph = {},
  components: richComponents = null,
  fullGenerators = null,
  onHover = null,
}) {
  const explorerThemeId = getActiveExplorerThemeId();
  const COLOR_V = notationColor('v_free');
  const COLOR_W = notationColor('w_summed');
  const EDGE_COLOR = explorerThemeColor(explorerThemeId, 'muted');
  const NODE_BORDER_COLOR = explorerThemeColor(explorerThemeId, 'surface');
  const n = allLabels.length;

  const vSet = useMemo(() => new Set(vLabels), [vLabels]);
  const { edges = [], components: graphComponents = [] } = interactionGraph;
  const uniqueEdges = useMemo(() => dedupEdges(edges), [edges]);
  const positions = useMemo(
    () => allLabels.map((_, idx) => circlePos(idx, allLabels.length, ORBIT_R)),
    [allLabels],
  );

  // Prefer the rich components (which carry a shape + regimeId, and so a
  // stable color shared with the DecisionLadder and CaseBadge). Fall back to
  // the raw index-array components from the interaction graph if they weren't
  // threaded through — in that case hulls use the rotating palette.
  const hullData = useMemo(
    () =>
      (richComponents ?? graphComponents).map((entry, compIdx) => {
        if (Array.isArray(entry)) {
          return {
            indices: entry,
            color: COMP_COLORS[compIdx % COMP_COLORS.length],
            comp: null,
          };
        }
        const indices = entry?.indices ?? [];
        const leafId = compLeafId(entry);
        const presentation = leafId ? getRegimePresentation(leafId) : null;
        return {
          indices,
          color: presentation?.color ?? COMP_COLORS[compIdx % COMP_COLORS.length],
          comp: entry,
        };
      }),
    [richComponents, graphComponents],
  );

  // Label index -> rich component (used inside node tooltips).
  const labelToComp = useMemo(() => {
    const map = new Array(n).fill(null);
    hullData.forEach((hull) => {
      if (!hull.comp) return;
      for (const idx of hull.indices) map[idx] = hull.comp;
    });
    return map;
  }, [hullData, n]);

  // ─── Tooltip state ────────────────────────────────────────────────
  const [hovered, setHovered] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const hideTimerRef = useRef(null);
  const hoveredKeyRef = useRef(null);
  const wrapRef = useRef(null);

  const cancelHide = useCallback(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
  }, []);

  // Emit a structured cross-highlight payload whenever the hover target
  // changes. Consumers (App → StickyBar / DecisionLadder) use this to halo
  // matching einsum letters in the top bar and spotlight matching ladder
  // leaves.
  const buildHoverPayload = useCallback(
    (target) => {
      if (!target) return null;
      if (target.kind === 'node') {
        const label = allLabels[target.idx];
        return label ? { labels: [label], leafKeys: [] } : null;
      }
      if (target.kind === 'edge') {
        const edge = uniqueEdges[target.idx];
        if (!edge) return null;
        const [a, b] = edge;
        const labs = [allLabels[a], allLabels[b]].filter(Boolean);
        return { labels: labs, leafKeys: [] };
      }
      if (target.kind === 'hull') {
        const hull = hullData[target.idx];
        if (!hull?.comp) return null;
        const leafKeys = [
          hull.comp.shape,
          hull.comp.accumulation?.regimeId,
        ].filter(Boolean);
        return { labels: hull.comp.labels ?? [], leafKeys };
      }
      return null;
    },
    [allLabels, uniqueEdges, hullData],
  );

  const hideTooltip = useCallback(() => {
    cancelHide();
    hideTimerRef.current = setTimeout(() => {
      hoveredKeyRef.current = null;
      setHovered(null);
      if (onHover) onHover(null);
    }, 80);
  }, [cancelHide, onHover]);

  const openTooltip = useCallback(
    (target, rect) => {
      if (!rect) return;
      const key = `${target.kind}:${target.idx}`;
      if (hoveredKeyRef.current === key) {
        cancelHide();
        return;
      }
      cancelHide();
      hoveredKeyRef.current = key;
      setHovered(target);
      setTooltipPos(computeGraphTooltipPos(rect));
      if (onHover) onHover(buildHoverPayload(target));
    },
    [cancelHide, onHover, buildHoverPayload],
  );

  useEffect(() => () => cancelHide(), [cancelHide]);

  // Defensive dismissal — causes where synthetic pointerleave never fires:
  // scroll, resize, Escape, pointerdown outside the SVG (includes pan gesture
  // start), blur.
  useEffect(() => {
    if (!hovered) return undefined;
    const dismiss = () => {
      cancelHide();
      hoveredKeyRef.current = null;
      setHovered(null);
      if (onHover) onHover(null);
    };
    const dismissOnEscape = (event) => {
      if (event.key === 'Escape') dismiss();
    };
    const dismissIfOutside = (event) => {
      const target = event.target;
      if (!(target instanceof Node)) return dismiss();
      if (wrapRef.current && wrapRef.current.contains(target)) return;
      dismiss();
    };

    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    window.addEventListener('keydown', dismissOnEscape);
    window.addEventListener('pointerdown', dismissIfOutside);
    window.addEventListener('blur', dismiss);

    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('keydown', dismissOnEscape);
      window.removeEventListener('pointerdown', dismissIfOutside);
      window.removeEventListener('blur', dismiss);
    };
  }, [hovered, cancelHide, onHover]);

  const tooltipContent = useMemo(() => {
    if (!hovered) return null;

    if (hovered.kind === 'node') {
      const label = allLabels[hovered.idx];
      if (label === undefined) return null;
      const isV = vSet.has(label);
      const comp = labelToComp[hovered.idx];
      const leafId = compLeafId(comp);
      const presentation = leafId ? getRegimePresentation(leafId) : null;
      return (
        <>
          <div className="mb-1 flex items-center gap-2">
            <span
              className="inline-block size-2.5 rounded-full"
              style={{ backgroundColor: isV ? COLOR_V : COLOR_W }}
            />
            <span className="text-xs font-bold uppercase tracking-wide">Label</span>
            <span className="font-mono text-sm text-stone-900">{label}</span>
          </div>
          <div className="text-[11px] text-stone-700">
            {isV
              ? `${notationText('v_free')} — appears in the output`
              : `${notationText('w_summed')} — contracted away`}
          </div>
          {comp && (
            <div className="mt-2 text-[11px] text-stone-700">
              <span className="text-stone-500">Component:</span>{' '}
              <span className="font-mono text-stone-900">{`{${comp.labels.join(', ')}}`}</span>
            </div>
          )}
          {presentation && (
            <div className="mt-1 text-[11px] text-stone-700">
              <span className="font-semibold text-stone-900">{presentation.label}</span>
              {presentation.tooltip?.body ? (
                <span className="text-stone-500"> — <InlineMathText>{presentation.tooltip.body}</InlineMathText></span>
              ) : null}
            </div>
          )}
        </>
      );
    }

    if (hovered.kind === 'edge') {
      const edge = uniqueEdges[hovered.idx];
      if (!edge) return null;
      const [a, b, genIdx] = edge;
      const gen = Number.isInteger(genIdx) ? fullGenerators?.[genIdx] : null;
      let cycle = null;
      try {
        cycle = gen?.cycleNotation?.(allLabels) ?? null;
      } catch {
        cycle = null;
      }
      return (
        <>
          <div className="mb-1 text-xs font-bold uppercase tracking-wide">
            {Number.isInteger(genIdx) ? `Generator σ${genIdx + 1}` : 'Generator'}
          </div>
          {cycle ? (
            <div className="font-mono text-[11px] text-stone-900">{cycle}</div>
          ) : null}
          <div className="mt-2 text-[11px] text-stone-700">
            Labels{' '}
            <span className="font-mono text-stone-900">{allLabels[a]}</span> and{' '}
            <span className="font-mono text-stone-900">{allLabels[b]}</span> are
            moved together by this generator of <span className="font-mono text-stone-900">G</span>.
          </div>
        </>
      );
    }

    if (hovered.kind === 'hull') {
      const hull = hullData[hovered.idx];
      const comp = hull?.comp;
      if (!comp) return null;
      const leafId = compLeafId(comp);
      const presentation = leafId ? getRegimePresentation(leafId) : null;
      return (
        <>
          <div className="mb-1 text-xs font-bold uppercase tracking-wide">
            {presentation?.label ?? 'Component'}
          </div>
          {presentation?.tooltip?.body ? (
            <div className="text-[11px] text-stone-700"><InlineMathText>{presentation.tooltip.body}</InlineMathText></div>
          ) : null}
          <div className="mt-2 space-y-1.5 text-[11px] text-stone-700">
            <span className="text-stone-500">Labels:</span>
            <div className="flex flex-wrap items-center gap-1.5">
              {(comp.labels?.length ? comp.labels : ['∅']).map((label) => {
                if (label === '∅') {
                  return (
                    <span key={`empty-${hovered.idx}`} className="font-mono text-stone-500">
                      ∅
                    </span>
                  );
                }
                const role = (comp.va ?? []).includes(label) ? 'v' : 'w';
                return (
                  <RoleBadge key={`tooltip-${hovered.idx}-${label}`} role={role}>
                    {label}
                  </RoleBadge>
                );
              })}
            </div>
          </div>
          <div className="mt-1 space-y-1.5 text-[11px] text-stone-700">
            <span className="text-stone-500">Symmetry:</span>
            <div className="flex items-center">
              <SymmetryBadge value={comp.groupName || 'trivial'} />
            </div>
          </div>
        </>
      );
    }

    return null;
  }, [hovered, allLabels, vSet, labelToComp, uniqueEdges, fullGenerators, hullData]);

  if (n === 0) return null;

  return (
    <>
      <svg
        ref={wrapRef}
        className="w-full max-w-[220px]"
        viewBox={`0 0 ${GRAPH_SIZE} ${GRAPH_SIZE}`}
        aria-label="Label interaction graph"
      >
        {hullData.map((hull, compIdx) => {
          if (hull.indices.length <= 1) return null;
          const points = hull.indices.map((idx) => positions[idx]);
          return (
            <polygon
              key={`comp-${compIdx}`}
              points={points.map((point) => `${point.x},${point.y}`).join(' ')}
              fill={hull.color}
              fillOpacity={0.08}
              stroke={hull.color}
              strokeDasharray="4 3"
              strokeOpacity={0.55}
              style={{ cursor: 'help' }}
              onMouseEnter={(e) =>
                openTooltip({ kind: 'hull', idx: compIdx }, e.currentTarget.getBoundingClientRect())
              }
              onMouseLeave={hideTooltip}
            />
          );
        })}

        {uniqueEdges.map(([a, b], edgeIdx) => {
          const pa = positions[a];
          const pb = positions[b];
          if (!pa || !pb) return null;
          return (
            <g key={`edge-${edgeIdx}`}>
              {/* Transparent wide hit area — 1 px edges are un-hoverable. */}
              <line
                x1={pa.x}
                y1={pa.y}
                x2={pb.x}
                y2={pb.y}
                stroke="transparent"
                strokeWidth={10}
                style={{ cursor: 'help' }}
                onMouseEnter={(e) =>
                  openTooltip(
                    { kind: 'edge', idx: edgeIdx },
                    e.currentTarget.getBoundingClientRect(),
                  )
                }
                onMouseLeave={hideTooltip}
              />
              <line
                x1={pa.x}
                y1={pa.y}
                x2={pb.x}
                y2={pb.y}
                stroke={EDGE_COLOR}
                strokeWidth={1}
                strokeOpacity={0.45}
                pointerEvents="none"
              />
            </g>
          );
        })}

        {allLabels.map((label, idx) => {
          const { x, y } = positions[idx];
          const isV = vSet.has(label);
          return (
            <g
              key={`node-${label}`}
              style={{ cursor: 'help' }}
              onMouseEnter={(e) =>
                openTooltip({ kind: 'node', idx }, e.currentTarget.getBoundingClientRect())
              }
              onMouseLeave={hideTooltip}
            >
              <circle cx={x} cy={y} r={NODE_R} fill={isV ? COLOR_V : COLOR_W} stroke={NODE_BORDER_COLOR} strokeWidth={2} />
              <text
                x={x}
                y={y}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={12}
                fontFamily="ui-monospace, monospace"
                fontWeight={600}
                fill={NODE_BORDER_COLOR}
                pointerEvents="none"
              >
                {label}
              </text>
            </g>
          );
        })}
      </svg>

      {hovered && typeof document !== 'undefined'
        ? createPortal(
            <div
              className="pointer-events-none fixed z-[9999] w-72 rounded-lg border border-stone-200 bg-white px-3.5 py-3 text-stone-900 shadow-[0_20px_48px_rgba(15,23,42,0.16)]"
              style={{
                left: tooltipPos.x,
                top: tooltipPos.y,
                transform: tooltipPos.flipped
                  ? 'translateX(-50%)'
                  : 'translateX(-50%) translateY(-100%)',
              }}
              role="tooltip"
            >
              {tooltipContent}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
