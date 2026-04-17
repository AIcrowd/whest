import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Handle, Panel, Position, ReactFlow, ReactFlowProvider, useReactFlow } from '@xyflow/react';
import PanZoomControls, { PanZoomHint } from './PanZoomControls.jsx';
import '@xyflow/react/dist/style.css';
import Latex from './Latex.jsx';
import GlossaryList from './GlossaryList.jsx';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';
import { REGIME_SPEC } from '../engine/regimeSpec.js';

// ─── DecisionLadder (two-stage hybrid) ─────────────────────────────────
//
// Same node types and styling as the classic ladder, but the tree is split
// into two named bands:
//
//   STAGE 1 — Structural checks. Decisions here only need (V, W, generators).
//             No group enumeration (dimino) required to REACH a leaf.
//             Leaves: allVisible · allSummed · trivial · directProduct.
//
//   STAGE 2 — Symmetry checks. We materialise G and run the remaining
//             ladder: singleton vs bruteForceOrbit fallback.
//
// An explicit "enumerate G" divider between the two bands makes the
// computational cost boundary obvious: everything above the line is
// (almost) free; everything below pays the dimino cost.

const LEAF_W = 210;
const LEAF_H = 44;
const QUESTION_W = 200;
const QUESTION_H = 44;
const LEAF_X = 0;
const LEAF_QUESTION_GAP = 56;
const QUESTION_X = LEAF_W + LEAF_QUESTION_GAP;
const LEAF_CENTER_OFFSET = (QUESTION_W - LEAF_W) / 2;
const SOURCE_W = 140;
const SOURCE_H = 38;
const SOURCE_X = QUESTION_X + (QUESTION_W - SOURCE_W) / 2;
const ROW_GAP = 88;

const BAND_WIDTH = QUESTION_X + QUESTION_W + 80; // leaves + gap + questions + right pad
const BAND_X = -40;                                // left pad
const STAGE_1_TOP_Y = 0;
const SOURCE_Y = 48;
const Q1_Y = SOURCE_Y + 88;                        // q_hasW
const Q2_Y = Q1_Y + ROW_GAP;                       // q_hasV
const Q3_Y = Q2_Y + ROW_GAP;                       // q_trivial
const Q4_Y = Q3_Y + ROW_GAP;                       // q_direct
const STAGE_1_BOTTOM_Y = Q4_Y + LEAF_H + 44;

const ENUMERATE_Y = STAGE_1_BOTTOM_Y + 12;
const ENUMERATE_H = 44;
const STAGE_2_TOP_Y = ENUMERATE_Y + ENUMERATE_H + 32;
const Q5_Y = STAGE_2_TOP_Y + 32;                   // q_singleton
const STAGE_2_BOTTOM_Y = Q5_Y + ROW_GAP + LEAF_H + 32;

const EDGE_YES = { color: '#23B761', label: 'yes' };
const EDGE_NO = { color: '#F0524D', label: 'no' };

// ─── Decision spec — reordered to dimino-free questions first ─────────

const QUESTIONS = [
  {
    id: 'q_hasW',
    short: 'W ≠ ∅ ?',
    long: 'Are there summed (contracted) labels? Cheap: checks wa.length alone — no group enumeration needed.',
    onTrue: 'q_hasV', onFalse: 'allVisible',
    stage: 1,
  },
  {
    id: 'q_hasV',
    short: 'V ≠ ∅ ?',
    long: 'Are there free (output) labels? Cheap: checks va.length alone — no group enumeration needed.',
    onTrue: 'q_trivial', onFalse: 'allSummed',
    stage: 1,
  },
  {
    id: 'q_trivial',
    short: '|G| = 1 ?',
    long: 'Is the symmetry group trivial? Cheap: detected from generators without running dimino (every gen is the identity, or no generators at all).',
    onTrue: 'trivial', onFalse: 'q_direct',
    stage: 1,
  },
  {
    id: 'q_direct',
    short: 'All gens V-only or W-only ?',
    long: 'Every generator moves only V-labels or only W-labels — cheap syntactic check on the generator list; decides directProduct without enumerating G.',
    onTrue: 'directProduct', onFalse: 'ENUMERATE',
    stage: 1,
  },
  {
    id: 'q_singleton',
    short: '|V| = 1 ?',
    long: 'Exactly one free label — singleton weighted Burnside applies. From here down, we have G materialised.',
    onTrue: 'singleton', onFalse: 'bruteForceOrbit',
    stage: 2,
  },
];

// ─── Helpers ──────────────────────────────────────────────────────────

function mixWithWhite(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const mix = (c) => Math.round(c + (255 - c) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

function specFor(leafId) {
  return SHAPE_SPEC[leafId] || REGIME_SPEC[leafId] || null;
}

// ─── Node renderers ──────────────────────────────────────────────────

function QuestionNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center justify-center rounded-md border border-gray-200 bg-white px-3 py-1.5 text-center text-sm leading-tight text-gray-900 shadow-sm transition-colors hover:border-gray-400"
      data-tree-node={data.nodeId}
    >
      <Handle id="top" type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      {data.text}
      <Handle id="bottom" type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
      <Handle id="side" type="source" position={Position.Left} className="pointer-events-none opacity-0" />
    </div>
  );
}

function SourceNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center justify-center rounded-full border border-gray-200 bg-white px-3 py-1.5 text-center text-sm font-semibold leading-tight text-gray-900 shadow-sm"
      data-tree-node={data.nodeId}
    >
      {data.text}
      <Handle id="bottom" type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
    </div>
  );
}

function LeafNode({ data }) {
  const bg = data.active ? mixWithWhite(data.color, 0.72) : mixWithWhite(data.color, 0.88);
  const SPOTLIGHT_RING = '#F0524D';
  const shadow = data.spotlight
    ? `0 0 0 10px ${SPOTLIGHT_RING}33, 0 0 0 6px ${mixWithWhite(data.color, 0.6)}`
    : data.active
      ? `0 0 0 6px ${mixWithWhite(data.color, 0.65)}`
      : undefined;
  const borderColor = data.spotlight ? SPOTLIGHT_RING : data.color;
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center justify-center whitespace-nowrap rounded-lg px-2 py-0.5 text-center text-sm font-bold leading-tight shadow-sm transition-all hover:shadow"
      style={{
        backgroundColor: bg,
        borderColor,
        borderWidth: data.spotlight ? 3 : data.active ? 3 : 2,
        borderStyle: 'solid',
        color: '#0F172A',
        boxShadow: shadow,
      }}
      data-tree-node={data.nodeId}
      data-leaf-spotlight={data.spotlight ? 'true' : undefined}
    >
      <Handle id="right" type="target" position={Position.Right} className="pointer-events-none opacity-0" />
      <Handle id="top" type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      {data.text}
    </div>
  );
}

// Stage band — visual container highlighting the "no-dimino" / "needs-dimino"
// regions. Rendered as a normal ReactFlow node so pan/zoom stays in sync.
//
// Both the stage label and the caption live in the top row — label pinned
// left, caption pinned right — so neither collides with edge labels near the
// bottom of the band (where the "no" label crosses into the enumerate node).
function StageBandNode({ data }) {
  const isStage1 = data.stage === 1;
  const accent = isStage1 ? '#16A34A' : '#6D28D9';
  return (
    <div
      className="pointer-events-none box-border h-full w-full rounded-xl"
      style={{
        background: isStage1
          ? 'linear-gradient(180deg, rgba(34,197,94,0.07) 0%, rgba(34,197,94,0.03) 100%)'
          : 'linear-gradient(180deg, rgba(139,92,246,0.08) 0%, rgba(139,92,246,0.03) 100%)',
        border: `1.5px dashed ${accent}`,
      }}
    >
      <div
        className="absolute left-4 top-2 text-[10px] font-bold uppercase tracking-[0.12em]"
        style={{ color: accent }}
      >
        {data.label}
      </div>
      <div
        className="absolute right-4 top-2 text-[10px] font-medium italic"
        style={{ color: accent }}
      >
        {data.caption}
      </div>
    </div>
  );
}

// "Enumerate G" divider — the moment dimino is paid.
function EnumerateNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center gap-2 rounded-md border-2 border-dashed border-violet-400 bg-violet-50 px-3 py-1.5 text-center text-[13px] font-semibold leading-tight text-violet-700 shadow-sm"
      data-tree-node={data.nodeId}
    >
      <Handle id="top" type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <circle cx="12" cy="12" r="3" />
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" />
      </svg>
      <span>{data.text}</span>
      <Handle id="bottom" type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
    </div>
  );
}

const dlNodeTypes = {
  source: SourceNode,
  question: QuestionNode,
  leaf: LeafNode,
  stageBand: StageBandNode,
  enumerate: EnumerateNode,
};

// ─── Tooltip ──────────────────────────────────────────────────────────

function tooltipFor(nodeId) {
  if (nodeId === 's0') {
    return {
      title: 'Component',
      whenText: null,
      body: 'Start with one independent component of the detected group action. The two-stage ladder tries structural (dimino-free) checks first, then enumerates G only if nothing lands.',
      latex: null,
      color: '#64748B',
    };
  }
  if (nodeId === 'enumerate') {
    return {
      title: 'Enumerate G (dimino)',
      whenText: 'Paid once, only when structural checks all refuse',
      body: 'Materialise every group element via Dimino\'s algorithm. After this point the remaining ladder has access to the full group and can use Burnside / orbit enumeration.',
      latex: null,
      color: '#8B5CF6',
    };
  }
  const q = QUESTIONS.find((x) => x.id === nodeId);
  if (q) {
    return {
      title: q.short,
      whenText: q.stage === 1 ? 'Stage 1 — structural check (no dimino)' : 'Stage 2 — symmetry check (dimino done)',
      body: q.long,
      latex: null,
      color: q.stage === 1 ? '#16A34A' : '#8B5CF6',
    };
  }
  const spec = specFor(nodeId);
  if (spec) {
    return {
      title: spec.label,
      whenText: spec.when,
      body: spec.description,
      latex: spec.latex,
      glossary: spec.glossary,
      color: spec.color,
    };
  }
  return null;
}

// ─── Layout ──────────────────────────────────────────────────────────
//
// Both stages share the same left-leaf / right-spine structure as the classic
// ladder, so the spatial grammar is unchanged. What's new:
//   - a pair of dashed "Stage 1 / Stage 2" bands behind the rows.
//   - an "enumerate G" node sitting on the spine between the two stages.

function buildLadderLayout(activeLeafIds, spotlightLeafIds) {
  const active = activeLeafIds instanceof Set
    ? activeLeafIds
    : new Set(activeLeafIds || []);
  const spotlight = spotlightLeafIds instanceof Set
    ? spotlightLeafIds
    : new Set(spotlightLeafIds || []);

  const nodes = [];
  const edges = [];

  function leafNode(leafId, y, centered = false) {
    const spec = specFor(leafId);
    return {
      id: leafId,
      position: { x: centered ? QUESTION_X + LEAF_CENTER_OFFSET : LEAF_X, y },
      type: 'leaf',
      style: { width: LEAF_W, height: LEAF_H },
      data: {
        text: spec.label,
        color: spec.color,
        active: active.has(leafId),
        spotlight: spotlight.has(leafId),
        nodeId: leafId,
      },
    };
  }

  // Background stage bands — added FIRST so they render behind everything.
  nodes.push({
    id: 'band_stage_1',
    position: { x: BAND_X, y: STAGE_1_TOP_Y },
    type: 'stageBand',
    style: { width: BAND_WIDTH, height: STAGE_1_BOTTOM_Y - STAGE_1_TOP_Y },
    data: { stage: 1, label: 'Stage 1 · Structural', caption: 'No dimino needed to decide' },
    draggable: false,
    selectable: false,
    zIndex: -1,
  });
  nodes.push({
    id: 'band_stage_2',
    position: { x: BAND_X, y: STAGE_2_TOP_Y - 16 },
    type: 'stageBand',
    style: { width: BAND_WIDTH, height: STAGE_2_BOTTOM_Y - STAGE_2_TOP_Y + 16 },
    // No caption needed: the immediately-preceding "enumerate G via dimino"
    // divider already tells the reader G is now materialised.
    data: { stage: 2, label: 'Stage 2 · Symmetry', caption: '' },
    draggable: false,
    selectable: false,
    zIndex: -1,
  });

  // Source.
  nodes.push({
    id: 's0',
    position: { x: SOURCE_X, y: SOURCE_Y },
    type: 'source',
    style: { width: SOURCE_W, height: SOURCE_H },
    data: { text: 'Component', nodeId: 's0' },
  });

  // For each Stage-1 question, decide which branch is the leaf and which
  // is the spine continuation. Terminal "spine" of Stage 1 is the enumerate
  // divider — emerges from whichever side of the last question continues.
  const spineYs = [Q1_Y, Q2_Y, Q3_Y, Q4_Y];
  const stage1 = QUESTIONS.filter((q) => q.stage === 1);

  function planStage1(q, nextIsEnumerate) {
    // Spine continuation is whichever branch is NOT a leaf in SHAPE_SPEC /
    // REGIME_SPEC — for q_direct the non-leaf branch is the sentinel
    // 'ENUMERATE' which leads to the divider node.
    const onTrueIsLeaf = q.onTrue !== 'ENUMERATE' && !!specFor(q.onTrue);
    const onFalseIsLeaf = q.onFalse !== 'ENUMERATE' && !!specFor(q.onFalse);
    if (onTrueIsLeaf && !onFalseIsLeaf) {
      return {
        sideLeaf: q.onTrue, sideEdge: EDGE_YES,
        spineTarget: q.onFalse, spineEdge: EDGE_NO,
      };
    }
    if (onFalseIsLeaf && !onTrueIsLeaf) {
      return {
        sideLeaf: q.onFalse, sideEdge: EDGE_NO,
        spineTarget: q.onTrue, spineEdge: EDGE_YES,
      };
    }
    // Both or neither — shouldn't happen in current spec, but be defensive.
    return { sideLeaf: null, spineTarget: q.onTrue, spineEdge: EDGE_YES };
  }

  stage1.forEach((q, i) => {
    const y = spineYs[i];
    nodes.push({
      id: q.id,
      position: { x: QUESTION_X, y },
      type: 'question',
      style: { width: QUESTION_W, height: QUESTION_H },
      data: { text: q.short, nodeId: q.id },
    });

    // Spine edge from previous node (source or prior question) to this q.
    if (i === 0) {
      edges.push({
        id: `s0-${q.id}`,
        source: 's0', sourceHandle: 'bottom',
        target: q.id, targetHandle: 'top',
        style: { stroke: '#94A3B8', strokeWidth: 1.5 },
      });
    } else {
      // Label with the prior question's SPINE edge (the branch that led to us).
      const prevQ = stage1[i - 1];
      const prevPlan = planStage1(prevQ, false);
      edges.push({
        id: `${prevQ.id}-${q.id}`,
        source: prevQ.id, sourceHandle: 'bottom',
        target: q.id, targetHandle: 'top',
        label: prevPlan.spineEdge.label,
        labelStyle: { fontSize: 11, fontWeight: 700, fill: prevPlan.spineEdge.color },
        style: { stroke: prevPlan.spineEdge.color, strokeWidth: 1.5 },
      });
    }

    // Side leaf (the non-spine branch).
    const plan = planStage1(q, false);
    if (plan.sideLeaf) {
      nodes.push(leafNode(plan.sideLeaf, y));
      edges.push({
        id: `${q.id}-${plan.sideLeaf}`,
        source: q.id, sourceHandle: 'side',
        target: plan.sideLeaf, targetHandle: 'right',
        label: plan.sideEdge.label,
        labelStyle: { fontSize: 11, fontWeight: 700, fill: plan.sideEdge.color },
        style: { stroke: plan.sideEdge.color, strokeWidth: 1.5 },
      });
    }
  });

  // Enumerate-G divider — reached via the last Stage-1 question's spine branch.
  const lastStage1 = stage1[stage1.length - 1];
  const lastPlan = planStage1(lastStage1, true);
  nodes.push({
    id: 'enumerate',
    position: { x: QUESTION_X - 12, y: ENUMERATE_Y },
    type: 'enumerate',
    style: { width: QUESTION_W + 24, height: ENUMERATE_H },
    data: { text: 'enumerate G via dimino', nodeId: 'enumerate' },
  });
  edges.push({
    id: `${lastStage1.id}-enumerate`,
    source: lastStage1.id, sourceHandle: 'bottom',
    target: 'enumerate', targetHandle: 'top',
    label: lastPlan.spineEdge.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: lastPlan.spineEdge.color },
    style: { stroke: lastPlan.spineEdge.color, strokeWidth: 1.5 },
  });

  // Stage 2 — one question, two leaves.
  const stage2Q = QUESTIONS.find((q) => q.id === 'q_singleton');
  nodes.push({
    id: stage2Q.id,
    position: { x: QUESTION_X, y: Q5_Y },
    type: 'question',
    style: { width: QUESTION_W, height: QUESTION_H },
    data: { text: stage2Q.short, nodeId: stage2Q.id },
  });
  edges.push({
    id: `enumerate-${stage2Q.id}`,
    source: 'enumerate', sourceHandle: 'bottom',
    target: stage2Q.id, targetHandle: 'top',
    style: { stroke: '#8B5CF6', strokeWidth: 1.5, strokeDasharray: '6 3' },
  });

  // q_singleton yes → singleton leaf on the left.
  nodes.push(leafNode('singleton', Q5_Y));
  edges.push({
    id: `${stage2Q.id}-singleton`,
    source: stage2Q.id, sourceHandle: 'side',
    target: 'singleton', targetHandle: 'right',
    label: EDGE_YES.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_YES.color },
    style: { stroke: EDGE_YES.color, strokeWidth: 1.5 },
  });

  // q_singleton no → bruteForceOrbit terminal centered under the spine.
  const bruteY = Q5_Y + ROW_GAP;
  nodes.push(leafNode('bruteForceOrbit', bruteY, true));
  edges.push({
    id: `${stage2Q.id}-bruteForceOrbit`,
    source: stage2Q.id, sourceHandle: 'bottom',
    target: 'bruteForceOrbit', targetHandle: 'top',
    label: EDGE_NO.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_NO.color },
    style: { stroke: EDGE_NO.color, strokeWidth: 1.5 },
  });

  return { nodes, edges };
}

// ─── Pan/zoom chrome ─────────────────────────────────────────────────

function LadderControlsPanel() {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  return (
    <Panel position="bottom-right" className="!m-2">
      <PanZoomControls
        onZoomIn={() => zoomIn({ duration: 150 })}
        onZoomOut={() => zoomOut({ duration: 150 })}
        onReset={() => fitView({ padding: 0.12, duration: 200, maxZoom: 1 })}
      />
    </Panel>
  );
}

function LadderHintPanel() {
  return (
    <Panel position="bottom-left" className="!m-2">
      <PanZoomHint />
    </Panel>
  );
}

// ─── Graph memo ──────────────────────────────────────────────────────

const DecisionLadderGraph = memo(function DecisionLadderGraph({
  nodes,
  edges,
  onNodeMouseEnter,
  onNodeMouseLeave,
}) {
  return (
    <ReactFlowProvider>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={dlNodeTypes}
        className="h-full w-full"
        defaultEdgeOptions={{ type: 'step', style: { strokeWidth: 2 } }}
        fitView
        fitViewOptions={{ padding: 0.12, maxZoom: 1, minZoom: 0.4 }}
        minZoom={0.4}
        maxZoom={2.5}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag
        panOnScroll={false}
        zoomOnScroll
        zoomOnPinch
        zoomOnDoubleClick
        preventScrolling={false}
        zoomActivationKeyCode={['Meta', 'Control']}
        proOptions={{ hideAttribution: true }}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
      >
        <LadderHintPanel />
        <LadderControlsPanel />
      </ReactFlow>
    </ReactFlowProvider>
  );
});

// ─── Component ────────────────────────────────────────────────────────

export default function DecisionLadder({
  activeRegimeId = null,
  activeShapeId = null,
  activeLeafIds = null,
  spotlightLeafIds = null,
}) {
  const effectiveLeafIds = useMemo(() => {
    if (Array.isArray(activeLeafIds) || activeLeafIds instanceof Set) {
      return new Set(activeLeafIds);
    }
    const legacy = [];
    if (activeRegimeId) legacy.push(activeRegimeId);
    if (activeShapeId) legacy.push(activeShapeId);
    return new Set(legacy);
  }, [activeLeafIds, activeRegimeId, activeShapeId]);

  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const wrapRef = useRef(null);
  const hideTimerRef = useRef(null);
  const hoveredNodeRef = useRef(null);

  const effectiveSpotlight = useMemo(() => {
    if (!spotlightLeafIds) return new Set();
    if (spotlightLeafIds instanceof Set) return spotlightLeafIds;
    return new Set(spotlightLeafIds);
  }, [spotlightLeafIds]);

  const { nodes, edges } = useMemo(
    () => buildLadderLayout(effectiveLeafIds, effectiveSpotlight),
    [effectiveLeafIds, effectiveSpotlight],
  );

  const openTooltipForNode = useCallback((nodeId, rect) => {
    if (hoveredNodeRef.current === nodeId) return;
    if (!rect) return;
    const tooltipW = 460;
    const tooltipH = 340;
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    let x = rect.left + rect.width / 2;
    x = Math.max(tooltipW / 2 + 8, Math.min(x, vw - tooltipW / 2 - 16));

    const roomAbove = rect.top;
    const roomBelow = vh - rect.bottom;
    let y;
    let flipped;
    if (roomAbove >= tooltipH + 16) {
      y = rect.top - 8;
      flipped = false;
    } else if (roomBelow >= tooltipH + 16) {
      y = rect.bottom + 8;
      flipped = true;
    } else {
      flipped = roomBelow > roomAbove;
      y = flipped ? rect.bottom + 8 : rect.top - 8;
    }

    setHoveredNode(nodeId);
    hoveredNodeRef.current = nodeId;
    setTooltipPos({ x, y, flipped });
  }, []);

  const cancelHide = useCallback(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
  }, []);

  const hideTooltip = useCallback(() => {
    cancelHide();
    hideTimerRef.current = setTimeout(() => {
      hoveredNodeRef.current = null;
      setHoveredNode(null);
    }, 80);
  }, [cancelHide]);

  const handleNodeMouseEnter = useCallback((evt, node) => {
    cancelHide();
    const nodeEl = evt?.target instanceof Element ? evt.target.closest('.react-flow__node') : null;
    if (nodeEl) {
      openTooltipForNode(node.id, nodeEl.getBoundingClientRect());
      return;
    }
    const fallbackEl = document.querySelector(`.react-flow__node[data-id="${node.id}"]`);
    if (fallbackEl) openTooltipForNode(node.id, fallbackEl.getBoundingClientRect());
  }, [cancelHide, openTooltipForNode]);

  const handleNodeMouseLeave = useCallback(() => hideTooltip(), [hideTooltip]);

  useEffect(() => () => cancelHide(), [cancelHide]);

  useEffect(() => {
    if (!hoveredNode) return undefined;
    const dismiss = () => {
      cancelHide();
      hoveredNodeRef.current = null;
      setHoveredNode(null);
    };
    const dismissOnEscape = (event) => { if (event.key === 'Escape') dismiss(); };
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
  }, [hoveredNode, cancelHide]);

  const activeTooltip = hoveredNode ? tooltipFor(hoveredNode) : null;

  return (
    <div ref={wrapRef} className="relative">
      <div className="h-[800px] w-full rounded-lg border border-gray-200 bg-white">
        <DecisionLadderGraph
          nodes={nodes}
          edges={edges}
          onNodeMouseEnter={handleNodeMouseEnter}
          onNodeMouseLeave={handleNodeMouseLeave}
        />
      </div>
      {activeTooltip && (
        <div
          className="pointer-events-none fixed z-[9999] w-[460px] max-w-[calc(100vw-2rem)] rounded-lg bg-gray-900 px-4 py-3.5 text-white shadow-2xl"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: tooltipPos.flipped
              ? 'translateX(-50%)'
              : 'translateX(-50%) translateY(-100%)',
          }}
        >
          <div className="mb-1 flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: activeTooltip.color }}
            />
            <span className="text-sm font-semibold">{activeTooltip.title}</span>
          </div>
          {activeTooltip.whenText && (
            <div className="mb-2 text-[11px] uppercase tracking-wider text-gray-400">
              {activeTooltip.whenText.toLowerCase().startsWith('when')
                ? activeTooltip.whenText
                : `When: ${activeTooltip.whenText}`}
            </div>
          )}
          <div className="whitespace-normal break-words text-sm leading-6 text-gray-300">
            {activeTooltip.body}
          </div>
          {activeTooltip.latex && (
            <div className="mt-3 overflow-x-auto border-t border-gray-700 pt-3 text-sm text-gray-100">
              <div className="min-w-0">
                <Latex math={activeTooltip.latex} display />
              </div>
            </div>
          )}
          {activeTooltip.glossary && (
            <div className="mt-3 whitespace-normal break-words border-t border-gray-700 pt-3 text-[11px] leading-relaxed text-gray-300">
              <div className="mb-1.5 text-[10px] uppercase tracking-wider text-gray-500">Where</div>
              <GlossaryList entries={activeTooltip.glossary} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
