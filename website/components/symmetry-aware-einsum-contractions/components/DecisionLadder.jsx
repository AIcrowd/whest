import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Controls, Handle, Position, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import Latex from './Latex.jsx';
import GlossaryList from './GlossaryList.jsx';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';
import { REGIME_SPEC } from '../engine/regimeSpec.js';

// ─── Layout constants ─────────────────────────────────────────────────
// Mirrors the old ComponentView.DecisionTree feel: leaves on the left,
// questions on the spine, terminal leaf at the bottom of the spine.

const LEAF_W = 210;
const LEAF_H = 44;
const QUESTION_W = 190;
const QUESTION_H = 44;
const LEAF_X = 0;
const LEAF_QUESTION_GAP = 56;
const QUESTION_X = LEAF_W + LEAF_QUESTION_GAP;
const LEAF_CENTER_OFFSET = (QUESTION_W - LEAF_W) / 2;
const SOURCE_W = 140;
const SOURCE_H = 38;
const SOURCE_X = QUESTION_X + (QUESTION_W - SOURCE_W) / 2;
const ROW_GAP = 88;
const TREE_ENTRY_Y = ROW_GAP;
const TREE_SPINE_START_Y = ROW_GAP * 2;

const EDGE_YES = { color: '#23B761', label: 'yes' };
const EDGE_NO = { color: '#F0524D', label: 'no' };

// ─── Decision spec ────────────────────────────────────────────────────
// 10 yes/no questions routing a component through shape → regime ladder.
// Each `onTrue`/`onFalse` is either a next question id or a leaf id.

const QUESTIONS = [
  {
    id: 'q_hasG',
    short: '|G| > 1 ?',
    long: 'Is the detected symmetry group nontrivial?',
    onTrue: 'q_hasW', onFalse: 'trivial',
  },
  {
    id: 'q_hasW',
    short: 'W ≠ ∅ ?',
    long: 'Are there summed (contracted) labels?',
    onTrue: 'q_hasV', onFalse: 'allVisible',
  },
  {
    id: 'q_hasV',
    short: 'V ≠ ∅ ?',
    long: 'Are there free (output) labels?',
    onTrue: 'q_singleton', onFalse: 'allSummed',
  },
  {
    id: 'q_singleton',
    short: '|V| = 1 ?',
    long: 'Exactly one free label — singleton weighted Burnside applies.',
    onTrue: 'singleton', onFalse: 'q_direct',
  },
  {
    id: 'q_direct',
    short: 'Split V/W ?',
    long: 'Every generator moves only V-labels or only W-labels — direct product.',
    onTrue: 'directProduct', onFalse: 'bruteForceOrbit',
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

function isLeafId(id) {
  return !!specFor(id);
}

function planRow(question) {
  const onTrueIsLeaf = isLeafId(question.onTrue);
  const onFalseIsLeaf = isLeafId(question.onFalse);
  if (onTrueIsLeaf && onFalseIsLeaf) {
    return {
      sideLeaf: question.onTrue,
      sideEdge: EDGE_YES,
      spineNext: question.onFalse,
      spineEdge: EDGE_NO,
      spineIsTerminalLeaf: true,
    };
  }
  if (onTrueIsLeaf) {
    return {
      sideLeaf: question.onTrue,
      sideEdge: EDGE_YES,
      spineNext: question.onFalse,
      spineEdge: EDGE_NO,
      spineIsTerminalLeaf: false,
    };
  }
  if (onFalseIsLeaf) {
    return {
      sideLeaf: question.onFalse,
      sideEdge: EDGE_NO,
      spineNext: question.onTrue,
      spineEdge: EDGE_YES,
      spineIsTerminalLeaf: false,
    };
  }
  return {
    sideLeaf: null,
    sideEdge: null,
    spineNext: question.onTrue,
    spineEdge: EDGE_YES,
    spineIsTerminalLeaf: false,
  };
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
  // Spotlight > active > idle. Spotlight is a coral ring (matches the
  // StickyBar halo color) so the cross-highlight reads as "these two
  // surfaces are talking about the same thing".
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

const dlNodeTypes = {
  source: SourceNode,
  question: QuestionNode,
  leaf: LeafNode,
};

// ─── Tooltip ──────────────────────────────────────────────────────────

function tooltipFor(nodeId) {
  if (nodeId === 's0') {
    return {
      title: 'Component',
      whenText: null,
      body: 'Start with one independent component of the detected group action. The ladder decides how to count its multiplications and output-bin updates.',
      latex: null,
      color: '#64748B',
    };
  }
  const q = QUESTIONS.find((x) => x.id === nodeId);
  if (q) {
    return {
      title: q.short,
      whenText: null,
      body: q.long,
      latex: null,
      color: '#64748B',
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

function buildLadderLayout(activeLeafIds, spotlightLeafIds) {
  const active = activeLeafIds instanceof Set
    ? activeLeafIds
    : new Set(activeLeafIds || []);
  const spotlight = spotlightLeafIds instanceof Set
    ? spotlightLeafIds
    : new Set(spotlightLeafIds || []);
  const nodes = [];
  const edges = [];

  function leafNodeData(leafId, y, centered = false) {
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

  nodes.push({
    id: 's0',
    position: { x: SOURCE_X, y: 0 },
    type: 'source',
    style: { width: SOURCE_W, height: SOURCE_H },
    data: { text: 'Component', nodeId: 's0' },
  });

  QUESTIONS.forEach((question, index) => {
    const y = index === 0 ? TREE_ENTRY_Y : TREE_SPINE_START_Y + ROW_GAP * (index - 1);
    nodes.push({
      id: question.id,
      position: { x: QUESTION_X, y },
      type: 'question',
      style: { width: QUESTION_W, height: QUESTION_H },
      data: { text: question.short, nodeId: question.id },
    });

    if (index === 0) {
      edges.push({
        id: 's0-q0',
        source: 's0',
        sourceHandle: 'bottom',
        target: question.id,
        targetHandle: 'top',
        style: { stroke: '#94A3B8', strokeWidth: 1.5 },
      });
    }

    const plan = planRow(question);

    if (plan.sideLeaf) {
      nodes.push(leafNodeData(plan.sideLeaf, y));
      edges.push({
        id: `${question.id}-${plan.sideLeaf}`,
        source: question.id,
        sourceHandle: 'side',
        target: plan.sideLeaf,
        targetHandle: 'right',
        label: plan.sideEdge.label,
        labelStyle: { fontSize: 11, fontWeight: 700, fill: plan.sideEdge.color },
        style: { stroke: plan.sideEdge.color, strokeWidth: 1.5 },
      });
    }

    if (plan.spineIsTerminalLeaf) {
      const terminalY = TREE_SPINE_START_Y + ROW_GAP * index;
      nodes.push(leafNodeData(plan.spineNext, terminalY, true));
      edges.push({
        id: `${question.id}-${plan.spineNext}`,
        source: question.id,
        sourceHandle: 'bottom',
        target: plan.spineNext,
        targetHandle: 'top',
        label: plan.spineEdge.label,
        labelStyle: { fontSize: 11, fontWeight: 700, fill: plan.spineEdge.color },
        style: { stroke: plan.spineEdge.color, strokeWidth: 1.5 },
      });
    } else if (plan.spineNext) {
      edges.push({
        id: `${question.id}-${plan.spineNext}`,
        source: question.id,
        sourceHandle: 'bottom',
        target: plan.spineNext,
        targetHandle: 'top',
        label: plan.spineEdge.label,
        labelStyle: { fontSize: 11, fontWeight: 700, fill: plan.spineEdge.color },
        style: { stroke: plan.spineEdge.color, strokeWidth: 1.5 },
      });
    }
  });

  return { nodes, edges };
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
        fitViewOptions={{ padding: 0.15, maxZoom: 1, minZoom: 0.4 }}
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
        preventScrolling
        proOptions={{ hideAttribution: true }}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
      >
        <Controls showInteractive={false} position="bottom-right" />
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
  // Highlighting policy: every detected leaf across all components gets a
  // halo (including shape leaves like `trivial` / `allVisible`). The
  // `activeLeafIds` prop is the authoritative source; legacy single-value
  // props stay as a fallback for callers not yet migrated.
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
    const nodeEl =
      evt?.target instanceof Element ? evt.target.closest('.react-flow__node') : null;
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
      <div className="h-[720px] w-full rounded-lg border border-gray-200 bg-white">
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
              When: {activeTooltip.whenText}
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
