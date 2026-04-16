import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Controls, Handle, Position, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import Latex from './Latex.jsx';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';
import { REGIME_SPEC, REGIME_PRIORITY } from '../engine/regimeSpec.js';

// Layout constants — visually dense like the old DecisionTree.
const SHAPE_NODE_W = 150;
const SHAPE_NODE_H = 56;
const REGIME_NODE_W = 250;
const REGIME_NODE_H = 68;
const SHAPE_Y = 0;
const SHAPE_GAP = 170;
const FIRST_REGIME_Y = 150;
const REGIME_ROW_GAP = 100;
const REGIME_X = 1.5 * SHAPE_GAP;

function mixWithWhite(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const mix = (c) => Math.round(c + (255 - c) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

function ShapeNode({ data }) {
  const bg = data.active ? mixWithWhite(data.color, 0.78) : '#FFFFFF';
  const border = data.active ? data.color : '#E5E7EB';
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center justify-center rounded-md px-3 py-2 text-center text-xs font-semibold shadow-sm transition-all hover:shadow"
      style={{
        backgroundColor: bg,
        borderWidth: 2,
        borderStyle: 'solid',
        borderColor: border,
        color: data.active ? data.color : '#334155',
      }}
      data-tree-node={data.nodeId}
    >
      <Handle type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      {data.label}
      <Handle type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
    </div>
  );
}

function RegimeNode({ data }) {
  const bg = data.active ? mixWithWhite(data.color, 0.78) : '#FFFFFF';
  const border = data.active ? data.color : mixWithWhite(data.color, 0.55);
  const text = data.active ? data.color : '#1F2937';
  return (
    <div
      className="box-border flex h-full w-full cursor-help flex-col justify-center rounded-md px-3 py-2 text-left shadow-sm transition-all hover:shadow"
      style={{
        backgroundColor: bg,
        borderWidth: 2,
        borderStyle: 'solid',
        borderColor: border,
      }}
      data-tree-node={data.nodeId}
    >
      <Handle type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-2 w-2 shrink-0 rounded-full"
          style={{ backgroundColor: data.color }}
        />
        <span className="text-xs font-semibold" style={{ color: text }}>
          {data.label}
        </span>
      </div>
      <div className="mt-0.5 truncate text-[10px] leading-tight text-gray-500">{data.when}</div>
      <Handle type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
    </div>
  );
}

const nodeTypes = { shape: ShapeNode, regime: RegimeNode };

function buildLadderLayout(activeRegimeId, activeShapeId) {
  const nodes = [];
  const edges = [];
  const shapeOrder = ['trivial', 'allVisible', 'allSummed', 'mixed'];

  shapeOrder.forEach((shapeId, i) => {
    const spec = SHAPE_SPEC[shapeId];
    nodes.push({
      id: `shape-${shapeId}`,
      type: 'shape',
      position: { x: i * SHAPE_GAP, y: SHAPE_Y },
      style: { width: SHAPE_NODE_W, height: SHAPE_NODE_H },
      data: {
        label: spec.label,
        color: spec.color,
        active: activeShapeId === shapeId,
        nodeId: `shape-${shapeId}`,
      },
    });
  });

  REGIME_PRIORITY.forEach((regimeId, i) => {
    const spec = REGIME_SPEC[regimeId];
    nodes.push({
      id: `regime-${regimeId}`,
      type: 'regime',
      position: { x: REGIME_X, y: FIRST_REGIME_Y + i * REGIME_ROW_GAP },
      style: { width: REGIME_NODE_W, height: REGIME_NODE_H },
      data: {
        label: spec.label,
        color: spec.color,
        when: spec.when,
        active: activeRegimeId === regimeId,
        nodeId: `regime-${regimeId}`,
      },
    });
    if (i > 0) {
      const prev = REGIME_PRIORITY[i - 1];
      edges.push({
        id: `e-${prev}-${regimeId}`,
        source: `regime-${prev}`,
        target: `regime-${regimeId}`,
        style: { stroke: '#CBD5E1', strokeWidth: 1.5 },
      });
    }
  });

  edges.push({
    id: 'shape-mixed-to-regime',
    source: 'shape-mixed',
    target: `regime-${REGIME_PRIORITY[0]}`,
    style: { stroke: SHAPE_SPEC.mixed.color, strokeWidth: 2, strokeDasharray: '6 4' },
    animated: activeShapeId === 'mixed',
  });

  return { nodes, edges };
}

function tooltipFor(nodeId) {
  if (nodeId?.startsWith('shape-')) {
    const spec = SHAPE_SPEC[nodeId.slice('shape-'.length)];
    if (!spec) return null;
    return {
      title: spec.label,
      whenText: spec.when,
      body: spec.description,
      latex: spec.latex,
      color: spec.color,
    };
  }
  if (nodeId?.startsWith('regime-')) {
    const spec = REGIME_SPEC[nodeId.slice('regime-'.length)];
    if (!spec) return null;
    return {
      title: spec.label,
      whenText: spec.when,
      body: spec.description,
      latex: spec.latex,
      color: spec.color,
    };
  }
  return null;
}

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
        nodeTypes={nodeTypes}
        className="h-full w-full"
        defaultEdgeOptions={{ type: 'step', style: { strokeWidth: 2 } }}
        fitView
        fitViewOptions={{ padding: 0.12, maxZoom: 1, minZoom: 0.4 }}
        minZoom={0.4}
        maxZoom={2.5}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnScroll={false}
        panOnScroll={false}
        preventScrolling={false}
        proOptions={{ hideAttribution: true }}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
      >
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
    </ReactFlowProvider>
  );
});

export default function DecisionLadder({ activeRegimeId = null, activeShapeId = null }) {
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const wrapRef = useRef(null);
  const hideTimerRef = useRef(null);
  const hoveredNodeRef = useRef(null);

  const { nodes, edges } = useMemo(
    () => buildLadderLayout(activeRegimeId, activeShapeId),
    [activeRegimeId, activeShapeId],
  );

  const openTooltipForNode = useCallback((nodeId, rect) => {
    if (hoveredNodeRef.current === nodeId) return;
    if (!rect) return;
    const tooltipW = 320;
    const tooltipH = 260;
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
          className="pointer-events-none fixed z-[9999] w-80 max-w-[calc(100vw-2rem)] rounded-lg bg-gray-900 px-4 py-3.5 text-white shadow-2xl"
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
              <Latex math={activeTooltip.latex} display />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
