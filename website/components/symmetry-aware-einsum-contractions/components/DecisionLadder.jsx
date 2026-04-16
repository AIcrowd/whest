import React, { useMemo } from 'react';
import { Controls, Handle, Position, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';
import { REGIME_SPEC, REGIME_PRIORITY } from '../engine/regimeSpec.js';

const SHAPE_NODE_W = 140;
const SHAPE_NODE_H = 52;
const REGIME_NODE_W = 220;
const REGIME_NODE_H = 60;
const SHAPE_Y = 0;
const SHAPE_GAP = 180;
const FIRST_REGIME_Y = 150;
const REGIME_ROW_GAP = 90;
const REGIME_X = 2 * SHAPE_GAP;

function ShapeNode({ data }) {
  return (
    <div
      className={`box-border flex h-full w-full items-center justify-center rounded-md border px-3 py-2 text-center text-xs font-medium ${
        data.active ? 'border-blue-500 bg-blue-50 text-blue-900' : 'border-gray-200 bg-white text-gray-700'
      }`}
      title={data.description || data.label}
    >
      <Handle type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      {data.label}
      <Handle type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
    </div>
  );
}

function RegimeNode({ data }) {
  return (
    <div
      className={`box-border flex h-full w-full flex-col items-start justify-center rounded-md border px-3 py-2 text-left text-xs ${
        data.active ? 'border-green-500 bg-green-50 text-green-900' : 'border-gray-200 bg-white text-gray-700'
      }`}
      title={data.description || data.label}
    >
      <Handle type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      <div className="font-semibold">{data.label}</div>
      <div className="text-[10px] text-gray-500">{data.when}</div>
      <Handle type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
    </div>
  );
}

const nodeTypes = { shape: ShapeNode, regime: RegimeNode };

function buildLadderLayout(activeRegimeId) {
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
      data: { label: spec.label, description: spec.description, active: activeRegimeId === shapeId },
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
        description: spec.description,
        when: spec.when,
        active: activeRegimeId === regimeId,
      },
    });
    if (i > 0) {
      const prev = REGIME_PRIORITY[i - 1];
      edges.push({
        id: `e-${prev}-${regimeId}`,
        source: `regime-${prev}`,
        target: `regime-${regimeId}`,
        style: { stroke: '#94A3B8', strokeWidth: 1.5 },
      });
    }
  });

  edges.push({
    id: 'shape-mixed-to-regime',
    source: 'shape-mixed',
    target: `regime-${REGIME_PRIORITY[0]}`,
    style: { stroke: '#94A3B8', strokeWidth: 1.5 },
    animated: true,
  });

  return { nodes, edges };
}

export default function DecisionLadder({ activeRegimeId = null }) {
  const { nodes, edges } = useMemo(() => buildLadderLayout(activeRegimeId), [activeRegimeId]);
  return (
    <div className="h-[700px] w-full rounded-lg border border-gray-200 bg-white">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.1 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          zoomOnScroll={false}
          panOnScroll={false}
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
        >
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}
