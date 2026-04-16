import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Controls, Handle, Position, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { cn } from '../lib/utils';
import { CASE_META } from '../engine/componentDecomposition.js';
import {
  CLASSIFICATION_LEAVES,
  CLASSIFICATION_QUESTIONS,
} from '../engine/classificationSpec.js';
import Latex from './Latex.jsx';
import GlossaryProse from './GlossaryProse.jsx';

const GRAPH_SIZE = 220;
const CENTER = GRAPH_SIZE / 2;
const ORBIT_R = 80;
const NODE_R = 14;
const COLOR_V = '#4A7CFF';
const COLOR_W = '#94A3B8';
const COMP_COLORS = ['#4A7CFF', '#23B761', '#FA9E33', '#7C3AED', '#F0524D'];
const CASE_NODE_COLORS = {
  trivial: CASE_META.trivial?.color ?? '#CBD5E1',
  A: CASE_META.A?.color ?? '#4A7CFF',
  B: CASE_META.B?.color ?? '#94A3B8',
  C: CASE_META.C?.color ?? '#FA9E33',
  D: CASE_META.D?.color ?? '#23B761',
  E: CASE_META.E?.color ?? '#F0524D',
};
const LEAF_W = 190;
const LEAF_H = 44;
const QUESTION_W = 170;
const QUESTION_H = 44;
const LEAF_X = 0;
const LEAF_QUESTION_GAP = 56;
const QUESTION_X = LEAF_W + LEAF_QUESTION_GAP;
const LEAF_CENTER_OFFSET = (QUESTION_W - LEAF_W) / 2;
const SOURCE_W = 120;
const SOURCE_H = 36;
const SOURCE_X = QUESTION_X + (QUESTION_W - SOURCE_W) / 2;
const ROW_GAP = 86;
const TREE_ENTRY_Y = ROW_GAP;
const TREE_SPINE_START_Y = ROW_GAP * 2;

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

export function LabelInteractionGraph({ allLabels = [], vLabels = [], interactionGraph = {} }) {
  const n = allLabels.length;
  if (n === 0) return null;

  const vSet = new Set(vLabels);
  const { edges = [], components = [] } = interactionGraph;
  const uniqueEdges = dedupEdges(edges);
  const positions = allLabels.map((_, idx) => circlePos(idx, n, ORBIT_R));
  const labelToComp = new Array(n).fill(-1);
  components.forEach((comp, compIdx) => {
    comp.forEach((idx) => {
      labelToComp[idx] = compIdx;
    });
  });

  return (
    <svg
      className="w-full max-w-[220px]"
      viewBox={`0 0 ${GRAPH_SIZE} ${GRAPH_SIZE}`}
      aria-label="Label interaction graph"
    >
      {components.map((comp, compIdx) => {
        if (comp.length <= 1) return null;
        const points = comp.map((idx) => positions[idx]);
        return (
          <polygon
            key={`comp-${compIdx}`}
            points={points.map((point) => `${point.x},${point.y}`).join(' ')}
            fill={COMP_COLORS[compIdx % COMP_COLORS.length]}
            fillOpacity={0.08}
            stroke={COMP_COLORS[compIdx % COMP_COLORS.length]}
            strokeDasharray="4 3"
            strokeOpacity={0.45}
          />
        );
      })}

      {uniqueEdges.map(([a, b], edgeIdx) => {
        const pa = positions[a];
        const pb = positions[b];
        if (!pa || !pb) return null;
        return (
          <line
            key={`edge-${edgeIdx}`}
            x1={pa.x}
            y1={pa.y}
            x2={pb.x}
            y2={pb.y}
            stroke="#6B7280"
            strokeWidth={1}
            strokeOpacity={0.45}
          />
        );
      })}

      {allLabels.map((label, idx) => {
        const { x, y } = positions[idx];
        const isV = vSet.has(label);
        return (
          <g key={`node-${label}`}>
            <circle cx={x} cy={y} r={NODE_R} fill={isV ? COLOR_V : COLOR_W} stroke="#F9FAFB" strokeWidth={2} />
            <text
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={12}
              fontFamily="ui-monospace, monospace"
              fontWeight={600}
              fill="#FFFFFF"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function QuestionNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center justify-center rounded-md border border-gray-200 bg-white px-3 py-1.5 text-center text-sm leading-tight text-gray-900 shadow-sm transition-colors hover:border-gray-400"
      title={data.title}
      data-tree-node={data.nodeId}
    >
      <Handle id="top" type="target" position={Position.Top} className="pointer-events-none opacity-0" />
      {data.text}
      <Handle id="bottom" type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
      <Handle type="source" position={Position.Left} id="yes" className="pointer-events-none opacity-0" />
    </div>
  );
}

function SourceNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center justify-center rounded-full border border-gray-200 bg-white px-3 py-1.5 text-center text-sm font-semibold leading-tight text-gray-900 shadow-sm transition-colors hover:border-gray-400"
      title={data.title}
      data-tree-node={data.nodeId}
    >
      {data.text}
      <Handle id="bottom" type="source" position={Position.Bottom} className="pointer-events-none opacity-0" />
    </div>
  );
}

function LeafNode({ data }) {
  return (
    <div
      className={cn(
        'box-border flex h-full w-full cursor-help items-center justify-center rounded-lg border px-2 py-0.5 text-center text-sm leading-tight font-accent font-bold whitespace-nowrap transition-colors text-gray-900',
      )}
      style={{
        backgroundColor: `${data.color}20`,
        borderColor: data.color,
        borderWidth: 2,
      }}
      title={data.title}
      data-tree-node={data.nodeId}
    >
      <Handle type="target" position={Position.Right} className="pointer-events-none opacity-0" id="right" />
      <Handle type="target" position={Position.Top} className="pointer-events-none opacity-0" id="top" />
      {data.text}
    </div>
  );
}

const dtNodeTypes = {
  source: SourceNode,
  question: QuestionNode,
  leaf: LeafNode,
};

const DecisionTreeGraph = memo(function DecisionTreeGraph({
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
        nodeTypes={dtNodeTypes}
        className="h-full w-full"
        defaultEdgeOptions={{ type: 'step', style: { strokeWidth: 2 } }}
        fitViewOptions={{ padding: 0.2, maxZoom: 1, minZoom: 0.4 }}
        minZoom={0.4}
        maxZoom={2.5}
        fitView
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
        panOnDrag
        panOnScroll={false}
        zoomOnScroll
        zoomOnPinch
        zoomOnDoubleClick
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        preventScrolling={false}
        proOptions={{ hideAttribution: true }}
      >
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
    </ReactFlowProvider>
  );
});

export function DecisionTree() {
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const wrapRef = useRef(null);
  const hideTimerRef = useRef(null);
  const hoveredNodeRef = useRef(null);

  // Tooltips are derived from the same CLASSIFICATION_* spec that the engine
  // uses, so the tree's node text and the engine's branching cannot drift.
  const tooltips = useMemo(() => {
    const entries = {
      s0: {
        title: 'Component',
        body: 'Start with one independent component of the detected group action. The tree decides how to count its multiplication representatives and accumulation updates.',
      },
    };
    for (const question of CLASSIFICATION_QUESTIONS) {
      entries[question.id] = {
        title: question.long,
        body: question.description,
        latex: question.latex ?? null,
      };
    }
    for (const leaf of Object.values(CLASSIFICATION_LEAVES)) {
      entries[leaf.id] = {
        title: leaf.label,
        body: leaf.description,
        latex: leaf.latex ?? null,
        glossary: leaf.glossary ?? null,
      };
    }
    return entries;
  }, []);

  const openTooltipForNode = useCallback((nodeId, rect) => {
    if (hoveredNodeRef.current === nodeId) {
      return;
    }
    if (!rect) return;
    // Current tooltip contents (description + KaTeX formula + glossary) are
    // around 360–400px tall. The flip-below decision must use this so we
    // don't place a tall tooltip above a node near the top of the viewport
    // and watch it clip off-screen.
    const tooltipW = 288;
    const tooltipH = 380;
    let x = rect.left + rect.width / 2;
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
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
      // Neither side has full clearance — pick the side with more room.
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
      evt?.target instanceof Element
        ? evt.target.closest('.react-flow__node')
        : null;
    if (nodeEl) {
      openTooltipForNode(node.id, nodeEl.getBoundingClientRect());
      return;
    }

    const selector = `.react-flow__node[data-id="${node.id}"]`;
    const fallbackEl = document.querySelector(selector);
    if (fallbackEl) {
      openTooltipForNode(node.id, fallbackEl.getBoundingClientRect());
    }
  }, [cancelHide, openTooltipForNode]);

  const handleNodeMouseLeave = useCallback(() => hideTooltip(), [hideTooltip]);

  useEffect(() => () => cancelHide(), [cancelHide]);

  // Defensive dismissal — identical to CaseBadge. Covers cases where a React
  // synthetic pointerleave never fires (scroll, focus change, programmatic
  // event, touch gesture), which would otherwise leave the tooltip stuck.
  useEffect(() => {
    if (!hoveredNode) return undefined;
    const dismiss = () => {
      cancelHide();
      hoveredNodeRef.current = null;
      setHoveredNode(null);
    };
    const dismissOnEscape = (event) => {
      if (event.key === 'Escape') dismiss();
    };
    const dismissIfOutsideGraph = (event) => {
      const target = event.target;
      if (!(target instanceof Node)) return dismiss();
      if (wrapRef.current && wrapRef.current.contains(target)) return;
      dismiss();
    };

    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    window.addEventListener('keydown', dismissOnEscape);
    window.addEventListener('pointerdown', dismissIfOutsideGraph);
    window.addEventListener('blur', dismiss);

    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('keydown', dismissOnEscape);
      window.removeEventListener('pointerdown', dismissIfOutsideGraph);
      window.removeEventListener('blur', dismiss);
    };
  }, [hoveredNode, cancelHide]);

  // Build nodes and edges from the shared classification spec. Visual layout
  // (x/y coordinates) stays here; branch structure comes from the spec.
  const { nodes, edges } = useMemo(() => {
    const isLeafId = (id) =>
      Object.prototype.hasOwnProperty.call(CLASSIFICATION_LEAVES, id);

    const EDGE_YES = { color: '#23B761', label: 'yes' };
    const EDGE_NO = { color: '#F0524D', label: 'no' };

    // For each question: decide which branch goes to a side-leaf (rendered
    // to the right) and which continues the spine downward.
    function planRow(question) {
      const onTrueIsLeaf = isLeafId(question.onTrue);
      const onFalseIsLeaf = isLeafId(question.onFalse);
      if (onTrueIsLeaf && onFalseIsLeaf) {
        // Both outcomes are leaves — onTrue goes to the side, onFalse terminates the spine.
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

    const leafNodeData = (leafId, y, centered = false) => {
      const leaf = CLASSIFICATION_LEAVES[leafId];
      const color = CASE_NODE_COLORS[leaf.caseType] ?? '#CBD5E1';
      return {
        id: leaf.id,
        position: {
          x: centered ? QUESTION_X + LEAF_CENTER_OFFSET : LEAF_X,
          y,
        },
        type: 'leaf',
        style: { width: LEAF_W, height: LEAF_H },
        data: {
          text: leaf.label,
          title: leaf.label,
          nodeId: leaf.id,
          color,
        },
      };
    };

    const builtNodes = [
      {
        id: 's0',
        position: { x: SOURCE_X, y: 0 },
        type: 'source',
        style: { width: SOURCE_W, height: SOURCE_H },
        data: { text: 'Component', title: 'Component', nodeId: 's0' },
      },
    ];
    const builtEdges = [];

    CLASSIFICATION_QUESTIONS.forEach((question, index) => {
      const y = index === 0 ? TREE_ENTRY_Y : TREE_SPINE_START_Y + ROW_GAP * (index - 1);

      // Question node on the spine.
      builtNodes.push({
        id: question.id,
        position: { x: QUESTION_X, y },
        type: 'question',
        style: { width: QUESTION_W, height: QUESTION_H },
        data: { text: question.short, title: question.long, nodeId: question.id },
      });

      // Edge coming in from above (s0 for q0, or from the previous question).
      if (index === 0) {
        builtEdges.push({
          id: 's0-q0',
          source: 's0',
          sourceHandle: 'bottom',
          target: question.id,
          targetHandle: 'top',
          style: { stroke: '#94A3B8', strokeWidth: 1.5 },
        });
      }

      const plan = planRow(question);

      // Side leaf (to the right of the question).
      if (plan.sideLeaf) {
        builtNodes.push(leafNodeData(plan.sideLeaf, y));
        builtEdges.push({
          id: `${question.id}-${plan.sideLeaf}`,
          source: question.id,
          sourceHandle: 'yes',
          target: plan.sideLeaf,
          targetHandle: 'right',
          label: plan.sideEdge.label,
          labelStyle: { fontSize: 10, fontWeight: 700, fill: plan.sideEdge.color },
          style: { stroke: plan.sideEdge.color, strokeWidth: 1.5 },
        });
      }

      // Spine continuation: either another question (next iteration will draw
      // its node) or a terminal leaf placed below the last question.
      if (plan.spineIsTerminalLeaf) {
        const terminalY = TREE_SPINE_START_Y + ROW_GAP * index;
        builtNodes.push(leafNodeData(plan.spineNext, terminalY, true));
        builtEdges.push({
          id: `${question.id}-${plan.spineNext}`,
          source: question.id,
          sourceHandle: 'bottom',
          target: plan.spineNext,
          targetHandle: 'top',
          label: plan.spineEdge.label,
          labelStyle: { fontSize: 10, fontWeight: 700, fill: plan.spineEdge.color },
          style: { stroke: plan.spineEdge.color, strokeWidth: 1.5 },
        });
      } else if (plan.spineNext) {
        builtEdges.push({
          id: `${question.id}-${plan.spineNext}`,
          source: question.id,
          sourceHandle: 'bottom',
          target: plan.spineNext,
          targetHandle: 'top',
          label: plan.spineEdge.label,
          labelStyle: { fontSize: 10, fontWeight: 700, fill: plan.spineEdge.color },
          style: { stroke: plan.spineEdge.color, strokeWidth: 1.5 },
        });
      }
    });

    return { nodes: builtNodes, edges: builtEdges };
  }, []);

  const activeTooltip = hoveredNode ? tooltips[hoveredNode] : null;
  const [isDeepDiveOpen, setIsDeepDiveOpen] = useState(true);

  return (
    <details
      open={isDeepDiveOpen}
      onToggle={(event) => setIsDeepDiveOpen(event.currentTarget.open)}
      className="overflow-hidden rounded-lg border border-gray-200 bg-white"
    >
      <summary className="cursor-pointer rounded-lg bg-gray-50 px-3.5 py-2.5 text-xs font-semibold text-gray-600">
        Deep dive: Classification decision tree
      </summary>
      {isDeepDiveOpen && (
        <div className="relative flex min-h-[620px] flex-col justify-center p-2" ref={wrapRef}>
          <p className="px-2 pb-2 pt-1 text-sm leading-6 text-muted-foreground">
            First check whether the component has any nontrivial detected symmetry at all. Only components that do
            flow into the A-E structural spine below.
          </p>
          <div className="flex h-[620px] w-full min-w-0 items-center justify-center">
            <DecisionTreeGraph
              nodes={nodes}
              edges={edges}
              onNodeMouseEnter={handleNodeMouseEnter}
              onNodeMouseLeave={handleNodeMouseLeave}
            />
          </div>

          {activeTooltip && (
            <div
              className="pointer-events-none fixed z-[9999] w-72 rounded-md bg-gray-900 px-3.5 py-3 text-white shadow-2xl"
              style={{
                left: tooltipPos.x,
                top: tooltipPos.y,
                transform: tooltipPos.flipped
                  ? 'translateX(-50%)'
                  : 'translateX(-50%) translateY(-100%)',
              }}
            >
              <div className="mb-1 text-xs font-bold">{activeTooltip.title}</div>
              <div className="text-[11px] leading-relaxed text-gray-300">{activeTooltip.body}</div>
              {activeTooltip.latex && (
                <div className="mt-2 text-xs">
                  <Latex math={activeTooltip.latex} />
                </div>
              )}
              {activeTooltip.glossary && (
                <div className="mt-2 text-[11px] leading-relaxed text-gray-300">
                  <GlossaryProse text={activeTooltip.glossary} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </details>
  );
}
