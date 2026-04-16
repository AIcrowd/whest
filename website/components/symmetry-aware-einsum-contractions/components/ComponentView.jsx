import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Handle, Position, ReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { cn } from '../lib/utils';
import { CASE_META } from '../engine/componentDecomposition.js';
import Latex from './Latex.jsx';

const GRAPH_SIZE = 220;
const CENTER = GRAPH_SIZE / 2;
const ORBIT_R = 80;
const NODE_R = 14;
const COLOR_V = '#4A7CFF';
const COLOR_W = '#94A3B8';
const COMP_COLORS = ['#4A7CFF', '#23B761', '#FA9E33', '#7C3AED', '#F0524D'];
const CASE_NODE_COLORS = {
  A: CASE_META.A?.color ?? '#4A7CFF',
  B: CASE_META.B?.color ?? '#94A3B8',
  C: CASE_META.C?.color ?? '#FA9E33',
  D: CASE_META.D?.color ?? '#23B761',
  E: CASE_META.E?.color ?? '#F0524D',
};
const LEAF_W = 146;
const LEAF_H = 44;
const QUESTION_W = 158;
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
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={dtNodeTypes}
      className="h-full w-full"
      defaultEdgeOptions={{ type: 'step', style: { strokeWidth: 2 } }}
      fitViewOptions={{ padding: 0.15 }}
      fitView
      onNodeMouseEnter={onNodeMouseEnter}
      onNodeMouseLeave={onNodeMouseLeave}
      panOnDrag={false}
      zoomOnScroll={false}
      zoomOnPinch={false}
      zoomOnDoubleClick={false}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      preventScrolling={false}
      proOptions={{ hideAttribution: true }}
    />
  );
});

export function DecisionTree() {
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const wrapRef = useRef(null);
  const hideTimerRef = useRef(null);
  const hoveredNodeRef = useRef(null);

  const tooltips = {
    s0: {
      title: 'Component',
      body: 'Start with one independent component of the detected group action. The tree decides how to count its multiplication representatives and accumulation updates.',
    },
    q0: {
      title: 'Check: nontrivial symmetry?',
      body: 'Does this component have any detected symmetry beyond the identity? If not, there is no quotienting to do and the count stays direct.',
    },
    t0: {
      title: 'Direct count (trivial)',
      body: 'The component symmetry is trivial, so every assignment remains distinct. Multiplication and accumulation counts are read off directly without Burnside or orbit enumeration.',
      latex: String.raw`|I_a / G_a| = |I_a| \quad \text{when } G_a = \{e\}`,
    },
    q1: {
      title: 'Check: W-labels present?',
      body: 'Does this component contain any summed (contracted) labels? If not, the symmetry only acts on output indices.',
    },
    q2: {
      title: 'Check: V-labels present?',
      body: 'Does this component contain any free (output) labels? If not, the symmetry only acts on summed indices.',
    },
    q3: {
      title: 'Check: cross-boundary generators?',
      body: 'Does any generator of Gₐ map a V-label to a W-label or vice versa? If no, V and W actions are correlated but partition-preserving.',
    },
    q4: {
      title: 'Check: full symmetric group?',
      body: 'Is Gₐ the full symmetric group on all labels in this component? That is, |Gₐ| = |Lₐ|! (factorial). If yes, the Young-tableau formula gives an analytic shortcut.',
      latex: String.raw`|G_a| = |L_a|! \implies \rho_a = |I_a / H_a|`,
    },
    a: {
      title: 'Case A: V-only',
      body: 'All labels are free (output). Symmetry reduces unique multiplications, but every output bin must still be written.',
      latex: String.raw`\rho_a = \prod_{\ell \in V_a} n_\ell`,
    },
    b: {
      title: 'Case B: W-only',
      body: 'All labels are summed. Orbits collapse both multiplications and accumulations equally.',
      latex: String.raw`\rho_a = |I_a / G_a| \text{ (Burnside)}`,
    },
    c: {
      title: 'Case C: Correlated',
      body: 'V and W labels are both present and generators act on both sides simultaneously, but no generator crosses the V/W boundary. No product formula exists — enumerate orbits exactly.',
      latex: String.raw`\rho_a = \sum_{O \in I_a/G_a} |\pi_{V_a}(O)|`,
    },
    d: {
      title: 'Case D: Cross (Young)',
      body: 'Cross-boundary generators with the full symmetric group. The V-stabilizer Hₐ gives an analytic Burnside count for accumulation cost.',
      latex: String.raw`\rho_a = |I_a / H_a|, \quad H_a = \mathrm{Stab}_{G_a}(V_a)`,
    },
    e: {
      title: 'Case E: Cross (general)',
      body: 'Cross-boundary generators but not the full symmetric group. Value coincidences can merge output bins unpredictably — must enumerate orbits.',
      latex: String.raw`\rho_a = \sum_{O \in I_a/G_a} |\pi_{V_a}(O)|`,
    },
  };

  const openTooltipForNode = useCallback((nodeId, rect) => {
    if (hoveredNodeRef.current === nodeId) {
      return;
    }
    if (!rect) return;
    const tooltipW = 280;
    const tooltipH = 126;
    let x = rect.left + rect.width / 2;
    const vw = document.documentElement.clientWidth;
    x = Math.max(tooltipW / 2 + 8, Math.min(x, vw - tooltipW / 2 - 16));

    let y = rect.top - 8;
    let flipped = false;
    if (y - tooltipH < 8) {
      y = rect.bottom + 8;
      flipped = true;
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

  const nodes = useMemo(
    () => {
      const questionNodes = [
        ['q0', 'Has symmetry?', 'Check: nontrivial symmetry?', TREE_ENTRY_Y],
        ['q1', 'W₀ = ∅ ?', 'Check: W-labels present?', TREE_SPINE_START_Y],
        ['q2', 'V₀ = ∅ ?', 'Check: V-labels present?', TREE_SPINE_START_Y + ROW_GAP],
        ['q3', 'Cross V/W gens?', 'Check: cross-boundary generators?', TREE_SPINE_START_Y + ROW_GAP * 2],
        ['q4', 'Gₐ = Sym(Lₐ) ?', 'Check: full symmetric group?', TREE_SPINE_START_Y + ROW_GAP * 3],
      ];

      const leafNodes = [
        ['t0', 'Direct count (trivial)', 'Direct count (trivial)', null, TREE_ENTRY_Y],
        ['a', 'Case A: V-only', 'Case A: V-only', CASE_NODE_COLORS.A, TREE_SPINE_START_Y],
        ['b', 'Case B: W-only', 'Case B: W-only', CASE_NODE_COLORS.B, TREE_SPINE_START_Y + ROW_GAP],
        ['c', 'Case C: Correlated', 'Case C: Correlated', CASE_NODE_COLORS.C, TREE_SPINE_START_Y + ROW_GAP * 2],
        ['d', 'Case D: Cross (Young)', 'Case D: Cross (Young)', CASE_NODE_COLORS.D, TREE_SPINE_START_Y + ROW_GAP * 3],
      ];

      return [
        {
          id: 's0',
          position: { x: SOURCE_X, y: 0 },
          type: 'source',
          style: { width: SOURCE_W, height: SOURCE_H },
          data: {
            text: 'Component',
            title: 'Component',
            nodeId: 's0',
          },
        },
        ...questionNodes.map(([id, text, title, y]) => ({
          id,
          position: { x: QUESTION_X, y },
          type: 'question',
          style: { width: QUESTION_W, height: QUESTION_H },
          data: { text, title, nodeId: id },
        })),
        ...leafNodes.map(([id, text, title, color, y]) => ({
          id,
          position: { x: LEAF_X, y },
          type: 'leaf',
          style: { width: LEAF_W, height: LEAF_H },
          data: { text, title, nodeId: id, color: color ?? '#CBD5E1' },
        })),
        {
          id: 'e',
          position: { x: QUESTION_X + LEAF_CENTER_OFFSET, y: TREE_SPINE_START_Y + ROW_GAP * 4 },
          type: 'leaf',
          style: { width: LEAF_W, height: LEAF_H },
          data: {
            text: 'Case E: Cross (general)',
            title: 'Case E: Cross (general)',
            nodeId: 'e',
            color: CASE_NODE_COLORS.E,
          },
        },
      ];
    },
    [],
  );

  const edges = useMemo(
    () => [
      {
        id: 's0-q0',
        source: 's0',
        sourceHandle: 'bottom',
        target: 'q0',
        targetHandle: 'top',
        style: { stroke: '#94A3B8', strokeWidth: 1.5 },
      },
      {
        id: 'q0-t0',
        source: 'q0',
        sourceHandle: 'yes',
        target: 't0',
        targetHandle: 'right',
        label: 'no',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#F0524D' },
        style: { stroke: '#F0524D', strokeWidth: 1.5 },
      },
      {
        id: 'q0-q1',
        source: 'q0',
        sourceHandle: 'bottom',
        target: 'q1',
        targetHandle: 'top',
        label: 'yes',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#23B761' },
        style: { stroke: '#23B761', strokeWidth: 1.5 },
      },
      {
        id: 'q1-a',
        source: 'q1',
        sourceHandle: 'yes',
        target: 'a',
        targetHandle: 'right',
        label: 'yes',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#23B761' },
        style: { stroke: '#23B761', strokeWidth: 1.5 },
      },
      {
        id: 'q1-q2',
        source: 'q1',
        sourceHandle: 'bottom',
        target: 'q2',
        targetHandle: 'top',
        label: 'no',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#F0524D' },
        style: { stroke: '#F0524D', strokeWidth: 1.5 },
      },
      {
        id: 'q2-b',
        source: 'q2',
        sourceHandle: 'yes',
        target: 'b',
        targetHandle: 'right',
        label: 'yes',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#23B761' },
        style: { stroke: '#23B761', strokeWidth: 1.5 },
      },
      {
        id: 'q2-q3',
        source: 'q2',
        sourceHandle: 'bottom',
        target: 'q3',
        targetHandle: 'top',
        label: 'no',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#F0524D' },
        style: { stroke: '#F0524D', strokeWidth: 1.5 },
      },
      {
        id: 'q3-c',
        source: 'q3',
        sourceHandle: 'yes',
        target: 'c',
        targetHandle: 'right',
        label: 'no',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#F0524D' },
        style: { stroke: '#F0524D', strokeWidth: 1.5 },
      },
      {
        id: 'q3-q4',
        source: 'q3',
        sourceHandle: 'bottom',
        target: 'q4',
        targetHandle: 'top',
        label: 'yes',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#23B761' },
        style: { stroke: '#23B761', strokeWidth: 1.5 },
      },
      {
        id: 'q4-d',
        source: 'q4',
        sourceHandle: 'yes',
        target: 'd',
        targetHandle: 'right',
        label: 'yes',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#23B761' },
        style: { stroke: '#23B761', strokeWidth: 1.5 },
      },
      {
        id: 'q4-e',
        source: 'q4',
        sourceHandle: 'bottom',
        target: 'e',
        targetHandle: 'top',
        label: 'no',
        labelStyle: { fontSize: 10, fontWeight: 700, fill: '#F0524D' },
        style: { stroke: '#F0524D', strokeWidth: 1.5 },
      },
    ],
    [],
  );

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
              className={cn(
                'pointer-events-none fixed z-[9999] w-72 bg-gray-900 px-3.5 py-3 text-white shadow-2xl',
                tooltipPos.flipped ? 'translate-y-0' : 'translate-y-[-100%]',
              )}
              style={{
                left: tooltipPos.x,
                top: tooltipPos.y,
                transform: tooltipPos.flipped ? 'translateX(-50%)' : 'translateX(-50%) translateY(-100%)',
              }}
            >
              <div className="mb-1 text-xs font-bold">{activeTooltip.title}</div>
              <div className="text-[11px] leading-relaxed text-gray-300">{activeTooltip.body}</div>
              {activeTooltip.latex && (
                <div className="mt-2 text-xs">
                  <Latex math={activeTooltip.latex} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </details>
  );
}
