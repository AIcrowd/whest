import { useMemo, useRef, useState } from 'react';
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
const LEAF_W = 146;
const LEAF_H = 44;
const QUESTION_W = 158;
const QUESTION_H = 44;
const LEAF_X = 0;
const LEAF_QUESTION_GAP = 56;
const QUESTION_X = LEAF_W + LEAF_QUESTION_GAP;
const LEAF_CENTER_OFFSET = (QUESTION_W - LEAF_W) / 2;
const ROW_GAP = 86;

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
      className="box-border flex h-full w-full items-center justify-center rounded-md border border-gray-200 bg-white px-3 py-1.5 text-center text-[11px] font-mono leading-tight text-gray-900 shadow-sm transition-colors hover:border-gray-400"
      onMouseEnter={data.onHover}
      onMouseLeave={data.onLeave}
    >
      <Handle id="top" type="target" position={Position.Top} className="opacity-0" />
      {data.text}
      <Handle id="bottom" type="source" position={Position.Bottom} className="opacity-0" />
      <Handle type="source" position={Position.Left} id="yes" className="opacity-0" />
    </div>
  );
}

function LeafNode({ data }) {
  return (
    <div
      className={cn(
        'box-border flex h-full w-full items-center justify-center rounded-lg border px-2 py-0.5 text-center text-[10px] leading-tight font-accent font-bold whitespace-nowrap transition-colors',
        data.hovered ? 'text-white' : 'text-gray-900',
      )}
      style={{
        backgroundColor: data.hovered ? data.color : `${data.color}20`,
        borderColor: data.color,
        borderWidth: 2,
      }}
      onMouseEnter={data.onHover}
      onMouseLeave={data.onLeave}
    >
      <Handle type="target" position={Position.Right} className="opacity-0" id="right" />
      <Handle type="target" position={Position.Top} className="opacity-0" id="top" />
      {data.text}
    </div>
  );
}

const dtNodeTypes = {
  question: QuestionNode,
  leaf: LeafNode,
};

export function DecisionTree() {
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const wrapRef = useRef(null);

  const tooltips = {
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

  const handleMouseEnter = (nodeId, evt) => {
    if (!evt?.currentTarget) return;
    const rect = evt.currentTarget.getBoundingClientRect();
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
    setTooltipPos({ x, y, flipped });
    setHoveredNode(nodeId);
  };

  const handleMouseLeave = () => setHoveredNode(null);

  const caseColors = {
    A: CASE_META.A?.color ?? '#4A7CFF',
    B: CASE_META.B?.color ?? '#94A3B8',
    C: CASE_META.C?.color ?? '#FA9E33',
    D: CASE_META.D?.color ?? '#23B761',
    E: CASE_META.E?.color ?? '#F0524D',
  };

  const nodes = useMemo(
    () => [
      {
        id: 'q1',
        position: { x: QUESTION_X, y: 0 },
        type: 'question',
        style: { width: QUESTION_W, height: QUESTION_H },
        data: {
          text: 'W₀ = ∅ ?',
          onHover: (e) => handleMouseEnter('q1', e),
          onLeave: handleMouseLeave,
        },
      },
      {
        id: 'a',
        position: { x: LEAF_X, y: 0 },
        type: 'leaf',
        data: {
          text: 'Case A: V-only',
          color: caseColors.A,
          hovered: hoveredNode === 'a',
          onHover: (e) => handleMouseEnter('a', e),
          onLeave: handleMouseLeave,
        },
        style: { width: LEAF_W, height: LEAF_H },
      },
      {
        id: 'q2',
        position: { x: QUESTION_X, y: ROW_GAP },
        type: 'question',
        style: { width: QUESTION_W, height: QUESTION_H },
        data: {
          text: 'V₀ = ∅ ?',
          onHover: (e) => handleMouseEnter('q2', e),
          onLeave: handleMouseLeave,
        },
      },
      {
        id: 'b',
        position: { x: LEAF_X, y: ROW_GAP },
        type: 'leaf',
        data: {
          text: 'Case B: W-only',
          color: caseColors.B,
          hovered: hoveredNode === 'b',
          onHover: (e) => handleMouseEnter('b', e),
          onLeave: handleMouseLeave,
        },
        style: { width: LEAF_W, height: LEAF_H },
      },
      {
        id: 'q3',
        position: { x: QUESTION_X, y: ROW_GAP * 2 },
        type: 'question',
        style: { width: QUESTION_W, height: QUESTION_H },
        data: {
          text: 'Cross V/W gens?',
          onHover: (e) => handleMouseEnter('q3', e),
          onLeave: handleMouseLeave,
        },
      },
      {
        id: 'c',
        position: { x: LEAF_X, y: ROW_GAP * 2 },
        type: 'leaf',
        data: {
          text: 'Case C: Correlated',
          color: caseColors.C,
          hovered: hoveredNode === 'c',
          onHover: (e) => handleMouseEnter('c', e),
          onLeave: handleMouseLeave,
        },
        style: { width: LEAF_W, height: LEAF_H },
      },
      {
        id: 'q4',
        position: { x: QUESTION_X, y: ROW_GAP * 3 },
        type: 'question',
        style: { width: QUESTION_W, height: QUESTION_H },
        data: {
          text: 'Gₐ = Sym(Lₐ) ?',
          onHover: (e) => handleMouseEnter('q4', e),
          onLeave: handleMouseLeave,
        },
      },
      {
        id: 'd',
        position: { x: LEAF_X, y: ROW_GAP * 3 },
        type: 'leaf',
        data: {
          text: 'Case D: Cross (Young)',
          color: caseColors.D,
          hovered: hoveredNode === 'd',
          onHover: (e) => handleMouseEnter('d', e),
          onLeave: handleMouseLeave,
        },
        style: { width: LEAF_W, height: LEAF_H },
      },
      {
        id: 'e',
        position: { x: QUESTION_X + LEAF_CENTER_OFFSET, y: ROW_GAP * 4 },
        type: 'leaf',
        data: {
          text: 'Case E: Cross (general)',
          color: caseColors.E,
          hovered: hoveredNode === 'e',
          onHover: (e) => handleMouseEnter('e', e),
          onLeave: handleMouseLeave,
        },
        style: { width: LEAF_W, height: LEAF_H },
      },
    ],
    [caseColors, hoveredNode],
  );

  const edges = useMemo(
    () => [
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

  return (
    <details open className="rounded-lg border border-gray-200 bg-white">
      <summary className="cursor-pointer rounded-lg bg-gray-50 px-3.5 py-2.5 text-xs font-semibold text-gray-600">
        Deep dive: Classification decision tree
      </summary>
      <div className="relative p-2" ref={wrapRef}>
        <div className="h-[440px] w-full min-w-0">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={dtNodeTypes}
            className="h-full w-full"
            defaultEdgeOptions={{ type: 'step', style: { strokeWidth: 2 } }}
            fitViewOptions={{ padding: 0.15 }}
            fitView
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
    </details>
  );
}
