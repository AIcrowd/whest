import { CASE_META } from '../engine/componentDecomposition.js';
import CaseBadge from './CaseBadge.jsx';

const GRAPH_SIZE = 220;
const CENTER = GRAPH_SIZE / 2;
const ORBIT_R = 80;
const NODE_R = 14;
const COLOR_V = '#4A7CFF';
const COLOR_W = '#94A3B8';
const COMP_COLORS = ['#4A7CFF', '#23B761', '#FA9E33', '#7C3AED', '#F0524D'];

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

export function DecisionTree({ components = [] }) {
  return (
    <div className="space-y-3 rounded-xl border border-gray-200 bg-white p-4">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500">Decision Tree</div>
        <p className="mt-1 text-sm text-gray-700">
          Each independent component lands in one of the A-E cases depending on whether it only touches free labels,
          only summed labels, or crosses the V/W boundary.
        </p>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {components.map((comp, idx) => (
          <div key={`${comp.caseType}-${idx}`} className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2">
            <div className="flex items-center gap-2">
              <CaseBadge caseType={comp.caseType} size="xs" interactive={false} />
              <span className="text-sm font-medium text-gray-900">{CASE_META[comp.caseType]?.label ?? `Case ${comp.caseType}`}</span>
            </div>
            <div className="mt-1 text-xs text-gray-600">
              labels: <code className="font-mono">{(comp.labels ?? []).join(', ')}</code>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
