import { useMemo } from 'react';
import { Handle, Position, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { explorerThemeColor, getActiveExplorerThemeId } from '../../lib/explorerTheme.js';
import { buildOrbitProjectionGraph } from './orbitProjectionLayout.js';

// OrbitProjectionGraph — info-vis-first unified diagram of one product orbit's
// projection π_V to its stored output representatives.
//
// Three columns, left to right:
//   1. Member dots (one per orbit member, derived from the orbit row's
//      outputs[].coeff so that members[i].repIndex tells us which rep that
//      member projects to).
//   2. The focused-orbit pill (one node showing orbit index, size, and the
//      branching factor = number of distinct reps reached).
//   3. Stored output rep nodes (one per rep, with a × weight badge).
//
// Edges:
//   - Each member -> the orbit center (thin static edge).
//   - The orbit center -> each rep (medium edge whose stroke width scales
//     with the rep's weight).
//
// The pure layout helper buildOrbitProjectionGraph lives in a sibling .js
// file so node-test can import it without a JSX loader.

// Members render as small monospace pills — chip language, not card language.
// 1px border, full radius, white-on-coloured fill so the orbit-side palette
// reads at a glance. Compare to design-system .chip rule (4px×10px padding,
// 11px text, 20px radius).
function MemberNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full items-center justify-center rounded-full border font-mono text-[11px]"
      style={{
        background: data.fill,
        borderColor: data.border,
        color: data.text,
      }}
      data-orbit-graph-node="member"
    >
      <Handle id="right" type="source" position={Position.Right} className="pointer-events-none opacity-0" />
      {data.label}
    </div>
  );
}

// Orbit-center pill — a soft inset card, not a heavy border. Matches the
// design-system .callout rule: gray-50 bg, 1px gray-200 border, no shadow.
// The branchSummary text uses the same gray-400 eyebrow voice as elsewhere.
function OrbitCenterNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full flex-col items-center justify-center rounded-lg border px-3 py-2 text-center leading-tight"
      style={{
        background: data.fill,
        borderColor: data.border,
        color: data.text,
      }}
      data-orbit-graph-node="orbit-center"
    >
      <Handle id="left" type="target" position={Position.Left} className="pointer-events-none opacity-0" />
      <Handle id="right" type="source" position={Position.Right} className="pointer-events-none opacity-0" />
      <div className="font-serif text-[13px] font-semibold">{data.title}</div>
      <div className="mt-0.5 font-mono text-[10px] opacity-70">size {data.size}</div>
      <div className="text-[9.5px] font-semibold uppercase tracking-[0.12em] opacity-70">{data.branchSummary}</div>
    </div>
  );
}

// Rep nodes — match the design-system chip rule (rounded-full, 1px border,
// font-mono, 11–12px). The × weight badge sits inline; no separate card.
function RepNode({ data }) {
  return (
    <div
      className="box-border flex h-full w-full items-center justify-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[12px]"
      style={{
        background: data.fill,
        borderColor: data.border,
        color: data.text,
      }}
      data-orbit-graph-node="rep"
    >
      <Handle id="left" type="target" position={Position.Left} className="pointer-events-none opacity-0" />
      <span className="font-semibold">{data.label}</span>
      <span className="opacity-70">{data.weightLabel}</span>
    </div>
  );
}

const opgNodeTypes = {
  member: MemberNode,
  orbitCenter: OrbitCenterNode,
  rep: RepNode,
};

export default function OrbitProjectionGraph({
  orbit = null,
  reachedReps = [],
  orbitIdx = 0,
  totalOrbits = 0,
}) {
  const themeId = getActiveExplorerThemeId();

  const { nodes, edges } = useMemo(
    () => buildOrbitProjectionGraph({ orbit, reachedReps, orbitIdx, totalOrbits, themeId }),
    [orbit, reachedReps, orbitIdx, totalOrbits, themeId],
  );

  if (nodes.length === 0) {
    return (
      <div
        className="flex h-[260px] w-full items-center justify-center text-[12px]"
        style={{ color: explorerThemeColor(themeId, 'muted') }}
        data-testid="orbit-projection-empty"
      >
        no orbit selected
      </div>
    );
  }

  // Float the graph in the prose flow — no card chrome, no shadow, no border.
  // Matches the visual rhythm of the μ + α-hard formula displays earlier in
  // Section 4: the diagram IS the visual, the surrounding card mustn't compete.
  return (
    <div
      className="h-[260px] w-full"
      data-testid="orbit-projection-graph"
    >
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={opgNodeTypes}
          className="h-full w-full"
          fitView
          fitViewOptions={{ padding: 0.15, maxZoom: 1.2, minZoom: 0.5 }}
          minZoom={0.4}
          maxZoom={2}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag={false}
          panOnScroll={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
        />
      </ReactFlowProvider>
    </div>
  );
}
