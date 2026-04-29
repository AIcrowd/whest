// Pure helper extracted from OrbitProjectionGraph.jsx so that node-test can
// import it directly (node --test cannot resolve .jsx without a loader).
//
// buildOrbitProjectionGraph turns a normalized orbit payload + its reachedReps
// into the {nodes, edges} shape ReactFlow expects. No React imports — all
// styling decisions are passed in via the themeId, and the consumer is
// expected to map the role names back to colours.

import { explorerThemeColor } from '../../lib/explorerTheme.js';
import { notationColor } from '../../lib/notationSystem.js';
import { readableTextOn } from '../../lib/nodeColorUtils.js';

export const NODE_W = 88;
export const MEMBER_NODE_H = 32;
export const ORBIT_NODE_W = 144;
export const ORBIT_NODE_H = 78;
export const REP_NODE_W = 132;
export const REP_NODE_H = 36;
export const COL_GAP = 84;
export const ROW_GAP = 24;

export const MEMBER_X = 0;
export const ORBIT_X = NODE_W + COL_GAP;
export const REP_X = ORBIT_X + ORBIT_NODE_W + COL_GAP;

export function buildOrbitProjectionGraph({
  orbit,
  reachedReps,
  orbitIdx,
  totalOrbits,
  themeId,
}) {
  if (!orbit || !Array.isArray(reachedReps) || reachedReps.length === 0) {
    return { nodes: [], edges: [] };
  }

  const memberFill = notationColor('m_component');
  const memberBorder = notationColor('m_component');
  const orbitFill = explorerThemeColor(themeId, 'surfaceInset');
  const orbitBorder = notationColor('m_component');
  const orbitText = explorerThemeColor(themeId, 'ink');
  const repFill = explorerThemeColor(themeId, 'surface');
  const repBorder = notationColor('h_output');
  const repText = notationColor('h_output');
  // WCAG-aware member text: white on dark member fills, dark on light fills.
  // Without this, the rotating notation palette could produce illegible
  // member labels (e.g. white text on a pale-yellow alphaFamily background).
  const memberText = readableTextOn(memberFill);
  const edgeColor = explorerThemeColor(themeId, 'muted');
  const projectionEdgeColor = notationColor('h_output');

  const members = orbit.members ?? [];
  const memberCount = members.length;
  const repCount = reachedReps.length;

  // Vertical centering: members and reps may have different counts, but the
  // orbit-center node sits at the midpoint of whichever column is taller.
  const memberColHeight = Math.max(memberCount, 1) * (MEMBER_NODE_H + ROW_GAP) - ROW_GAP;
  const repColHeight = Math.max(repCount, 1) * (REP_NODE_H + ROW_GAP) - ROW_GAP;
  const colMaxHeight = Math.max(memberColHeight, repColHeight, ORBIT_NODE_H);

  const memberStartY = (colMaxHeight - memberColHeight) / 2;
  const repStartY = (colMaxHeight - repColHeight) / 2;
  const orbitY = (colMaxHeight - ORBIT_NODE_H) / 2;

  const branchSummary = repCount === 1
    ? 'reaches 1 rep'
    : `branches to ${repCount} reps`;

  const nodes = [
    // Members (left column).
    ...members.map((_, idx) => ({
      id: `member-${idx}`,
      type: 'member',
      position: { x: MEMBER_X, y: memberStartY + idx * (MEMBER_NODE_H + ROW_GAP) },
      width: NODE_W,
      height: MEMBER_NODE_H,
      data: {
        label: `m${idx + 1}`,
        fill: memberFill,
        border: memberBorder,
        text: memberText,
      },
    })),
    // Focused-orbit pill (center).
    {
      id: 'orbit-center',
      type: 'orbitCenter',
      position: { x: ORBIT_X, y: orbitY },
      width: ORBIT_NODE_W,
      height: ORBIT_NODE_H,
      data: {
        title: typeof orbitIdx === 'number' && typeof totalOrbits === 'number'
          ? `Orbit ${orbitIdx + 1} of ${totalOrbits}`
          : 'Orbit',
        size: orbit.size ?? memberCount,
        branchSummary,
        fill: orbitFill,
        border: orbitBorder,
        text: orbitText,
      },
    },
    // Reps (right column).
    ...reachedReps.map((rep, idx) => ({
      id: `rep-${idx}`,
      type: 'rep',
      position: { x: REP_X, y: repStartY + idx * (REP_NODE_H + ROW_GAP) },
      width: REP_NODE_W,
      height: REP_NODE_H,
      data: {
        label: `Q${idx + 1}`,
        weightLabel: `(×${rep.weight ?? 1})`,
        fill: repFill,
        border: repBorder,
        text: repText,
      },
    })),
  ];

  const edges = [
    // Member -> orbit center (thin, neutral).
    ...members.map((_, idx) => ({
      id: `e-member-${idx}`,
      source: `member-${idx}`,
      target: 'orbit-center',
      sourceHandle: 'right',
      targetHandle: 'left',
      type: 'default',
      style: { stroke: edgeColor, strokeWidth: 1.2 },
    })),
    // Orbit center -> rep (medium, thickness ~ weight, projection-coloured).
    ...reachedReps.map((rep, idx) => ({
      id: `e-rep-${idx}`,
      source: 'orbit-center',
      target: `rep-${idx}`,
      sourceHandle: 'right',
      targetHandle: 'left',
      type: 'default',
      style: {
        stroke: projectionEdgeColor,
        strokeWidth: 1.4 + 0.6 * Math.min(rep.weight ?? 1, 6),
      },
    })),
  ];

  return { nodes, edges };
}
