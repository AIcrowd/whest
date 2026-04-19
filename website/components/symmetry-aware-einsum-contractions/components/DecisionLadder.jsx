import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Handle, Panel, Position, ReactFlow, ReactFlowProvider, useReactFlow } from '@xyflow/react';
import PanZoomControls, { PanZoomHint } from './PanZoomControls.jsx';
import '@xyflow/react/dist/style.css';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
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

// Horizontal padding: must be large enough to clear the q_crossVW → BFO
// edge arc in Stage 2 (exits q_crossVW's right, loops right-down-left back
// into BFO's right handle). Kept at 120 for arc clearance.
// Vertical padding: smaller — the dashed band just needs to breathe above
// the first row and below the last row; no arcs cross top/bottom.
const BAND_PAD_X = 120;
const BAND_PAD_Y = 60;
const BAND_X = -BAND_PAD_X;
const BAND_WIDTH = QUESTION_X + QUESTION_W + 2 * BAND_PAD_X;

const STAGE_1_TOP_Y = 0;
const SOURCE_Y = BAND_PAD_Y;                         // top pad
const Q1_Y = SOURCE_Y + 88;                          // q_trivial
const Q2_Y = Q1_Y + ROW_GAP;                         // q_hasW
const Q3_Y = Q2_Y + ROW_GAP;                         // q_hasV
const STAGE_1_BOTTOM_Y = Q3_Y + LEAF_H + BAND_PAD_Y; // bottom pad

const ENUMERATE_Y = STAGE_1_BOTTOM_Y + 12;
const ENUMERATE_H = 44;
const STAGE_2_TOP_Y = ENUMERATE_Y + ENUMERATE_H + 32;
const Q_SINGLETON_Y = STAGE_2_TOP_Y + BAND_PAD_Y;    // top pad
const Q_DIRECT_Y = Q_SINGLETON_Y + ROW_GAP;          // q_direct (F-check) — post-dimino
const Q_CROSSVW_Y = Q_DIRECT_Y + ROW_GAP;            // q_crossVW
const Q_FULLSYM_Y = Q_CROSSVW_Y + ROW_GAP;           // q_fullSym
const STAGE_2_BOTTOM_Y = Q_FULLSYM_Y + ROW_GAP + LEAF_H + BAND_PAD_Y; // bottom pad

const EDGE_YES = { color: '#23B761', label: 'yes' };
const EDGE_NO = { color: '#F0524D', label: 'no' };

// Edge-label offsets. Once the label background went transparent (so the
// dashed stage bands read through), the naturally centered label text
// started overlapping the stroke it was annotating. These offsets shift
// the text perpendicular to its edge so the yes/no "sits" next to the
// line instead of crossing it:
//
//   VERTICAL edges  (spine: question → question, question → enumerate,
//                    ENUM → Q_singleton, singleton → direct, etc.)
//     → label moves to the RIGHT of the vertical line
//
//   HORIZONTAL edges (side: question → leaf, also the crossVW → BFO
//                     right-hand loop whose midpoint sits on a vertical
//                     segment of the step path — treated as vertical)
//     → label moves ABOVE the horizontal line
//
// Applied via the labelStyle `transform` CSS property. Because
// labelBgStyle is transparent the invisible background rect stays put
// at the edge midpoint — only the visible text shifts.
const LABEL_OFFSET_VERTICAL = { transform: 'translate(18px, 0)' };
const LABEL_OFFSET_HORIZONTAL = { transform: 'translate(0, -11px)' };

// ─── Decision spec — mirrors the engine's `detectShape` order ─────────
//
// engine/shapeLayer.js checks (in order):
//     |G| ≤ 1  → trivial
//     wa == [] → allVisible
//     va == [] → allSummed
//     else     → mixed (drop into Stage 2)
//
// Stage 1 here asks the same three questions in the same order so the UI
// narrative matches the code path the engine actually takes. Beyond
// pedagogy this matters because activeLeafIds are driven by the engine's
// output: if the ladder's path disagreed with the engine's, the reader
// could highlight one leaf while having arrived there via a different
// branching story than the code.
//
// Each QUESTION carries three tooltip fields that the `tooltipFor`
// resolver stitches into labeled sections:
//
//   checks    — precisely what's tested, in engine terms.
//   why       — the connection between the yes/no answer and the
//               closed-form (or fallback) that gates it.
//   intuition — common misconceptions, concrete preset examples, and
//               the subtleties careful readers trip on. Optional.
//
// These were originally squeezed into a single `long` string; splitting
// them makes the tooltip easier to scan and lets each section stand on
// its own (especially "intuition", which is where the reader builds the
// model that prevents future confusion).

const QUESTIONS = [
  {
    id: 'q_trivial',
    short: '|G| = 1 ?',
    checks: 'Does the σ-loop produce any non-identity generator for $G_{\\text{pt}}$? This is the cheap pre-dimino version of the check: if Sources A+B yield zero generators (or only the identity), $|G|=1$ is known immediately; otherwise dimino is run later to materialise the full group.',
    why: 'With no symmetry no two tuples are equivalent, so every full assignment is its own accumulation bin. $\\alpha$ collapses to the product of all label sizes, $\\prod_{\\ell \\in L} n_\\ell$, and there is nothing further to classify — regardless of $V$/$W$ shape.',
    intuition: 'Generic operands with no declared axis symmetries, and no repeated-operand structure the σ-loop can exploit, land here. Examples: $\\texttt{frobenius}$ ($\\texttt{ij,ij→}$ — two identical $A$\'s but matching subscripts kill the swap), $\\texttt{matrix-chain}$ ($\\texttt{ij,jk→ik}$ — identical $A$\'s but different subscript structure), $\\texttt{mixed-chain}$ ($A$,$B$,$A$ with $B$ breaking the chain).',
    onTrue: 'trivial', onFalse: 'q_hasW',
    stage: 1,
  },
  {
    id: 'q_hasW',
    short: 'W ≠ ∅ ?',
    checks: 'Purely structural: does the einsum sum over any label, i.e. does any label appear in the input subscripts but not in the output? $|W| > 0$ tests that.',
    why: 'If $W = \\emptyset$, every label is free; the expression is an elementwise product, each output cell is touched exactly once, and $\\alpha$ collapses to $\\prod_{\\ell \\in V} n_\\ell$ — no orbit work needed.',
    intuition: 'Examples with $W = \\emptyset$ (all labels free): $\\texttt{outer}$ ($\\texttt{ab,cd→abcd}$), $\\texttt{triangle}$, $\\texttt{four-cycle}$, $\\texttt{triple-outer}$, $\\texttt{declared-c3}$. These are "visible-only" einsums.',
    onTrue: 'q_hasV', onFalse: 'allVisible',
    stage: 1,
  },
  {
    id: 'q_hasV',
    short: 'V ≠ ∅ ?',
    checks: 'Purely structural: does the output expression contain any labels? $|V| > 0$ tests that.',
    why: 'If $V = \\emptyset$, the output is a single scalar, so there is exactly one accumulation bin and $\\alpha$ equals $|X/G|$ — the classical Burnside orbit count on the full tuple space.',
    intuition: 'Examples with $V = \\emptyset$: $\\texttt{frobenius}$ ($\\texttt{ij,ij→}$) and $\\texttt{trace-product}$ ($\\texttt{ij,ji→}$) — both produce scalars. "Yes" here means both $V$ and $W$ are populated and $G$ is non-trivial; this is the only branch that pays the cost of dimino and proceeds into Stage 2.',
    onTrue: 'ENUMERATE', onFalse: 'allSummed',
    stage: 1,
  },
  {
    id: 'q_singleton',
    short: '|V| = 1 ?',
    checks: 'Does the output have exactly one axis on this component? $|V_c| = 1$.',
    why: 'With a single free label, the $V$-projection of every orbit is just that label\'s $G$-orbit — small and structured. A closed-form "point-stabilizer weighted Burnside" gives $\\alpha$ in one shot, with no orbit enumeration on the full tuple space.',
    intuition: 'Examples: $\\texttt{cross-s3}$ ($\\texttt{ijk→i}$ with $T$ fully symmetric — $S_3$ on $\\{i,j,k\\}$ reducing to a vector) and $\\texttt{cyclic-cross}$ (same shape with $C_3$). Any $\\texttt{...→i}$ einsum with non-trivial $G$ that mixes $i$ with summed labels lands here.',
    onTrue: 'singleton', onFalse: 'q_direct',
    stage: 2,
  },
  {
    id: 'q_direct',
    short: 'F-check passes ?',
    checks: 'Three post-dimino conditions, all required: (1) no element of $G$ maps a $V$-label to a $W$-label (no cross), (2) $|G| = |G_V| \\cdot |G_W|$ where $G_V$, $G_W$ are the projections onto $V$ and $W$, and (3) both $|G_V| > 1$ and $|G_W| > 1$ (meaningfulness guard).',
    why: 'All three together say $G$ is exactly the direct product of its $V$-action and its $W$-action, with no coupling. $\\alpha$ then decomposes: for each $V$-tuple, Burnside runs independently on the $W$-side. Closed form: $\\alpha = (\\prod_{\\ell \\in V} n_\\ell) \\cdot |[n]^W / G_W|$.',
    intuition: '"No cross" alone is not enough — $\\texttt{bilinear-trace}$ has no cross elements but $|G|=2 \\neq |G_V| \\cdot |G_W| = 4$ because its single non-identity element $(i\\;j)(k\\;l)$ is a *coupled* $\\mathbb{Z}_2$ (swapping $V$ forces swapping $W$). Passing examples: $\\texttt{four-A-grid}$, $\\texttt{direct-s2-s2}$, $\\texttt{direct-s2-c3}$, $\\texttt{direct-s3-s2}$.',
    onTrue: 'directProduct', onFalse: 'q_crossVW',
    stage: 2,
  },
  {
    id: 'q_crossVW',
    short: 'Cross-V/W element ?',
    checks: 'Does any element $g \\in G$ move at least one $V$-label to a $W$-label (or vice versa)? An element-level scan: for each $g$, does there exist $\\ell \\in V$ with $g(\\ell) \\in W$?',
    why: 'The Young regime (the leaf reached via "yes" → "G = Sym(L_c)" → "yes") requires at least one cross element — its multinomial closed form assumes $G$ genuinely mixes $V$ and $W$. Without cross, the Young regime cannot fire and brute-force orbit is the only remaining option.',
    intuition: 'Reaching this node with answer "no" happens when the F-check above refused for a *numeric* reason, not because of cross. Examples: $\\texttt{four-A-grid}$\'s $S_2\\{a,b\\} \\times S_2\\{i,j\\}$ has zero cross elements, but it already fired the direct-product regime at the F-check and never reaches here. $\\texttt{bilinear-trace}$\'s coupled $\\mathbb{Z}_2$ also has zero cross elements, but the F-check refused on $|G| = 2 \\neq |G_V| \\cdot |G_W| = 4$ — that is a "no-cross but reach here" case, and it falls straight to brute-force orbit. Cross elements themselves arise from declared axis symmetries spanning $V$ and $W$ on an operand, or from identical-operand swaps pairing a $V$-label with a $W$-label.',
    onTrue: 'q_fullSym',
    onFalse: 'bruteForceOrbit',
    stage: 2,
  },
  {
    id: 'q_fullSym',
    short: 'G = Sym(L_c) ?',
    checks: 'Is $G$ the full symmetric group on the component\'s label set — every permutation of $L_c$? The cardinality test $|G| = |L_c|!$ decides this.',
    why: 'When $G = \\mathrm{Sym}(L)$, the pointwise $V$-stabilizer is exactly the Young subgroup $\\mathrm{Sym}(W)$. The orbit count collapses to a multinomial: $\\alpha = n_L^{|V|} \\cdot \\binom{n_L + |W| - 1}{|W|}$, with no orbit enumeration.',
    intuition: 'Examples that fire the Young regime: $\\texttt{young-s3}$ ($\\texttt{abc→ab}$ with $T$ fully symmetric, $|G|=6=3!$), $\\texttt{young-s4-v2w2}$, $\\texttt{young-s4-v3w1}$. Examples with cross elements but $G \\neq \\mathrm{Sym}$: $\\texttt{cross-c3-partial}$ ($C_3 \\subsetneq S_3$), $\\texttt{cross-s2}$ ($S_2 \\subsetneq S_3$ on a 3-label component). These fall to brute-force orbit because the closed form requires the full symmetric group.',
    onTrue: 'young',
    onFalse: 'bruteForceOrbit',
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
      <Handle id="right" type="source" position={Position.Right} className="pointer-events-none opacity-0" />
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
  // Allow 2-line wrap for longer leaf labels (e.g. the Young regime's
  // "Young subgroup (full Sym, cross V/W)"). Shorter labels render on one
  // line as usual because justify-center + items-center vertically centers.
  return (
    <div
      className="box-border flex h-full w-full cursor-help items-center justify-center whitespace-normal break-words rounded-lg px-2 py-0.5 text-center text-xs font-bold leading-tight shadow-sm transition-all hover:shadow"
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

function tooltipFor(nodeId, liveReasonsByLeaf = null) {
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
      // Structured sections replace the old single `body` string for
      // question nodes. The render path displays `checks`, `why`, and
      // `intuition` under their own captioned headings when present.
      checks: q.checks,
      why: q.why,
      intuition: q.intuition,
      latex: null,
      color: q.stage === 1 ? '#16A34A' : '#8B5CF6',
    };
  }
  const spec = specFor(nodeId);
  if (spec) {
    const liveReasons = liveReasonsByLeaf?.get?.(nodeId) ?? [];
    return {
      title: spec.label,
      whenText: spec.when,
      body: spec.description,
      latex: spec.latex,
      glossary: spec.glossary,
      color: spec.color,
      liveReasons,
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

  // leafNode creates a ReactFlow node for a regime/shape leaf. Pass a
  // non-null `nodeIdSuffix` to make the instance visually distinct while
  // still pulling label/color from the regime/shape spec. This is used when
  // the same regime appears as a leaf in multiple branches (e.g.,
  // bruteForceOrbit is reachable from both q_crossVW "no" and q_fullSym "no").
  function leafNode(leafId, y, centered = false, nodeIdSuffix = null) {
    const spec = specFor(leafId);
    const nodeId = nodeIdSuffix ? `${leafId}__${nodeIdSuffix}` : leafId;
    return {
      id: nodeId,
      position: { x: centered ? QUESTION_X + LEAF_CENTER_OFFSET : LEAF_X, y },
      type: 'leaf',
      style: { width: LEAF_W, height: LEAF_H },
      data: {
        text: spec.label,
        color: spec.color,
        active: active.has(leafId),
        spotlight: spotlight.has(leafId),
        nodeId: leafId, // keep the canonical id for active/spotlight testing
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
  const spineYs = [Q1_Y, Q2_Y, Q3_Y];
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
        labelStyle: { fontSize: 11, fontWeight: 700, fill: prevPlan.spineEdge.color, ...LABEL_OFFSET_VERTICAL },
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
        labelStyle: { fontSize: 11, fontWeight: 700, fill: plan.sideEdge.color, ...LABEL_OFFSET_HORIZONTAL },
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
    labelStyle: { fontSize: 11, fontWeight: 700, fill: lastPlan.spineEdge.color, ...LABEL_OFFSET_VERTICAL },
    style: { stroke: lastPlan.spineEdge.color, strokeWidth: 1.5 },
  });

  // Stage 2 — q_singleton → q_direct (F-check) → q_crossVW → q_fullSym.
  // All four questions need post-Dimino group elements.
  const singletonQ = QUESTIONS.find((q) => q.id === 'q_singleton');
  nodes.push({
    id: singletonQ.id,
    position: { x: QUESTION_X, y: Q_SINGLETON_Y },
    type: 'question',
    style: { width: QUESTION_W, height: QUESTION_H },
    data: { text: singletonQ.short, nodeId: singletonQ.id },
  });
  edges.push({
    id: `enumerate-${singletonQ.id}`,
    source: 'enumerate', sourceHandle: 'bottom',
    target: singletonQ.id, targetHandle: 'top',
    style: { stroke: '#8B5CF6', strokeWidth: 1.5, strokeDasharray: '6 3' },
  });

  // q_singleton yes → singleton leaf on the left.
  nodes.push(leafNode('singleton', Q_SINGLETON_Y));
  edges.push({
    id: `${singletonQ.id}-singleton`,
    source: singletonQ.id, sourceHandle: 'side',
    target: 'singleton', targetHandle: 'right',
    label: EDGE_YES.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_YES.color, ...LABEL_OFFSET_HORIZONTAL },
    style: { stroke: EDGE_YES.color, strokeWidth: 1.5 },
  });

  // q_singleton no → q_direct (F-check) on the spine.
  const directQ = QUESTIONS.find((q) => q.id === 'q_direct');
  nodes.push({
    id: directQ.id,
    position: { x: QUESTION_X, y: Q_DIRECT_Y },
    type: 'question',
    style: { width: QUESTION_W, height: QUESTION_H },
    data: { text: directQ.short, nodeId: directQ.id },
  });
  edges.push({
    id: `${singletonQ.id}-${directQ.id}`,
    source: singletonQ.id, sourceHandle: 'bottom',
    target: directQ.id, targetHandle: 'top',
    label: EDGE_NO.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_NO.color, ...LABEL_OFFSET_VERTICAL },
    style: { stroke: EDGE_NO.color, strokeWidth: 1.5 },
  });

  // q_direct yes → directProduct leaf on the left.
  nodes.push(leafNode('directProduct', Q_DIRECT_Y));
  edges.push({
    id: `${directQ.id}-directProduct`,
    source: directQ.id, sourceHandle: 'side',
    target: 'directProduct', targetHandle: 'right',
    label: EDGE_YES.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_YES.color, ...LABEL_OFFSET_HORIZONTAL },
    style: { stroke: EDGE_YES.color, strokeWidth: 1.5 },
  });

  // q_direct no → q_crossVW on the spine.
  const crossVWQ = QUESTIONS.find((q) => q.id === 'q_crossVW');
  nodes.push({
    id: crossVWQ.id,
    position: { x: QUESTION_X, y: Q_CROSSVW_Y },
    type: 'question',
    style: { width: QUESTION_W, height: QUESTION_H },
    data: { text: crossVWQ.short, nodeId: crossVWQ.id },
  });
  edges.push({
    id: `${directQ.id}-${crossVWQ.id}`,
    source: directQ.id, sourceHandle: 'bottom',
    target: crossVWQ.id, targetHandle: 'top',
    label: EDGE_NO.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_NO.color, ...LABEL_OFFSET_VERTICAL },
    style: { stroke: EDGE_NO.color, strokeWidth: 1.5 },
  });

  // Single bruteForceOrbit leaf serves BOTH "no" branches (q_crossVW-no and
  // q_fullSym-no). Positioned at the bottom of the tree (centered under
  // the spine, below q_fullSym) so:
  //   - q_fullSym-no connects cleanly via a short vertical edge from its
  //     bottom handle into BFO's top handle.
  //   - q_crossVW-no exits the RIGHT side of the question, sweeps down and
  //     around the spine, and connects into BFO's right handle.
  const bruteY = Q_FULLSYM_Y + ROW_GAP;
  nodes.push(leafNode('bruteForceOrbit', bruteY, true));
  edges.push({
    id: `${crossVWQ.id}-bruteForceOrbit`,
    source: crossVWQ.id, sourceHandle: 'right',
    target: 'bruteForceOrbit', targetHandle: 'right',
    label: EDGE_NO.label,
    // This edge's step path has a long vertical segment on the RIGHT side
    // of the tree, so the midpoint sits on a vertical line — treat it as
    // a vertical edge and shift the label horizontally.
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_NO.color, ...LABEL_OFFSET_VERTICAL },
    style: { stroke: EDGE_NO.color, strokeWidth: 1.5 },
  });

  // q_crossVW yes → q_fullSym question on the spine.
  const fullSymQ = QUESTIONS.find((q) => q.id === 'q_fullSym');
  nodes.push({
    id: fullSymQ.id,
    position: { x: QUESTION_X, y: Q_FULLSYM_Y },
    type: 'question',
    style: { width: QUESTION_W, height: QUESTION_H },
    data: { text: fullSymQ.short, nodeId: fullSymQ.id },
  });
  edges.push({
    id: `${crossVWQ.id}-${fullSymQ.id}`,
    source: crossVWQ.id, sourceHandle: 'bottom',
    target: fullSymQ.id, targetHandle: 'top',
    label: EDGE_YES.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_YES.color, ...LABEL_OFFSET_VERTICAL },
    style: { stroke: EDGE_YES.color, strokeWidth: 1.5 },
  });

  // q_fullSym yes → young leaf on the left.
  nodes.push(leafNode('young', Q_FULLSYM_Y));
  edges.push({
    id: `${fullSymQ.id}-young`,
    source: fullSymQ.id, sourceHandle: 'side',
    target: 'young', targetHandle: 'right',
    label: EDGE_YES.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_YES.color, ...LABEL_OFFSET_HORIZONTAL },
    style: { stroke: EDGE_YES.color, strokeWidth: 1.5 },
  });

  // q_fullSym no → same single bruteForceOrbit leaf at the bottom. Short
  // vertical edge from q_fullSym's bottom handle into BFO's top handle.
  edges.push({
    id: `${fullSymQ.id}-bruteForceOrbit`,
    source: fullSymQ.id, sourceHandle: 'bottom',
    target: 'bruteForceOrbit', targetHandle: 'top',
    label: EDGE_NO.label,
    labelStyle: { fontSize: 11, fontWeight: 700, fill: EDGE_NO.color, ...LABEL_OFFSET_VERTICAL },
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
        defaultEdgeOptions={{
          type: 'step',
          style: { strokeWidth: 2 },
          // Kill the default white rectangle ReactFlow draws behind every
          // edge label. The yes/no text sits directly on the dashed stage
          // bands, and the opaque (even low-opacity) rect breaks the band's
          // visual continuity. `fillOpacity: 0` plus zero padding leaves
          // just the letterform glyphs, which read cleanly against both
          // stage bands and the white page background.
          labelBgStyle: { fill: 'transparent', fillOpacity: 0 },
          labelBgPadding: [0, 0],
          labelBgBorderRadius: 0,
        }}
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
  // Optional: { Map<leafId, string[]> } — verdict.reason strings collected
  // from each component's accumulation.trace. When the user hovers a leaf
  // that has live reasons, the tooltip appends them so the reader sees the
  // concrete numbers (e.g. "estimate 800,000,000 exceeds budget").
  liveReasonsByLeaf = null,
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
    // Bumped from 340 → 520: the restructured question tooltips carry
    // three stacked sections (Checks · Why · Intuition), and the
    // intuition paragraphs for q_crossVW / q_direct run long by design
    // because they're where readers build the missing mental model.
    // Under-estimating the height made the placer flip tooltips above
    // short nodes even when plenty of room below would have fit.
    const tooltipH = 520;
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

  const activeTooltip = hoveredNode ? tooltipFor(hoveredNode, liveReasonsByLeaf) : null;

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
              <InlineMathText>
                {activeTooltip.whenText.toLowerCase().startsWith('when')
                  ? activeTooltip.whenText
                  : `When: ${activeTooltip.whenText}`}
              </InlineMathText>
            </div>
          )}
          {activeTooltip.body && (
            <div className="whitespace-normal break-words text-sm leading-6 text-gray-300">
              <InlineMathText>{activeTooltip.body}</InlineMathText>
            </div>
          )}
          {activeTooltip.checks && (
            <div className="whitespace-normal break-words text-sm leading-6 text-gray-300">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-500">
                What's checked
              </div>
              <InlineMathText>{activeTooltip.checks}</InlineMathText>
            </div>
          )}
          {activeTooltip.why && (
            <div className="mt-3 whitespace-normal break-words border-t border-gray-700 pt-3 text-sm leading-6 text-gray-300">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-500">
                Why it matters
              </div>
              <InlineMathText>{activeTooltip.why}</InlineMathText>
            </div>
          )}
          {activeTooltip.intuition && (
            <div className="mt-3 whitespace-normal break-words border-t border-gray-700 pt-3 text-sm leading-6 text-gray-300">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-500">
                Intuition
              </div>
              <InlineMathText>{activeTooltip.intuition}</InlineMathText>
            </div>
          )}
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
          {activeTooltip.liveReasons && activeTooltip.liveReasons.length > 0 && (
            <div className="mt-3 whitespace-normal break-words border-t border-gray-700 pt-3 text-[11px] leading-relaxed text-gray-300">
              <div className="mb-1.5 text-[10px] uppercase tracking-wider text-gray-500">
                Live for this example
              </div>
              <ul className="list-none space-y-0.5 font-mono text-[11px] text-gray-300">
                {activeTooltip.liveReasons.map((r, i) => (
                  <li key={i}>· {r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
