/**
 * Bipartite graph rendered as polished SVG with bezier edges,
 * labeled group boxes for U axis-classes (left), V free and W summed (right).
 */

import PanZoomCanvas from './PanZoomCanvas.jsx';
import Latex from './Latex.jsx';
import {
  explorerThemeColor,
  explorerThemeTint,
  getExplorerThemeOperandPalette,
} from '../lib/explorerTheme.js';
import {
  getActiveExplorerThemeId,
  notationLatex,
} from '../lib/notationSystem.js';

export default function BipartiteGraph({ graph, example, variableColors, highlightedLabels = new Set() }) {
  const explorerThemeId = getActiveExplorerThemeId();
  const V_COLOR = explorerThemeColor(explorerThemeId, 'hero');
  const W_COLOR = explorerThemeColor(explorerThemeId, 'summedSide');
  const U_FALLBACK_COLOR = explorerThemeColor(explorerThemeId, 'heroMuted');
  const HIGHLIGHT_COLOR = explorerThemeColor(explorerThemeId, 'heroMuted');
  const MUTED_COLOR = explorerThemeColor(explorerThemeId, 'muted');
  const BORDER_COLOR = explorerThemeColor(explorerThemeId, 'border');
  const isHighlighted = (label) =>
    highlightedLabels instanceof Set
      ? highlightedLabels.has(label)
      : Array.isArray(highlightedLabels) && highlightedLabels.includes(label);
  const { uVertices, incidence, freeLabels, summedLabels, identicalGroups } = graph;

  const vLabels = [...freeLabels].sort();
  const wLabels = [...summedLabels].sort();

  // ── Layout constants ──
  const leftX = 150;
  const rightX = 520;
  const ySpacing = 46;
  const nodeR = 15;
  const groupGap = 50;
  const topMargin = 30; // space above first box for labels

  // ── Left: U-vertices, grouped by operand ──
  const opGroups = {};
  uVertices.forEach((u, i) => {
    (opGroups[u.opIdx] ??= []).push(i);
  });
  const opKeys = Object.keys(opGroups).sort((a, b) => a - b);

  // Compute pill widths for each U-vertex
  const charW = 7.5;
  const uPillWidths = uVertices.map(u => {
    const labelStr = [...u.labels].sort().join(',');
    return Math.max(32, labelStr.length * charW + 18);
  });

  // Left box width adapts to widest pill
  const maxPillW = Math.max(32, ...uPillWidths);

  // Compute left positions: stack operand groups with gap
  const uPositions = new Array(uVertices.length);
  let leftY = topMargin + 20;
  const opBoxes = [];
  const leftBoxPadX = 24;
  const leftBoxPadY = 24;
  const leftBoxW = maxPillW + 10; // pad around widest pill

  opKeys.forEach((opIdx) => {
    const indices = opGroups[opIdx];
    const vcName = example.operandNames?.[opIdx];
    const vc = variableColors?.[vcName];
    const colors = vc
      ? { fill: `${vc.color}08`, stroke: `${vc.color}4D` }
      : identicalGroups.length > 0
        ? getGroupColor(opIdx, identicalGroups, explorerThemeId)
        : {
            fill: explorerThemeTint(explorerThemeId, 'heroMuted', 0.04),
            stroke: explorerThemeTint(explorerThemeId, 'heroMuted', 0.25),
          };

    const boxTop = leftY;
    indices.forEach((uIdx, j) => {
      uPositions[uIdx] = boxTop + leftBoxPadY + nodeR + j * ySpacing;
    });
    const boxBottom = boxTop + leftBoxPadY * 2 + nodeR * 2 + (indices.length - 1) * ySpacing;
    opBoxes.push({ opIdx, top: boxTop, bottom: boxBottom, indices, colors, vc });
    leftY = boxBottom + 16; // gap between operand groups
  });

  // ── Right: V and W positions ──
  const rightStartY = topMargin + 20;
  const rightBoxPadX = 28;
  const rightBoxPadY = 28;
  const rightBoxW = 100;

  const vPositions = vLabels.map((_, i) => rightStartY + rightBoxPadY + nodeR + i * ySpacing);
  const vBoxTop = rightStartY;
  const vBoxH = rightBoxPadY * 2 + nodeR * 2 + Math.max(0, vLabels.length - 1) * ySpacing;

  const wBoxGapTop = vBoxTop + vBoxH + groupGap;
  const wPositions = wLabels.map((_, i) => wBoxGapTop + rightBoxPadY + nodeR + i * ySpacing);
  const wBoxTop = wBoxGapTop;
  const wBoxH = wLabels.length > 0
    ? rightBoxPadY * 2 + nodeR * 2 + (wLabels.length - 1) * ySpacing
    : 50;

  // Map label → y
  const labelY = {};
  vLabels.forEach((l, i) => { labelY[l] = vPositions[i]; });
  wLabels.forEach((l, i) => { labelY[l] = wPositions[i]; });

  // ── SVG dimensions ──
  const rightEdge = rightX + rightBoxW / 2 + rightBoxPadX + 50;
  const leftBottom = opBoxes.length > 0 ? opBoxes[opBoxes.length - 1].bottom + 40 : 200;
  const rightBottom = wBoxTop + wBoxH + 40;
  const W = Math.max(660, rightEdge);
  const H = Math.max(300, Math.max(leftBottom, rightBottom) + 20);

  // Box x helpers
  const lbLeft = leftX - leftBoxW / 2 - leftBoxPadX;
  const lbFullW = leftBoxW + leftBoxPadX * 2;
  const rbLeft = rightX - rightBoxW / 2 - rightBoxPadX;
  const rbFullW = rightBoxW + rightBoxPadX * 2;

  // Bezier
  const bezier = (x1, y1, x2, y2) => {
    const cx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`;
  };

  return (
    <PanZoomCanvas
      className="graph-container h-[560px]"
      ariaLabel="Bipartite graph (zoomable)"
    >
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: W, height: H, maxWidth: 'none' }}>
        <defs>
          <filter id="node-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="1" stdDeviation="2" floodOpacity="0.07" />
          </filter>
        </defs>

        {/* ── U operand group boxes (left) ── */}
        {opBoxes.map(({ opIdx, top, bottom, colors, vc }) => {
          const opName = example.operandNames?.[opIdx] || `Op${opIdx}`;
          const label = `${opName}${example.operandNames?.filter((n) => n === opName).length > 1 ? ` (${opIdx})` : ''}`;
          return (
            <g key={`ubox-${opIdx}`}>
              <rect
                x={lbLeft} y={top}
                width={lbFullW} height={bottom - top}
                rx={14} fill={colors.fill}
              />
              <LabelBadge
                x={lbLeft + 12} y={top}
                text={label}
                color={vc?.color || U_FALLBACK_COLOR}
              />
            </g>
          );
        })}

        {/* ── V free group box (right) ── */}
        <rect
          x={rbLeft} y={vBoxTop}
          width={rbFullW} height={vBoxH}
          rx={14}
          fill={explorerThemeTint(explorerThemeId, 'hero', 0.03)}
          stroke={explorerThemeTint(explorerThemeId, 'hero', 0.2)}
          strokeWidth={1.5}
        />
        <MathLabelBadge
          x={rbLeft + 12}
          y={vBoxTop}
          math={notationLatex('v_free')}
          color={V_COLOR}
          width={92}
        />

        {/* ── W summed group box (right) ── */}
        {wLabels.length > 0 ? (
          <g>
            <rect
              x={rbLeft} y={wBoxTop}
              width={rbFullW} height={wBoxH}
              rx={14}
              fill={explorerThemeTint(explorerThemeId, 'summedSide', 0.03)}
              stroke={explorerThemeTint(explorerThemeId, 'summedSide', 0.2)}
              strokeWidth={1.5}
            />
            <MathLabelBadge
              x={rbLeft + 12}
              y={wBoxTop}
              math={notationLatex('w_summed')}
              color={W_COLOR}
              width={122}
            />
          </g>
        ) : (
          <g>
            <rect
              x={rbLeft} y={wBoxTop}
              width={rbFullW} height={wBoxH}
              rx={14}
              fill={explorerThemeTint(explorerThemeId, 'summedSide', 0.02)}
              stroke={explorerThemeTint(explorerThemeId, 'summedSide', 0.15)}
              strokeWidth={1.5}
              strokeDasharray="6,4"
            />
            <MathLabelBadge
              x={rbLeft + 12}
              y={wBoxTop}
              math={notationLatex('w_summed')}
              color={W_COLOR}
              width={122}
            />
            <text x={rightX} y={wBoxTop + wBoxH / 2 + 4}
              textAnchor="middle" fontSize={11} fill={MUTED_COLOR}
              fontFamily="'Inter', sans-serif" fontStyle="italic">
              (empty)
            </text>
          </g>
        )}

        {/* ── Bezier edges ── */}
        {uVertices.map((u, uIdx) => {
          const uy = uPositions[uIdx];
          const labelStr = [...u.labels].sort().join(',');
          const pillW = Math.max(32, labelStr.length * 7.5 + 18);
          const edgeStartX = leftX + pillW / 2 + 2;
          const eOpName = example.operandNames?.[u.opIdx];
          const eVc = variableColors?.[eOpName];
          const edgeColor = eVc?.color || BORDER_COLOR;
          return [...vLabels, ...wLabels].map((lbl) => {
            const mult = incidence[uIdx][lbl] || 0;
            if (mult === 0) return null;
            const ry = labelY[lbl];
            return (
              <path key={`e-${uIdx}-${lbl}`}
                d={bezier(edgeStartX, uy, rightX - nodeR - 4, ry)}
                fill="none" stroke={edgeColor}
                strokeWidth={Math.max(1.5, mult * 1.5)}
                opacity={0.45}
              />
            );
          });
        })}

        {/* ── U-vertex nodes (left) — pill-shaped, auto-sized ── */}
        {uVertices.map((u, i) => {
          const y = uPositions[i];
          const labelStr = [...u.labels].sort().join(',');
          const charW = 7.5;
          const pillW = Math.max(32, labelStr.length * charW + 18);
          const pillH = 28;
          const pillR = pillH / 2;
          const nOpName = example.operandNames?.[u.opIdx];
          const nVc = variableColors?.[nOpName];
          const nodeColor = nVc?.color || U_FALLBACK_COLOR;
          const hasSymmetry = nVc?.symmetry && nVc.symmetry !== 'none';
          return (
            <g key={`u-${i}`}>
              <rect x={leftX - pillW / 2} y={y - pillH / 2}
                width={pillW} height={pillH} rx={pillR}
                fill="white" stroke={nodeColor} strokeWidth={hasSymmetry ? 2.5 : 1.5}
                filter="url(#node-shadow)" />
              <text x={leftX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill={nodeColor} fontSize={11} fontWeight={500}
                fontFamily="'IBM Plex Mono', monospace">
                {labelStr}
              </text>
            </g>
          );
        })}

        {/* ── V label nodes (right) ── */}
        {vLabels.map((lbl, i) => {
          const y = vPositions[i];
          const hl = isHighlighted(lbl);
          return (
            <g key={`v-${lbl}`}>
              <circle cx={rightX} cy={y} r={nodeR}
                fill="white" stroke={hl ? HIGHLIGHT_COLOR : V_COLOR} strokeWidth={hl ? 3 : 1.5}
                filter="url(#node-shadow)" />
              <text x={rightX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill={hl ? HIGHLIGHT_COLOR : V_COLOR} fontSize={11} fontWeight={600}
                fontFamily="'IBM Plex Mono', monospace">
                {lbl}
              </text>
            </g>
          );
        })}

        {/* ── W label nodes (right) ── */}
        {wLabels.map((lbl, i) => {
          const y = wPositions[i];
          const hl = isHighlighted(lbl);
          return (
            <g key={`w-${lbl}`}>
              <circle cx={rightX} cy={y} r={nodeR}
                fill="white" stroke={hl ? HIGHLIGHT_COLOR : W_COLOR} strokeWidth={hl ? 3 : 1.5}
                filter="url(#node-shadow)" />
              <text x={rightX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill={hl ? HIGHLIGHT_COLOR : W_COLOR} fontSize={11} fontWeight={600}
                fontFamily="'IBM Plex Mono', monospace">
                {lbl}
              </text>
            </g>
          );
        })}

        {/* ── Column labels ── */}
        <foreignObject x={leftX - 72} y={H - 30} width={144} height={28}>
          <div
            xmlns="http://www.w3.org/1999/xhtml"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              fontSize: '10px',
              fontWeight: 600,
              letterSpacing: '0.06em',
              color: MUTED_COLOR,
              textTransform: 'uppercase',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <span style={{ color: explorerThemeColor(explorerThemeId, 'heroMuted'), textTransform: 'none' }}>
              <Latex math={notationLatex('u_axis_classes')} />
            </span>
            <span>axis classes</span>
          </div>
        </foreignObject>
        <foreignObject x={rightX - 72} y={H - 30} width={144} height={28}>
          <div
            xmlns="http://www.w3.org/1999/xhtml"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              fontSize: '10px',
              fontWeight: 600,
              letterSpacing: '0.06em',
              color: MUTED_COLOR,
              textTransform: 'uppercase',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <span style={{ color: explorerThemeColor(explorerThemeId, 'summedSide'), textTransform: 'none' }}>
              <Latex math={notationLatex('l_labels')} />
            </span>
            <span>index labels</span>
          </div>
        </foreignObject>
      </svg>
    </PanZoomCanvas>
  );
}

/** Small label badge that sits on the top edge of a group box */
function LabelBadge({ x, y, text, color }) {
  return (
    <g>
      <text x={x} y={y - 4}
        fontSize={10} fontWeight={600} fill={color}
        fontFamily="'Inter', sans-serif" letterSpacing="0.06em">
        {text}
      </text>
    </g>
  );
}

function MathLabelBadge({ x, y, math, color, width }) {
  return (
    <g>
      <foreignObject x={x} y={y - 18} width={width} height={18}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-start',
            height: '18px',
            color,
            fontSize: '11px',
            lineHeight: 1,
            textTransform: 'none',
          }}
        >
          <Latex math={math} />
        </div>
      </foreignObject>
    </g>
  );
}

function getGroupColor(opIdx, identicalGroups, explorerThemeId) {
  const palette = getExplorerThemeOperandPalette(explorerThemeId);
  for (let gi = 0; gi < identicalGroups.length; gi++) {
    if (identicalGroups[gi].includes(Number(opIdx))) {
      const color = palette[gi % palette.length];
      return {
        fill: colorWithAlpha(color, '0F'),
        stroke: colorWithAlpha(color, '4D'),
      };
    }
  }
  return {
    fill: explorerThemeTint(explorerThemeId, 'heroMuted', 0.04),
    stroke: explorerThemeTint(explorerThemeId, 'heroMuted', 0.25),
  };
}

function colorWithAlpha(color, alphaHex) {
  return /^#[0-9a-f]{6}$/i.test(color) ? `${color}${alphaHex}` : color;
}
