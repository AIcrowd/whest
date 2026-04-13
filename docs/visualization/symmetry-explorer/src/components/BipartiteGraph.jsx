/**
 * Bipartite graph rendered as polished SVG with bezier edges,
 * labeled group boxes for U axis-classes (left), V free and W summed (right).
 */

export default function BipartiteGraph({ graph, example }) {
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

  opKeys.forEach((opIdx, gi) => {
    const indices = opGroups[opIdx];
    const colors = identicalGroups.length > 0
      ? getGroupColor(opIdx, identicalGroups)
      : { fill: 'rgba(250,158,51,0.04)', stroke: 'rgba(250,158,51,0.25)' };

    const boxTop = leftY;
    indices.forEach((uIdx, j) => {
      uPositions[uIdx] = boxTop + leftBoxPadY + nodeR + j * ySpacing;
    });
    const boxBottom = boxTop + leftBoxPadY * 2 + nodeR * 2 + (indices.length - 1) * ySpacing;
    opBoxes.push({ opIdx, top: boxTop, bottom: boxBottom, indices, colors });
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
    <div className="graph-container">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <defs>
          <filter id="node-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="1" stdDeviation="2" floodOpacity="0.07" />
          </filter>
        </defs>

        {/* ── U operand group boxes (left) ── */}
        {opBoxes.map(({ opIdx, top, bottom, colors }) => {
          const opName = example.operandNames?.[opIdx] || `Op${opIdx}`;
          const label = `${opName}${example.operandNames?.filter((n, i) => n === opName).length > 1 ? ` (${opIdx})` : ''}`;
          return (
            <g key={`ubox-${opIdx}`}>
              <rect
                x={lbLeft} y={top}
                width={lbFullW} height={bottom - top}
                rx={14} fill={colors.fill} stroke={colors.stroke} strokeWidth={1.5}
              />
              <LabelBadge
                x={lbLeft + 12} y={top}
                text={label}
                color="#FA9E33" bg="#F8F9F9"
              />
            </g>
          );
        })}

        {/* ── V free group box (right) ── */}
        <rect
          x={rbLeft} y={vBoxTop}
          width={rbFullW} height={vBoxH}
          rx={14} fill="rgba(74,124,255,0.03)" stroke="rgba(74,124,255,0.2)" strokeWidth={1.5}
        />
        <LabelBadge x={rbLeft + 12} y={vBoxTop} text="V (free)" color="#4A7CFF" bg="#F8F9F9" />

        {/* ── W summed group box (right) ── */}
        {wLabels.length > 0 ? (
          <g>
            <rect
              x={rbLeft} y={wBoxTop}
              width={rbFullW} height={wBoxH}
              rx={14} fill="rgba(35,183,97,0.03)" stroke="rgba(35,183,97,0.2)" strokeWidth={1.5}
            />
            <LabelBadge x={rbLeft + 12} y={wBoxTop} text="W (summed)" color="#23B761" bg="#F8F9F9" />
          </g>
        ) : (
          <g>
            <rect
              x={rbLeft} y={wBoxTop}
              width={rbFullW} height={wBoxH}
              rx={14} fill="rgba(35,183,97,0.02)"
              stroke="rgba(35,183,97,0.15)" strokeWidth={1.5} strokeDasharray="6,4"
            />
            <LabelBadge x={rbLeft + 12} y={wBoxTop} text="W (summed)" color="#23B761" bg="#F8F9F9" />
            <text x={rightX} y={wBoxTop + wBoxH / 2 + 4}
              textAnchor="middle" fontSize={11} fill="#AAACAD"
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
          return [...vLabels, ...wLabels].map((lbl) => {
            const mult = incidence[uIdx][lbl] || 0;
            if (mult === 0) return null;
            const ry = labelY[lbl];
            return (
              <path key={`e-${uIdx}-${lbl}`}
                d={bezier(edgeStartX, uy, rightX - nodeR - 4, ry)}
                fill="none" stroke="#D9DCDC"
                strokeWidth={Math.max(1.5, mult * 1.5)}
                opacity={0.7}
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
          return (
            <g key={`u-${i}`}>
              <rect x={leftX - pillW / 2} y={y - pillH / 2}
                width={pillW} height={pillH} rx={pillR}
                fill="white" stroke="#FA9E33" strokeWidth={1.5}
                filter="url(#node-shadow)" />
              <text x={leftX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill="#FA9E33" fontSize={11} fontWeight={500}
                fontFamily="'IBM Plex Mono', monospace">
                {labelStr}
              </text>
            </g>
          );
        })}

        {/* ── V label nodes (right) ── */}
        {vLabels.map((lbl, i) => {
          const y = vPositions[i];
          return (
            <g key={`v-${lbl}`}>
              <circle cx={rightX} cy={y} r={nodeR}
                fill="white" stroke="#4A7CFF" strokeWidth={1.5}
                filter="url(#node-shadow)" />
              <text x={rightX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill="#4A7CFF" fontSize={11} fontWeight={600}
                fontFamily="'IBM Plex Mono', monospace">
                {lbl}
              </text>
            </g>
          );
        })}

        {/* ── W label nodes (right) ── */}
        {wLabels.map((lbl, i) => {
          const y = wPositions[i];
          return (
            <g key={`w-${lbl}`}>
              <circle cx={rightX} cy={y} r={nodeR}
                fill="white" stroke="#23B761" strokeWidth={1.5}
                filter="url(#node-shadow)" />
              <text x={rightX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill="#23B761" fontSize={11} fontWeight={600}
                fontFamily="'IBM Plex Mono', monospace">
                {lbl}
              </text>
            </g>
          );
        })}

        {/* ── Column labels ── */}
        <text x={leftX} y={H - 8} textAnchor="middle" fill="#AAACAD" fontSize={10}
          fontFamily="'Inter', sans-serif" fontWeight={600} letterSpacing="0.06em">
          U (AXIS CLASSES)
        </text>
        <text x={rightX} y={H - 8} textAnchor="middle" fill="#AAACAD" fontSize={10}
          fontFamily="'Inter', sans-serif" fontWeight={600} letterSpacing="0.06em">
          INDEX LABELS
        </text>
      </svg>
    </div>
  );
}

/** Small label badge that sits on the top edge of a group box */
function LabelBadge({ x, y, text, color, bg }) {
  // Measure approximate width (10px per char is close enough for SVG)
  const w = text.length * 7.5 + 16;
  return (
    <g>
      <rect x={x} y={y - 10} width={w} height={18} rx={4} fill={bg} />
      <text x={x + 8} y={y + 3}
        fontSize={10} fontWeight={600} fill={color}
        fontFamily="'Inter', sans-serif" letterSpacing="0.06em">
        {text}
      </text>
    </g>
  );
}

function getGroupColor(opIdx, identicalGroups) {
  const fills = [
    'rgba(74,124,255,0.06)', 'rgba(35,183,97,0.06)',
    'rgba(250,158,51,0.06)', 'rgba(124,58,237,0.06)',
  ];
  const strokes = [
    'rgba(74,124,255,0.3)', 'rgba(35,183,97,0.3)',
    'rgba(250,158,51,0.3)', 'rgba(124,58,237,0.3)',
  ];
  for (let gi = 0; gi < identicalGroups.length; gi++) {
    if (identicalGroups[gi].includes(Number(opIdx))) {
      return { fill: fills[gi % fills.length], stroke: strokes[gi % strokes.length] };
    }
  }
  return { fill: 'rgba(250,158,51,0.04)', stroke: 'rgba(250,158,51,0.25)' };
}
