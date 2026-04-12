export default function BipartiteGraph({ graph, example }) {
  const { uVertices, incidence, freeLabels, summedLabels, identicalGroups, allLabels } = graph;

  const W = 650, H = 420;
  const uX = 120, labelX = 530;
  const uSpacing = Math.min(55, (H - 60) / Math.max(uVertices.length, 1));
  const uStartY = (H - uVertices.length * uSpacing) / 2 + uSpacing / 2;

  const vLabels = [...freeLabels].sort();
  const wLabels = [...summedLabels].sort();
  const allRight = [...vLabels, ...wLabels];
  const rSpacing = Math.min(55, (H - 60) / Math.max(allRight.length, 1));
  const rStartY = (H - allRight.length * rSpacing) / 2 + rSpacing / 2;

  // Group U-vertices by operand for background rectangles
  const opGroups = {};
  uVertices.forEach((u, i) => {
    (opGroups[u.opIdx] ??= []).push(i);
  });

  // Identical-group coloring
  const opToGroupColor = {};
  const groupColors = ['#4a7cff44', '#3ddc8444', '#ffb74d44', '#bb86fc44'];
  identicalGroups.forEach((grp, gi) => {
    grp.forEach(opIdx => { opToGroupColor[opIdx] = groupColors[gi % groupColors.length]; });
  });

  return (
    <div className="graph-container">
      <svg viewBox={`0 0 ${W} ${H}`} className="bipartite-svg">
        {/* Operand group backgrounds */}
        {Object.entries(opGroups).map(([opIdx, uIndices]) => {
          const ys = uIndices.map(i => uStartY + i * uSpacing);
          const minY = Math.min(...ys) - 22;
          const maxY = Math.max(...ys) + 22;
          const col = opToGroupColor[opIdx] || '#ffffff08';
          return (
            <rect key={`bg-${opIdx}`}
              x={uX - 70} y={minY} width={140} height={maxY - minY}
              rx={10} fill={col} stroke={col.replace('44', '88')} strokeWidth={1}
            />
          );
        })}

        {/* Edges */}
        {uVertices.map((u, uIdx) => {
          const uy = uStartY + uIdx * uSpacing;
          return allRight.map((lbl, rIdx) => {
            const mult = incidence[uIdx][lbl] || 0;
            if (mult === 0) return null;
            const ry = rStartY + rIdx * rSpacing;
            return (
              <line key={`e-${uIdx}-${lbl}`}
                x1={uX + 40} y1={uy} x2={labelX - 30} y2={ry}
                stroke="#ffffff20" strokeWidth={mult * 2}
                className="graph-edge"
              />
            );
          });
        })}

        {/* U-vertices */}
        {uVertices.map((u, i) => {
          const y = uStartY + i * uSpacing;
          const labelStr = [...u.labels].sort().join(',');
          return (
            <g key={`u-${i}`} className="graph-node">
              <circle cx={uX} cy={y} r={18} fill="#ffb74d22" stroke="#ffb74d" strokeWidth={1.5} />
              <text x={uX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill="#ffb74d" fontSize={11} fontFamily="'JetBrains Mono', monospace">
                {labelStr}
              </text>
              <text x={uX - 50} y={y + 1} textAnchor="end" dominantBaseline="middle"
                fill="#888" fontSize={10} fontFamily="'JetBrains Mono', monospace">
                Op{u.opIdx}
              </text>
            </g>
          );
        })}

        {/* V/W divider */}
        {wLabels.length > 0 && vLabels.length > 0 && (
          <line
            x1={labelX - 30} y1={rStartY + vLabels.length * rSpacing - rSpacing / 2}
            x2={labelX + 60} y2={rStartY + vLabels.length * rSpacing - rSpacing / 2}
            stroke="#ffffff15" strokeDasharray="4,4"
          />
        )}

        {/* Label vertices */}
        {allRight.map((lbl, i) => {
          const y = rStartY + i * rSpacing;
          const isV = freeLabels.has(lbl);
          const col = isV ? '#4a7cff' : '#3ddc84';
          return (
            <g key={`r-${lbl}`} className="graph-node">
              <circle cx={labelX} cy={y} r={18} fill={col + '22'} stroke={col} strokeWidth={1.5} />
              <text x={labelX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill={col} fontSize={14} fontWeight="600"
                fontFamily="'JetBrains Mono', monospace">
                {lbl}
              </text>
              <text x={labelX + 40} y={y + 1} textAnchor="start" dominantBaseline="middle"
                fill="#666" fontSize={10}>
                {isV ? 'V' : 'W'}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <text x={uX} y={H - 10} textAnchor="middle" fill="#666" fontSize={10}>
          U (axis classes)
        </text>
        <text x={labelX} y={H - 10} textAnchor="middle" fill="#666" fontSize={10}>
          Labels
        </text>
      </svg>
    </div>
  );
}
