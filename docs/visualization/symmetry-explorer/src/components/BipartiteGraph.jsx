/**
 * Bipartite graph rendered as polished SVG with bezier edges,
 * labeled group boxes for U axis-classes (left), V free and W summed (right).
 *
 * Symmetry buttons appear below the graph in two categories:
 * - INPUT: per-operand declared symmetries (axis swaps within each operand)
 * - OUTPUT: detected group elements (label permutations from the σ-loop)
 * Clicking a button overlays arrows on the graph.
 */

import { useState, useMemo } from 'react';

export default function BipartiteGraph({ graph, example, variableColors, group }) {
  const { uVertices, incidence, freeLabels, summedLabels, identicalGroups } = graph;

  const vLabels = [...freeLabels].sort();
  const wLabels = [...summedLabels].sort();

  // ── Build unified symmetry list ──
  const [activeSymIdx, setActiveSymIdx] = useState(null);

  // Input symmetries: one entry per operand with declared symmetry
  const inputSyms = useMemo(() => {
    try {
    const perOp = example.perOpSymmetry;
    const subs = example.subscripts;
    if (!perOp || !subs) return [];

    const syms = [];
    for (let opIdx = 0; opIdx < subs.length; opIdx++) {
      const sym = perOp[opIdx];
      if (!sym) continue;
      const sub = subs[opIdx];
      const opName = example.operandNames?.[opIdx] || `Op${opIdx}`;
      const hasMultiple = example.operandNames?.filter(n => n === opName).length > 1;
      const opLabel = hasMultiple ? `${opName}(${opIdx})` : opName;

      let axes, type;
      if (sym === 'symmetric') {
        axes = [...Array(sub.length).keys()];
        type = 'symmetric';
      } else if (typeof sym === 'object') {
        axes = sym.axes || [...Array(sub.length).keys()];
        type = sym.type;
      } else continue;

      if (axes.length < 2) continue;
      // Guard against out-of-range axes
      if (axes.some(a => a >= sub.length)) continue;

      const labels = axes.map(a => sub[a]);
      const k = axes.length;
      const subscriptDigits = '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089';
      let notation;
      if (type === 'symmetric' || type === 'cyclic' || type === 'dihedral') {
        const prefix = type === 'symmetric' ? 'S' : type === 'cyclic' ? 'C' : 'D';
        const sub_k = k < 10 ? subscriptDigits[k] : String(k);
        notation = `${prefix}${sub_k}{${labels.join(',')}}`;
      } else {
        // Custom: show generator cycles with label names
        notation = `\u27e8${labels.join(',')}\u27e9`;
      }

      syms.push({
        source: 'input', opIdx, axes, labels, type,
        notation,
        opLabel,
      });
    }
    return syms;
    } catch (e) { console.error('inputSyms error:', e); return []; }
  }, [example]);

  // Output symmetries: detected V/W group elements
  const outputSyms = useMemo(() => {
    if (!group) return [];
    const syms = [];

    if (group.vElements) {
      for (const el of group.vElements) {
        if (el.isIdentity) continue;
        const cycles = el.cyclicForm();
        const labelCycles = cycles.map(c => c.map(i => group.vLabels[i]));
        syms.push({
          source: 'output', side: 'V',
          cycles: labelCycles,
          notation: formatCycles(labelCycles),
        });
      }
    }

    if (group.wElements) {
      for (const el of group.wElements) {
        if (el.isIdentity) continue;
        const cycles = el.cyclicForm();
        const labelCycles = cycles.map(c => c.map(i => group.wLabels[i]));
        syms.push({
          source: 'output', side: 'W',
          cycles: labelCycles,
          notation: formatCycles(labelCycles),
        });
      }
    }

    return syms;
  }, [group]);

  // Combined list
  const allSymmetries = useMemo(
    () => [...inputSyms, ...outputSyms],
    [inputSyms, outputSyms]
  );

  // Reset selection when data changes
  useMemo(() => setActiveSymIdx(null), [group, example]);

  const activeSym = activeSymIdx != null ? allSymmetries[activeSymIdx] : null;

  // Highlight labels for output symmetries (right-side nodes)
  const highlightLabels = useMemo(() => {
    if (!activeSym || activeSym.source !== 'output') return new Set();
    const s = new Set();
    for (const cycle of activeSym.cycles) {
      for (const lbl of cycle) s.add(lbl);
    }
    return s;
  }, [activeSym]);

  // Highlight U-vertex indices for input symmetries (left-side nodes)
  const highlightUIndices = useMemo(() => {
    if (!activeSym || activeSym.source !== 'input') return new Set();
    return new Set(); // highlight rings drawn inline below
  }, [activeSym]);

  // ── Layout constants ──
  const leftX = 150;
  const rightX = 520;
  const ySpacing = 46;
  const nodeR = 15;
  const groupGap = 50;
  const topMargin = 30;

  // ── Left: U-vertices, grouped by operand ──
  const opGroups = {};
  uVertices.forEach((u, i) => {
    (opGroups[u.opIdx] ??= []).push(i);
  });
  const opKeys = Object.keys(opGroups).sort((a, b) => a - b);

  const charW = 7.5;
  const uPillWidths = uVertices.map(u => {
    const labelStr = [...u.labels].sort().join(',');
    return Math.max(32, labelStr.length * charW + 18);
  });
  const maxPillW = Math.max(32, ...uPillWidths);

  const uPositions = new Array(uVertices.length);
  let leftY = topMargin + 20;
  const opBoxes = [];
  const leftBoxPadX = 24;
  const leftBoxPadY = 24;
  const leftBoxW = maxPillW + 10;

  opKeys.forEach((opIdx) => {
    const indices = opGroups[opIdx];
    const vcName = example.operandNames?.[opIdx];
    const vc = variableColors?.[vcName];
    const colors = vc
      ? { fill: `${vc.color}08`, stroke: `${vc.color}4D` }
      : identicalGroups.length > 0
        ? getGroupColor(opIdx, identicalGroups)
        : { fill: 'rgba(250,158,51,0.04)', stroke: 'rgba(250,158,51,0.25)' };

    const boxTop = leftY;
    indices.forEach((uIdx, j) => {
      uPositions[uIdx] = boxTop + leftBoxPadY + nodeR + j * ySpacing;
    });
    const boxBottom = boxTop + leftBoxPadY * 2 + nodeR * 2 + (indices.length - 1) * ySpacing;
    opBoxes.push({ opIdx, top: boxTop, bottom: boxBottom, indices, colors, vc });
    leftY = boxBottom + 16;
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

  const labelY = {};
  vLabels.forEach((l, i) => { labelY[l] = vPositions[i]; });
  wLabels.forEach((l, i) => { labelY[l] = wPositions[i]; });

  // ── SVG dimensions ──
  const arrowPad = (activeSym?.source === 'output') ? 80 : 0;
  const rightEdge = rightX + rightBoxW / 2 + rightBoxPadX + 50 + arrowPad;
  const leftBottom = opBoxes.length > 0 ? opBoxes[opBoxes.length - 1].bottom + 40 : 200;
  const rightBottom = wBoxTop + wBoxH + 40;
  const W = Math.max(660 + arrowPad, rightEdge);
  const H = Math.max(300, Math.max(leftBottom, rightBottom) + 20);

  const lbLeft = leftX - leftBoxW / 2 - leftBoxPadX;
  const lbFullW = leftBoxW + leftBoxPadX * 2;
  const rbLeft = rightX - rightBoxW / 2 - rightBoxPadX;
  const rbFullW = rightBoxW + rightBoxPadX * 2;

  const bezier = (x1, y1, x2, y2) => {
    const cx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`;
  };

  // Active output arrow color
  const outputArrowColor = activeSym?.source === 'output'
    ? (activeSym.side === 'V' ? '#4A7CFF' : '#64748B')
    : null;

  // Active input arrow color
  const inputArrowColor = activeSym?.source === 'input'
    ? (variableColors?.[example.operandNames?.[activeSym.opIdx]]?.color || '#FA9E33')
    : null;

  return (
    <div className="graph-container">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <defs>
          <filter id="node-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="1" stdDeviation="2" floodOpacity="0.07" />
          </filter>
          <marker id="sym-arrowhead-v" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M 0 0 L 8 3 L 0 6 Z" fill="#4A7CFF" />
          </marker>
          <marker id="sym-arrowhead-w" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M 0 0 L 8 3 L 0 6 Z" fill="#64748B" />
          </marker>
          <marker id="sym-arrowhead-input" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
            <path d="M 0 0 L 7 2.5 L 0 5 Z" fill={inputArrowColor || '#FA9E33'} opacity="0.7" />
          </marker>
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
                rx={14} fill={colors.fill} stroke={colors.stroke} strokeWidth={1.5}
              />
              <LabelBadge
                x={lbLeft + 12} y={top}
                text={label}
                color={vc?.color || '#FA9E33'} bg="#F8F9F9"
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
              rx={14} fill="rgba(100,116,139,0.03)" stroke="rgba(100,116,139,0.2)" strokeWidth={1.5}
            />
            <LabelBadge x={rbLeft + 12} y={wBoxTop} text="W (summed)" color="#64748B" bg="#F8F9F9" />
          </g>
        ) : (
          <g>
            <rect
              x={rbLeft} y={wBoxTop}
              width={rbFullW} height={wBoxH}
              rx={14} fill="rgba(100,116,139,0.02)"
              stroke="rgba(100,116,139,0.15)" strokeWidth={1.5} strokeDasharray="6,4"
            />
            <LabelBadge x={rbLeft + 12} y={wBoxTop} text="W (summed)" color="#64748B" bg="#F8F9F9" />
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
          const eOpName = example.operandNames?.[u.opIdx];
          const eVc = variableColors?.[eOpName];
          const edgeColor = eVc?.color || '#D9DCDC';
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

        {/* ── INPUT symmetry arrows (left side, when active) ── */}
        {activeSym?.source === 'input' && (() => {
          const { opIdx, axes: symAxes, type } = activeSym;
          const indices = opGroups[opIdx];
          if (!indices || !symAxes) return null;
          const color = inputArrowColor;

          if (type === 'cyclic') {
            // Cyclic: directed arrows forming a cycle through axes
            return (
              <g>
                {symAxes.map((ax, j) => {
                  const nextAx = symAxes[(j + 1) % symAxes.length];
                  if (ax >= indices.length || nextAx >= indices.length) return null;
                  const y1 = uPositions[indices[ax]];
                  const y2 = uPositions[indices[nextAx]];
                  const bowX = lbLeft - 18 - j * 6;
                  const midY = (y1 + y2) / 2;
                  return (
                    <path key={`inp-${j}`}
                      d={`M ${lbLeft} ${y1} Q ${bowX} ${midY}, ${lbLeft} ${y2}`}
                      fill="none" stroke={color} strokeWidth={2}
                      strokeDasharray="5,3" opacity={0.7}
                      markerEnd="url(#sym-arrowhead-input)"
                    />
                  );
                })}
              </g>
            );
          } else {
            // Symmetric/Dihedral: double-headed arrows between all pairs
            const pairs = [];
            for (let i = 0; i < symAxes.length; i++) {
              for (let j = i + 1; j < symAxes.length; j++) {
                pairs.push([symAxes[i], symAxes[j]]);
              }
            }
            return (
              <g>
                {pairs.map(([ax0, ax1], pi) => {
                  if (ax0 >= indices.length || ax1 >= indices.length) return null;
                  const y1 = uPositions[indices[ax0]];
                  const y2 = uPositions[indices[ax1]];
                  const bowX = lbLeft - 18 - pi * 8;
                  const midY = (y1 + y2) / 2;
                  return (
                    <g key={`inp-${pi}`}>
                      <path
                        d={`M ${lbLeft} ${y1} Q ${bowX} ${midY}, ${lbLeft} ${y2}`}
                        fill="none" stroke={color} strokeWidth={2}
                        strokeDasharray="5,3" opacity={0.7}
                      />
                      <circle cx={lbLeft} cy={y1} r={3} fill={color} opacity={0.7} />
                      <circle cx={lbLeft} cy={y2} r={3} fill={color} opacity={0.7} />
                    </g>
                  );
                })}
              </g>
            );
          }
        })()}

        {/* ── OUTPUT symmetry arrows (right side, when active) ── */}
        {activeSym?.source === 'output' && activeSym.cycles.map((cycle, ci) => {
          const arrowStartX = rightX + nodeR + 8;
          const markerId = activeSym.side === 'V' ? 'sym-arrowhead-v' : 'sym-arrowhead-w';

          if (cycle.length === 2) {
            const [a, b] = cycle;
            const y1 = labelY[a];
            const y2 = labelY[b];
            const bowX = arrowStartX + 30 + ci * 18;
            const midY = (y1 + y2) / 2;
            return (
              <g key={`sym-${ci}`}>
                <path
                  d={`M ${arrowStartX} ${y1} Q ${bowX} ${midY}, ${arrowStartX} ${y2}`}
                  fill="none" stroke={outputArrowColor} strokeWidth={2}
                  strokeDasharray="5,3" opacity={0.8}
                />
                <circle cx={arrowStartX} cy={y1} r={3} fill={outputArrowColor} opacity={0.8} />
                <circle cx={arrowStartX} cy={y2} r={3} fill={outputArrowColor} opacity={0.8} />
                <text x={bowX + 6} y={midY + 4}
                  fontSize={10} fontWeight={600} fill={outputArrowColor}
                  fontFamily="'IBM Plex Mono', monospace" opacity={0.7}>
                  {a}↔{b}
                </text>
              </g>
            );
          } else {
            return (
              <g key={`sym-${ci}`}>
                {cycle.map((lbl, j) => {
                  const nextLbl = cycle[(j + 1) % cycle.length];
                  const y1 = labelY[lbl];
                  const y2 = labelY[nextLbl];
                  const bowX = arrowStartX + 30 + ci * 18;
                  const midY = (y1 + y2) / 2;
                  return (
                    <path key={`sym-${ci}-${j}`}
                      d={`M ${arrowStartX} ${y1} Q ${bowX} ${midY}, ${arrowStartX} ${y2}`}
                      fill="none" stroke={outputArrowColor} strokeWidth={2}
                      strokeDasharray="5,3" opacity={0.8}
                      markerEnd={`url(#${markerId})`}
                    />
                  );
                })}
                <text x={arrowStartX + 30 + ci * 18 + 6}
                  y={labelY[cycle[0]] + (labelY[cycle[1]] - labelY[cycle[0]]) / 2 + 4}
                  fontSize={10} fontWeight={600} fill={outputArrowColor}
                  fontFamily="'IBM Plex Mono', monospace" opacity={0.7}>
                  {cycle.join('→')}
                </text>
              </g>
            );
          }
        })}

        {/* ── U-vertex nodes (left) — pill-shaped ── */}
        {uVertices.map((u, i) => {
          const y = uPositions[i];
          const labelStr = [...u.labels].sort().join(',');
          const pillW = Math.max(32, labelStr.length * 7.5 + 18);
          const pillH = 28;
          const pillR = pillH / 2;
          const nOpName = example.operandNames?.[u.opIdx];
          const nVc = variableColors?.[nOpName];
          const nodeColor = nVc?.color || '#FA9E33';
          const hasSymmetry = nVc?.symmetry && nVc.symmetry !== 'none';
          // Highlight if this U-vertex is involved in the active input symmetry
          const isInputHighlighted = activeSym?.source === 'input' && activeSym.opIdx === u.opIdx;
          return (
            <g key={`u-${i}`}>
              {isInputHighlighted && (
                <rect x={leftX - pillW / 2 - 4} y={y - pillH / 2 - 4}
                  width={pillW + 8} height={pillH + 8} rx={pillR + 2}
                  fill="none" stroke={inputArrowColor} strokeWidth={2.5}
                  strokeDasharray="4,2" opacity={0.4} />
              )}
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
          const isHighlighted = highlightLabels.has(lbl);
          return (
            <g key={`v-${lbl}`}>
              {isHighlighted && (
                <circle cx={rightX} cy={y} r={nodeR + 4}
                  fill="none" stroke="#4A7CFF" strokeWidth={2.5}
                  strokeDasharray="4,2" opacity={0.5} />
              )}
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
          const isHighlighted = highlightLabels.has(lbl);
          return (
            <g key={`w-${lbl}`}>
              {isHighlighted && (
                <circle cx={rightX} cy={y} r={nodeR + 4}
                  fill="none" stroke="#64748B" strokeWidth={2.5}
                  strokeDasharray="4,2" opacity={0.5} />
              )}
              <circle cx={rightX} cy={y} r={nodeR}
                fill="white" stroke="#64748B" strokeWidth={1.5}
                filter="url(#node-shadow)" />
              <text x={rightX} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                fill="#64748B" fontSize={11} fontWeight={600}
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

      {/* ── Symmetry buttons (two rows) ── */}
      {allSymmetries.length > 0 && (
        <div className="sym-buttons-container">
          {inputSyms.length > 0 && (
            <div className="sym-row">
              <span className="sym-section-label">Input</span>
              {inputSyms.map((sym, i) => {
                const globalIdx = i;
                const vc = variableColors?.[example.operandNames?.[sym.opIdx]];
                const color = vc?.color || '#FA9E33';
                return (
                  <button
                    key={`in-${i}`}
                    className={`sym-btn sym-btn-input ${activeSymIdx === globalIdx ? 'active' : ''}`}
                    style={{
                      borderColor: activeSymIdx === globalIdx ? color : `${color}4D`,
                      color: activeSymIdx === globalIdx ? 'white' : color,
                      backgroundColor: activeSymIdx === globalIdx ? color : 'white',
                    }}
                    onClick={() => setActiveSymIdx(activeSymIdx === globalIdx ? null : globalIdx)}
                    title={`Input: ${sym.notation} on ${sym.opLabel}`}
                  >
                    {sym.notation} <span className="sym-btn-op">{sym.opLabel}</span>
                  </button>
                );
              })}
            </div>
          )}
          {outputSyms.length > 0 && (
            <div className="sym-row">
              <span className="sym-section-label">{inputSyms.length > 0 ? 'Output' : 'Symmetries'}</span>
              {outputSyms.map((sym, i) => {
                const globalIdx = inputSyms.length + i;
                return (
                  <button
                    key={`out-${i}`}
                    className={`sym-btn ${sym.side === 'V' ? 'sym-btn-v' : 'sym-btn-w'} ${activeSymIdx === globalIdx ? 'active' : ''}`}
                    onClick={() => setActiveSymIdx(activeSymIdx === globalIdx ? null : globalIdx)}
                    title={`${sym.side}-side: ${sym.notation}`}
                  >
                    {sym.notation}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Format cycles as human-readable notation */
function formatCycles(labelCycles) {
  return labelCycles.map(c => {
    if (c.length === 2) return `(${c[0]} ${c[1]})`;
    return '(' + c.join(' ') + ')';
  }).join('');
}

/** Small label badge that sits on the top edge of a group box */
function LabelBadge({ x, y, text, color, bg }) {
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
