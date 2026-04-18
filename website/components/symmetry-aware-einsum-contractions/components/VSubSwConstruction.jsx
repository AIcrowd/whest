import { useState } from 'react';

/**
 * Format a Permutation (with `.arr` field) in disjoint cycle notation over
 * the given label array. Fixed points are omitted. Identity renders as "id".
 *
 * @param {Permutation} perm   - permutation object with .arr
 * @param {string[]}    labels - labels[i] corresponds to position i
 * @returns {string}
 */
function toCycleNotation(perm, labels) {
  if (!perm || !perm.arr) return 'id';
  const arr = perm.arr;
  const visited = new Set();
  const cycles = [];

  for (let i = 0; i < arr.length; i++) {
    if (visited.has(i) || arr[i] === i) {
      visited.add(i);
      continue;
    }
    const cycle = [];
    let j = i;
    while (!visited.has(j)) {
      cycle.push(j);
      visited.add(j);
      j = arr[j];
    }
    if (cycle.length > 1) cycles.push(cycle);
  }

  if (cycles.length === 0) return 'id';
  return cycles.map((c) => '(' + c.map((idx) => labels[idx] ?? String(idx)).join(' ') + ')').join('');
}

/**
 * Three-column widget displaying V-sub, S(W), and their Cartesian product
 * G_EXPR = V-sub x S(W) in cycle notation. Hover a left or middle element
 * to highlight the corresponding product rows.
 *
 * Props:
 *   expressionGroup - { elements, vSub, sw, order } from analysis.expressionGroup
 *   vLabels         - free labels (string[])
 *   wLabels         - summed labels (string[])
 */
export default function VSubSwConstruction({ expressionGroup, vLabels = [], wLabels = [] }) {
  const [hoveredVIdx, setHoveredVIdx] = useState(null);
  const [hoveredWIdx, setHoveredWIdx] = useState(null);

  if (!expressionGroup || expressionGroup.order <= 1) {
    return (
      <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm text-muted-foreground">
        V-sub &times; S(W) is trivial for this einsum (only the identity permutation).
      </div>
    );
  }

  const { vSub, sw } = expressionGroup;
  const swCount = sw?.length ?? 0;

  // Determine which product rows to highlight based on hover state.
  // Products are laid out in vSub-major order: row i*swCount + j corresponds
  // to vSub[i] x sw[j].
  const isHighlightedRow = (productIdx) => {
    if (hoveredVIdx !== null && hoveredWIdx !== null) {
      return productIdx === hoveredVIdx * swCount + hoveredWIdx;
    }
    if (hoveredVIdx !== null) {
      return Math.floor(productIdx / swCount) === hoveredVIdx;
    }
    if (hoveredWIdx !== null) {
      return productIdx % swCount === hoveredWIdx;
    }
    return false;
  };

  const colHead = 'text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground mb-2';
  const cellBase = 'font-mono text-sm px-2 py-1 rounded cursor-default transition-colors';
  const cellNormal = 'text-foreground hover:bg-muted/40';
  const cellActive = 'bg-coral-light/60 text-foreground font-semibold';

  return (
    <div className="overflow-x-auto">
      <div className="grid grid-cols-3 gap-4 min-w-[480px]">
        {/* Left column: V-sub */}
        <div>
          <div className={colHead}>
            V-sub
            <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
              (G_PT projected to V-labels)
            </span>
          </div>
          <div className="flex flex-col gap-0.5">
            {vSub.map((vElem, vi) => (
              <div
                key={vi}
                className={`${cellBase} ${cellNormal} ${hoveredVIdx === vi ? 'bg-sky-100 text-sky-800' : ''}`}
                onMouseEnter={() => setHoveredVIdx(vi)}
                onMouseLeave={() => setHoveredVIdx(null)}
              >
                {toCycleNotation(vElem, vLabels)}
              </div>
            ))}
          </div>
        </div>

        {/* Middle column: S(W) */}
        <div>
          <div className={colHead}>
            S(W)
            <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
              (all dummy-rename perms)
            </span>
          </div>
          <div className="flex flex-col gap-0.5">
            {sw.map((wElem, wi) => (
              <div
                key={wi}
                className={`${cellBase} ${cellNormal} ${hoveredWIdx === wi ? 'bg-amber-100 text-amber-800' : ''}`}
                onMouseEnter={() => setHoveredWIdx(wi)}
                onMouseLeave={() => setHoveredWIdx(null)}
              >
                {toCycleNotation(wElem, wLabels)}
              </div>
            ))}
          </div>
        </div>

        {/* Right column: G_EXPR product */}
        <div>
          <div className={colHead}>
            G_EXPR = V-sub &times; S(W)
            <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
              (expression-level group)
            </span>
          </div>
          <div className="flex flex-col gap-0.5">
            {vSub.flatMap((_, vi) =>
              sw.map((_, wi) => {
                const productIdx = vi * swCount + wi;
                const elem = expressionGroup.elements[productIdx];
                const allLabels = [...vLabels, ...wLabels];
                const highlighted = isHighlightedRow(productIdx);
                return (
                  <div
                    key={productIdx}
                    className={`${cellBase} ${highlighted ? cellActive : cellNormal}`}
                  >
                    {toCycleNotation(elem, allLabels)}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
