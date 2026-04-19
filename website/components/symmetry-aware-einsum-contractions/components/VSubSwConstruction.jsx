import { useMemo, useState } from 'react';
import Latex from './Latex.jsx';

/**
 * Render a Permutation in disjoint cycle notation as an array of JSX tokens.
 * Every label is wrapped in a span whose color is set by its V/W class:
 *   - V-label  → sky
 *   - W-label  → amber
 * Parens, spaces, and the literal "id" fall back to neutral foreground.
 *
 * @param {Permutation} perm       - permutation object with .arr
 * @param {string[]}    labels     - labels[i] is the label at position i
 * @param {Set<string>} vLabelSet  - V-labels (for color lookup)
 * @param {Set<string>} wLabelSet  - W-labels (for color lookup)
 * @returns {JSX.Element[]} tokens ready to render inside a font-mono span
 */
function renderCycleTokens(perm, labels, vLabelSet, wLabelSet) {
  const arr = perm?.arr;
  if (!arr) return [<span key="id" className="text-muted-foreground">id</span>];

  const visited = new Set();
  const cycles = [];
  for (let i = 0; i < arr.length; i += 1) {
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

  if (cycles.length === 0) {
    return [<span key="id" className="text-muted-foreground italic">id</span>];
  }

  const tokens = [];
  let key = 0;
  for (const cycle of cycles) {
    tokens.push(<span key={key++} className="text-gray-400">(</span>);
    for (let k = 0; k < cycle.length; k += 1) {
      const label = labels[cycle[k]] ?? String(cycle[k]);
      const colorClass = vLabelSet.has(label)
        ? 'text-sky-700'
        : wLabelSet.has(label)
          ? 'text-amber-700'
          : 'text-foreground';
      tokens.push(
        <span key={key++} className={`${colorClass} font-semibold`}>
          {label}
        </span>,
      );
      if (k < cycle.length - 1) tokens.push(<span key={key++} className="text-gray-400"> </span>);
    }
    tokens.push(<span key={key++} className="text-gray-400">)</span>);
  }
  return tokens;
}

/**
 * V-sub × S(W) construction widget. Renders three columns of permutations
 * plus visible × / = separators so the equation reads left-to-right:
 *
 *   V-sub   ×   S(W)   =   G_expr
 *
 * Each label in a cycle is colored by its V/W class (sky/amber), matching
 * the site-wide convention. Hover any cell in any of the three columns to
 * highlight its contribution(s) in the other two.
 *
 * Props:
 *   expressionGroup - { elements, vSub, sw, order } from analysis.expressionGroup
 *   vLabels         - free labels (string[])
 *   wLabels         - summed labels (string[])
 */
export default function VSubSwConstruction({ expressionGroup, vLabels = [], wLabels = [] }) {
  const [hoveredVIdx, setHoveredVIdx] = useState(null);
  const [hoveredWIdx, setHoveredWIdx] = useState(null);
  const [hoveredProductIdx, setHoveredProductIdx] = useState(null);

  const vLabelSet = useMemo(() => new Set(vLabels), [vLabels]);
  const wLabelSet = useMemo(() => new Set(wLabels), [wLabels]);

  if (!expressionGroup || expressionGroup.order <= 1) {
    return (
      <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm text-muted-foreground">
        V-sub × S(W) is trivial for this einsum (only the identity permutation).
      </div>
    );
  }

  const { vSub, sw } = expressionGroup;
  const swCount = sw?.length ?? 0;

  // Derive the effective (vIdx, wIdx) pair being highlighted. A product-row
  // hover resolves to a unique (vIdx, wIdx); a column-cell hover leaves the
  // other index null.
  const effectiveVIdx =
    hoveredProductIdx !== null ? Math.floor(hoveredProductIdx / swCount) : hoveredVIdx;
  const effectiveWIdx =
    hoveredProductIdx !== null ? hoveredProductIdx % swCount : hoveredWIdx;

  const isHighlightedProduct = (productIdx) => {
    if (hoveredProductIdx !== null) return productIdx === hoveredProductIdx;
    if (effectiveVIdx !== null && effectiveWIdx !== null) {
      return productIdx === effectiveVIdx * swCount + effectiveWIdx;
    }
    if (effectiveVIdx !== null) return Math.floor(productIdx / swCount) === effectiveVIdx;
    if (effectiveWIdx !== null) return productIdx % swCount === effectiveWIdx;
    return false;
  };

  const colHead =
    'text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground mb-2';
  const cellBase =
    'font-mono text-sm px-2 py-1 rounded cursor-default transition-colors border border-transparent';
  const cellNormal = 'hover:border-border/60 hover:bg-muted/30';
  const vActive = 'bg-sky-50 border-sky-300';
  const wActive = 'bg-amber-50 border-amber-300';
  const prodActive = 'bg-emerald-50 border-emerald-400';
  const prodDim = 'opacity-40';
  const sepBase = 'flex h-8 items-center justify-center text-xl font-light text-muted-foreground select-none';

  const allLabels = [...vLabels, ...wLabels];

  return (
    <div className="overflow-x-auto">
      <div
        className="grid min-w-[560px] items-start gap-3"
        style={{ gridTemplateColumns: 'minmax(0,1fr) 20px minmax(0,1fr) 20px minmax(0,1.4fr)' }}
      >
        {/* Column headers */}
        <div>
          <div className={colHead}>
            <Latex math="V_{\text{sub}}" />
            <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
              (<Latex math="G_{\text{pt}}" /> restricted to V)
            </span>
          </div>
        </div>
        <div className={colHead} />
        <div>
          <div className={colHead}>
            <Latex math="S(W)" />
            <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
              (all dummy-rename permutations)
            </span>
          </div>
        </div>
        <div className={colHead} />
        <div>
          <div className={colHead}>
            <Latex math="G_{\text{expr}} = V_{\text{sub}} \times S(W)" />
            <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
              (expression-level group)
            </span>
          </div>
        </div>

        {/* Row content — stacked vertically inside each column cell */}

        {/* V-sub column */}
        <div className="flex flex-col gap-0.5">
          {vSub.map((vElem, vi) => (
            <button
              type="button"
              key={vi}
              className={`${cellBase} ${cellNormal} flex items-center gap-2 text-left ${
                effectiveVIdx === vi ? vActive : ''
              }`}
              onMouseEnter={() => setHoveredVIdx(vi)}
              onMouseLeave={() => setHoveredVIdx(null)}
              onFocus={() => setHoveredVIdx(vi)}
              onBlur={() => setHoveredVIdx(null)}
            >
              <span className="w-5 shrink-0 text-right text-[11px] font-semibold text-muted-foreground">
                {vi + 1}
              </span>
              <span className="flex flex-wrap items-baseline">
                {renderCycleTokens(vElem, vLabels, vLabelSet, wLabelSet)}
              </span>
            </button>
          ))}
        </div>

        {/* × separator (repeated on every V-sub row) */}
        <div className="flex flex-col gap-0.5">
          {vSub.map((_, vi) => (
            <div
              key={vi}
              className={`${sepBase} ${effectiveVIdx === vi ? 'text-sky-600' : ''}`}
              aria-hidden
            >
              ×
            </div>
          ))}
        </div>

        {/* S(W) column */}
        <div className="flex flex-col gap-0.5">
          {sw.map((wElem, wi) => (
            <button
              type="button"
              key={wi}
              className={`${cellBase} ${cellNormal} flex items-center gap-2 text-left ${
                effectiveWIdx === wi ? wActive : ''
              }`}
              onMouseEnter={() => setHoveredWIdx(wi)}
              onMouseLeave={() => setHoveredWIdx(null)}
              onFocus={() => setHoveredWIdx(wi)}
              onBlur={() => setHoveredWIdx(null)}
            >
              <span className="w-5 shrink-0 text-right text-[11px] font-semibold text-muted-foreground">
                {wi + 1}
              </span>
              <span className="flex flex-wrap items-baseline">
                {renderCycleTokens(wElem, wLabels, vLabelSet, wLabelSet)}
              </span>
            </button>
          ))}
        </div>

        {/* = separator */}
        <div className="flex flex-col gap-0.5">
          {vSub.flatMap((_, vi) =>
            sw.map((_, wi) => {
              const productIdx = vi * swCount + wi;
              const active = isHighlightedProduct(productIdx);
              const dim = hoveredProductIdx !== null && hoveredProductIdx !== productIdx;
              return (
                <div
                  key={productIdx}
                  className={`${sepBase} ${active ? 'text-emerald-700' : ''} ${
                    dim ? 'opacity-30' : ''
                  }`}
                  aria-hidden
                >
                  =
                </div>
              );
            }),
          )}
        </div>

        {/* G_expr product column — every (vi, wi) pair */}
        <div className="flex flex-col gap-0.5">
          {vSub.flatMap((_, vi) =>
            sw.map((_, wi) => {
              const productIdx = vi * swCount + wi;
              const elem = expressionGroup.elements[productIdx];
              const active = isHighlightedProduct(productIdx);
              const dim =
                (effectiveVIdx !== null || effectiveWIdx !== null || hoveredProductIdx !== null) &&
                !active;
              return (
                <button
                  type="button"
                  key={productIdx}
                  className={`${cellBase} flex items-center gap-2 text-left ${
                    active ? prodActive : cellNormal
                  } ${dim ? prodDim : ''}`}
                  onMouseEnter={() => setHoveredProductIdx(productIdx)}
                  onMouseLeave={() => setHoveredProductIdx(null)}
                  onFocus={() => setHoveredProductIdx(productIdx)}
                  onBlur={() => setHoveredProductIdx(null)}
                >
                  <span className="w-10 shrink-0 text-right text-[10px] font-mono text-muted-foreground">
                    {vi + 1}·{wi + 1}
                  </span>
                  <span className="flex flex-wrap items-baseline">
                    {renderCycleTokens(elem, allLabels, vLabelSet, wLabelSet)}
                  </span>
                </button>
              );
            }),
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm bg-sky-200 border border-sky-400" />
          V-label
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm bg-amber-200 border border-amber-400" />
          W-label
        </span>
        <span className="ml-auto italic">
          Hover a row in any column — the other two columns highlight the pairing.
        </span>
      </div>
    </div>
  );
}
