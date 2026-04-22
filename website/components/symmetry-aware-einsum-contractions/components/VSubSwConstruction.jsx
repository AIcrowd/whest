import { Fragment, useMemo, useState } from 'react';
import Latex from './Latex.jsx';
import { explorerThemeColor, explorerThemeTint } from '../lib/explorerTheme.js';
import {
  getActiveExplorerThemeId,
  notationColor,
  notationColoredLatex,
  notationLatex,
} from '../lib/notationSystem.js';

/**
 * Format a Permutation in standard disjoint cycle notation. Fixed points
 * are omitted; identity renders as "id". Labels inside a cycle are
 * space-separated so "(i j)" unambiguously reads "i maps to j, j maps to i"
 * — not a single two-character label "ij" that happens to be fixed.
 */
function toCycleString(perm, labels) {
  const arr = perm?.arr;
  if (!arr) return 'id';
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
  if (cycles.length === 0) return 'id';
  return cycles.map((c) => '(' + c.map((idx) => labels[idx] ?? String(idx)).join(' ') + ')').join('');
}

/**
 * Render a cycle-notation string with every label re-colored by V/W class.
 * Parens, spaces, and the "id" token pass through uncolored. Inline spans
 * (no flex container) so the inter-label spaces render verbatim.
 */
function ColoredLabels({ text, vSet, wSet }) {
  const COLOR_V = notationColor('v_free');
  const COLOR_W = notationColor('w_summed');
  if (!text) return null;
  if (text === 'id') {
    return <span className="italic text-muted-foreground">id</span>;
  }
  // Tokenize on any non-label character; keep separators in the token list.
  const tokens = text.split(/([() ])/);
  return (
    <>
      {tokens.map((tok, i) => {
        if (!tok) return null;
        if (vSet.has(tok)) {
          return (
            <span key={i} style={{ color: COLOR_V, fontWeight: 600 }}>
              {tok}
            </span>
          );
        }
        if (wSet.has(tok)) {
          return (
            <span key={i} style={{ color: COLOR_W, fontWeight: 600 }}>
              {tok}
            </span>
          );
        }
        if (tok === '(' || tok === ')') {
          return (
            <span key={i} className="text-gray-400">
              {tok}
            </span>
          );
        }
        return <Fragment key={i}>{tok}</Fragment>;
      })}
    </>
  );
}

/**
 * G_f = G_pt|_V × S(W) construction widget. Renders a 5-column grid:
 *
 *   G_pt|_V   ×   S(W)   =   G_f
 *
 * where G_pt|_V is the induced permutation group on V (the V-restriction
 * of G_pt), S(W) is the symmetric group on W-labels, and G_f is the
 * formal symmetry group. Every label inside a cycle is colored by its
 * V/W class (V → blue, W → slate) using the canonical hexes the rest of
 * the explorer uses. Hover is bidirectional: hover any row in any column
 * to highlight its counterparts in the other two.
 */
export default function VSubSwConstruction({
  expressionGroup,
  vLabels = [],
  wLabels = [],
  showHeading = true,
}) {
  const explorerThemeId = getActiveExplorerThemeId();
  const SHORT_V_LATEX = notationColoredLatex('v_free', 'V');
  const SHORT_S_W_LATEX = notationColoredLatex('s_w_summed', 'S(W)');
  const SHORT_G_OUT_EQUALS_RESTRICTION_LATEX = `${notationLatex('g_output')} = ${notationLatex('g_pointwise')}\\big|_{${SHORT_V_LATEX}}`;
  const COLOR_V = notationColor('v_free');
  const COLOR_W = notationColor('w_summed');
  const PRODUCT_ACCENT = explorerThemeColor(explorerThemeId, 'quantity');
  const [hoveredVIdx, setHoveredVIdx] = useState(null);
  const [hoveredWIdx, setHoveredWIdx] = useState(null);
  const [hoveredProductIdx, setHoveredProductIdx] = useState(null);

  const vLabelSet = useMemo(() => new Set(vLabels), [vLabels]);
  const wLabelSet = useMemo(() => new Set(wLabels), [wLabels]);

  if (!expressionGroup || expressionGroup.order <= 1) {
    return (
      <div className="mt-4 text-[15px] text-gray-600">
        <Latex math={SHORT_G_OUT_EQUALS_RESTRICTION_LATEX} /> is trivial for this einsum. <Latex math={SHORT_S_W_LATEX} /> is trivial as well.
      </div>
    );
  }

  const { vSub, sw } = expressionGroup;
  const swCount = sw?.length ?? 0;
  const allLabels = [...vLabels, ...wLabels];

  // Derive the effective (vIdx, wIdx) pair being highlighted. A product-row
  // hover resolves to a unique pair; a source-column hover leaves the other
  // coordinate null.
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
  const sepBase =
    'flex h-8 items-center justify-center text-xl font-light text-muted-foreground select-none';

  const activeVStyle = {
    background: `${COLOR_V}15`, // ~8% alpha
    borderColor: `${COLOR_V}66`,
  };
  const activeWStyle = {
    background: `${COLOR_W}1F`, // ~12% alpha
    borderColor: `${COLOR_W}80`,
  };
  const activeProductStyle = {
    background: explorerThemeTint(explorerThemeId, 'quantity', 0.1),
    borderColor: explorerThemeTint(explorerThemeId, 'quantity', 0.32),
  };

  return (
    <div className="space-y-3">
      {showHeading ? (
        <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-500">
          Formal-group construction
        </div>
      ) : null}
      <div className="overflow-x-auto">
        <div
          className="grid min-w-[560px] items-start gap-3"
          style={{ gridTemplateColumns: 'minmax(0,1fr) 20px minmax(0,1fr) 20px minmax(0,1.4fr)' }}
        >
          {/* Column headers */}
          <div>
            <div className={colHead}>
              <Latex math={SHORT_G_OUT_EQUALS_RESTRICTION_LATEX} />
              <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
                (visible output action, inherited from G_pt)
              </span>
              <span className="block text-[10px] text-muted-foreground normal-case tracking-normal">
                (row-witnessed)
              </span>
            </div>
          </div>
          <div className={colHead} />
          <div>
            <div className={colHead}>
              <Latex math={SHORT_S_W_LATEX} />
              <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
                (dummy-label renamings, available only after summation)
              </span>
              <span className="block text-[10px] text-muted-foreground normal-case tracking-normal">
                (row-unwitnessed)
              </span>
            </div>
          </div>
          <div className={colHead} />
          <div>
            <div className={colHead}>
              <Latex math={`${notationLatex('g_formal')} = ${notationLatex('g_output')} \\times ${SHORT_S_W_LATEX}`} />
              <span className="ml-1 text-[10px] text-muted-foreground normal-case tracking-normal">
                (label-renaming formal group)
              </span>
              <span className="block text-[10px] text-muted-foreground normal-case tracking-normal">
                (row-witnessed × row-unwitnessed)
              </span>
            </div>
          </div>
          {/* G_pt|_V column */}
          <div className="flex flex-col gap-0.5">
            {vSub.map((vElem, vi) => {
              const active = effectiveVIdx === vi;
              return (
                <button
                  type="button"
                  key={vi}
                  className={`${cellBase} ${cellNormal} flex items-center gap-2 text-left`}
                  style={active ? activeVStyle : undefined}
                  onMouseEnter={() => setHoveredVIdx(vi)}
                  onMouseLeave={() => setHoveredVIdx(null)}
                  onFocus={() => setHoveredVIdx(vi)}
                  onBlur={() => setHoveredVIdx(null)}
                >
                  <span className="w-5 shrink-0 text-right text-[11px] font-semibold text-muted-foreground">
                    {vi + 1}
                  </span>
                  <span>
                    <ColoredLabels
                      text={toCycleString(vElem, vLabels)}
                      vSet={vLabelSet}
                      wSet={wLabelSet}
                    />
                  </span>
                </button>
              );
            })}
          </div>

          {/* × separator (one per G_pt|_V row) */}
          <div className="flex flex-col gap-0.5">
            {vSub.map((_, vi) => (
              <div
                key={vi}
                className={sepBase}
                style={effectiveVIdx === vi ? { color: COLOR_V } : undefined}
                aria-hidden
              >
                ×
              </div>
            ))}
          </div>

          {/* S(W) column */}
          <div className="flex flex-col gap-0.5">
            {sw.map((wElem, wi) => {
              const active = effectiveWIdx === wi;
              return (
                <button
                  type="button"
                  key={wi}
                  className={`${cellBase} ${cellNormal} flex items-center gap-2 text-left`}
                  style={active ? activeWStyle : undefined}
                  onMouseEnter={() => setHoveredWIdx(wi)}
                  onMouseLeave={() => setHoveredWIdx(null)}
                  onFocus={() => setHoveredWIdx(wi)}
                  onBlur={() => setHoveredWIdx(null)}
                >
                  <span className="w-5 shrink-0 text-right text-[11px] font-semibold text-muted-foreground">
                    {wi + 1}
                  </span>
                  <span>
                    <ColoredLabels
                      text={toCycleString(wElem, wLabels)}
                      vSet={vLabelSet}
                      wSet={wLabelSet}
                    />
                  </span>
                </button>
              );
            })}
          </div>

          {/* = separator (one per product row) */}
          <div className="flex flex-col gap-0.5">
            {vSub.flatMap((_, vi) =>
              sw.map((_, wi) => {
                const productIdx = vi * swCount + wi;
                const active = isHighlightedProduct(productIdx);
                const dim = hoveredProductIdx !== null && hoveredProductIdx !== productIdx;
                return (
                  <div
                    key={productIdx}
                    className={sepBase}
                    style={{
                      color: active ? '#10B981' : undefined,
                      opacity: dim ? 0.3 : 1,
                    }}
                    aria-hidden
                  >
                    =
                  </div>
                );
              }),
            )}
          </div>

          {/* G_f product column */}
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
                      active ? '' : cellNormal
                    }`}
                    style={{
                      ...(active ? activeProductStyle : {}),
                      opacity: dim ? 0.4 : 1,
                    }}
                    onMouseEnter={() => setHoveredProductIdx(productIdx)}
                    onMouseLeave={() => setHoveredProductIdx(null)}
                    onFocus={() => setHoveredProductIdx(productIdx)}
                    onBlur={() => setHoveredProductIdx(null)}
                  >
                    <span className="w-10 shrink-0 text-right text-[10px] font-mono text-muted-foreground">
                      {vi + 1}·{wi + 1}
                    </span>
                    <span>
                      <ColoredLabels
                        text={toCycleString(elem, allLabels)}
                        vSet={vLabelSet}
                        wSet={wLabelSet}
                      />
                    </span>
                  </button>
                );
              }),
            )}
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ background: `${COLOR_V}33`, border: `1px solid ${COLOR_V}` }}
            />
            V-label
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ background: `${COLOR_W}33`, border: `1px solid ${COLOR_W}` }}
            />
            W-label
          </span>
          <span className="ml-auto italic">
            Hover a row in any column; the other two columns highlight the pairing.
          </span>
        </div>
      </div>
    </div>
  );
}
