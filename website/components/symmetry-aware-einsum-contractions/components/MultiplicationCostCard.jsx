import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import CaseBadge from './CaseBadge.jsx';
import RoleBadge from './RoleBadge.jsx';

/**
 * Teaching card: "Multiplication Events (μ)".
 *
 * Canonical vocabulary (established by the Counting Convention band above):
 *   Mₐ — per-component orbit count = |Xₐ/Gₐ|  (computed via size-aware Burnside)
 *   M  — global orbit count = ∏ₐ Mₐ
 *   μ  — Multiplication events = (num_terms − 1) · M = (num_terms − 1) · ∏ₐ Mₐ
 *   α  — Accumulation Cost (a separate quantity; see the Accumulation card)
 *
 * The card shows the per-component Mₐ list followed by the global aggregation
 * μ = (num_terms − 1) · ∏ₐ Mₐ. There is no per-component "μₐ" — the (k−1)
 * factor lives globally on the product of orbit counts, not on each component.
 *
 * A hover tooltip on the formula explains the two properties users most need
 * to internalise:
 *   1. Each Mₐ is structure-agnostic — it only sees Gₐ's action on Lₐ, not
 *      the V/W split, so the same Burnside sum covers every regime (trivial,
 *      all-visible, all-summed, and every mixed regime).
 *   2. (num_terms − 1) captures the k-operand chain cost:
 *      forming one product per orbit rep costs k−1 binary multiplies, applied
 *      once at the global level.
 */

// ─── Intuition tooltip ────────────────────────────────────────────────

function FormulaIntuitionTooltip({ anchorRect, onDismiss }) {
  const TOOLTIP_W = 440;
  const TOOLTIP_MIN_H = 320;

  const [pos, setPos] = useState(null);

  useEffect(() => {
    if (!anchorRect) {
      setPos(null);
      return;
    }

    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;

    let x = anchorRect.left + anchorRect.width / 2;
    x = Math.max(TOOLTIP_W / 2 + 12, Math.min(x, vw - TOOLTIP_W / 2 - 12));

    const roomAbove = anchorRect.top;
    const roomBelow = vh - anchorRect.bottom;
    const flipped = roomAbove < TOOLTIP_MIN_H + 16 && roomBelow > roomAbove;
    const y = flipped ? anchorRect.bottom + 8 : anchorRect.top - 8;

    setPos({ x, y, flipped });
  }, [anchorRect]);

  if (!anchorRect || !pos) return null;

  return (
    <div
      className="pointer-events-none fixed z-[9999] w-[440px] max-w-[calc(100vw-2rem)] rounded-xl border border-stone-200 bg-white px-4 py-3.5 text-stone-900 shadow-[0_24px_60px_rgba(15,23,42,0.16)]"
      style={{
        left: pos.x,
        top: pos.y,
        transform: pos.flipped
          ? 'translateX(-50%)'
          : 'translateX(-50%) translateY(-100%)',
      }}
      onClick={onDismiss}
      role="tooltip"
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="inline-block h-2.5 w-2.5 rounded-full bg-primary" />
        <span className="text-sm font-semibold">Why this equation is universal</span>
      </div>

      {/* Property 1 — structure-agnostic. */}
      <div className="text-[12px] leading-5 text-stone-700">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
          <InlineMathText>{String.raw`$M_a$ doesn't see the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split`}</InlineMathText>
        </div>
        <p>
          <InlineMathText>{String.raw`The Burnside sum only knows about $G_a$ acting on the component's label set $L_a$. It does not read the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split at all, so the same equation applies to every regime: trivial, all-visible, all-summed, and every mixed case. Contraction structure never enters $M_a$.`}</InlineMathText>
        </p>
      </div>

      {/* Property 2 — (k-1) factor. */}
      <div className="mt-3 border-t border-stone-200 pt-3 text-[12px] leading-5 text-stone-700">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
          Where <span className="font-mono normal-case">(num_terms − 1)</span> comes from
        </div>
        <p>
          <InlineMathText>{String.raw`$M = \prod_a M_a$ counts global orbits, one product **to compute** per orbit representative. Each product is a chain of num_terms operand entries, so it costs num_terms - 1 binary multiplies. The factor applies once globally, not per component.`}</InlineMathText>
        </p>
      </div>

      {/* Punchline connecting both. */}
      <div className="mt-3 border-t border-stone-200 pt-3 text-[11.5px] leading-5 text-stone-600">
        <span className="mr-1 font-semibold uppercase tracking-wider text-stone-500">Net</span>
        <span><InlineMathText>{String.raw`symmetry shrinks $M$; the einsum's operand count scales $M$ by num_terms - 1 to get $\mu$.`}</InlineMathText></span>
      </div>
    </div>
  );
}

// ─── Burnside Card (V3.1 §10) ─────────────────────────────────────────
//
// One row per element g of the per-component group G_a, showing:
//   - Group element (cycle notation on labels L_a, identity rendered as "e")
//   - Label cycles  (full cyclic form including 1-cycles, on label names)
//   - Fixed assignments  (∏ over each cycle of n_{label}, since all labels
//                          in a cycle must take the same value to be fixed)
//   - Contribution  (the same product, displayed numerically once sizes
//                    are known — this is what the Burnside sum averages)
//
// Below the table: M_a = (1/|G_a|) Σ_g |Fix(g)|.
//
// Hover writes the cycle's labels to the shared hoveredLabels bus. No dense
// grid exists yet (TODO: highlight grid cells for hovered element once a
// dense-grid view lands in this card).
function BurnsideTable({
  comp,
  testIdSuffix = '',
  onHoveredLabelsChange = null,
}) {
  const elements = comp?.elements ?? [];
  const labels = comp?.labels ?? [];
  const sizes = comp?.sizes ?? [];
  const groupOrder = elements.length || 1;

  const rows = useMemo(() => elements.map((g, idx) => {
    const cycles = g.fullCyclicForm();
    // Cycle as labels (for display + hover bus payload).
    const cycleLabels = cycles.map((c) => c.map((i) => labels[i]));
    // Cycle notation: identity → "e"; 1-cycles dropped (sympy convention).
    const movedCycles = cycleLabels.filter((c) => c.length > 1);
    const elementText = movedCycles.length === 0
      ? 'e'
      : movedCycles.map((c) => `(${c.join(' ')})`).join('');
    // Full cycle text including 1-cycles, e.g. "(i j)(k)".
    const fullCycleText = cycleLabels.map((c) => `(${c.join(' ')})`).join('');
    // Fixed-assignments factor list, e.g. ["n_i n_k"] — one factor per cycle.
    // All labels in a cycle must agree, so the cycle contributes n_{first}.
    const factors = cycles.map((c) => labels[c[0]]);
    const fixedSymbolic = factors.length === 0
      ? '1'
      : factors.map((lbl) => `n_{${lbl}}`).join(' \\cdot ');
    // Numeric fixed count (uses comp.sizes).
    const fixedNumeric = cycles.reduce((acc, c) => acc * (sizes[c[0]] ?? 1), 1);
    // Flat label set on this row (for the hover bus).
    const flatLabels = cycleLabels.flat();
    return {
      idx,
      elementText,
      fullCycleText,
      fixedSymbolic,
      fixedNumeric,
      flatLabels,
    };
  }), [elements, labels, sizes]);

  const burnsideTotal = rows.reduce((acc, r) => acc + r.fixedNumeric, 0);
  const burnsideAverage = burnsideTotal / groupOrder;

  if (elements.length === 0) return null;

  const writeHover = (set) => {
    if (typeof onHoveredLabelsChange === 'function') {
      onHoveredLabelsChange(set);
    }
  };

  const testId = testIdSuffix ? `burnside-table-${testIdSuffix}` : 'burnside-table';

  return (
    <div className="mt-3 rounded-md border border-stone-200 bg-white">
      <div className="flex items-center justify-between gap-2 border-b border-stone-200 px-3 py-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Burnside Table
        </span>
        <span
          className="cursor-help text-[11px] text-stone-500"
          title="Burnside counts products, not updates."
          aria-label="Burnside counts products, not updates."
          tabIndex={0}
          role="note"
        >
          (?)
        </span>
      </div>
      <div className="overflow-x-auto">
        <table
          data-testid={testId}
          className="w-full border-collapse text-left text-[12px]"
        >
        <thead>
          <tr className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            <th className="px-3 py-1.5 font-semibold">Group element</th>
            <th className="px-3 py-1.5 font-semibold">Label cycles</th>
            <th className="px-3 py-1.5 font-semibold">Fixed assignments</th>
            <th className="px-3 py-1.5 text-right font-semibold">Contribution</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={`burnside-row-${row.idx}`}
              className="border-t border-stone-100 cursor-pointer transition-colors hover:bg-stone-50 focus:bg-stone-50 focus:outline-none"
              tabIndex={0}
              onMouseEnter={() => writeHover(new Set(row.flatLabels))}
              onMouseLeave={() => writeHover(null)}
              onFocus={() => writeHover(new Set(row.flatLabels))}
              onBlur={() => writeHover(null)}
            >
              <td className="px-3 py-1.5 font-mono">{row.elementText}</td>
              <td className="px-3 py-1.5 font-mono text-stone-700">{row.fullCycleText}</td>
              <td className="px-3 py-1.5">
                <Latex math={row.fixedSymbolic} />
              </td>
              <td className="px-3 py-1.5 text-right font-mono text-stone-900">
                {row.fixedNumeric.toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
        </table>
      </div>
      <div className="border-t border-stone-200 px-3 py-2">
        <Latex
          math={String.raw`M_a \;=\; \frac{1}{|G_a|}\,\sum_{g \in G_a}\,|\mathrm{Fix}(g)| \;=\; ${burnsideAverage.toLocaleString()}`}
          display
        />
      </div>
    </div>
  );
}

// ─── C11 Product Savings Metric Row (V3.1 §11) ────────────────────────
//
// Four metric cards in a row, each surfacing one number that lives at
// the top of the Multiplication card so the reader sees the product-side
// savings story before they read the Burnside derivation:
//
//   1. Dense product chains  — ∏ over all labels of n_label (the dense
//                              baseline: one product per full assignment).
//   2. Product representatives M  — ∏_a M_a  (the global orbit-quotient).
//   3. Multiplication-chain events μ  — (num_terms − 1) · M.
//   4. Product-side reduction  — (1 − M / dense) · 100, signed honestly.
//
// Plus a warning pill: "Products are rows, not updates." This is the
// single most common reader confusion at this point in the page — they
// see big numbers and assume they're updates (α). The pill is clickable
// and scrolls back to the O→Q matrix where row vs column is settled.
//
// Hover wiring:
//   - M card  → writes the union of all component labels to the hover
//               bus so upstream views can highlight the product-orbit
//               rows. (Token: 'product-rep-M'.)
//   - μ card  → writes a special token 'mu-k-minus-1' so views that
//               know about it (e.g. the formula block below) can
//               emphasize the (k−1) factor.
// All four cards are tabIndex=0 / role="button" so the bus also fires
// for keyboard users.

const PILL_LABELS = {
  dense: 'Dense product chains',
  M: 'Product representatives M',
  mu: 'Multiplication-chain events mu',
  reduction: 'Product-side reduction',
};

const WARNING_PILL_TEXT = 'Products are rows, not updates.';

function MetricCard({ label, value, sublabel, accent, testId, onHover, onLeave }) {
  return (
    <div
      tabIndex={0}
      role="button"
      aria-label={`${label}: ${value}`}
      data-testid={testId}
      onMouseEnter={onHover}
      onMouseLeave={onLeave}
      onFocus={onHover}
      onBlur={onLeave}
      className="flex flex-col gap-0.5 rounded-md border border-stone-200 bg-white px-3 py-2 text-left transition-colors hover:border-stone-400 focus:border-stone-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
      style={accent ? { borderLeft: `3px solid ${accent}` } : undefined}
    >
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="font-mono text-sm font-semibold text-stone-900">
        {value}
      </div>
      {sublabel ? (
        <div className="text-[10px] text-stone-500">{sublabel}</div>
      ) : null}
    </div>
  );
}

function ProductSavingsMetricRow({
  components,
  numTerms,
  onHoveredLabelsChange,
}) {
  // Dense baseline = ∏_a ∏_{label in a} n_label — number of full label
  // assignments before symmetry. The dense direct path forms one product
  // chain per assignment.
  const dense = useMemo(() => {
    if (!components?.length) return null;
    let prod = 1;
    for (const comp of components) {
      const sizes = comp?.sizes ?? [];
      for (const n of sizes) {
        if (typeof n === 'number' && Number.isFinite(n)) prod *= n;
      }
    }
    return prod;
  }, [components]);

  // Global product-rep count M = ∏_a M_a.
  const M = useMemo(() => {
    if (!components?.length) return null;
    let prod = 1;
    let any = false;
    for (const comp of components) {
      const Ma = comp?.multiplicationCount ?? comp?.multiplication?.count ?? null;
      if (typeof Ma === 'number' && Number.isFinite(Ma)) {
        prod *= Ma;
        any = true;
      } else {
        return null;
      }
    }
    return any ? prod : null;
  }, [components]);

  // μ = (num_terms − 1) · M.
  const k = typeof numTerms === 'number' && numTerms > 0 ? numTerms : 2;
  const mu = M != null ? (k - 1) * M : null;

  // Product-side reduction: how much M shrinks the dense baseline.
  const reductionPct = useMemo(() => {
    if (typeof dense !== 'number' || dense <= 0) return null;
    if (typeof M !== 'number' || !Number.isFinite(M)) return null;
    return ((dense - M) / dense) * 100;
  }, [dense, M]);

  const fmt = (n) => {
    if (typeof n !== 'number' || !Number.isFinite(n)) return '—';
    return n.toLocaleString();
  };
  const fmtPct = (p) => {
    if (typeof p !== 'number' || !Number.isFinite(p)) return '—';
    return `${p.toFixed(1)}%`;
  };

  // Hover bus payloads.
  const allLabels = useMemo(() => {
    const set = new Set();
    for (const comp of components ?? []) {
      for (const lbl of comp?.labels ?? []) set.add(lbl);
    }
    return set;
  }, [components]);

  const writeHover = (payload) => {
    if (typeof onHoveredLabelsChange === 'function') {
      onHoveredLabelsChange(payload);
    }
  };

  const onHoverM = () => writeHover(allLabels.size ? new Set(allLabels) : null);
  const onLeaveM = () => writeHover(null);
  // μ hover writes a single sentinel label 'mu-k-minus-1' that downstream
  // views can branch on without disturbing the existing label-name bus.
  const onHoverMu = () => writeHover(new Set(['mu-k-minus-1']));
  const onLeaveMu = () => writeHover(null);

  return (
    <div
      data-testid="product-savings-metric-row"
      className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4"
    >
      <MetricCard
        label={PILL_LABELS.dense}
        value={fmt(dense)}
        sublabel="baseline"
        testId="product-savings-dense"
      />
      <MetricCard
        label={PILL_LABELS.M}
        value={fmt(M)}
        sublabel="prod_a M_a"
        accent="var(--ein-v, #2c5f7c)"
        testId="product-savings-M"
        onHover={onHoverM}
        onLeave={onLeaveM}
      />
      <MetricCard
        label={PILL_LABELS.mu}
        value={fmt(mu)}
        sublabel={`(k - 1) M, k = ${k}`}
        accent="var(--ein-v, #2c5f7c)"
        testId="product-savings-mu"
        onHover={onHoverMu}
        onLeave={onLeaveMu}
      />
      <MetricCard
        label={PILL_LABELS.reduction}
        value={fmtPct(reductionPct)}
        sublabel="vs dense"
        testId="product-savings-reduction"
      />
    </div>
  );
}

// Warning pill — clickable button that scrolls the O→Q matrix into view.
// V3.1 §11 verbatim copy: "Products are rows, not updates."
function ProductsAreRowsPill() {
  const onClick = useCallback(() => {
    if (typeof document === 'undefined') return;
    // Prefer the data-testid the matrix already exposes; fall back to
    // the BranchingDemo container by id, then to its anchor id.
    const target =
      document.querySelector('[data-testid="orbit-rep-matrix"]') ||
      document.querySelector('[data-orbit-rep-matrix]') ||
      document.getElementById('orbit-rep-matrix');
    if (target && typeof target.scrollIntoView === 'function') {
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Products are rows, not updates. Click to scroll to O→Q matrix."
      data-testid="products-are-rows-pill"
      className="mt-3 inline-flex items-center gap-2 rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-[11px] font-medium text-amber-900 transition-colors hover:bg-amber-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
    >
      <span aria-hidden="true">⚠</span>
      <span>{WARNING_PILL_TEXT}</span>
      <span aria-hidden="true" className="text-amber-700">→</span>
    </button>
  );
}

export default function MultiplicationCostCard({
  components = [],
  numTerms = 2,
  onHoveredLabelsChange = null,
}) {
  const formulaRef = useRef(null);
  const [anchorRect, setAnchorRect] = useState(null);
  const hideTimerRef = useRef(null);

  const openTooltip = useCallback(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
    if (formulaRef.current) {
      setAnchorRect(formulaRef.current.getBoundingClientRect());
    }
  }, []);

  const scheduleHide = useCallback(() => {
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    hideTimerRef.current = setTimeout(() => setAnchorRect(null), 80);
  }, []);

  useEffect(() => {
    if (!anchorRect) return undefined;
    const dismiss = () => setAnchorRect(null);
    const onKey = (e) => { if (e.key === 'Escape') dismiss(); };
    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('keydown', onKey);
    };
  }, [anchorRect]);

  useEffect(() => () => {
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
  }, []);

  return (
    <div id="multiplication-cost" className="bg-white p-4 scroll-mt-sticky">
      <ExplorerSubsectionHeader anchorId="multiplication-cost" labelText="Multiplication Events (μ)">
        <InlineMathText>{`Multiplication Events ($${String.raw`\mu`}$)`}</InlineMathText>
      </ExplorerSubsectionHeader>
      <p className="explorer-support-prose mt-2">
        <InlineMathText>{String.raw`Every component gets the same treatment. Size-aware Burnside counts the per-component orbit count $M_a$ once each; the multiplication cost is then $\mu = (\texttt{num\_terms} - 1)\cdot \prod_a M_a$, one product per global orbit representative, where forming that product costs num_terms - 1 binary multiplies. No regime dispatch is needed because the equation ignores the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split.`}</InlineMathText>
      </p>
      <p className="mt-3 text-[12px] leading-5 text-muted-foreground">
        <InlineMathText>{String.raw`$M$ counts representative product values. $\mu = (k-1)M$ counts multiplication-chain events. $\alpha$ counts accumulation updates from product-orbit representatives into stored output representatives.`}</InlineMathText>
      </p>

      {/* C11 Product Savings Metric Row — four cards summarising the
          dense baseline, the orbit-quotient M, μ, and the % reduction. */}
      <ProductSavingsMetricRow
        components={components}
        numTerms={numTerms}
        onHoveredLabelsChange={onHoveredLabelsChange}
      />
      <ProductsAreRowsPill />

      {/* Hover-wrapped formula — shows μ on top, the per-component
          Mₐ/Burnside identity below, so the reader sees how each
          per-component count rolls up to the global μ without leaving
          the card. */}
      <div
        ref={formulaRef}
        className="group relative mt-3 cursor-help rounded-md bg-white px-3 py-2.5 transition-colors hover:bg-white"
        onMouseEnter={openTooltip}
        onMouseLeave={scheduleHide}
        onFocus={openTooltip}
        onBlur={scheduleHide}
        tabIndex={0}
        role="button"
        aria-describedby="mult-formula-intuition"
      >
        <Latex
          math={String.raw`\mu \;=\; (\texttt{num\_terms} - 1)\,\cdot\, \prod_a M_a`}
          display
        />
        <div className="mt-1 text-[11px] text-muted-foreground text-center">
          where
        </div>
        <Latex
          math={String.raw`M_a \;=\; \frac{1}{|G_a|} \sum_{g \in G_a} \prod_{c \,\in\, \mathrm{cycles}(g)} n_c`}
          display
        />
      </div>

      {components.length > 0 && (
        <div className="mt-3 space-y-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Live for this example
          </div>
          {components.map((comp, i) => {
            const orbits = comp.multiplicationCount ?? comp.multiplication?.count ?? null;
            const hasGroupAction = Array.isArray(comp.elements) && comp.elements.length > 0;
            return (
              <React.Fragment key={`mult-${i}`}>
                <div className="flex items-center gap-2 text-xs">
                  <CaseBadge
                    regimeId={comp.accumulation?.regimeId ?? comp.shape}
                    size="xs"
                    variant="pill"
                  />
                  <span className="flex flex-1 flex-wrap items-center gap-1.5">
                    {(comp.labels?.length ? comp.labels : ['∅']).map((label) => {
                      if (label === '∅') {
                        return (
                          <span key={`empty-${i}`} className="text-gray-500">
                            ∅
                          </span>
                        );
                      }
                      const role = (comp.va ?? []).includes(label) ? 'v' : 'w';
                      return (
                        <RoleBadge key={`mult-${i}-${label}`} role={role}>
                          {label}
                        </RoleBadge>
                      );
                    })}
                  </span>
                  <span className="ml-auto font-mono text-gray-900">
                    <Latex math={`M_a = ${orbits != null ? orbits.toLocaleString() : '\\text{—}'}`} />
                  </span>
                </div>
                {hasGroupAction && (
                  <BurnsideTable
                    comp={comp}
                    testIdSuffix={String(i)}
                    onHoveredLabelsChange={onHoveredLabelsChange}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>
      )}

      <FormulaIntuitionTooltip
        anchorRect={anchorRect}
        onDismiss={() => setAnchorRect(null)}
      />
    </div>
  );
}
