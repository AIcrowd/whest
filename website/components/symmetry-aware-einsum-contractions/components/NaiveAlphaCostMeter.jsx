/**
 * NaiveAlphaCostMeter — §7 Counting Shortcuts (C28)
 *
 * Motivates the classification tree by showing the cost of literal
 * incidence-counting: how many tuple/group touches, product orbits, and
 * projected outputs are canonicalized when computing α naively.
 *
 * Four metrics (all live from dimensionN + preset labels):
 *   1. product orbits         ≈ tupleSpace / |G|        (rough Burnside lower bound)
 *   2. tuple/group touches    = tupleSpace * |G|         (canonicalization cost)
 *   3. projected outputs canonicalized = productOrbits * |H|  (output-rep cost)
 *   4. interactive budget     = 1 000 000               (constant threshold)
 *
 * Four-tier gauge (small / feasible / expensive / unavailable) keyed on
 * tuple/group touches against the budget.
 *
 * "Show literal algorithm" toggle expands a Python pseudocode block.
 *
 * All colours via design-system tokens. No raw hex outside TOKEN.
 * Accessibility: aria-expanded on toggle, role/tabindex on gauge segments.
 */

import { useState, useCallback, useEffect, useRef } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors colors_and_type.css tier 1 + 3A)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:      'var(--coral)',        // #F0524D
  coralLight: 'var(--coral-light)',  // #FEF2F1
  gray900:    'var(--gray-900)',     // #292C2D
  gray700:    'var(--gray-700)',     // #3F4142
  gray600:    'var(--gray-600)',     // #5D5F60
  gray500:    'var(--gray-500)',     // #888B8D
  gray400:    'var(--gray-400)',     // #AAACAD
  gray200:    'var(--gray-200)',     // #D9DCDC
  gray100:    'var(--gray-100)',     // #F1F3F5
  gray50:     'var(--gray-50)',      // #F8F9F9
  white:      'var(--white)',        // #FFFFFF
  einV:       'var(--ein-v)',        // #4A7CFF  (free-label / visible)
};

/* ─────────────────────────────────────────────────────────────────────────────
   Gauge tier definitions
   ───────────────────────────────────────────────────────────────────────────── */
const TIERS = [
  {
    id: 'small',
    label: 'small',
    threshold: 1e3,
    color: TOKEN.einV,
    bg: TOKEN.gray100,
    description: 'Fewer than 1,000 tuple/group touches — α is trivially enumerable.',
  },
  {
    id: 'feasible',
    label: 'feasible',
    threshold: 1e5,
    color: TOKEN.gray700,
    bg: TOKEN.gray200,
    description: 'Under 100,000 touches — manageable with a tight loop but shortcuts help.',
  },
  {
    id: 'expensive',
    label: 'expensive',
    threshold: 1e6,
    color: TOKEN.coralLight,
    bg: TOKEN.coralLight,
    description: 'Approaching 1M touches — the interactive budget limit. Classification shortcuts are valuable here.',
  },
  {
    id: 'unavailable',
    label: 'unavailable',
    threshold: Infinity,
    color: TOKEN.coral,
    bg: TOKEN.coral,
    description: '1M or more touches — naive enumeration is infeasible. The classification tree is necessary.',
  },
];

const INTERACTIVE_BUDGET = 1_000_000;

/* ─────────────────────────────────────────────────────────────────────────────
   Cost calculation helpers
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * Compute naive incidence-counting cost metrics from dimensionN and
 * the preset's label / group information.
 *
 * Parameters:
 *   dimensionN   — active dimension size (positive integer)
 *   allLabels    — array of all label strings (e.g. ['i','j','k'])
 *   groupSize    — |G_pt|: number of pointwise-group elements
 *   hSize        — |H|: number of output-symmetry elements (Stab restriction)
 *
 * Returns object with { tupleSpace, productOrbits, tupleGroupTouches,
 *                        projectedOutputsCanonicalized, interactiveBudget }.
 */
function computeNaiveCost({ dimensionN, allLabels, groupSize, hSize }) {
  const n = Math.max(1, dimensionN);
  const k = allLabels.length;
  const tupleSpace = Math.pow(n, k);
  // Rough Burnside lower bound: orbit count ≈ tupleSpace / groupSize
  const productOrbits = Math.ceil(tupleSpace / Math.max(1, groupSize));
  // Total canonicalization work: each assignment touched by each group element
  const tupleGroupTouches = tupleSpace * Math.max(1, groupSize);
  // Output-rep cost: each product orbit may fan out to |H| stored output reps
  const projectedOutputsCanonicalized = productOrbits * Math.max(1, hSize);

  return {
    tupleSpace,
    productOrbits,
    tupleGroupTouches,
    projectedOutputsCanonicalized,
    interactiveBudget: INTERACTIVE_BUDGET,
  };
}

/** Pick the active gauge tier based on tupleGroupTouches. */
function activeTier(tupleGroupTouches) {
  if (tupleGroupTouches < 1e3) return 'small';
  if (tupleGroupTouches < 1e5) return 'feasible';
  if (tupleGroupTouches < 1e6) return 'expensive';
  return 'unavailable';
}

/** Format large numbers in a compact, readable way. */
function fmt(n) {
  if (!Number.isFinite(n) || n > 1e18) return '> 10¹⁸';
  if (n >= 1e12) return `${(n / 1e12).toPrecision(3)}T`;
  if (n >= 1e9)  return `${(n / 1e9).toPrecision(3)}B`;
  if (n >= 1e6)  return `${(n / 1e6).toPrecision(3)}M`;
  if (n >= 1e3)  return `${(n / 1e3).toPrecision(3)}k`;
  return String(Math.round(n));
}

/* ─────────────────────────────────────────────────────────────────────────────
   Pseudocode literal algorithm
   ───────────────────────────────────────────────────────────────────────────── */
const PSEUDOCODE = `alpha = 0
for O in product_orbits(X, G_pt):
    reached = set()
    for x in members(O):
        y = project_to_V(x)        # drop summed labels
        reached.add(canonical_output_rep(y, H))
    alpha += len(reached)`;

/* ─────────────────────────────────────────────────────────────────────────────
   Gauge tooltip (hover → fixed portal)
   ───────────────────────────────────────────────────────────────────────────── */
function GaugeTooltip({ anchorRect, content }) {
  const TOOLTIP_W = 320;

  const [pos, setPos] = useState(null);

  useEffect(() => {
    if (!anchorRect) { setPos(null); return; }
    const vw = document.documentElement.clientWidth;
    let x = anchorRect.left + anchorRect.width / 2;
    x = Math.max(TOOLTIP_W / 2 + 8, Math.min(x, vw - TOOLTIP_W / 2 - 8));
    const y = anchorRect.top - 8;
    setPos({ x, y });
  }, [anchorRect]);

  if (!anchorRect || !pos) return null;

  return (
    <div
      className="pointer-events-none fixed z-[9999] max-w-[calc(100vw-2rem)] rounded-lg border border-stone-200 bg-white px-3 py-2.5 text-xs leading-5 text-stone-700 shadow-[0_8px_32px_rgba(15,23,42,0.14)]"
      style={{
        left: pos.x,
        top: pos.y,
        width: TOOLTIP_W,
        transform: 'translateX(-50%) translateY(-100%)',
      }}
      role="tooltip"
      id="naive-cost-gauge-tooltip"
    >
      {content}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Gauge bar — 4 segments, active one highlighted
   ───────────────────────────────────────────────────────────────────────────── */
function GaugeBar({ active, onHover }) {
  return (
    <div
      className="mt-4"
      role="group"
      aria-label="Cost tier gauge"
    >
      <div className="flex gap-1" role="list">
        {TIERS.map((tier) => {
          const isActive = tier.id === active;
          return (
            <div
              key={tier.id}
              role="listitem"
              tabIndex={0}
              aria-label={`Cost tier: ${tier.label}${isActive ? ' (active)' : ''}`}
              aria-current={isActive ? 'true' : undefined}
              className="flex-1 cursor-default select-none rounded-sm py-1.5 text-center text-[10px] font-semibold uppercase tracking-wide transition-opacity focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
              style={{
                backgroundColor: isActive ? tier.color : TOKEN.gray100,
                color: isActive ? TOKEN.white : TOKEN.gray500,
                opacity: isActive ? 1 : 0.55,
              }}
              onMouseEnter={(e) => onHover(tier.description, e.currentTarget.getBoundingClientRect())}
              onMouseLeave={() => onHover(null, null)}
              onFocus={(e) => onHover(tier.description, e.currentTarget.getBoundingClientRect())}
              onBlur={() => onHover(null, null)}
            >
              {tier.label}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Metric row — horizontal log-scaled bar + numeric badge.
   Visual register mirrors TuplePatternMeter.BarRow (label left, log-scaled
   bar middle, mono value right) so the two §7 / §8 cost meters read in the
   same idiom rather than two different "cost dashboards" side by side.
   The earlier label+value-only MetricRow read more like a key/value table
   than a comparable-magnitude visualization.
   ───────────────────────────────────────────────────────────────────────────── */
function MetricRow({ label, value, rawValue, maxValue, accent, prefersReducedMotion }) {
  const numeric = typeof rawValue === 'number' ? rawValue : null;
  const isUnavailable = numeric === null || Number.isNaN(numeric);
  // log10(x + 1) keeps small values visible against a 1M budget.
  const logMax = Math.log10(Math.max(2, (maxValue ?? 1) + 1));
  const logVal = isUnavailable ? 0 : Math.log10(numeric + 1);
  const widthPct = isUnavailable ? 0 : Math.max(2, (logVal / logMax) * 100);
  const fill = accent || TOKEN.gray500;

  return (
    <div className="flex items-center gap-3 py-1.5">
      <span
        className="w-[180px] shrink-0 text-[11.5px] leading-5"
        style={{ color: TOKEN.gray700 }}
      >
        {label}
      </span>
      <div
        className="relative h-5 flex-1 overflow-hidden rounded-sm"
        style={{ backgroundColor: TOKEN.gray100 }}
        aria-label={`${label}: ${value}`}
      >
        <div
          style={{
            width: `${widthPct}%`,
            height: '100%',
            backgroundColor: fill,
            transition: prefersReducedMotion ? 'none' : 'width 0.35s ease-out',
          }}
          aria-hidden="true"
        />
      </div>
      <span
        className="w-[78px] shrink-0 text-right font-mono text-[11.5px] font-semibold tabular-nums"
        style={{ color: isUnavailable ? TOKEN.gray400 : (accent || TOKEN.gray900) }}
        aria-live="polite"
        aria-atomic="true"
      >
        {value}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * NaiveAlphaCostMeter
 *
 * Props:
 *   dimensionN   {number}   active dimension (from App state)
 *   allLabels    {string[]} all label strings from group.allLabels
 *   groupSize    {number}   |G_pt| — pointwise group element count
 *   hSize        {number}   |H| — output-symmetry group element count
 */
export default function NaiveAlphaCostMeter({
  dimensionN = 3,
  allLabels = [],
  groupSize = 1,
  hSize = 1,
}) {
  const [showCode, setShowCode] = useState(false);
  const [tooltipContent, setTooltipContent] = useState(null);
  const [tooltipAnchor, setTooltipAnchor] = useState(null);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
  const codeId = 'naive-alpha-pseudocode';
  const buttonRef = useRef(null);

  // Honour prefers-reduced-motion for bar growth animations (mirrors
  // TuplePatternMeter so the two §7 / §8 cost meters behave identically).
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mq.matches);
    const handler = (e) => setPrefersReducedMotion(e.matches);
    mq.addEventListener?.('change', handler);
    return () => mq.removeEventListener?.('change', handler);
  }, []);

  const cost = computeNaiveCost({ dimensionN, allLabels, groupSize, hSize });
  const tier = activeTier(cost.tupleGroupTouches);
  const activeTierObj = TIERS.find((t) => t.id === tier);
  // Common max for bar log-scaling. The 1M interactive budget is NOT
  // included here — it would dominate the scale and compress every real
  // cost into a tiny sliver. The gauge strip below already surfaces
  // budget compliance via the small/feasible/expensive/unavailable tier
  // pill, so the budget doesn't need its own bar to compete with costs.
  const barMax = Math.max(
    cost.productOrbits ?? 0,
    cost.tupleGroupTouches ?? 0,
    cost.projectedOutputsCanonicalized ?? 0,
    1,
  );

  const handleGaugeHover = useCallback((content, rect) => {
    setTooltipContent(content);
    setTooltipAnchor(rect);
  }, []);

  const handleToggle = useCallback(() => {
    setShowCode((v) => !v);
  }, []);

  // Dismiss tooltip on scroll / resize
  useEffect(() => {
    if (!tooltipAnchor) return undefined;
    const dismiss = () => { setTooltipContent(null); setTooltipAnchor(null); };
    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
    };
  }, [tooltipAnchor]);

  // Escape dismisses the pseudocode panel and returns focus
  useEffect(() => {
    if (!showCode) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setShowCode(false);
        buttonRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showCode]);

  return (
    <div
      className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm"
      data-testid="naive-alpha-cost-meter"
    >
      {/* Header */}
      <div className="mb-3">
        <p
          className="text-[10px] font-semibold uppercase tracking-[0.15em]"
          style={{ color: TOKEN.gray500 }}
        >
          Naive incidence enumeration
        </p>
        <p
          className="mt-0.5 text-[11.5px] leading-5"
          style={{ color: TOKEN.gray600 }}
        >
          Without shortcuts: N tuple/group touches; with shortcuts: M pattern orbits &mdash; same &alpha;.
        </p>
      </div>

      {/* Divider */}
      <div
        className="mb-3 h-px"
        style={{ backgroundColor: TOKEN.gray200 }}
        aria-hidden="true"
      />

      {/* Metrics — horizontal bars matching TuplePatternMeter's visual
          register. Same color grammar as §8: coral for the
          after-shortcuts count (product orbits — the savings story),
          gray-400 for baseline naive-enumeration costs (tuple/group
          touches, projected outputs canonicalized). gray-400 is the
          token that actually resolves in the explorer theme — gray-700
          rendered transparent. */}
      <div role="list" aria-label="Naive enumeration cost metrics">
        <div role="listitem">
          <MetricRow
            label="product orbits"
            value={fmt(cost.productOrbits)}
            rawValue={cost.productOrbits}
            maxValue={barMax}
            accent={TOKEN.coral}
            prefersReducedMotion={prefersReducedMotion}
          />
        </div>
        <div role="listitem">
          <MetricRow
            label="tuple/group touches"
            value={fmt(cost.tupleGroupTouches)}
            rawValue={cost.tupleGroupTouches}
            maxValue={barMax}
            accent={TOKEN.gray400}
            prefersReducedMotion={prefersReducedMotion}
          />
        </div>
        <div role="listitem">
          <MetricRow
            label="projected outputs canonicalized"
            value={fmt(cost.projectedOutputsCanonicalized)}
            rawValue={cost.projectedOutputsCanonicalized}
            maxValue={barMax}
            accent={TOKEN.gray400}
            prefersReducedMotion={prefersReducedMotion}
          />
        </div>
        {/* The "interactive budget" bar (always 1M) was removed — it
            saturated the bar scale, compressing all real cost bars into
            slivers. Budget compliance is now surfaced solely through the
            tier strip beneath (small / feasible / expensive / unavailable),
            which is what V3.1 §C28 asks for anyway. */}
      </div>

      {/* Gauge bar */}
      <GaugeBar active={tier} onHover={handleGaugeHover} />

      {/* Divider */}
      <div
        className="mb-3 mt-4 h-px"
        style={{ backgroundColor: TOKEN.gray200 }}
        aria-hidden="true"
      />

      {/* Toggle button */}
      <button
        ref={buttonRef}
        type="button"
        aria-expanded={showCode}
        aria-controls={codeId}
        onClick={handleToggle}
        className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-[12px] font-medium transition-colors hover:bg-stone-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
        style={{ color: TOKEN.gray700 }}
      >
        <span>Show literal algorithm</span>
        <span
          aria-hidden="true"
          className="text-[10px]"
          style={{
            display: 'inline-block',
            transform: showCode ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s',
          }}
        >
          ▼
        </span>
      </button>

      {/* Pseudocode block */}
      {showCode && (
        <div id={codeId} className="mt-2">
          <pre
            className="overflow-x-auto rounded-lg p-3 text-[11.5px] leading-6"
            style={{
              backgroundColor: TOKEN.gray100,
              color: TOKEN.gray900,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            }}
          >
            <code>{PSEUDOCODE}</code>
          </pre>
          <p
            className="mt-2 text-[11px] leading-5"
            style={{ color: TOKEN.gray500 }}
          >
            Each iteration of the inner loop canonicalizes one full assignment under G_pt
            to find its product-orbit representative, then projects to V to find the
            stored output representative under H. Total work: tupleSpace &times; |G_pt|.
          </p>
        </div>
      )}

      {/* Portal tooltip */}
      <GaugeTooltip anchorRect={tooltipAnchor} content={tooltipContent} />
    </div>
  );
}
