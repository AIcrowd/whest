// V3.1 §40 — Cost Savings Spread (L5.T2.6 / C40).
//
// Standalone C40 card/table spread retained for the V3.1 component contract.
// Section 9's publish-ready page now uses the later EditorialComparisonSpread
// as the single visible dense-vs-symmetry summary, so TotalCostView does not
// mount this component directly.
//
//   Dense Direct            = (k-1)|X| + |X|
//   Symmetry-Aware Direct   = μ + α
//
// Below the columns: a small supporting-rows table that lights up the
// matching visual elsewhere in §5 on hover.
//
//   Product chains        — dense |X| → μ
//   Updates               — dense |X| → α
//   Active alpha method   — leaf badge + name
//   Speedup               — denseDirect / (μ+α)
//   Savings               — (1 − (μ+α)/denseDirect) × 100%
//
// Interactions:
//   • Hover a row → savingsCategoryBus publishes the row's category id.
//     Downstream visuals subscribe via useSyncExternalStore. The
//     active-method row also extends the existing alphaMethodBus so the
//     classification tree + partition counter highlight without a new bus.
//   • Toggle [Linear ↔ Log]. Log scale is used to render the bar widths
//     when the difference between dense and symmetry-aware spans many
//     orders of magnitude.
//
// Token discipline: colors via CSS variables on the explorer theme. No
// raw notation hex outside the `var(--…)` references in the TOKEN map
// (mirrors NaiveAlphaCostMeter.jsx, sibling).
//
// Accessibility: each row is focusable (tabIndex=0), carries an aria-label,
// and respects prefers-reduced-motion (no transition on the bar widths
// when the user opts in).

import { useEffect, useMemo, useState } from 'react';
import CaseBadge from './CaseBadge.jsx';
import { setActiveAlphaMethodBus } from '../lib/alphaMethodBus.js';
import { setActiveSavingsCategory } from '../lib/savingsCategoryBus.js';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map — CSS variables only.
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:      'var(--coral)',
  coralLight: 'var(--coral-light)',
  gray900:    'var(--gray-900)',
  gray700:    'var(--gray-700)',
  gray600:    'var(--gray-600)',
  gray500:    'var(--gray-500)',
  gray400:    'var(--gray-400)',
  gray200:    'var(--gray-200)',
  gray100:    'var(--gray-100)',
  gray50:     'var(--gray-50)',
  white:      'var(--white)',
};

const COLUMN_HEADINGS = {
  dense: 'Dense Direct',
  symmetryAware: 'Symmetry-Aware Direct',
};

/* ─────────────────────────────────────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────────────────────────────────────── */

/** prefers-reduced-motion gate. Mirrors TwoQuotientSchematic / OrbitRepMatrix. */
function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handler = (e) => setReduced(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return reduced;
}

/** Locale-aware integer formatter with a finite-number guard. */
function fmt(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return String(n);
  return n.toLocaleString();
}

/**
 * Map a non-negative integer to a 0-100% bar width.
 *
 *   linear → value / max
 *   log    → log(1 + value) / log(1 + max)
 *
 * Both forms guard against div-by-zero by returning 0 when max ≤ 0.
 */
function barWidthPct({ value, max, scale }) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return 0;
  if (typeof max !== 'number' || !Number.isFinite(max) || max <= 0) return 0;
  if (scale === 'log') {
    const num = Math.log1p(value);
    const den = Math.log1p(max);
    if (den <= 0) return 0;
    return Math.max(0, Math.min(100, (num / den) * 100));
  }
  return Math.max(0, Math.min(100, (value / max) * 100));
}

/* ─────────────────────────────────────────────────────────────────────────────
   ColumnCard — one of the two large side-by-side columns.
   ───────────────────────────────────────────────────────────────────────────── */
function ColumnCard({
  heading,
  formula,
  value,
  accentColor,
  barWidth,
  reducedMotion,
}) {
  return (
    <div
      className="flex flex-col items-center justify-between rounded-xl border bg-white px-5 py-6 text-center"
      style={{ borderColor: TOKEN.gray200 }}
      data-savings-column={heading}
    >
      <p
        className="text-[10px] font-semibold uppercase tracking-[0.2em]"
        style={{ color: TOKEN.gray500 }}
      >
        {heading}
      </p>
      <p
        className="mt-2 font-mono text-[12px] leading-5"
        style={{ color: TOKEN.gray600 }}
      >
        {formula}
      </p>
      <p
        className="mt-3 font-serif text-[40px] leading-[1.05] tracking-[-0.02em] tabular-nums"
        style={{ color: accentColor }}
        aria-live="polite"
        aria-atomic="true"
      >
        {fmt(value)}
      </p>
      <div
        className="mt-4 h-1.5 w-full overflow-hidden rounded-full"
        style={{ backgroundColor: TOKEN.gray100 }}
        role="presentation"
        aria-hidden="true"
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${barWidth}%`,
            backgroundColor: accentColor,
            transition: reducedMotion ? 'none' : 'width 0.35s ease-out',
          }}
        />
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   ROW_CLASSNAME — shared className for every supporting row. Defined once
   so each row body below stays focused on its literal data-savings-row="<id>"
   attribute (source-greppable per id, not hidden behind a prop).
   ───────────────────────────────────────────────────────────────────────────── */
const ROW_CLASSNAME =
  'grid cursor-default grid-cols-[1fr_auto_auto] items-baseline gap-x-4 rounded-md px-2 py-2 text-[12.5px] outline-none transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current hover:bg-stone-50';

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * CostSavingsSpread
 *
 * Props
 * ─────
 * mu                  — multiplication-chain count (number)
 * alpha               — accumulation count (number)
 * total               — μ + α (number)
 * denseBaseline       — dense-direct event cost (number)
 * k                   — operand-tensor count (number; 'k' in spec)
 * assignmentSpaceSize — |X|, full assignment space size (number)
 * activeAlphaMethod   — { regimeId?, label? } for the active α method
 */
export default function CostSavingsSpread({
  mu = 0,
  alpha = 0,
  total = 0,
  denseBaseline = 0,
  k = 0,
  assignmentSpaceSize = 0,
  activeAlphaMethod = null,
}) {
  const reducedMotion = usePrefersReducedMotion();
  const [scale, setScale] = useState('linear');

  // Derived metrics — guarded so a zero baseline doesn't NaN.
  const speedup = useMemo(() => {
    if (!Number.isFinite(denseBaseline) || !Number.isFinite(total) || total <= 0) {
      return 1;
    }
    return denseBaseline / total;
  }, [denseBaseline, total]);

  const savingsPct = useMemo(() => {
    if (!Number.isFinite(denseBaseline) || denseBaseline <= 0) return 0;
    if (!Number.isFinite(total)) return 0;
    return (1 - total / denseBaseline) * 100;
  }, [denseBaseline, total]);

  // Bar widths (relative to the larger of the two columns under the
  // current scale) so dense always anchors the right-hand edge in linear.
  const maxValue = Math.max(denseBaseline || 0, total || 0);
  const denseBarWidth = barWidthPct({ value: denseBaseline, max: maxValue, scale });
  const symAwareBarWidth = barWidthPct({ value: total, max: maxValue, scale });

  // Bus publish handlers — the active-method row ALSO extends alphaMethodBus
  // so the classification tree + partition counter highlight as if the user
  // were hovering the leaf directly.
  const handleRowActivate = (categoryId) => {
    setActiveSavingsCategory(categoryId);
    if (categoryId === 'active-method' && activeAlphaMethod?.regimeId) {
      setActiveAlphaMethodBus(activeAlphaMethod.regimeId);
    }
  };
  const handleRowDeactivate = () => {
    setActiveSavingsCategory(null);
    // Don't clobber alphaMethodBus on leave — the App owns the canonical
    // hover state and may have a different value live. We only PUBLISH on
    // enter for the active-method row; the App's existing setter clears
    // it normally when the user moves elsewhere.
  };

  // Active-method label — falls back to a neutral phrase if no method is
  // currently selected (which shouldn't happen on the live page, but the
  // component must render in isolation for tests/storybook).
  const methodLabel = activeAlphaMethod?.label ?? 'the selected α regime';
  const methodRegimeId = activeAlphaMethod?.regimeId ?? null;

  // Dense formula values (for the two product/updates rows). The dense
  // direct evaluator pays |X| once for the multiplication chain and |X|
  // once for the accumulation pass, mirroring the (k-1)|X| + |X| split.
  const denseProductChainCost = Math.max(k - 1, 0) * (assignmentSpaceSize || 0);
  const denseUpdateCost = assignmentSpaceSize || 0;

  return (
    <section
      data-testid="cost-savings-spread"
      aria-label="Cost savings spread"
      className="mx-auto max-w-[44rem] space-y-4"
    >
      {/* Scale toggle — Linear ↔ Log. Literal data-savings-scale attributes
          give source-grep tests + e2e selectors a stable hook. */}
      <div
        role="group"
        aria-label="Bar-width scale"
        className="flex items-center justify-end gap-2"
      >
        <span
          className="text-[10px] font-semibold uppercase tracking-[0.18em]"
          style={{ color: TOKEN.gray500 }}
        >
          Scale
        </span>
        <button
          type="button"
          data-savings-scale="linear"
          aria-pressed={scale === 'linear'}
          onClick={() => setScale('linear')}
          className="rounded-md border px-2 py-0.5 text-[11px] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
          style={{
            borderColor: scale === 'linear' ? TOKEN.coral : TOKEN.gray200,
            color: scale === 'linear' ? TOKEN.coral : TOKEN.gray600,
            backgroundColor: scale === 'linear' ? TOKEN.coralLight : TOKEN.white,
            fontWeight: scale === 'linear' ? 600 : 400,
          }}
        >
          Linear
        </button>
        <button
          type="button"
          data-savings-scale="log"
          aria-pressed={scale === 'log'}
          onClick={() => setScale('log')}
          className="rounded-md border px-2 py-0.5 text-[11px] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
          style={{
            borderColor: scale === 'log' ? TOKEN.coral : TOKEN.gray200,
            color: scale === 'log' ? TOKEN.coral : TOKEN.gray600,
            backgroundColor: scale === 'log' ? TOKEN.coralLight : TOKEN.white,
            fontWeight: scale === 'log' ? 600 : 400,
          }}
        >
          Log
        </button>
      </div>

      {/* Two large columns */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <ColumnCard
          heading={COLUMN_HEADINGS.dense}
          formula="(k − 1)·|X| + |X|"
          value={denseBaseline}
          accentColor={TOKEN.gray700}
          barWidth={denseBarWidth}
          reducedMotion={reducedMotion}
        />
        <ColumnCard
          heading={COLUMN_HEADINGS.symmetryAware}
          formula="μ + α"
          value={total}
          accentColor={TOKEN.coral}
          barWidth={symAwareBarWidth}
          reducedMotion={reducedMotion}
        />
      </div>

      {/* Supporting rows — hover lights the matching visual */}
      <div
        role="table"
        aria-label="Cost savings supporting rows"
        className="rounded-xl border bg-white px-2 py-2"
        style={{ borderColor: TOKEN.gray200 }}
      >
        {/* Header row */}
        <div
          role="row"
          className="grid grid-cols-[1fr_auto_auto] gap-x-4 px-2 pb-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
          style={{ color: TOKEN.gray500 }}
        >
          <span className="text-left">Component</span>
          <span className="text-right">Dense</span>
          <span className="text-right">Symmetry-aware</span>
        </div>
        <div
          className="mb-1 h-px"
          style={{ backgroundColor: TOKEN.gray200 }}
          aria-hidden="true"
        />

        {/* Row 1 — Product chains (dense |X| → μ). Hover highlights the
            product-orbit visuals via the savings-category bus. */}
        <div
          role="row"
          tabIndex={0}
          aria-label={`Product chains: dense ${fmt(denseProductChainCost)}, symmetry-aware mu ${fmt(mu)}`}
          data-savings-row="product-chains"
          className={ROW_CLASSNAME}
          style={{ color: TOKEN.gray700 }}
          onMouseEnter={() => handleRowActivate('product-chains')}
          onMouseLeave={() => handleRowDeactivate('product-chains')}
          onFocus={() => handleRowActivate('product-chains')}
          onBlur={() => handleRowDeactivate('product-chains')}
        >
          <span className="text-left">Product chains</span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray500 }}
          >
            {fmt(denseProductChainCost)}
          </span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray900 }}
          >
            {fmt(mu)}
          </span>
        </div>

        {/* Row 2 — Updates (dense |X| → α). Hover highlights the O→Q matrix. */}
        <div
          role="row"
          tabIndex={0}
          aria-label={`Updates: dense ${fmt(denseUpdateCost)}, symmetry-aware alpha ${fmt(alpha)}`}
          data-savings-row="updates"
          className={ROW_CLASSNAME}
          style={{ color: TOKEN.gray700 }}
          onMouseEnter={() => handleRowActivate('updates')}
          onMouseLeave={() => handleRowDeactivate('updates')}
          onFocus={() => handleRowActivate('updates')}
          onBlur={() => handleRowDeactivate('updates')}
        >
          <span className="text-left">Updates</span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray500 }}
          >
            {fmt(denseUpdateCost)}
          </span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray900 }}
          >
            {fmt(alpha)}
          </span>
        </div>

        {/* Row 3 — Active alpha method. Also extends alphaMethodBus so the
            classification tree + partition counter highlight on hover. */}
        <div
          role="row"
          tabIndex={0}
          aria-label={`Active alpha method: ${methodLabel}`}
          data-savings-row="active-method"
          className={ROW_CLASSNAME}
          style={{ color: TOKEN.gray700 }}
          onMouseEnter={() => handleRowActivate('active-method')}
          onMouseLeave={() => handleRowDeactivate('active-method')}
          onFocus={() => handleRowActivate('active-method')}
          onBlur={() => handleRowDeactivate('active-method')}
        >
          <span className="text-left">Active alpha method</span>
          <span
            className="text-right font-mono"
            style={{ color: TOKEN.gray400 }}
          >
            —
          </span>
          <span className="flex items-center justify-end gap-2">
            {methodRegimeId ? (
              <CaseBadge
                regimeId={methodRegimeId}
                size="xs"
                themeOverride="editorial-noir-math"
                presentationThemeOverride={null}
              />
            ) : (
              <span
                className="font-mono text-[12px]"
                style={{ color: TOKEN.gray700 }}
              >
                {methodLabel}
              </span>
            )}
          </span>
        </div>

        {/* Row 4 — Speedup. Live = denseBaseline / total. */}
        <div
          role="row"
          tabIndex={0}
          aria-label={`Speedup: dense 1.0x, symmetry-aware ${speedup.toFixed(1)}x`}
          data-savings-row="speedup"
          className={ROW_CLASSNAME}
          style={{ color: TOKEN.gray700 }}
          onMouseEnter={() => handleRowActivate('speedup')}
          onMouseLeave={() => handleRowDeactivate('speedup')}
          onFocus={() => handleRowActivate('speedup')}
          onBlur={() => handleRowDeactivate('speedup')}
        >
          <span className="text-left">Speedup</span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray500 }}
          >
            1.0×
          </span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray900 }}
          >
            {`${speedup.toFixed(1)}×`}
          </span>
        </div>

        {/* Row 5 — Savings. Live = (1 - total / denseBaseline) × 100%. */}
        <div
          role="row"
          tabIndex={0}
          aria-label={`Savings: dense 0.0%, symmetry-aware ${savingsPct.toFixed(1)}%`}
          data-savings-row="savings"
          className={ROW_CLASSNAME}
          style={{ color: TOKEN.gray700 }}
          onMouseEnter={() => handleRowActivate('savings')}
          onMouseLeave={() => handleRowDeactivate('savings')}
          onFocus={() => handleRowActivate('savings')}
          onBlur={() => handleRowDeactivate('savings')}
        >
          <span className="text-left">Savings</span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray500 }}
          >
            0.0%
          </span>
          <span
            className="text-right font-mono tabular-nums"
            style={{ color: TOKEN.gray900 }}
          >
            {`${savingsPct.toFixed(1)}%`}
          </span>
        </div>
      </div>
    </section>
  );
}

// Exported for tests — pins the verbatim column headings.
export { COLUMN_HEADINGS };
