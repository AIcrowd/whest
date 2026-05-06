/**
 * CertificationSummaryStrip — V3.1 §27 (C27)
 *
 * A bridge strip that lives at the bottom of the §3 Certification section.
 * Surfaces the seven headline numbers the certifier produces (candidate
 * σ moves, accepted witnesses, identity-only, rejected, |G_pt|, H, and
 * components) so the reader can take stock before continuing into the
 * cost / classification panels below.
 *
 * Each pill is hover-aware: hovering a pill briefly highlights the
 * upstream panel it summarizes (wreath view, audit table, generator
 * closure, column-action panel, label interaction graph). Highlight
 * targets that don't exist on the page silently no-op — the spec
 * explicitly calls out this fallback.
 *
 * Pill → highlight target:
 *   candidate σ moves    → #wreath-structure
 *   accepted witnesses   → .witness-gallery-mount  (audit table)
 *   identity-only        → .witness-gallery-mount  (audit table)
 *   rejected             → .witness-gallery-mount  (audit table)
 *   |G_pt|               → #generator-construction (generator closure)
 *   H                    → (column-action panel — no-op when absent)
 *   components           → writes activeComponentId via prop callback
 *
 * All colors via TOKEN map (CSS variables). No raw notation hex.
 * Accessibility: each pill has tabIndex=0, role="button", aria-label.
 */

import { useCallback, useState } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors colors_and_type.css tier 1 + 3A)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:       'var(--coral)',
  coralLight:  'var(--coral-light)',
  gray900:     'var(--gray-900)',
  gray700:     'var(--gray-700)',
  gray600:     'var(--gray-600)',
  gray500:     'var(--gray-500)',
  gray400:     'var(--gray-400)',
  gray200:     'var(--gray-200)',
  gray100:     'var(--gray-100)',
  gray50:      'var(--gray-50)',
  white:       'var(--white)',
  einV:        'var(--ein-v)',
  einW:        'var(--ein-w)',
  success:     'var(--success, var(--ein-v))',
};

/* ─────────────────────────────────────────────────────────────────────────────
   DOM-based highlight bus — adds a transient class to the target element so
   the upstream panel pulses when the user hovers a pill. We do this via
   plain DOM (no React subscription) because the strip is a passive bridge:
   it should not own any of the panels' state.

   Targets are queried by selector when the user hovers; if no element
   matches (e.g. column-action panel doesn't exist on this page) the call
   silently no-ops — exactly the behavior V3.1 §27 calls for.
   ───────────────────────────────────────────────────────────────────────────── */
const HIGHLIGHT_CLASS = 'cert-summary-strip-highlight';

function setHighlight(selector, on) {
  if (!selector || typeof document === 'undefined') return;
  const el = document.querySelector(selector);
  if (!el) return; // no-op when target doesn't exist
  if (on) el.classList.add(HIGHLIGHT_CLASS);
  else el.classList.remove(HIGHLIGHT_CLASS);
}

/* ─────────────────────────────────────────────────────────────────────────────
   V3.1 §27 verbatim labels — DO NOT EDIT. The seven pills, in order.
   ───────────────────────────────────────────────────────────────────────────── */
const PILL_LABELS = {
  candidateMoves:   'candidate sigma moves',
  accepted:         'accepted witnesses',
  identityOnly:     'identity-only',
  rejected:         'rejected',
  gPtSize:          '|G_pt|',
  hSize:            'H',
  components:       'components',
};

/* ─────────────────────────────────────────────────────────────────────────────
   Single pill — keyboard-focusable, hover-aware. Renders label + value.
   ───────────────────────────────────────────────────────────────────────────── */
function MetricPill({
  label,
  value,
  accent,
  ariaLabel,
  onHover,
  onLeave,
  testId,
}) {
  const [isHover, setIsHover] = useState(false);

  const fireOn = useCallback(() => {
    setIsHover(true);
    onHover?.();
  }, [onHover]);

  const fireOff = useCallback(() => {
    setIsHover(false);
    onLeave?.();
  }, [onLeave]);

  return (
    <div
      tabIndex={0}
      role="button"
      aria-label={ariaLabel}
      data-testid={testId}
      onMouseEnter={fireOn}
      onMouseLeave={fireOff}
      onFocus={fireOn}
      onBlur={fireOff}
      className="flex h-full flex-col items-start justify-between gap-1 rounded-md border px-2.5 py-1.5 text-left transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
      style={{
        borderColor: isHover ? accent || TOKEN.gray400 : TOKEN.gray200,
        backgroundColor: isHover ? TOKEN.gray50 : TOKEN.white,
        minWidth: 96,
        cursor: 'default',
      }}
    >
      <span
        className="text-[9.5px] font-semibold uppercase tracking-[0.12em]"
        style={{ color: TOKEN.gray500 }}
      >
        {label}
      </span>
      <span className="flex items-baseline gap-1">
        <span
          className="text-[9px] font-normal uppercase tracking-wider"
          style={{ color: TOKEN.gray400 }}
        >
          live
        </span>
        <span
          className="font-mono text-[14px] font-semibold tabular-nums"
          style={{ color: accent || TOKEN.gray900 }}
          aria-live="polite"
        >
          {value}
        </span>
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * CertificationSummaryStrip
 *
 * Props:
 *   candidateMoves        {number} non-skipped σ candidates (wreath enumeration)
 *   accepted              {number} accepted witnesses (isValid + non-skipped)
 *   identityOnly          {number} identity (skipped) σ entries
 *   rejected              {number} rejected σ entries (non-skipped, !isValid)
 *   gPtSize               {number} |G_pt| — pointwise group element count
 *   hSize                 {number} |H|    — column-action stabilizer size
 *   componentsCount       {number} number of independent label components
 *   setActiveComponentId  {(id|null) => void}  bus writer for the components hover
 */
export default function CertificationSummaryStrip({
  candidateMoves = 0,
  accepted = 0,
  identityOnly = 0,
  rejected = 0,
  gPtSize = 1,
  hSize = 1,
  componentsCount = 0,
  setActiveComponentId = null,
}) {
  // Memoized highlight handlers per pill. Each writes a transient class to
  // the upstream target so the user sees which panel a metric came from.
  const onCandHover  = useCallback(() => setHighlight('#wreath-structure', true), []);
  const onCandLeave  = useCallback(() => setHighlight('#wreath-structure', false), []);

  const onAccHover   = useCallback(() => setHighlight('.witness-gallery-mount', true), []);
  const onAccLeave   = useCallback(() => setHighlight('.witness-gallery-mount', false), []);

  const onIdHover    = useCallback(() => setHighlight('.witness-gallery-mount', true), []);
  const onIdLeave    = useCallback(() => setHighlight('.witness-gallery-mount', false), []);

  const onRejHover   = useCallback(() => setHighlight('.witness-gallery-mount', true), []);
  const onRejLeave   = useCallback(() => setHighlight('.witness-gallery-mount', false), []);

  const onGptHover   = useCallback(() => setHighlight('#generator-construction', true), []);
  const onGptLeave   = useCallback(() => setHighlight('#generator-construction', false), []);

  // Column-action panel is not rendered on this page — these calls no-op.
  const onHHover     = useCallback(() => setHighlight('#column-action-panel', true), []);
  const onHLeave     = useCallback(() => setHighlight('#column-action-panel', false), []);

  // Components pill writes to the existing activeComponentId bus.
  // Sentinel '*' means "all components" (highlights the whole label
  // interaction graph). The bus accepts any string identifier; downstream
  // ComponentView treats unknown ids as "no component matched", which
  // gives a soft halo on the graph wrapper without fighting per-hull state.
  const onCompsHover = useCallback(() => {
    if (typeof setActiveComponentId === 'function') setActiveComponentId('*');
  }, [setActiveComponentId]);
  const onCompsLeave = useCallback(() => {
    if (typeof setActiveComponentId === 'function') setActiveComponentId(null);
  }, [setActiveComponentId]);

  return (
    <div
      data-testid="certification-summary-strip"
      role="group"
      aria-label="Certification summary — seven live metrics produced by the σ-loop and pointwise-group construction"
      className="rounded-xl border border-stone-200 bg-white p-3 shadow-sm"
    >
      <div className="mb-2">
        <p
          className="text-[10px] font-semibold uppercase tracking-[0.15em]"
          style={{ color: TOKEN.gray500 }}
        >
          Certification summary
        </p>
        <p
          className="mt-0.5 text-[11.5px] leading-5"
          style={{ color: TOKEN.gray600 }}
        >
          Live tally of what the σ-loop produced; hover a pill to spotlight the panel it came from.
        </p>
      </div>

      <div
        className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7"
        role="list"
        aria-label="Certification summary metrics"
      >
        <div role="listitem">
          <MetricPill
            label={PILL_LABELS.candidateMoves}
            value={candidateMoves.toLocaleString()}
            accent={TOKEN.gray700}
            ariaLabel={`Candidate sigma moves: ${candidateMoves}. Hover or focus to highlight the wreath structure view.`}
            testId="cert-summary-pill-candidate-moves"
            onHover={onCandHover}
            onLeave={onCandLeave}
          />
        </div>

        <div role="listitem">
          <MetricPill
            label={PILL_LABELS.accepted}
            value={accepted.toLocaleString()}
            accent={TOKEN.success}
            ariaLabel={`Accepted witnesses: ${accepted}. Hover or focus to highlight the audit table.`}
            testId="cert-summary-pill-accepted"
            onHover={onAccHover}
            onLeave={onAccLeave}
          />
        </div>

        <div role="listitem">
          <MetricPill
            label={PILL_LABELS.identityOnly}
            value={identityOnly.toLocaleString()}
            accent={TOKEN.gray500}
            ariaLabel={`Identity-only entries: ${identityOnly}. Hover or focus to highlight the audit table.`}
            testId="cert-summary-pill-identity-only"
            onHover={onIdHover}
            onLeave={onIdLeave}
          />
        </div>

        <div role="listitem">
          <MetricPill
            label={PILL_LABELS.rejected}
            value={rejected.toLocaleString()}
            accent={TOKEN.coral}
            ariaLabel={`Rejected entries: ${rejected}. Hover or focus to highlight the audit table.`}
            testId="cert-summary-pill-rejected"
            onHover={onRejHover}
            onLeave={onRejLeave}
          />
        </div>

        <div role="listitem">
          <MetricPill
            label={PILL_LABELS.gPtSize}
            value={gPtSize.toLocaleString()}
            accent={TOKEN.einV}
            ariaLabel={`Pointwise group order |G_pt|: ${gPtSize}. Hover or focus to highlight the generator closure panel.`}
            testId="cert-summary-pill-g-pt"
            onHover={onGptHover}
            onLeave={onGptLeave}
          />
        </div>

        <div role="listitem">
          <MetricPill
            label={PILL_LABELS.hSize}
            value={hSize.toLocaleString()}
            accent={TOKEN.einW}
            ariaLabel={`Output-symmetry order |H|: ${hSize}. Hover or focus to highlight the column-action panel.`}
            testId="cert-summary-pill-h"
            onHover={onHHover}
            onLeave={onHLeave}
          />
        </div>

        <div role="listitem">
          <MetricPill
            label={PILL_LABELS.components}
            value={componentsCount.toLocaleString()}
            accent={TOKEN.gray900}
            ariaLabel={`Independent label components: ${componentsCount}. Hover or focus to highlight the label interaction graph.`}
            testId="cert-summary-pill-components"
            onHover={onCompsHover}
            onLeave={onCompsLeave}
          />
        </div>
      </div>
    </div>
  );
}
