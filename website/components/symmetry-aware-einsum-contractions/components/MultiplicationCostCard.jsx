import React, { useCallback, useEffect, useRef, useState } from 'react';
import Latex from './Latex.jsx';
import CaseBadge from './CaseBadge.jsx';
import { AnchorLink } from './ExplorerSectionCard.jsx';

/**
 * Teaching card: "Calculating Multiplication Cost (μ)".
 *
 * Canonical vocabulary (established by the Counting Convention band above):
 *   Mₐ — per-component orbit count = |Xₐ/Gₐ|  (computed via size-aware Burnside)
 *   M  — global orbit count = ∏ₐ Mₐ
 *   μ  — Multiplication Cost = (num_terms − 1) · M = (num_terms − 1) · ∏ₐ Mₐ
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
      className="pointer-events-none fixed z-[9999] w-[440px] max-w-[calc(100vw-2rem)] rounded-lg bg-gray-900 px-4 py-3.5 text-white shadow-2xl"
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
        <span className="text-sm font-semibold">Why this formula is universal</span>
      </div>

      {/* Property 1 — structure-agnostic. */}
      <div className="text-[12px] leading-5 text-gray-300">
        <div className="font-semibold uppercase tracking-wider text-gray-400 text-[10px] mb-1">
          Mₐ doesn&apos;t see V vs W
        </div>
        <p>
          The Burnside sum only knows about Gₐ acting on the component&apos;s label
          set Lₐ. It doesn&apos;t read the V/W split at all — so the exact same
          formula applies to every regime: trivial, all-visible, all-summed,
          and every mixed case. Contraction structure never enters Mₐ.
        </p>
      </div>

      {/* Property 2 — (k-1) factor. */}
      <div className="mt-3 border-t border-gray-700 pt-3 text-[12px] leading-5 text-gray-300">
        <div className="font-semibold uppercase tracking-wider text-gray-400 text-[10px] mb-1">
          Where <span className="font-mono normal-case">(num_terms − 1)</span> comes from
        </div>
        <p>
          M = ∏ₐ Mₐ counts global orbits — one product <span className="italic">to compute</span> per
          orbit rep. Each product is a chain of <span className="font-mono">num_terms</span>{' '}
          operand entries (2-operand einsum → chain of 2, 3-operand → chain of 3, etc.),
          which costs <span className="font-mono">num_terms − 1</span> binary multiplies. The
          factor applies once globally, not per component.
        </p>
      </div>

      {/* Punchline connecting both. */}
      <div className="mt-3 border-t border-gray-700 pt-3 text-[11.5px] leading-5 text-gray-400">
        <span className="mr-1 font-semibold uppercase tracking-wider text-gray-500">Net</span>
        <span>
          symmetry shrinks M (fewer distinct products); the einsum&apos;s operand count
          scales M up by{' '}
          <span className="font-mono">num_terms − 1</span> to get μ.
        </span>
      </div>
    </div>
  );
}

export default function MultiplicationCostCard({ components = [] }) {
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
    <div id="multiplication-cost" className="rounded-xl border border-gray-200 bg-white p-4 scroll-mt-24">
      <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">
        <AnchorLink anchorId="multiplication-cost" labelText="Calculating Multiplication Cost (μ)">
          Calculating Multiplication Cost <span className="normal-case">(μ)</span>
        </AnchorLink>
      </div>
      <p className="mt-2 text-sm leading-6 text-foreground">
        Every component gets the same treatment. Size-aware Burnside counts the
        per-component orbit count <span className="font-mono">Mₐ</span> once each;
        the multiplication cost is then
        <span className="font-mono"> μ = (num_terms − 1) · ∏ₐ Mₐ</span> — one
        product per global orbit rep, where forming that product costs
        <span className="font-mono"> num_terms − 1</span> binary multiplies.
        No regime dispatch; one formula covers trivial, all-visible, all-summed,
        and every mixed regime.
      </p>

      {/* Hover-wrapped formula — shows μ on top, the per-component
          Mₐ/Burnside identity below, so the reader sees how each
          per-component count rolls up to the global μ without leaving
          the card. */}
      <div
        ref={formulaRef}
        className="group relative mt-3 cursor-help rounded-md bg-gray-50 px-3 py-2.5 transition-colors hover:bg-blue-50/60"
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
        <div
          className="pointer-events-none absolute right-2 top-1.5 text-[10px] font-medium uppercase tracking-wider text-blue-500/70 opacity-0 transition-opacity group-hover:opacity-100"
          aria-hidden
        >
          hover for intuition
        </div>
      </div>

      {components.length > 0 && (
        <div className="mt-3 space-y-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Live for this example
          </div>
          {components.map((comp, i) => {
            const orbits = comp.multiplicationCount ?? comp.multiplication?.count ?? null;
            return (
              <div
                key={`mult-${i}`}
                className="flex items-center gap-2 text-xs"
              >
                <CaseBadge
                  regimeId={comp.accumulation?.regimeId ?? comp.shape ?? comp.caseType}
                  caseType={comp.caseType}
                  size="xs"
                  variant="pill"
                />
                <span className="truncate text-gray-600">
                  {comp.labels?.join(', ') || '∅'}
                </span>
                <span className="ml-auto font-mono text-gray-900">
                  Mₐ = {orbits != null ? orbits.toLocaleString() : '—'}
                </span>
              </div>
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
