import React, { useCallback, useEffect, useRef, useState } from 'react';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import CaseBadge from './CaseBadge.jsx';
import RoleBadge from './RoleBadge.jsx';
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
        <span className="text-sm font-semibold">Why this formula is universal</span>
      </div>

      {/* Property 1 — structure-agnostic. */}
      <div className="text-[12px] leading-5 text-stone-700">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
          <InlineMathText>{String.raw`$M_a$ doesn't see the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split`}</InlineMathText>
        </div>
        <p>
          <InlineMathText>{String.raw`The Burnside sum only knows about $G_a$ acting on the component's label set $L_a$. It does not read the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split at all, so the same formula applies to every regime: trivial, all-visible, all-summed, and every mixed case. Contraction structure never enters $M_a$.`}</InlineMathText>
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
      <h3 className="font-sans text-[15px] font-semibold leading-tight tracking-[-0.01em] text-gray-900">
        <AnchorLink anchorId="multiplication-cost" labelText="Calculating Multiplication Cost (μ)">
          Calculating Multiplication Cost <span>(<Latex math={String.raw`\mu`} />)</span>
        </AnchorLink>
      </h3>
      <p className="mt-2 text-sm leading-6 text-foreground">
        <InlineMathText>{String.raw`Every component gets the same treatment. Size-aware Burnside counts the per-component orbit count $M_a$ once each; the multiplication cost is then $\mu = (\texttt{num\_terms} - 1)\cdot \prod_a M_a$, one product per global orbit representative, where forming that product costs num_terms - 1 binary multiplies. No regime dispatch is needed because the formula ignores the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split.`}</InlineMathText>
      </p>

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
            return (
              <div
                key={`mult-${i}`}
                className="flex items-center gap-2 text-xs"
              >
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
