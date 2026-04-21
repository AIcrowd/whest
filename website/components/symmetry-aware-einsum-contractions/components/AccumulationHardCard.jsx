import React, { useCallback, useEffect, useRef, useState } from 'react';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';

/**
 * Teaching card: "Why Accumulation Cost (α) is Hard".
 *
 * Static prose explaining why αₐ (per-component accumulation cost) resists
 * the uniform Burnside treatment that Mₐ (per-component orbit count) enjoys —
 * and explains why the Classification Tree below exists to route each
 * component to its cheapest applicable closed form.
 *
 * Symbols follow the canonical page vocabulary set by the Counting
 * Convention band: μ is multiplication cost, α is accumulation cost (with
 * αₐ per-component), Mₐ is the per-component orbit count.
 *
 * The α formula has a hover tooltip (same visual shape as the μ tooltip
 * in MultiplicationCostCard) that makes the contrast explicit: αₐ sees the
 * `V_free / W_summed` split, Mₐ doesn't.
 */

// ─── Intuition tooltip ────────────────────────────────────────────────
//
// Mirrors the FormulaIntuitionTooltip in MultiplicationCostCard: same
// viewport-aware positioning and dismiss logic, different body content.

function FormulaIntuitionTooltip({ anchorRect, onDismiss }) {
  const TOOLTIP_W = 440;
  const TOOLTIP_MIN_H = 340;

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
        <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-400" />
        <span className="text-sm font-semibold">Why <Latex math="\alpha" /> is structure-sensitive</span>
      </div>

      {/* Property 1 — αₐ sees the V_free / W_summed split. */}
      <div className="text-[12px] leading-5 text-stone-700">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
          <InlineMathText>{String.raw`$\alpha_a$ reads the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split`}</InlineMathText>
        </div>
        <p>
          <InlineMathText>{String.raw`Unlike $M_a$, $\alpha_a$ depends on how each orbit projects onto the free labels $V_{\mathrm{free},a}$. One orbit can contribute $1$ or its full orbit size, depending on whether the component's symmetry is only on $W_{\mathrm{summed},a}$, only on $V_{\mathrm{free},a}$, or crosses that boundary.`}</InlineMathText>
        </p>
      </div>

      {/* Property 2 — the generic cost. */}
      <div className="mt-3 border-t border-stone-200 pt-3 text-[12px] leading-5 text-stone-700">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
          No universal Burnside shortcut
        </div>
        <p>
          <InlineMathText>{String.raw`The generic formula $\sum_{O \in X_a/G_a} |\pi_{V_{\mathrm{free}}}(O)|$ still works, but reading it costs $O(|X_a| \cdot |G_a|)$ because you walk every orbit and measure its $V_{\mathrm{free}}$-projection size. Specialized regimes short-circuit this when $G_a$ has recognizable structure on $V_{\mathrm{free},a}$ and $W_{\mathrm{summed},a}$.`}</InlineMathText>
        </p>
      </div>

      {/* Punchline connecting both. */}
      <div className="mt-3 border-t border-stone-200 pt-3 text-[11.5px] leading-5 text-stone-600">
        <span className="mr-1 font-semibold uppercase tracking-wider text-stone-500">Net</span>
        <span><InlineMathText>{String.raw`$M_a$ cares only about $G_a$. $\alpha_a$ cares about $G_a$ and the $V_{\mathrm{free}} / W_{\mathrm{summed}}$ split, so the Classification Tree below exists to find the right closed form for each split.`}</InlineMathText></span>
      </div>
    </div>
  );
}

export default function AccumulationHardCard() {
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
    <div id="accumulation-cost" className="bg-white p-4 scroll-mt-24">
      <ExplorerSubsectionHeader anchorId="accumulation-cost" labelText="Why Accumulation Cost (α) is Hard">
        <InlineMathText>{`Why Accumulation Cost ($${String.raw`\alpha`}$) is Hard`}</InlineMathText>
      </ExplorerSubsectionHeader>
      <p className="explorer-support-prose mt-2">
        <InlineMathText>{String.raw`Unlike $M_a$, the per-component accumulation cost $\alpha_a$ depends on how each orbit projects onto the free labels $V_{\mathrm{free},a}$. That projection is not a simple Burnside sum, and the global total is $\prod_a \alpha_a$.`}</InlineMathText>
      </p>

      {/* Hover-wrapped formula — mirrors the μ card so the two cards share
          a visual rhythm. The tooltip explains what makes α different. */}
      <div
        ref={formulaRef}
        className="group relative mt-3 cursor-help rounded-md bg-white px-3 py-2.5 transition-colors hover:bg-white"
        onMouseEnter={openTooltip}
        onMouseLeave={scheduleHide}
        onFocus={openTooltip}
        onBlur={scheduleHide}
        tabIndex={0}
        role="button"
        aria-describedby="acc-formula-intuition"
      >
        <Latex math={String.raw`\alpha_a \;=\; \sum_{O \in X_a/G_a} |\pi_{V_{\mathrm{free}}}(O)|`} display />
      </div>

      <p className="explorer-support-prose mt-3">
        <InlineMathText>{String.raw`The generic formula requires enumerating every orbit and counting its distinct $V_{\mathrm{free}}$-projection, which is $O\!\left(\prod_{\ell} n_\ell \cdot |G|\right)$ work in the worst case.`}</InlineMathText>
      </p>
      <p className="explorer-support-prose mt-2">
        <InlineMathText>{`We dodge that cost when the group has a recognizable structure. The classification tree below routes each component to its cheapest applicable closed form, or falls back to brute-force enumeration only when nothing else fits.`}</InlineMathText>
      </p>
      <FormulaIntuitionTooltip
        anchorRect={anchorRect}
        onDismiss={() => setAnchorRect(null)}
      />
    </div>
  );
}
