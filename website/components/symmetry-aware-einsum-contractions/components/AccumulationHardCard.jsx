import React, { useCallback, useEffect, useRef, useState } from 'react';
import Latex from './Latex.jsx';
import { AnchorLink } from './ExplorerSectionCard.jsx';

/**
 * Teaching card: "Why Accumulation Cost (α) is Hard".
 *
 * Static prose explaining why αₐ (per-component accumulation cost) resists
 * the uniform Burnside treatment that Mₐ (per-component orbit count) enjoys —
 * and points the reader to the Classification Tree below, which routes each
 * component to its cheapest applicable closed form.
 *
 * Symbols follow the canonical page vocabulary set by the Counting
 * Convention band: μ is multiplication cost, α is accumulation cost (with
 * αₐ per-component), Mₐ is the per-component orbit count.
 *
 * The α formula has a hover tooltip (same visual shape as the μ tooltip
 * in MultiplicationCostCard) that makes the contrast explicit: αₐ sees the
 * V/W split, Mₐ doesn't.
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
        <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-400" />
        <span className="text-sm font-semibold">Why <Latex math="\alpha" /> is structure-sensitive</span>
      </div>

      {/* Property 1 — αₐ sees the V/W split. */}
      <div className="text-[12px] leading-5 text-gray-300">
        <div className="font-semibold uppercase tracking-wider text-gray-400 text-[10px] mb-1">
          <Latex math="\alpha_a" /> reads the V/W split
        </div>
        <p>
          Unlike Mₐ, αₐ depends on <span className="italic">how</span> each orbit
          projects onto the free labels Vₐ. One orbit can contribute{' '}
          <span className="font-mono">1</span> (all elements share one output
          bin) or the full orbit size (every element lands in a different bin),
          depending on whether the component&apos;s symmetry is W-only, V-only,
          or crosses the V/W boundary.
        </p>
      </div>

      {/* Property 2 — the generic cost. */}
      <div className="mt-3 border-t border-gray-700 pt-3 text-[12px] leading-5 text-gray-300">
        <div className="font-semibold uppercase tracking-wider text-gray-400 text-[10px] mb-1">
          No universal Burnside shortcut
        </div>
        <p>
          The generic formula <span className="font-mono">Σ_O |π_V(O)|</span> still
          works, but reading it costs <span className="font-mono">O(|Xₐ|·|Gₐ|)</span>{' '}
          — you walk every orbit and measure its V-projection size. Specialized
          regimes (direct-product, singleton, …) short-circuit this when Gₐ has
          recognisable structure on Vₐ and Wₐ.
        </p>
      </div>

      {/* Punchline connecting both. */}
      <div className="mt-3 border-t border-gray-700 pt-3 text-[11.5px] leading-5 text-gray-400">
        <span className="mr-1 font-semibold uppercase tracking-wider text-gray-500">Net</span>
        <span>
          Mₐ cares only about Gₐ. αₐ cares about Gₐ <span className="italic">and</span>{' '}
          the V/W split — so the Classification Tree below exists to find the
          right closed form for each split.
        </span>
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
    <div id="accumulation-cost" className="rounded-xl border border-gray-200 bg-white p-4 scroll-mt-24">
      <h3 className="font-sans text-[15px] font-semibold leading-tight tracking-[-0.01em] text-gray-900">
        <AnchorLink anchorId="accumulation-cost" labelText="Why Accumulation Cost (α) is Hard">
          Why Accumulation Cost <span>(α)</span> is Hard
        </AnchorLink>
      </h3>
      <p className="mt-2 text-sm leading-6 text-foreground">
        Unlike <span className="font-mono">Mₐ</span>, the per-component
        accumulation cost <span className="font-mono">αₐ</span> depends on how
        each orbit <em>projects</em> onto the free labels Vₐ. That projection
        isn&apos;t a simple Burnside sum, and the global α is{' '}
        <span className="font-mono">∏ₐ αₐ</span>.
      </p>

      {/* Hover-wrapped formula — mirrors the μ card so the two cards share
          a visual rhythm. The tooltip explains what makes α different. */}
      <div
        ref={formulaRef}
        className="group relative mt-3 cursor-help rounded-md bg-gray-50 px-3 py-2.5 transition-colors hover:bg-amber-50/60"
        onMouseEnter={openTooltip}
        onMouseLeave={scheduleHide}
        onFocus={openTooltip}
        onBlur={scheduleHide}
        tabIndex={0}
        role="button"
        aria-describedby="acc-formula-intuition"
      >
        <Latex math={String.raw`\alpha_a \;=\; \sum_{O \in X_a/G_a} |\pi_V(O)|`} display />
        <div
          className="pointer-events-none absolute right-2 top-1.5 text-[10px] font-medium uppercase tracking-wider text-amber-600/70 opacity-0 transition-opacity group-hover:opacity-100"
          aria-hidden
        >
          hover for intuition
        </div>
      </div>

      <p className="mt-3 text-sm leading-6 text-foreground">
        The generic formula requires enumerating every orbit and counting
        its distinct V-projection — <span className="font-mono">O(Π n<sub>ℓ</sub> · |G|)</span>{' '}
        work in the worst case.
      </p>
      <p className="mt-2 text-sm leading-6 text-foreground">
        We dodge that cost when the group has a recognizable structure
        (full-symmetric, direct-product, setwise-stable, …). The classification
        tree below routes each component to its cheapest applicable closed form,
        or falls back to brute-force enumeration only when nothing else fits.
      </p>
      <div className="mt-3 border-t border-gray-100 pt-3 text-xs italic text-muted-foreground">
        → See the Classification Tree below.
      </div>

      <FormulaIntuitionTooltip
        anchorRect={anchorRect}
        onDismiss={() => setAnchorRect(null)}
      />
    </div>
  );
}
