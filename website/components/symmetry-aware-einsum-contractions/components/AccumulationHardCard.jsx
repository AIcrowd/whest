import React from 'react';
import Latex from './Latex.jsx';

/**
 * Teaching card: "Why Accumulation Cost is Hard".
 *
 * Static prose explaining why A (accumulation count) resists the uniform
 * Burnside treatment that M enjoys — and points the reader to the
 * Classification Tree below, which routes each component to its cheapest
 * applicable closed form.
 */
export default function AccumulationHardCard() {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground">
        Why Accumulation Cost is Hard
      </div>
      <p className="mt-2 text-sm leading-6 text-foreground">
        Unlike <span className="font-mono">M</span>, the accumulation count{' '}
        <span className="font-mono">A</span> depends on how each orbit{' '}
        <em>projects</em> onto the free labels V. That projection isn&apos;t a
        simple Burnside sum.
      </p>
      <div className="mt-3 rounded-md bg-gray-50 px-3 py-2">
        <Latex math={String.raw`A \;=\; \sum_{O \in X/G} |\pi_V(O)|`} display />
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
    </div>
  );
}
