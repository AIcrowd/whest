import InlineMathText from './InlineMathText.jsx';

// NarrativeCallout — three tones, three distinct patterns from the
// design-system reference (preview/components.html + Whest Einsum
// Explorer.html). The previous implementation wrapped all three in
// the same shadcn Card shell (ExplorerSectionCard: ring-1 + border +
// bg + CardHeader), which made short editorial observations read as
// UI panels rather than as part of the essay.
//
//   muted      → `.callout` default — warm editorial-accent framed/tinted box
//   algorithm  → `.callout--accent` — coral-light callout box
//                (the Approach side of an Interpretation/Approach pair,
//                per the reference template's §2/§3 markup)
//   accent     → `.produces` capstone — NO box, just a dashed top
//                border + inline uppercase label + wrapping prose.
//                Used for the "What this produces" paragraph that
//                lives under a visualization as its sign-off.
//
// Body prose stays in the paper register (Source Serif 4 17/1.75) so
// the observations read as part of the surrounding essay, not as
// dense-UI captions inside a panel.

export default function NarrativeCallout({ label, tone = 'muted', children }) {
  if (tone === 'accent') {
    // `.produces` capstone — dashed top divider, inline kicker on the
    // left, paragraph wrapping to the right. No box, no bg.
    //
    // Important: the kicker and the sentence use very different font
    // metrics (10px uppercase sans vs 17px serif). Aligning the parent
    // row is not enough; each text run needs its own equal-height lane so
    // the visible center of the glyphs is centered, not just the DOM box.
    return (
      <div className="mt-5 flex flex-col gap-3 border-t border-dashed border-gray-200 pt-5 sm:grid sm:grid-cols-[180px_minmax(0,1fr)] sm:items-start sm:gap-x-5 sm:gap-y-3">
        {label ? (
          <div className="shrink-0">
            <span className="block whitespace-nowrap font-sans text-[10px] font-semibold uppercase leading-none tracking-[0.2em] text-gray-400">
              {label}
            </span>
          </div>
        ) : null}
        <p className="m-0 font-serif text-[17px] leading-[1.45] text-gray-700">
          <InlineMathText>{children}</InlineMathText>
        </p>
      </div>
    );
  }

  const isCoral = tone === 'algorithm';

  // `.callout` box — 1px hairline, 11.2px radius, with non-coral
  // tones framed/tinted in warm editorial-accent hues and coral tones
  // staying coral-light. Height stretches in its grid row so sibling
  // callouts line up; the body floats at the top (no more vertical
  // centering which made short paragraphs feel adrift inside oversized
  // cards).
  return (
    <div
      className={[
        'h-full rounded-[var(--radius-xl)] border px-5 py-4',
        isCoral
          ? 'border-[color:color-mix(in_oklab,var(--coral)_25%,transparent)] bg-[var(--coral-light)]'
          : 'border-[color:color-mix(in_oklab,var(--explorer-editorial-accent)_28%,var(--explorer-border))] bg-[color:color-mix(in_oklab,var(--explorer-editorial-accent)_10%,var(--explorer-surface))]',
      ].join(' ')}
    >
      {label ? (
        <div
          className={[
            'mb-2 font-sans text-[10px] font-semibold uppercase tracking-[0.2em]',
            isCoral ? 'text-coral' : 'text-[var(--explorer-editorial-accent)]',
          ].join(' ')}
        >
          {label}
        </div>
      ) : null}
      <p className="m-0 font-serif text-[17px] leading-[1.75] text-gray-700">
        <InlineMathText>{children}</InlineMathText>
      </p>
    </div>
  );
}
