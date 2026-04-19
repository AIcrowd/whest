import InlineMathText from './InlineMathText.jsx';

// NarrativeCallout — three tones, three distinct patterns from the
// design-system reference (preview/components.html + Whest Einsum
// Explorer.html). The previous implementation wrapped all three in
// the same shadcn Card shell (ExplorerSectionCard: ring-1 + border +
// bg + CardHeader), which made short editorial observations read as
// UI panels rather than as part of the essay.
//
//   muted      → `.callout` default — light gray-50 callout box
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
    return (
      <div className="mt-5 flex flex-col gap-3 border-t border-dashed border-gray-200 pt-5 sm:flex-row sm:items-start sm:gap-5">
        {label ? (
          <div className="shrink-0 pt-1 font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400 sm:min-w-[130px]">
            {label}
          </div>
        ) : null}
        <p className="m-0 font-serif text-[17px] leading-[1.75] text-gray-700">
          <InlineMathText>{children}</InlineMathText>
        </p>
      </div>
    );
  }

  const isCoral = tone === 'algorithm';

  // `.callout` box — 1px hairline, 11.2px radius, gray-50 (or coral-
  // light), quiet inline kicker. Height stretches in its grid row so
  // sibling callouts line up; the body floats at the top (no more
  // vertical centering which made short paragraphs feel adrift inside
  // oversized cards).
  return (
    <div
      className={[
        'h-full rounded-[var(--radius-xl)] border px-5 py-4',
        isCoral
          ? 'border-[color:color-mix(in_oklab,var(--coral)_25%,transparent)] bg-[var(--coral-light)]'
          : 'border-gray-200 bg-gray-50',
      ].join(' ')}
    >
      {label ? (
        <div
          className={[
            'mb-2 font-sans text-[10px] font-semibold uppercase tracking-[0.2em]',
            isCoral ? 'text-coral' : 'text-gray-400',
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
