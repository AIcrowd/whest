import { AnchorLink, SectionEyebrow } from './ExplorerSectionCard.jsx';
import InlineMathText from './InlineMathText.jsx';

// Appendix sub-section kicker — same `.sec-kicker` token grammar as
// SectionEyebrow (uppercase 11px / 0.16em / coral) but rendered as the
// label only, without a "Section N · " prefix. Used for the redesigned
// V3.1 letter-grouped modal where eyebrows read "A.1", "A.2", "B", … "E".
const APPENDIX_SUB_KICKER_CLASS = 'text-[11px] font-semibold uppercase tracking-[0.16em] text-coral';

export default function AppendixSection({
  n,
  label,
  title,
  deck = null,
  children,
  className = '',
  contentClassName = '',
  anchorId = '',
  deckClassName = '',
  // When true, render a label-only kicker (no "Section N ·" prefix) and
  // skip the cross-section divider. Used by the V3.1 letter-grouped modal,
  // where each AppendixGroup wrapper supplies its own divider + letter
  // kicker. Default false preserves the legacy behaviour for any other
  // call sites.
  subEyebrow = false,
  // When true, render a compact "passing-disclaimer" chrome: smaller h3
  // (15px font-medium instead of 24px font-semibold), tighter top margin.
  // Used by Appendix E so its h3 doesn't visually compete with the
  // AppendixGroup wrapper's H2 above it. Default false preserves the
  // legacy 24px treatment for every other section.
  compactChrome = false,
  // When true, skip the inner chrome block entirely (no eyebrow, no h3,
  // no deck). The outer <section id={anchorId}> still renders so the
  // anchor target survives for deep links, and the `title` prop can
  // still be passed for tests that pin its presence — it just isn't
  // visually rendered. Used by Appendix E, where the AppendixGroup
  // wrapper already supplies "Appendix E" + "Scope, assumptions, and
  // non-goals" and the inner copy would be a duplicate.
  hideChrome = false,
}) {
  const renderedTitle = typeof title === 'string' ? <InlineMathText>{title}</InlineMathText> : title;
  const eyebrowAriaLabel = typeof label === 'string' && label ? label : 'subsection';

  return (
    <section id={anchorId || undefined} className={['pt-8 first:pt-0', className].join(' ')}>
      {n > 1 && !subEyebrow ? <div className="mb-8 border-t border-gray-100" /> : null}
      {hideChrome ? null : (
        <div className={compactChrome ? 'mb-3' : 'mb-4'}>
          {subEyebrow ? (
            <AnchorLink anchorId={anchorId} labelText={eyebrowAriaLabel}>
              <span className={APPENDIX_SUB_KICKER_CLASS}>{label}</span>
            </AnchorLink>
          ) : (
            <SectionEyebrow n={n} label={label} anchorId={anchorId} />
          )}
          {compactChrome ? (
            <h3 className="mt-1 font-heading text-[15px] font-medium leading-tight text-gray-900">
              {renderedTitle}
            </h3>
          ) : (
            <h3 className="mt-2 font-heading text-[24px] font-semibold leading-tight text-gray-900">
              {renderedTitle}
            </h3>
          )}
          {deck ? (
            <p className={['mt-3 font-serif text-[17px] leading-[1.75] text-gray-700', deckClassName || 'max-w-[70ch]'].filter(Boolean).join(' ')}>
              {deck}
            </p>
          ) : null}
        </div>
      )}
      <div className={contentClassName}>
        {children}
      </div>
    </section>
  );
}
