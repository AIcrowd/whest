import InlineMathText from './InlineMathText.jsx';
import { AnchorLink } from './ExplorerSectionCard.jsx';

/**
 * Letter-level wrapper for the Appendix modal. Renders a flush horizontal
 * divider, a wide-set "Appendix X" kicker (uppercase, letterspaced, coral —
 * the same `.sec-kicker` token grammar SectionEyebrow uses on the main page),
 * and a one-line title sourced from V3.1's APPENDIX_MAP.
 *
 * The outer `id="appendix-letter-{a..e}"` is the per-letter deep-link
 * target — kept stable so future surfaces (a TOC sidebar, a different
 * sticky nav, etc.) can address each letter without revisiting this file.
 *
 * Inner `<AppendixSection>` children keep their own `appendix-section-{1..8}`
 * anchors so the main page's `openAppendix(hash)` deep links continue to
 * land on the per-section content as before.
 */
const KICKER_CLASS =
  'text-[11px] font-semibold uppercase tracking-[0.16em] text-coral';

export default function AppendixGroup({
  letter,
  title,
  anchorId,
  children,
  className = '',
}) {
  return (
    <section
      id={anchorId || undefined}
      className={['pt-12 first:pt-2', className].join(' ')}
    >
      <div className="mb-6 border-t border-gray-200" />
      <div className="mb-4">
        <AnchorLink anchorId={anchorId} labelText={`Appendix ${letter}`}>
          <span className={KICKER_CLASS}>
            <span
              aria-hidden
              className="mr-2 inline-block h-px w-8 align-middle bg-gray-300"
            />
            Appendix {letter}
          </span>
        </AnchorLink>
        <h2 className="mt-3 font-heading text-[28px] font-semibold leading-tight text-gray-900">
          <InlineMathText>{title}</InlineMathText>
        </h2>
      </div>
      <div className="space-y-10">{children}</div>
    </section>
  );
}
