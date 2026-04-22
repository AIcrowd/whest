import { SectionEyebrow } from './ExplorerSectionCard.jsx';

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
}) {
  return (
    <section id={anchorId || undefined} className={['pt-8 first:pt-0', className].join(' ')}>
      {n > 1 ? <div className="mb-8 border-t border-gray-100" /> : null}
      <div className="mb-4">
        <SectionEyebrow n={n} label={label} anchorId={anchorId} />
        <h3 className="mt-2 font-heading text-[24px] font-semibold leading-tight text-gray-900">
          {title}
        </h3>
        {deck ? (
          <p className={['mt-3 font-serif text-[17px] leading-[1.75] text-gray-700', deckClassName || 'max-w-[70ch]'].filter(Boolean).join(' ')}>
            {deck}
          </p>
        ) : null}
      </div>
      <div className={contentClassName}>
        {children}
      </div>
    </section>
  );
}
